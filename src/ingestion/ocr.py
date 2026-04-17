# src/ingestion/ocr.py
"""
OCR engine + table extraction.

Handles:
- Scanned PDFs (page → image → Tesseract)
- Standalone image files
- Table extraction from PDFs via pdfplumber
- Arabic + English (Tesseract with ara+eng lang pack)
- Low-quality / skewed scans (preprocessing with OpenCV when available)
- Graceful fallback when OpenCV is absent
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from .loaders import RawDocument, _clean_text, _detect_language_hint

logger = logging.getLogger('uvicorn.error')


# ─────────────────────────────────────────────
# Image pre-processing helpers
# ─────────────────────────────────────────────
def _preprocess_image(image):
    """
    Improve OCR accuracy:
    1. Convert to grayscale
    2. Denoise
    3. Threshold (binarize) — handles uneven lighting
    4. Deskew if heavily rotated

    Falls back gracefully if OpenCV is not installed.
    Returns PIL Image.
    """
    try:
        import cv2
        import numpy as np

        img_array = np.array(image)

        # Handle RGBA → RGB
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Denoise (light denoising to keep text crisp)
        gray = cv2.fastNlMeansDenoising(gray, h=10)

        # Adaptive threshold — handles shadows and uneven lighting
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=10,
        )

        # Deskew
        coords = np.column_stack(np.where(binary < 128))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if abs(angle) < 45:
                h, w = binary.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                binary = cv2.warpAffine(binary, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)

        from PIL import Image
        return Image.fromarray(binary)

    except ImportError:
        # OpenCV not available — return as-is
        return image
    except Exception as exc:
        logger.debug(f"[OCR] Preprocessing warning (non-fatal): {exc}")
        return image


# ─────────────────────────────────────────────
# OCR Engine
# ─────────────────────────────────────────────
class OCREngine:
    """
    Tesseract-based OCR supporting Arabic + English.

    Constructor parameters
    ----------------------
    lang        : Tesseract language string. 'ara+eng' covers bilingual docs.
    dpi         : Resolution for PDF→image conversion (300 is standard for OCR).
    min_confidence: Discard text blocks below this Tesseract confidence score.
    """

    def __init__(
        self,
        lang: str = "ara+eng",
        dpi:  int = 300,
        min_confidence: int = 40,
    ):
        self.lang           = lang
        self.dpi            = dpi
        self.min_confidence = min_confidence
        self._verify_tesseract()

    # ── setup ─────────────────────────────────
    def _verify_tesseract(self) -> None:
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            logger.info(f"[OCR] Tesseract version: {version}")
        except Exception:
            logger.warning(
                "[OCR] Tesseract not found or not installed. "
                "OCR calls will return empty text. "
                "Install: sudo apt-get install tesseract-ocr tesseract-ocr-ara"
            )

    # ── core OCR on a single PIL image ────────
    def _ocr_image(self, image) -> str:
        try:
            import pytesseract

            processed = _preprocess_image(image)

            # Get per-word confidence data to filter noise
            data = pytesseract.image_to_data(
                processed,
                lang=self.lang,
                config="--oem 3 --psm 3",   # LSTM engine, fully automatic page segmentation
                output_type=pytesseract.Output.DICT,
            )

            words = []
            for i, word in enumerate(data["text"]):
                conf = int(data["conf"][i])
                if conf >= self.min_confidence and word.strip():
                    words.append(word)

            return " ".join(words)

        except ImportError:
            logger.error("[OCR] pytesseract not installed. Run: pip install pytesseract")
            return ""
        except Exception as exc:
            logger.error(f"[OCR] OCR failed: {exc}")
            return ""

    # ── OCR a scanned PDF ─────────────────────
    def ocr_pdf_page(self, pdf_path: str | Path, page_num: int) -> str:
        """
        Convert a single PDF page to an image then OCR it.
        Uses PyMuPDF for high-quality rasterization.
        """
        try:
            import fitz
            from PIL import Image

            pdf = fitz.open(str(pdf_path))
            page = pdf[page_num]

            # Render at target DPI (72 is PDF default; scale accordingly)
            zoom   = self.dpi / 72
            matrix = fitz.Matrix(zoom, zoom)
            pix    = page.get_pixmap(matrix=matrix, alpha=False)

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pdf.close()

            text = self._ocr_image(img)
            return _clean_text(text)

        except Exception as exc:
            logger.error(f"[OCR] PDF page OCR failed (page {page_num}): {exc}")
            return ""

    # ── OCR a standalone image file ───────────
    def ocr_image_file(self, image_path: str | Path) -> str:
        try:
            from PIL import Image
            img  = Image.open(str(image_path))
            text = self._ocr_image(img)
            return _clean_text(text)
        except Exception as exc:
            logger.error(f"[OCR] Image OCR failed ({image_path}): {exc}")
            return ""

    # ── Process a list of RawDocuments ────────
    def process_documents(
        self,
        documents: List[RawDocument],
        source_path: Optional[str] = None,
    ) -> List[RawDocument]:
        """
        For each document that needs OCR, fill in its text field.
        source_path is needed when the doc came from a PDF page.
        
        Handles:
        - Scanned PDF pages (full page OCR)
        - Embedded images within PDF pages (extract + OCR)
        - Embedded images in Word documents (OCR from image_data)
        - Embedded images in PowerPoint slides (OCR from image_data)
        - Standalone image files
        """
        results: List[RawDocument] = []

        for doc in documents:
            if not doc.needs_ocr:
                results.append(doc)
                continue

            src_type = doc.metadata.get("source_type", "")
            page_num = doc.metadata.get("page", 1) - 1  # 0-indexed for fitz

            if src_type == "pdf" and source_path:
                # Full page is scanned image
                ocr_text = self.ocr_pdf_page(source_path, page_num)
            elif src_type == "pdf_embedded_image" and source_path:
                # Embedded image within a PDF page
                ocr_text = self._ocr_embedded_pdf_image(source_path, doc.metadata)
            elif src_type in ("docx_embedded_image", "pptx_embedded_image"):
                # Embedded image in Word or PowerPoint (image data in metadata)
                ocr_text = self._ocr_from_image_data(doc.metadata)
            elif src_type == "image":
                # Standalone image file
                path = doc.metadata.get("source_path", "")
                ocr_text = self.ocr_image_file(path) if path else ""
            else:
                logger.warning(f"[OCR] Cannot OCR doc of type '{src_type}' without source_path")
                ocr_text = ""

            doc.text      = ocr_text
            doc.needs_ocr = False
            doc.metadata["ocr_applied"]  = True
            doc.metadata["language"]     = _detect_language_hint(ocr_text)
            
            # Clean up image_data from metadata (no longer needed, saves memory)
            if "image_data" in doc.metadata:
                del doc.metadata["image_data"]

            results.append(doc)

        return results

    # ── OCR embedded image from PDF ───────────
    def _ocr_embedded_pdf_image(self, pdf_path: str | Path, metadata: dict) -> str:
        """
        Extract and OCR a specific embedded image from a PDF page.
        """
        try:
            import fitz
            from PIL import Image
            import io

            pdf = fitz.open(str(pdf_path))
            page_num = metadata.get("page", 1) - 1
            xref = metadata.get("image_xref")

            if xref is None:
                logger.warning("[OCR] No image_xref in metadata for embedded image")
                pdf.close()
                return ""

            # Extract the image
            base_image = pdf.extract_image(xref)
            if not base_image:
                pdf.close()
                return ""

            # Convert to PIL Image
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes))

            pdf.close()

            # OCR the image
            text = self._ocr_image(img)
            return _clean_text(text)

        except Exception as exc:
            logger.error(f"[OCR] Embedded image OCR failed: {exc}")
            return ""
    
    # ── OCR from image data (Word/PowerPoint) ─
    def _ocr_from_image_data(self, metadata: dict) -> str:
        """
        OCR an image from raw image data stored in metadata.
        Used for Word and PowerPoint embedded images.
        """
        try:
            from PIL import Image
            import io
            
            image_data = metadata.get("image_data")
            if not image_data:
                logger.warning("[OCR] No image_data in metadata")
                return ""
            
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(image_data))
            
            # OCR the image
            text = self._ocr_image(img)
            return _clean_text(text)
            
        except Exception as exc:
            logger.error(f"[OCR] Image data OCR failed: {exc}")
            return ""


# ─────────────────────────────────────────────
# Table Extractor
# ─────────────────────────────────────────────
class TableExtractor:
    """
    Extracts tables from PDFs using pdfplumber.
    Converts each table to a structured text block that the LLM can read.

    Output format (per table):
    ──────────────
    [TABLE: page N, table M]
    Header1 | Header2 | Header3
    val1    | val2    | val3
    ...
    ──────────────
    """

    def extract_from_pdf(self, pdf_path: str | Path) -> List[RawDocument]:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber not installed. Run: pip install pdfplumber")

        path = Path(pdf_path)
        docs: List[RawDocument] = []

        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    if not tables:
                        continue

                    for table_idx, table in enumerate(tables, start=1):
                        if not table:
                            continue

                        rows = []
                        for row in table:
                            # Replace None cells with empty string
                            clean_row = [str(cell).strip() if cell else "" for cell in row]
                            # Skip completely empty rows
                            if any(clean_row):
                                rows.append(" | ".join(clean_row))

                        if not rows:
                            continue

                        text = (
                            f"[TABLE: page {page_num}, table {table_idx}]\n"
                            + "\n".join(rows)
                        )
                        text = _clean_text(text)

                        docs.append(RawDocument(
                            text=text,
                            metadata={
                                "source":       path.name,
                                "source_path":  str(path),
                                "source_type":  "table",
                                "page":         page_num,
                                "table_index":  table_idx,
                                "language":     _detect_language_hint(text),
                            }
                        ))

        except Exception as exc:
            logger.error(f"[TableExtractor] Failed on {path.name}: {exc}")

        logger.info(f"[TableExtractor] {path.name}: {len(docs)} tables extracted")
        return docs