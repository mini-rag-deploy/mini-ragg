# src/ingestion/hybrid_chunker.py
"""
Hybrid chunker that creates context-aware chunks by merging image OCR text
back into the original document text at appropriate positions.

This solves the problem where images and text are processed separately,
losing their contextual relationship during retrieval.

Key features:
- Merges image OCR text back into original text context
- Creates hybrid chunks: [Text paragraph] [Image caption/description] [Nearby text]
- Maintains document structure and context
- Handles multiple images per page/document
- Preserves metadata for both text and image sources
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .chunker import ContextAwareChunker, DocumentChunk
from .loaders import RawDocument

logger = logging.getLogger('uvicorn.error')


@dataclass
class ImageContext:
    """Represents an image and its extracted text within a document."""
    image_text: str
    image_index: int
    page_number: int
    source_type: str  # pdf_embedded_image, docx_embedded_image, etc.
    metadata: dict = field(default_factory=dict)


@dataclass
class HybridDocument:
    """A document with text and associated images merged contextually."""
    text: str
    images: List[ImageContext]
    metadata: dict = field(default_factory=dict)


class HybridChunker(ContextAwareChunker):
    """
    Extends ContextAwareChunker to create hybrid chunks that maintain
    the relationship between text and images.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 60,
        deduplicate: bool = True,
        image_placeholder_pattern: str = r"\[IMAGE_(\d+)\]",
    ):
        super().__init__(chunk_size, chunk_overlap, min_chunk_size, deduplicate)
        self.image_placeholder_pattern = image_placeholder_pattern

    def create_hybrid_documents(self, documents: List[RawDocument]) -> List[HybridDocument]:
        """
        Group documents by source and page, merging text with associated images.
        
        Returns a list of HybridDocument objects where each represents a logical
        document unit (page/slide/document) with text and images combined.
        """
        # Group documents by source and page
        doc_groups: Dict[Tuple[str, int], Dict[str, List[RawDocument]]] = {}
        
        for doc in documents:
            source = doc.metadata.get("source_path", "")
            page = doc.metadata.get("page", 1)
            source_type = doc.metadata.get("source_type", "")
            
            key = (source, page)
            if key not in doc_groups:
                doc_groups[key] = {"text": [], "images": []}
            
            if source_type.endswith("_embedded_image"):
                doc_groups[key]["images"].append(doc)
            else:
                doc_groups[key]["text"].append(doc)
        
        hybrid_docs: List[HybridDocument] = []
        
        for (source, page), group in doc_groups.items():
            text_docs = group["text"]
            image_docs = group["images"]
            
            # Combine text from all text documents for this source/page
            combined_text = ""
            base_metadata = {}
            
            for text_doc in text_docs:
                if text_doc.text.strip():
                    combined_text += text_doc.text + "\n\n"
                base_metadata.update(text_doc.metadata)
            
            # Create image contexts
            image_contexts: List[ImageContext] = []
            for img_doc in image_docs:
                if img_doc.text.strip():  # Only include images with OCR text
                    image_contexts.append(ImageContext(
                        image_text=img_doc.text,
                        image_index=img_doc.metadata.get("image_index", 0),
                        page_number=page,
                        source_type=img_doc.metadata.get("source_type", ""),
                        metadata=img_doc.metadata
                    ))
            
            # Create hybrid document even if no images (maintains consistency)
            hybrid_doc = HybridDocument(
                text=combined_text.strip(),
                images=image_contexts,
                metadata=base_metadata
            )
            
            if hybrid_doc.text or hybrid_doc.images:
                hybrid_docs.append(hybrid_doc)
        
        logger.info(f"[HybridChunker] Created {len(hybrid_docs)} hybrid documents from {len(documents)} raw documents")
        return hybrid_docs

    def chunk_hybrid_documents(self, hybrid_docs: List[HybridDocument]) -> List[DocumentChunk]:
        """
        Create chunks from hybrid documents, inserting image text at appropriate positions.
        """
        self.reset_dedup()
        all_chunks: List[DocumentChunk] = []
        
        for hybrid_doc in hybrid_docs:
            chunks = self._chunk_single_hybrid_document(hybrid_doc)
            all_chunks.extend(chunks)
        
        logger.info(f"[HybridChunker] Created {len(all_chunks)} hybrid chunks from {len(hybrid_docs)} hybrid documents")
        return all_chunks

    def _chunk_single_hybrid_document(self, hybrid_doc: HybridDocument) -> List[DocumentChunk]:
        """
        Chunk a single hybrid document, strategically placing image text
        to maintain context.
        """
        if not hybrid_doc.text and not hybrid_doc.images:
            return []
        
        # If no images, use standard chunking
        if not hybrid_doc.images:
            return self._chunk_text_only(hybrid_doc)
        
        # If no text, create chunks from images only
        if not hybrid_doc.text.strip():
            return self._chunk_images_only(hybrid_doc)
        
        # Merge images into text context
        enhanced_text = self._merge_images_into_text(hybrid_doc)
        
        # Create a temporary RawDocument for chunking
        temp_doc = RawDocument(
            text=enhanced_text,
            metadata=hybrid_doc.metadata
        )
        
        # Use parent chunking logic
        base_chunks = self.chunk_document(temp_doc)
        
        # Enhance chunks with hybrid metadata
        enhanced_chunks = []
        for chunk in base_chunks:
            # Add hybrid-specific metadata
            chunk.metadata["is_hybrid"] = True
            chunk.metadata["has_images"] = len(hybrid_doc.images) > 0
            chunk.metadata["image_count"] = len(hybrid_doc.images)
            
            # Track which images are referenced in this chunk
            referenced_images = self._find_referenced_images(chunk.chunk_text, hybrid_doc.images)
            if referenced_images:
                chunk.metadata["referenced_images"] = [
                    {
                        "image_index": img.image_index,
                        "source_type": img.source_type,
                        "text_preview": img.image_text[:100] + "..." if len(img.image_text) > 100 else img.image_text
                    }
                    for img in referenced_images
                ]
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks

    def _merge_images_into_text(self, hybrid_doc: HybridDocument) -> str:
        """
        Strategically insert image text into the document text to maintain context.
        
        Improved Strategy:
        1. Split text into logical sections (by sentences or lines)
        2. Distribute images evenly throughout the text
        3. Place images at natural break points (after sentences/paragraphs)
        4. Format: [Text section] [Image: description] [More text]
        """
        text = hybrid_doc.text
        images = sorted(hybrid_doc.images, key=lambda x: x.image_index)
        
        if not images:
            return text
        
        # Try multiple splitting strategies to find natural break points
        sections = self._split_text_into_sections(text)
        
        if len(sections) <= 1:
            # If we can't split meaningfully, distribute images by character position
            return self._distribute_images_by_position(text, images)
        
        # Distribute images evenly among sections
        enhanced_sections = []
        total_sections = len(sections)
        total_images = len(images)
        
        # Calculate insertion points - distribute images evenly
        if total_images >= total_sections:
            # More images than sections - insert multiple images per section
            images_per_section = total_images // total_sections
            remaining_images = total_images % total_sections
        else:
            # More sections than images - insert one image every few sections
            sections_per_image = total_sections // total_images
            images_per_section = 1
            remaining_images = 0
        
        image_idx = 0
        
        for i, section in enumerate(sections):
            enhanced_sections.append(section)
            
            # Determine if we should insert an image after this section
            should_insert_image = False
            
            if total_images >= total_sections:
                # Insert images regularly, with some sections getting extra images
                images_to_insert = images_per_section
                if i < remaining_images:
                    images_to_insert += 1
                
                for _ in range(images_to_insert):
                    if image_idx < len(images):
                        image = images[image_idx]
                        image_text = self._format_image_text(image)
                        enhanced_sections.append(image_text)
                        image_idx += 1
            else:
                # Insert one image every few sections
                section_position = i + 1
                insert_at_positions = []
                
                # Calculate evenly spaced positions
                for img_num in range(total_images):
                    position = ((img_num + 1) * total_sections) // (total_images + 1)
                    insert_at_positions.append(position)
                
                if section_position in insert_at_positions and image_idx < len(images):
                    image = images[image_idx]
                    image_text = self._format_image_text(image)
                    enhanced_sections.append(image_text)
                    image_idx += 1
        
        # Insert any remaining images at the end (fallback)
        while image_idx < len(images):
            image = images[image_idx]
            image_text = self._format_image_text(image)
            enhanced_sections.append(image_text)
            image_idx += 1
        
        return '\n\n'.join(enhanced_sections)

    def _split_text_into_sections(self, text: str) -> List[str]:
        """
        Split text into logical sections using multiple strategies.
        Returns the best split found, prioritizing natural boundaries.
        """
        # Strategy 1: Split by double newlines (paragraphs) - most natural
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            return paragraphs
        
        # Strategy 2: Split by single newlines (lines) - preserve line structure
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) > 3:  # Only use if we have enough lines
            return lines
        
        # Strategy 3: Split by sentences (periods followed by space/newline) - preserve sentence boundaries
        import re
        # Enhanced sentence splitting that handles more punctuation and languages
        sentences = re.split(r'(?<=[.!?؟،\u06D4])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 2:
            return sentences
        
        # Strategy 4: Smart length-based splitting (improved fallback)
        if len(text) > 500:  # Increased threshold for better chunks
            # Try to find natural break points in order of preference
            break_points = []
            
            # Look for paragraph breaks first (double newlines)
            for match in re.finditer(r'\n\n', text):
                break_points.append(match.start())
            
            # Then single newlines
            if not break_points:
                for match in re.finditer(r'\n', text):
                    break_points.append(match.start())
            
            # Then sentence endings
            if not break_points:
                for match in re.finditer(r'[.!?؟،\u06D4]\s+', text):
                    break_points.append(match.end())
            
            # Then word boundaries (spaces)
            if not break_points:
                for match in re.finditer(r'\s+', text):
                    break_points.append(match.start())
            
            if break_points:
                # Find the break point closest to the middle
                mid_point = len(text) // 2
                best_break = min(break_points, key=lambda x: abs(x - mid_point))
                
                # Make sure we don't create tiny chunks
                if best_break > 100 and (len(text) - best_break) > 100:
                    part1 = text[:best_break].strip()
                    part2 = text[best_break:].strip()
                    if part1 and part2:
                        return [part1, part2]
        
        # Fallback: return as single section
        return [text]

    def _distribute_images_by_position(self, text: str, images: List[ImageContext]) -> str:
        """
        Distribute images by character position when we can't find natural sections.
        """
        if not images:
            return text
        
        # Calculate positions to insert images
        text_length = len(text)
        num_images = len(images)
        
        # Create insertion points evenly distributed through the text
        insertion_points = []
        for i in range(num_images):
            # Position each image at (i+1)/(num_images+1) through the text
            position = int(((i + 1) * text_length) / (num_images + 1))
            
            # Find a good break point near this position (space, newline, or punctuation)
            best_position = position
            search_range = min(50, text_length // 10)  # Search within 50 chars or 10% of text
            
            for offset in range(search_range):
                # Check positions before and after the target
                for pos in [position - offset, position + offset]:
                    if 0 <= pos < text_length and text[pos] in ' \n.!?':
                        best_position = pos
                        break
                if best_position != position:
                    break
            
            insertion_points.append(best_position)
        
        # Sort insertion points in reverse order so we can insert without affecting positions
        insertion_data = list(zip(insertion_points, images))
        insertion_data.sort(reverse=True, key=lambda x: x[0])
        
        # Insert images at calculated positions
        result_text = text
        for position, image in insertion_data:
            image_text = self._format_image_text(image)
            # Insert with proper spacing
            result_text = (result_text[:position] + 
                          '\n\n' + image_text + '\n\n' + 
                          result_text[position:])
        
        return result_text

    def _append_images_to_text(self, text: str, images: List[ImageContext]) -> str:
        """Append images to text when no clear structure is found."""
        parts = [text] if text.strip() else []
        
        for image in images:
            image_text = self._format_image_text(image)
            parts.append(image_text)
        
        return '\n\n'.join(parts)

    def _format_image_text(self, image: ImageContext) -> str:
        """Format image text for insertion into document."""
        # Clean and truncate image text if too long
        clean_text = image.image_text.strip()
        if len(clean_text) > 300:
            clean_text = clean_text[:297] + "..."
        
        # Format with clear image marker
        return f"[Image {image.image_index}: {clean_text}]"

    def _find_referenced_images(self, chunk_text: str, images: List[ImageContext]) -> List[ImageContext]:
        """Find which images are referenced in a chunk."""
        referenced = []
        for image in images:
            # Check if image marker is in chunk
            image_marker = f"[Image {image.image_index}:"
            if image_marker in chunk_text:
                referenced.append(image)
        return referenced

    def _chunk_text_only(self, hybrid_doc: HybridDocument) -> List[DocumentChunk]:
        """Chunk document with text only (no images)."""
        temp_doc = RawDocument(
            text=hybrid_doc.text,
            metadata=hybrid_doc.metadata
        )
        chunks = self.chunk_document(temp_doc)
        print("done chunkdocument")
        
        # Mark as hybrid for consistency
        for chunk in chunks:
            chunk.metadata["is_hybrid"] = True
            chunk.metadata["has_images"] = False
            chunk.metadata["image_count"] = 0
        
        return chunks

    def _chunk_images_only(self, hybrid_doc: HybridDocument) -> List[DocumentChunk]:
        """Create chunks from images only (no text)."""
        chunks = []
        
        for i, image in enumerate(hybrid_doc.images):
            if not image.image_text.strip():
                continue
            
            # Create chunk from image text
            chunk_text = self._format_image_text(image)
            chunk_hash = self._make_hash(chunk_text)
            
            if self.deduplicate and chunk_hash in self._seen_hashes:
                continue
            
            if self.deduplicate:
                self._seen_hashes.add(chunk_hash)
            
            chunk_meta = {
                **hybrid_doc.metadata,
                **image.metadata,
                "chunk_index": i,
                "chunk_size": len(chunk_text),
                "is_hybrid": True,
                "has_images": True,
                "image_count": 1,
                "is_image_only": True,
                "referenced_images": [{
                    "image_index": image.image_index,
                    "source_type": image.source_type,
                    "text_preview": image.image_text[:100] + "..." if len(image.image_text) > 100 else image.image_text
                }]
            }
            
            chunk = DocumentChunk(
                chunk_text=chunk_text,
                chunk_index=i,
                chunk_hash=chunk_hash,
                chunk_context="",
                metadata=chunk_meta
            )
            
            chunks.append(chunk)
        
        return chunks

    def _make_hash(self, text: str) -> str:
        """Create hash for deduplication."""
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# Convenience function for easy integration
def create_hybrid_chunks(documents: List[RawDocument], **chunker_kwargs) -> List[DocumentChunk]:
    """
    Convenience function to create hybrid chunks from raw documents.
    
    Args:
        documents: List of RawDocument objects (mix of text and image documents)
        **chunker_kwargs: Additional arguments for HybridChunker
    
    Returns:
        List of DocumentChunk objects with hybrid text-image content
    """
    chunker = HybridChunker(**chunker_kwargs)
    hybrid_docs = chunker.create_hybrid_documents(documents)
    return chunker.chunk_hybrid_documents(hybrid_docs)