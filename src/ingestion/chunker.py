# src/ingestion/chunker.py
"""
Context-aware document chunker.

Problems solved vs naive fixed-size chunking:
- Never splits mid-sentence or mid-word
- Respects Arabic right-to-left sentence boundaries
- Preserves structural cues (headings, article numbers, list items)
- Carries rich metadata into every chunk
- Handles documents that are shorter than chunk_size (no empty chunks)
- Deduplicates chunks from the same source (repeated headers, footers)
- Adds chunk_context: the heading/section title the chunk belongs to
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .loaders import RawDocument, _detect_language_hint

logger = logging.getLogger(uvicorn.error)


# ─────────────────────────────────────────────
# Data contract
# ─────────────────────────────────────────────
@dataclass
class DocumentChunk:
    chunk_text:    str
    chunk_index:   int
    chunk_hash:    str                     # SHA-256 of text — for dedup
    chunk_context: str = ""               # section heading this chunk belongs to
    metadata:      dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.chunk_text or not self.chunk_text.strip()


# ─────────────────────────────────────────────
# Sentence splitters (language-aware)
# ─────────────────────────────────────────────
# Arabic sentence enders: full stop, Arabic comma variations, question mark, exclamation
_ARABIC_SENTENCE_END = re.compile(r"(?<=[.!?؟،\u06D4])\s+")

# English sentence ender
_ENGLISH_SENTENCE_END = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+"
)

# Structural cues that should never be split from the content that follows
_HEADING_PATTERN = re.compile(
    r"^(?:"
    r"(?:Article|Section|Chapter|Clause|مادة|باب|فصل|بند)\s*\d+"
    r"|(?:\d+\.)+\s+"          # numbered headings like "1.2.3 "
    r"|[A-Z][A-Z\s]{2,}$"      # ALL-CAPS headings
    r")",
    re.MULTILINE | re.IGNORECASE,
)


def _split_sentences(text: str, language: str) -> List[str]:
    """Split text into sentences respecting language."""
    if language == "arabic":
        parts = _ARABIC_SENTENCE_END.split(text)
    elif language == "english":
        parts = _ENGLISH_SENTENCE_END.split(text)
    else:
        # Mixed: use both patterns
        parts = re.split(r"(?<=[.!?؟،])\s+", text)

    return [p.strip() for p in parts if p.strip()]


def _extract_section_heading(text: str) -> str:
    """
    Return the first structural heading found in the text, or empty string.
    Used to populate chunk_context.
    """
    match = _HEADING_PATTERN.search(text)
    if match:
        # Return just the heading line (up to first newline)
        heading = text[match.start():].split("\n")[0].strip()
        return heading[:120]  # cap length
    return ""


def _make_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────
class ContextAwareChunker:
    """
    Parameters
    ----------
    chunk_size    : target character count per chunk (not a hard limit)
    chunk_overlap : characters of overlap between consecutive chunks
    min_chunk_size: discard chunks shorter than this (noise filter)
    deduplicate   : skip chunks whose text hash already appeared
    """

    def __init__(
        self,
        chunk_size:     int  = 512,
        chunk_overlap:  int  = 64,
        min_chunk_size: int  = 60,
        deduplicate:    bool = True,
    ):
        self.chunk_size     = chunk_size
        self.chunk_overlap  = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.deduplicate    = deduplicate
        self._seen_hashes:  set = set()

    def reset_dedup(self) -> None:
        """Call between different documents to reset dedup state."""
        self._seen_hashes.clear()

    # ── Main entry point ──────────────────────
    def chunk_document(self, doc: RawDocument) -> List[DocumentChunk]:
        """Chunk a single RawDocument into DocumentChunks."""
        if doc.is_empty():
            return []

        text     = doc.text
        language = doc.metadata.get("language", "english")
        sentences = _split_sentences(text, language)

        if not sentences:
            return []

        # Group sentences into chunks respecting size + overlap
        raw_chunks = self._group_sentences(sentences)

        # Build DocumentChunk objects with full metadata
        chunks:   List[DocumentChunk] = []
        section   = ""

        for idx, chunk_text in enumerate(raw_chunks):
            if len(chunk_text) < self.min_chunk_size:
                continue

            chunk_hash = _make_hash(chunk_text)

            if self.deduplicate:
                if chunk_hash in self._seen_hashes:
                    continue
                self._seen_hashes.add(chunk_hash)

            # Update section context when we encounter a heading
            heading = _extract_section_heading(chunk_text)
            if heading:
                section = heading

            chunk_meta = {
                **doc.metadata,
                "chunk_index":   idx,
                "chunk_size":    len(chunk_text),
                "chunk_context": section,
                "language":      _detect_language_hint(chunk_text),
            }

            chunks.append(DocumentChunk(
                chunk_text=chunk_text,
                chunk_index=idx,
                chunk_hash=chunk_hash,
                chunk_context=section,
                metadata=chunk_meta,
            ))

        return chunks

    # ── Batch entry point ─────────────────────
    def chunk_documents(self, docs: List[RawDocument]) -> List[DocumentChunk]:
        """Chunk a list of documents. Resets dedup per call."""
        self.reset_dedup()
        all_chunks: List[DocumentChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"[Chunker] {len(docs)} docs → {len(all_chunks)} chunks")
        return all_chunks

    # ── Internal grouping logic ───────────────
    def _group_sentences(self, sentences: List[str]) -> List[str]:
        """
        Greedily pack sentences into chunks.
        When a chunk would exceed chunk_size, start a new one
        but carry `chunk_overlap` characters from the previous chunk.
        """
        chunks: List[str] = []
        current_parts: List[str] = []
        current_len: int = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # Single sentence larger than chunk_size → emit as its own chunk
            if sentence_len > self.chunk_size and not current_parts:
                chunks.append(sentence)
                continue

            if current_len + sentence_len > self.chunk_size and current_parts:
                # Emit current chunk
                chunk_text = " ".join(current_parts)
                chunks.append(chunk_text)

                # Overlap: keep tail of previous chunk
                overlap_text = chunk_text[-self.chunk_overlap:]
                current_parts = [overlap_text] if overlap_text.strip() else []
                current_len   = len(overlap_text)

            current_parts.append(sentence)
            current_len += sentence_len + 1  # +1 for space

        # Emit any remaining sentences
        if current_parts:
            chunks.append(" ".join(current_parts))

        return chunks