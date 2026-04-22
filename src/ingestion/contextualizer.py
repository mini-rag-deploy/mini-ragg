# src/ingestion/contextualizer.py
"""
Contextual Retrieval — Chunk Contextualization.

When a document is chunked, individual chunks lose their surrounding context:

  Full doc: "Egypt Labor Law No. 12/2003... Article 47: The termination
             shall be effective immediately..."

  Chunk:    "The termination shall be effective immediately."
                ↑ What termination? Of what? Under which law?

The embedding of this isolated chunk will be weak and hard to retrieve.

"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from .chunker import DocumentChunk

logger = logging.getLogger('uvicorn.error')


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────
CONTEXTUALIZE_PROMPT = """\
You are a document analysis assistant.

Below is a full document (or a large excerpt) followed by a specific chunk from that document.
Your task is to write 2-3 concise sentences that explain:
1. Where this chunk appears in the document (e.g., section, article, topic).
2. What broader topic or theme this chunk belongs to.
3. Why this chunk is important in the context of the full document.

Rules:
- Write in the SAME language as the document chunk.
  Arabic chunk → context in Arabic. English chunk → context in English.
- Be specific — mention article numbers, section names, or key terms if visible.
- Do NOT repeat the chunk content — only describe its context.
- Return ONLY the context sentences, nothing else.

Full document (excerpt):
<document>
{document_excerpt}
</document>

Chunk to contextualize:
<chunk>
{chunk_text}
</chunk>

Context (2-3 sentences only):
"""

# Prefix added to contextualized chunks (language-aware)
_CONTEXT_PREFIX_EN = "[Context: {context}]\n\n"
_CONTEXT_PREFIX_AR = "[السياق: {context}]\n\n"


def _context_prefix(language: str, context: str) -> str:
    if language == "arabic":
        return _CONTEXT_PREFIX_AR.format(context=context)
    return _CONTEXT_PREFIX_EN.format(context=context)


# ─────────────────────────────────────────────
# Contextualizer
# ─────────────────────────────────────────────
class ChunkContextualizer:
    """
    Prepends AI-generated context to each chunk before indexing.

    Parameters
    ----------
    generation_client  : LLM with generate_text()
    doc_excerpt_chars  : How many chars of the source document to send
                         as context to the LLM (longer = better context,
                         more tokens consumed)
    min_chunk_chars    : Skip contextualization for very short chunks
    max_concurrency    : Parallel async LLM calls (rate-limit friendly)
    """

    def __init__(
        self,
        generation_client,
        doc_excerpt_chars: int = 3000,
        min_chunk_chars:   int = 80,
        max_concurrency:   int = 5,
    ):
        self.generation_client = generation_client
        self.doc_excerpt_chars = doc_excerpt_chars
        self.min_chunk_chars   = min_chunk_chars
        self.max_concurrency   = max_concurrency

    # ── Single chunk contextualization ────────
    def _contextualize_one(
        self,
        chunk,  # Can be DocumentChunk or DataChunk (SQLAlchemy)
        document_excerpt: str,
    ):
        """
        Synchronous per-chunk contextualization.
        Returns the chunk with updated chunk_text (prepended context).
        Handles both DocumentChunk objects and DataChunk SQLAlchemy objects.
        """
        # Handle different chunk types
        if hasattr(chunk, 'chunk_text'):
            # SQLAlchemy DataChunk object
            chunk_text = chunk.chunk_text
            metadata = chunk.chunk_metadata or {}
        elif hasattr(chunk, 'text'):
            # DocumentChunk object  
            chunk_text = chunk.text
            metadata = chunk.metadata or {}
        else:
            logger.warning(f"[Contextualizer] Unknown chunk type: {type(chunk)}")
            return chunk

        # Skip if already contextualized
        if "[Context:" in chunk_text or "[السياق:" in chunk_text:
            return chunk

        # Skip very short chunks
        if len(chunk_text.strip()) < self.min_chunk_chars:
            return chunk

        # Handle metadata access - support both dict and SQLAlchemy JSONB
        try:
            if hasattr(metadata, 'get'):
                # Dictionary-like access
                language = metadata.get("language", "english")
            else:
                # SQLAlchemy JSONB or other object - try to convert to dict
                if metadata and hasattr(metadata, '__iter__'):
                    metadata_dict = dict(metadata) if metadata else {}
                    language = metadata_dict.get("language", "english")
                else:
                    language = "english"  # fallback
        except Exception as e:
            logger.debug(f"[Contextualizer] Metadata access error: {e}, using default language")
            language = "english"

        try:
            context = self.generation_client.generate_text(
                prompt=CONTEXTUALIZE_PROMPT.format(
                    document_excerpt=document_excerpt[: self.doc_excerpt_chars],
                    chunk_text=chunk_text[:1500],
                ),
                chat_history=[],
                temperature=0.0,       # deterministic — context should be factual
                max_output_tokens=200,
            )

            if not context or not context.strip():
                if hasattr(chunk, 'chunk_id'):
                    logger.debug(f"[Contextualizer] Empty context for chunk {chunk.chunk_id}")
                else:
                    logger.debug(f"[Contextualizer] Empty context for chunk {getattr(chunk, 'chunk_index', 'unknown')}")
                return chunk

            context = context.strip()

            # Prepend context to chunk text
            prefix = _context_prefix(language, context)
            new_chunk_text = prefix + chunk_text

            # Update chunk text based on chunk type
            if hasattr(chunk, 'chunk_text'):
                # SQLAlchemy DataChunk object
                chunk.chunk_text = new_chunk_text
                # Update metadata
                if chunk.chunk_metadata is None:
                    chunk.chunk_metadata = {}
                chunk.chunk_metadata["contextualized"] = True
                chunk.chunk_metadata["context_summary"] = context
                chunk.chunk_metadata["original_text"] = chunk_text
            else:
                # DocumentChunk object
                chunk.chunk_text = new_chunk_text
                # Update metadata
                if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata["contextualized"] = True
                chunk.metadata["context_summary"] = context
                chunk.metadata["original_text"] = chunk_text

        except Exception as exc:
            chunk_id = getattr(chunk, 'chunk_id', getattr(chunk, 'chunk_index', 'unknown'))
            logger.warning(
                f"[Contextualizer] Failed for chunk {chunk_id} "
                f"(using original): {exc}"
            )

        return chunk

    # ── Helper method for metadata access ──────
    def _get_metadata_value(self, chunk, key: str, default=None):
        """
        Safely get metadata value from either DocumentChunk or DataChunk objects.
        """
        try:
            if hasattr(chunk, 'chunk_metadata'):
                # SQLAlchemy DataChunk object
                metadata = chunk.chunk_metadata or {}
                if hasattr(metadata, 'get'):
                    return metadata.get(key, default)
                else:
                    # Convert to dict if needed
                    metadata_dict = dict(metadata) if metadata else {}
                    return metadata_dict.get(key, default)
            elif hasattr(chunk, 'metadata'):
                # DocumentChunk object
                metadata = chunk.metadata or {}
                if hasattr(metadata, 'get'):
                    return metadata.get(key, default)
                else:
                    metadata_dict = dict(metadata) if metadata else {}
                    return metadata_dict.get(key, default)
            else:
                return default
        except Exception as e:
            logger.debug(f"[Contextualizer] Error accessing metadata key '{key}': {e}")
            return default

    # ── Build document excerpt ─────────────────
    @staticmethod
    def _build_excerpt(chunks) -> str:
        """
        Reconstruct a document excerpt from chunks to give the LLM
        enough context about the full document.
        We take: first 3 chunks + last 2 chunks for a representative sample.
        Handles both DocumentChunk and DataChunk objects.
        """
        if not chunks:
            return ""

        selected = []
        if len(chunks) <= 5:
            selected = chunks
        else:
            selected = chunks[:3] + chunks[-2:]

        # Extract text from chunks, handling both types
        texts = []
        for c in selected:
            if hasattr(c, 'chunk_text'):
                texts.append(c.chunk_text)
            elif hasattr(c, 'text'):
                texts.append(c.text)
            else:
                texts.append(str(c))  # fallback
        
        return "\n\n".join(texts)

    # ── Synchronous batch (for Celery tasks) ──
    def contextualize_chunks(
        self,
        chunks: List[DocumentChunk],
        document_excerpt: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Synchronous batch contextualization.
        Use this in Celery workers (no async support needed).
        """
        if not chunks:
            return chunks

        excerpt = document_excerpt or self._build_excerpt(chunks)

        contextualized = []
        for i, chunk in enumerate(chunks):
            updated = self._contextualize_one(chunk, excerpt)
            contextualized.append(updated)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"[Contextualizer] Processed {i+1}/{len(chunks)} chunks"
                )

        n_done = sum(
            1 for c in contextualized 
            if self._get_metadata_value(c, "contextualized", False)
        )
        logger.info(
            f"[Contextualizer] Done: {n_done}/{len(chunks)} chunks contextualized"
        )
        return contextualized

    # ── Async batch (for FastAPI endpoints) ───
    async def contextualize_chunks_async(
        self,
        chunks: List[DocumentChunk],
        document_excerpt: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """
        Async batch with concurrency control.
        Respects max_concurrency to avoid hammering the LLM API.
        """
        if not chunks:
            return chunks

        excerpt = document_excerpt or self._build_excerpt(chunks)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _process(chunk: DocumentChunk) -> DocumentChunk:
            async with semaphore:
                return await asyncio.to_thread(
                    self._contextualize_one, chunk, excerpt
                )

        tasks = [_process(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: List[DocumentChunk] = []
        for chunk, result in zip(chunks, results):
            if isinstance(result, Exception):
                logger.warning(
                    f"[Contextualizer] Async error for chunk "
                    f"{chunk.chunk_index}: {result}"
                )
                output.append(chunk)   # use original on error
            else:
                output.append(result)

        n_done = sum(1 for c in output if self._get_metadata_value(c, "contextualized", False))
        logger.info(
            f"[Contextualizer] Async done: {n_done}/{len(chunks)} chunks contextualized"
        )
        return output