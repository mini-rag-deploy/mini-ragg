# src/retrieval/hyde.py
"""
HyDE — Hypothetical Document Embedding.

Queries and documents live in different embedding "spaces":
  - Query  : short, conversational, question-form
  - Document: long, formal, declarative

Even if the meaning is identical, the vectors are often far apart —
causing relevant documents to rank low.

"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger('uvicorn.error')


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────
HYDE_PROMPT = """\
You are a legal document assistant. A user has asked the following question.
Your task is to write a short, realistic passage (2-4 sentences) that would
appear in a legal or official document and directly answers the question.

Rules:
- Write in the SAME language as the question (Arabic → Arabic, English → English).
- Write as if you are the document itself — declarative, formal tone.
- Do NOT say "According to..." or "The answer is..." — write the passage directly.
- Do NOT add citations or article numbers you don't know — keep it plausible.
- If the question is in Arabic, the passage must be in Arabic.

Question: {query}

Document passage:
"""


# ─────────────────────────────────────────────
# HyDE Engine
# ─────────────────────────────────────────────
class HyDEEngine:
    """
    Generates a hypothetical document for a query and returns
    an embedding vector aligned with the document space.

    Parameters
    ----------
    generation_client : LLM provider with generate_text()
    embedding_client  : embedding model with embed_text()
    max_hypo_chars    : cap hypothetical length before embedding
    """

    def __init__(
        self,
        generation_client,
        embedding_client,
        max_hypo_chars: int = 800,
    ):
        self.generation_client = generation_client
        self.embedding_client  = embedding_client
        self.max_hypo_chars    = max_hypo_chars

    # ── Generate hypothetical document ────────
    def _generate_hypothetical(self, query: str) -> Optional[str]:
        try:
            hypo = self.generation_client.generate_text(
                prompt=HYDE_PROMPT.format(query=query),
                chat_history=[],
                temperature=0.4,       # some creativity, still factual
                max_output_tokens=300,
            )

            if not hypo or not hypo.strip():
                logger.warning("[HyDE] Empty hypothetical document generated")
                return None

            # Truncate to embedding model safe length
            hypo = hypo.strip()[: self.max_hypo_chars]
            logger.debug(f"[HyDE] Hypothetical: {hypo[:100]}...")
            return hypo

        except Exception as exc:
            logger.error(f"[HyDE] Generation failed: {exc}")
            return None

    # ── Embed with fallback ────────────────────
    def _embed(self, text: str) -> Optional[list]:
        try:
            from stores.llm.LLMEnums import DocumentTypeEnums
            vector = self.embedding_client.embed_text(
                text=text,
                document_type=DocumentTypeEnums.QUERY.value,
            )
            if not vector:
                return None
            # embed_text returns list-of-lists for batch; unwrap single
            if isinstance(vector[0], list):
                return vector[0]
            return vector
        except Exception as exc:
            logger.error(f"[HyDE] Embedding failed: {exc}")
            return None

    # ── Main: get HyDE vector ─────────────────
    def get_hyde_vector(self, query: str) -> tuple[Optional[list], str]:
        """
        Returns (vector, source) where source is 'hyde' or 'raw_query'.
        Always returns a usable vector — falls back to raw query on error.
        """
        hypo = self._generate_hypothetical(query)

        if hypo:
            vector = self._embed(hypo)
            if vector:
                logger.info("[HyDE] Using hypothetical document vector")
                return vector, "hyde"
            logger.warning("[HyDE] Hypothetical embedding failed — falling back to raw query")

        # Fallback: embed raw query
        vector = self._embed(query)
        if vector:
            return vector, "raw_query"

        logger.error("[HyDE] Both HyDE and raw query embedding failed")
        return None, "failed"

    # ── Batch: expand query into multiple HyDE vectors ────────
    def get_hyde_vectors_batch(
        self, queries: List[str]
    ) -> List[tuple[Optional[list], str]]:
        """
        For use with multi-query expansion:
        each query variant gets its own HyDE vector.
        """
        return [self.get_hyde_vector(q) for q in queries]