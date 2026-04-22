"""
Hybrid search: dense (semantic) + sparse (BM25) combined.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Use uvicorn logger to ensure logs appear
logger = logging.getLogger("uvicorn.error")


# Result container
@dataclass
class SearchResult:
    text:       str
    score:      float
    rank:       int = 0
    metadata:   dict = field(default_factory=dict)
    source:     str = "unknown"    # "semantic" | "bm25" | "fused" | "reranked"


# BM25 index
class BM25Index:
    """
    Thin wrapper around rank_bm25.BM25Okapi.
    Handles tokenization, index build, and scoring.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        k1 : term frequency saturation (1.5 = balanced)
        b  : length normalization (0.75 = standard)
        """
        self.k1       = k1
        self.b        = b
        self._index   = None
        self._corpus  : List[str] = []

    # ── Tokenizer (Arabic + English) ──────────
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple but effective tokenizer for BM25.
        - Lowercase Latin
        - Split on whitespace + punctuation
        - Keep Arabic words intact (already space-delimited)
        - Remove tokens shorter than 2 chars (noise)
        """
        text  = text.lower()
        # Remove punctuation except Arabic-specific chars
        text  = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
        tokens = text.split()
        return [t for t in tokens if len(t) >= 2]

    # ── Build ─────────────────────────────────
    def build(self, texts: List[str]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank_bm25 not installed. Run: pip install rank-bm25")

        if not texts:
            logger.warning("[BM25] No texts provided to build index")
            return

        self._corpus = texts
        tokenized    = [self._tokenize(t) for t in texts]
        self._index  = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        logger.info(f"[BM25] Index built on {len(texts)} documents")

    # ── Search ────────────────────────────────
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Returns list of (doc_index, bm25_score) sorted descending.
        """
        if self._index is None:
            logger.warning("[BM25] Index not built. Call build() first.")
            return []

        if not query.strip():
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)

        # Return only non-zero scores, sorted
        results = [
            (idx, float(score))
            for idx, score in enumerate(scores)
            if score > 0
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# Hybrid Search Engine
class HybridSearchEngine:
    """
    Combines dense (vector) + sparse (BM25) retrieval.

    Usage
    -----
    1. Call build_bm25_index(texts, metadatas) once after indexing.
    2. Call search(query, project, limit) for each query.

    The BM25 index lives in memory — it rebuilds automatically
    if the object is re-created (e.g., after server restart).
    For production, consider persisting it with pickle.
    """

    def __init__(
        self,
        vectordb_client,
        embedding_client,
        collection_name: str,
        bm25_k1:  float = 1.5,
        bm25_b:   float = 0.75,
        hyde_engine  = None,    # HyDEEngine | None
    ):
        from stores.llm.LLMEnums import DocumentTypeEnums
        self.DocumentTypeEnums = DocumentTypeEnums

        self.vectordb_client  = vectordb_client
        self.embedding_client = embedding_client
        self.collection_name  = collection_name
        self.hyde_engine      = hyde_engine

        self.bm25 = BM25Index(k1=bm25_k1, b=bm25_b)

        # Corpus mirror (needed to map BM25 doc_idx → text + metadata)
        self._corpus_texts:     List[str]  = []
        self._corpus_metadatas: List[dict] = []
    
    # ── Get query vector (HyDE → raw fallback) ─
    def _get_query_vector(self, query: str) -> Optional[list]:
        """
        Priority:
        1. HyDE vector (hypothetical document embedding)
        2. Raw query embedding
        """
        if self.hyde_engine:
            vector, source = self.hyde_engine.get_hyde_vector(query)
            if vector:
                logger.info(f"[HybridSearch] Dense vector source: {source}")
                return vector
            logger.warning("[HybridSearch] HyDE failed — falling back to raw embedding")
 
        try:
            vector = self.embedding_client.embed_text(
                text=query,
                document_type=self.DocumentTypeEnums.QUERY.value,
            )
            if not vector:
                return None
            if isinstance(vector[0], list):
                return vector[0]
            return vector
        except Exception as exc:
            logger.error(f"[HybridSearch] Raw embedding failed: {exc}")
            return None

    # ── Index management ──────────────────────
    def build_bm25_index(
        self,
        texts:     List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """
        Call this after ingesting documents.
        texts and metadatas must be parallel lists.
        """
        self._corpus_texts     = texts
        self._corpus_metadatas = metadatas or [{} for _ in texts]
        self.bm25.build(texts)

    # ── Dense search ──────────────────────────
    async def _dense_search(
        self, query: str, top_k: int
    ) -> List[SearchResult]:
        try:

            vector = self._get_query_vector(query)

            # vector = self.embedding_client.embed_text(
            #     text=query,
            #     document_type=self.DocumentTypeEnums.QUERY.value,
            # )

            if not vector or len(vector) == 0:
                logger.warning("[HybridSearch] Embedding returned empty vector")
                return []

            # embed_text returns list of vectors; take first for a single query
            if isinstance(vector[0], list):
                vector = vector[0]

            raw = await self.vectordb_client.search_by_vector(
                collection_name=self.collection_name,
                vector=vector,
                limit=top_k,
            )

            results = []
            for rank, doc in enumerate(raw or []):
                # Get metadata and add chunk_id if available
                metadata = getattr(doc, "metadata", {})
                if hasattr(doc, 'chunk_id') and doc.chunk_id:
                    metadata['chunk_id'] = doc.chunk_id
                
                results.append(SearchResult(
                    text     = doc.text,
                    score    = float(getattr(doc, "score", 0.0)),
                    rank     = rank,
                    metadata = metadata,
                    source   = "semantic",
                ))
            return results

        except Exception as exc:
            logger.error(f"[HybridSearch] Dense search failed: {exc}")
            return []

    # ── Sparse search ─────────────────────────
    def _sparse_search(
        self, query: str, top_k: int
    ) -> List[SearchResult]:
        if not self._corpus_texts:
            logger.warning("[HybridSearch] BM25 corpus is empty. Hybrid search will use dense-only results.")
            return []

        try:
            hits = self.bm25.search(query, top_k=top_k)
            results = []
            for rank, (idx, score) in enumerate(hits):
                results.append(SearchResult(
                    text     = self._corpus_texts[idx],
                    score    = score,
                    rank     = rank,
                    metadata = self._corpus_metadatas[idx] if idx < len(self._corpus_metadatas) else {},
                    source   = "bm25",
                ))
            return results

        except Exception as exc:
            logger.error(f"[HybridSearch] BM25 search failed: {exc}")
            return []

    # ── Hybrid search (main entry point) ──────
    async def search(
        self,
        query:       str,
        top_k:       int = 10,
        dense_weight: float = 0.6,
        auto_build_bm25: bool = True,  # NEW: Auto-build if empty
    ) -> Tuple[List[SearchResult], List[SearchResult]]:
        """
        Returns (dense_results, sparse_results).
        These are passed to RRF for fusion — kept separate intentionally.

        dense_weight is stored for reference but fusion is done in rrf.py.
        
        auto_build_bm25: If True and BM25 corpus is empty, fetch all docs
                        from vector DB and build the index automatically.
        """
        if not query or not query.strip():
            logger.warning("[HybridSearch] Empty query received")
            return [], []

        # Truncate very long queries (embedding models have token limits)
        query = query[:1000]

        # NEW: Auto-build BM25 index if empty (lazy loading)
        if auto_build_bm25 and not self._corpus_texts:
            logger.info("[HybridSearch] BM25 index empty, attempting auto-build...")
            try:
                await self._auto_build_bm25_index()
            except Exception as exc:
                logger.warning(f"[HybridSearch] Auto-build BM25 failed: {exc}")

        dense_results  = await self._dense_search(query, top_k)
        sparse_results = self._sparse_search(query, top_k)

        logger.info(
            f"[HybridSearch] Dense: {len(dense_results)}, "
            f"Sparse (BM25): {len(sparse_results)}"
        )
        return dense_results, sparse_results
    
    # ── Auto-build BM25 index ─────────────────
    async def _auto_build_bm25_index(self) -> None:
        """
        Fetch all documents from vector DB and build BM25 index.
        Called automatically on first search if index is empty.
        """
        try:
            # Get embedding size from embedding client
            embedding_size = self.embedding_client.embedding_size
            
            # Fetch all documents from vector database
            # Use a dummy vector to get all docs (not ideal but works)
            all_docs = await self.vectordb_client.search_by_vector(
                collection_name=self.collection_name,
                vector=[0.0] * embedding_size,  # Dummy vector with correct size
                limit=100000,  # Large limit to get all docs
            )
            
            if not all_docs:
                logger.warning(f"[HybridSearch] No documents found in {self.collection_name}")
                return
            
            texts = [doc.text for doc in all_docs]
            metadatas = []
            for doc in all_docs:
                metadata = getattr(doc, "metadata", {}).copy()
                # Add chunk_id to metadata if available
                if hasattr(doc, 'chunk_id') and doc.chunk_id:
                    metadata['chunk_id'] = doc.chunk_id
                metadatas.append(metadata)
            
            self.build_bm25_index(texts, metadatas)
            logger.info(f"[HybridSearch] Auto-built BM25 index with {len(texts)} documents")
            
        except Exception as exc:
            logger.error(f"[HybridSearch] Auto-build BM25 failed: {exc}")
            raise