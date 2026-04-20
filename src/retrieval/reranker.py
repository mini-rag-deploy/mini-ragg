# src/retrieval/reranker.py
"""
Cross-encoder re-ranking.

Why cross-encoder?
------------------
Bi-encoder (vector search) encodes query and document independently.
Cross-encoder reads [query + document] together — like a human reading
both side-by-side. It is 10-100× slower but dramatically more accurate.

Pipeline position: after RRF fusion, before final top-k selection.
Typical flow: retrieve top-20 → re-rank → keep top-5.

Models (ranked by Arabic+English quality):
- BAAI/bge-reranker-v2-m3        ← best multilingual, recommended
- cross-encoder/ms-marco-MiniLM-L-6-v2  ← fast, English-only fallback
- Cohere Rerank API               ← API-based fallback (no local GPU needed)

Edge cases handled
------------------
- Model not available (graceful fallback: return input unchanged)
- Empty candidate list
- Query or document exceeds model max tokens (auto-truncated)
- Batch size management (avoids OOM on large candidate sets)
- Cohere API errors (falls back to local model or original order)
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from .hybrid_search import SearchResult

# Use uvicorn logger to ensure logs appear
logger = logging.getLogger("uvicorn.error")

# Max tokens most cross-encoders accept
_MAX_INPUT_CHARS = 4096


def _truncate(text: str, max_chars: int = _MAX_INPUT_CHARS) -> str:
    return text[:max_chars] if len(text) > max_chars else text


# ─────────────────────────────────────────────
# Local cross-encoder (HuggingFace)
# ─────────────────────────────────────────────
class LocalReranker:
    """
    Uses sentence-transformers CrossEncoder locally.
    Recommended model: BAAI/bge-reranker-v2-m3
    (supports Arabic, English, and 100+ other languages)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device:     str = "cpu",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device     = device
        self.batch_size = batch_size
        self._model     = None

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512,
            )
            logger.info(f"[Reranker] Loaded model: {self.model_name} on {self.device}")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        except Exception as exc:
            logger.error(f"[Reranker] Failed to load model: {exc}")
            self._model = None

    def rerank(
        self,
        query:      str,
        candidates: List[SearchResult],
        top_k:      int = 5,
    ) -> List[SearchResult]:
        if not candidates:
            return []

        self._load_model()

        if self._model is None:
            logger.warning("[Reranker] Model unavailable — returning original order")
            return candidates[:top_k]

        try:
            query_trunc = _truncate(query, 512)

            # Build pairs for cross-encoder
            pairs = [
                (query_trunc, _truncate(doc.text, _MAX_INPUT_CHARS))
                for doc in candidates
            ]

            # Score in batches to manage memory
            all_scores: List[float] = []
            for i in range(0, len(pairs), self.batch_size):
                batch  = pairs[i : i + self.batch_size]
                scores = self._model.predict(batch, show_progress_bar=False)
                all_scores.extend(scores.tolist())

            # Sort candidates by cross-encoder score
            scored = sorted(
                zip(candidates, all_scores),
                key=lambda x: x[1],
                reverse=True,
            )

            results = []
            for rank, (doc, score) in enumerate(scored[:top_k]):
                results.append(SearchResult(
                    text     = doc.text,
                    score    = float(score),
                    rank     = rank,
                    metadata = doc.metadata,
                    source   = "reranked",
                ))

            logger.info(
                f"[Reranker] {len(candidates)} candidates → "
                f"{len(results)} after re-ranking"
            )
            return results

        except Exception as exc:
            logger.error(f"[Reranker] Re-ranking failed: {exc}")
            return candidates[:top_k]


# ─────────────────────────────────────────────
# Cohere Rerank API (no local GPU needed)
# ─────────────────────────────────────────────
class CohereReranker:
    """
    Uses Cohere's /rerank endpoint.
    Requires COHERE_API_KEY.
    Model: rerank-multilingual-v3.0 (supports Arabic)
    """

    def __init__(
        self,
        api_key:    str,
        model:      str = "rerank-multilingual-v3.0",
        batch_size: int = 20,
    ):
        self.api_key    = api_key
        self.model      = model
        self.batch_size = batch_size

    def _simplify_query(self, query: str) -> str:
        """
        Simplify query for better reranking by focusing on the core question.
        Removes context prefixes and emphasizes key relationships.
        """
        import re
        
        # Remove common context prefixes
        patterns = [
            r"^Consider\s+[\"']?[^\"']+[\"']?'?s?\s+privacy\s+policy[;:,]\s*",
            r"^According\s+to\s+[\"']?[^\"']+[\"']?[;:,]\s*",
            r"^In\s+[\"']?[^\"']+[\"']?'?s?\s+policy[;:,]\s*",
            r"^Based\s+on\s+[\"']?[^\"']+[\"']?[;:,]\s*",
        ]
        
        simplified = query
        for pattern in patterns:
            simplified = re.sub(pattern, "", simplified, flags=re.IGNORECASE)
        
        # Emphasize key relationships by adding explicit keywords
        # This helps the reranker understand the specific focus
        key_phrases = {
            r'\b(passed|shared|exchanged|transferred)\s+(between|among|with)\s+users\b': 'user-to-user information sharing',
            r'\b(visible|accessible|available)\s+to\s+(all\s+)?users\b': 'information visible to users',
            r'\b(public|publicly\s+available)\b.*\busers?\b': 'public user information',
        }
        
        for pattern, emphasis in key_phrases.items():
            if re.search(pattern, simplified, re.IGNORECASE):
                # Add emphasis to help reranker focus
                simplified = f"{simplified} (focus: {emphasis})"
                logger.debug(f"[CohereReranker] Added emphasis: {emphasis}")
                break
        
        # If we removed something and the result is still meaningful, use it
        if simplified != query and len(simplified.strip()) > 10:
            logger.debug(f"[CohereReranker] Simplified query: '{query}' -> '{simplified}'")
            return simplified.strip()
        
        return query

    def rerank(
        self,
        query:      str,
        candidates: List[SearchResult],
        top_k:      int = 5,
    ) -> List[SearchResult]:
        if not candidates:
            return []

        try:
            import cohere
            client = cohere.Client(self.api_key)

            # Simplify query to focus on core question
            simplified_query = self._simplify_query(query)

            # Cohere has a max of 1000 docs per call — chunk if needed
            texts = [_truncate(doc.text) for doc in candidates]

            response = client.rerank(
                query     = _truncate(simplified_query, 512),
                documents = texts,
                model     = self.model,
                top_n     = min(top_k, len(candidates)),
            )

            results: List[SearchResult] = []
            for rank, hit in enumerate(response.results):
                original = candidates[hit.index]
                results.append(SearchResult(
                    text     = original.text,
                    score    = float(hit.relevance_score),
                    rank     = rank,
                    metadata = original.metadata,
                    source   = "reranked_cohere",
                ))

            logger.info(
                f"[CohereReranker] {len(candidates)} candidates → "
                f"{len(results)} after re-ranking"
            )
            return results

        except ImportError:
            logger.error("[CohereReranker] cohere not installed. Run: pip install cohere")
            return candidates[:top_k]
        except Exception as exc:
            logger.error(f"[CohereReranker] Re-ranking failed: {exc}")
            return candidates[:top_k]


# ─────────────────────────────────────────────
# Unified Reranker — auto-selects backend
# ─────────────────────────────────────────────
class Reranker:
    """
    Auto-selects the best available re-ranker:
    1. LocalReranker  (if sentence-transformers available)
    2. CohereReranker (if cohere_api_key provided)
    3. No-op          (returns original order, logs a warning)

    Parameters
    ----------
    model_name     : HuggingFace cross-encoder model
    device         : 'cpu' or 'cuda'
    cohere_api_key : optional — enables Cohere fallback
    prefer_cohere  : if True, tries Cohere before local model
    """

    def __init__(
        self,
        model_name:     str = "BAAI/bge-reranker-v2-m3",
        device:         str = "cpu",
        batch_size:     int = 16,
        cohere_api_key: Optional[str] = None,
        prefer_cohere:  bool = False,
    ):
        self._local   = LocalReranker(model_name, device, batch_size)
        self._cohere  = CohereReranker(cohere_api_key) if cohere_api_key else None
        self._prefer_cohere = prefer_cohere

    def rerank(
        self,
        query:      str,
        candidates: List[SearchResult],
        top_k:      int = 5,
    ) -> List[SearchResult]:

        if not candidates:
            return []

        if self._prefer_cohere and self._cohere:
            logger.info("[Reranker] Trying Cohere Reranker first (prefer_cohere=True)")
            results = self._cohere.rerank(query, candidates, top_k)
            if results:
                return results

        # Try local
        try:
            results = self._local.rerank(query, candidates, top_k)
            if results:
                return results
        except Exception as exc:
            logger.warning(f"[Reranker] Local reranker error: {exc}")

        # Try Cohere as fallback
        if self._cohere:
            results = self._cohere.rerank(query, candidates, top_k)
            if results:
                return results

        # Last resort: return original order
        logger.warning("[Reranker] All re-rankers failed — returning original order")
        for rank, doc in enumerate(candidates[:top_k]):
            doc.rank   = rank
            doc.source = "no_rerank"
        return candidates[:top_k]