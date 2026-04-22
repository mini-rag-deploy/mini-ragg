# src/retrieval/reranker.py
"""
Cross-encoder re-ranking.

Why cross-encoder?
------------------
Bi-encoder (vector search) encodes query and document independently.
Cross-encoder reads [query + document] together — like a human reading
both side-by-side.

Pipeline position: after RRF fusion, before final top-k selection.
Typical flow: retrieve top-20 → re-rank → keep top-5.

Models (ranked by Arabic+English quality):
- BAAI/bge-reranker-v2-m3        ← best multilingual
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
import time
from typing import List, Optional
from threading import Lock
from collections import deque

from .hybrid_search import SearchResult

# Use uvicorn logger to ensure logs appear
logger = logging.getLogger("uvicorn.error")

# Max tokens most cross-encoders accept
_MAX_INPUT_CHARS = 4096


def _truncate(text: str, max_chars: int = _MAX_INPUT_CHARS) -> str:
    return text[:max_chars] if len(text) > max_chars else text


# Rate Limiter (shared utility)
class RateLimiter:
    """Simple rate limiter using sliding window"""
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests = deque()
        self.lock = Lock()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            
            # Remove old requests outside the time window
            while self.requests and self.requests[0] < now - self.time_window:
                self.requests.popleft()
            
            # If at limit, wait until oldest request expires
            if len(self.requests) >= self.max_requests:
                sleep_time = self.requests[0] + self.time_window - now + 0.1  # +0.1 for safety
                if sleep_time > 0:
                    logger.info(f"[RateLimiter] Waiting {sleep_time:.1f}s to respect rate limit...")
                    time.sleep(sleep_time)
                    # Clean up again after waiting
                    now = time.time()
                    while self.requests and self.requests[0] < now - self.time_window:
                        self.requests.popleft()
            
            # Record this request
            self.requests.append(now)


# Local cross-encoder (HuggingFace)
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


# Cohere Rerank API (no local GPU needed)
class CohereReranker:
    """
    Uses Cohere's /rerank endpoint.
    Requires COHERE_API_KEY.
    Model: rerank-multilingual-v3.0 (supports Arabic)
    """

    def __init__(
        self,
        api_key:    str,
        backup_api_key: str = None,  # NEW: Backup API key
        backup_api_key2: str = None,
        backup_api_key3: str = None,
        model:      str = "rerank-multilingual-v3.0",
        batch_size: int = 20,
    ):
        self.api_key    = api_key
        self.backup_api_key = backup_api_key
        self.backup_api_key2 = backup_api_key2
        self.backup_api_key3 = backup_api_key3
        self.current_api_key = api_key  # Track which key is active
        
        self.using_backup = False
        self.using_backup2 = False
        self.using_backup3 = False
        
        self.model      = model
        self.batch_size = batch_size
        
        # Rate limiter for Cohere Rerank API: 10 requests/minute
        self.rate_limiter = RateLimiter(max_requests=10, time_window=60.0)
        logger.info("[CohereReranker] Rate limiter initialized: Rerank(10/min)")
    
    def _switch_to_backup(self):
        """Switch to backup API key when primary fails"""
        if self.backup_api_key and not self.using_backup:
            logger.warning("⚠️  [CohereReranker] PRIMARY API KEY EXHAUSTED - Switching to BACKUP API key...")
            self.current_api_key = self.backup_api_key
            self.using_backup = True
            logger.info("✅ [CohereReranker] Successfully switched to BACKUP API key")
            return True
        
        if self.backup_api_key2 and not self.using_backup2:
            logger.warning("⚠️  [CohereReranker] BACKUP API KEY EXHAUSTED - Switching to BACKUP2 API key...")
            self.current_api_key = self.backup_api_key2
            self.using_backup2 = True
            logger.info("✅ [CohereReranker] Successfully switched to BACKUP2 API key")
            return True
        
        if self.backup_api_key3 and not self.using_backup3:
            logger.warning("⚠️  [CohereReranker] BACKUP2 API KEY EXHAUSTED - Switching to BACKUP3 API key...")
            self.current_api_key = self.backup_api_key3
            self.using_backup3 = True
            logger.info("✅ [CohereReranker] Successfully switched to BACKUP3 API key")
            return True

        return False
    
    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """Check if error is a rate limit / quota exhausted error"""
        rate_limit_indicators = [
            "429", "TooManyRequests", "rate limit", "quota", 
            "limit exceeded", "too many requests"
        ]
        return any(indicator.lower() in error_msg.lower() for indicator in rate_limit_indicators)

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
            import time
            client = cohere.Client(self.current_api_key)  # Use current_api_key instead of api_key

            # Simplify query to focus on core question
            simplified_query = self._simplify_query(query)

            # Cohere has a max of 1000 docs per call — chunk if needed
            texts = [_truncate(doc.text) for doc in candidates]

            # Retry logic for rate limits with backup key failover
            response = None
            for attempt in range(3):
                try:
                    # Wait if needed to respect rate limit
                    self.rate_limiter.wait_if_needed()
                    
                    response = client.rerank(
                        query     = _truncate(simplified_query, 512),
                        documents = texts,
                        model     = self.model,
                        top_n     = min(top_k, len(candidates)),
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    error_msg = str(e)
                    if self._is_rate_limit_error(error_msg):
                        # Try to switch to backup key
                        if self._switch_to_backup():
                            logger.info("[CohereReranker] Retrying with backup API key...")
                            client = cohere.Client(self.current_api_key)  # Create new client with backup key
                            continue  # Retry immediately with backup key
                        else:
                            # No backup available, wait and retry
                            if attempt < 2:  # Don't sleep on last attempt
                                logger.warning(f"[CohereReranker] Rate limit hit (429). Retrying in 20 seconds... (Attempt {attempt+1}/3)")
                                time.sleep(20)
                            else:
                                logger.error(f"[CohereReranker] Rate limit exhausted after 3 attempts")
                                raise
                    else:
                        raise  # Re-raise non-rate-limit errors immediately
            
            if not response:
                logger.error("[CohereReranker] No response after retries")
                return candidates[:top_k]

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


# Unified Reranker — auto-selects backend
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
        cohere_backup_key: Optional[str] = None,  # NEW: Backup keys
        cohere_backup_key2: Optional[str] = None,
        cohere_backup_key3: Optional[str] = None,
        prefer_cohere:  bool = False,
    ):
        self._local   = LocalReranker(model_name, device, batch_size)
        self._cohere  = CohereReranker(
            api_key=cohere_api_key,
            backup_api_key=cohere_backup_key,
            backup_api_key2=cohere_backup_key2,
            backup_api_key3=cohere_backup_key3,
        ) if cohere_api_key else None
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