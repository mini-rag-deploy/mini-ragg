# src/retrieval/rrf.py
"""
Reciprocal Rank Fusion (RRF).

Why RRF?
--------
Dense scores (cosine similarity) and BM25 scores live on completely
different scales — you can't average them directly. RRF bypasses this
by only using the *rank* of each result in each list, not its raw score.

Formula:  RRF_score(doc) = Σ  1 / (k + rank_i)
          where k=60 (standard constant that dampens top-rank dominance)

Edge cases handled
------------------
- One list completely empty (fuses the other list as-is)
- Both lists empty (returns empty list)
- Duplicate documents across lists (merged, scores accumulated)
- Document identity: matched by text content hash (not object identity)
- k parameter validation
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Tuple

from .hybrid_search import SearchResult

# Use uvicorn logger to ensure logs appear
logger = logging.getLogger("uvicorn.error")


def _text_key(text: str) -> str:
    """
    Stable identity key for a document.
    Using first 200 chars hash — fast and collision-resistant enough.
    """
    return hashlib.sha256(text[:200].encode("utf-8")).hexdigest()[:12]


class RRFFusion:
    """
    Fuses an arbitrary number of ranked result lists using RRF.

    Parameters
    ----------
    k : RRF constant. 60 is the standard default (from the original paper).
        Lower k → top-ranked documents get much higher scores (more aggressive).
        Higher k → scores are more evenly distributed.
    """

    def __init__(self, k: int = 60):
        if k <= 0:
            raise ValueError(f"RRF k must be positive, got {k}")
        self.k = k

    def fuse(
        self,
        *ranked_lists: List[SearchResult],
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Fuse any number of ranked lists.

        Each list should be sorted by relevance descending (index 0 = best).
        Lists can be of different lengths or empty.

        Returns a single fused list sorted by RRF score descending.
        """
        if not ranked_lists:
            logger.warning("[RRF] No ranked lists provided for fusion")
            return []

        # Filter out empty lists
        valid_lists = [lst for lst in ranked_lists if lst]
        if not valid_lists:
            logger.warning("[RRF] All ranked lists are empty")
            return []

        # Single non-empty list — nothing to fuse, just re-rank by position
        if len(valid_lists) == 1:
            logger.info("[RRF] Only one non-empty list, applying RRF scoring by rank")
            results = []
            for rank, doc in enumerate(valid_lists[0]):
                doc.score  = 1.0 / (self.k + rank)
                doc.rank   = rank
                doc.source = "fused"
                results.append(doc)
            return results[:top_k]

        # Accumulate RRF scores by document identity
        score_map:    Dict[str, float]        = {}
        metadata_map: Dict[str, SearchResult] = {}

        for ranked_list in valid_lists:
            for rank, doc in enumerate(ranked_list):
                key   = _text_key(doc.text)
                score = 1.0 / (self.k + rank)

                score_map[key]    = score_map.get(key, 0.0) + score
                # Keep the metadata from the first time we see this doc
                if key not in metadata_map:
                    metadata_map[key] = doc

        # Build output list sorted by accumulated RRF score
        fused: List[SearchResult] = []
        for rank, (key, rrf_score) in enumerate(
            sorted(score_map.items(), key=lambda x: x[1], reverse=True)
        ):
            original = metadata_map[key]
            fused.append(SearchResult(
                text     = original.text,
                score    = rrf_score,
                rank     = rank,
                metadata = original.metadata,
                source   = "fused",
            ))

        result = fused[:top_k]
        logger.info(
            f"[RRF] {len(valid_lists)} lists × "
            f"{[len(l) for l in valid_lists]} docs → "
            f"{len(result)} fused results (k={self.k})"
        )
        return result
