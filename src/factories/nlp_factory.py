# src/factories/nlp_factory.py
"""
Factory that wires all retrieval components together.
Call build_nlp_controller() in your FastAPI app startup.

Usage in main.py / app startup:
    from factories.nlp_factory import build_nlp_controller
    app.nlp_controller = build_nlp_controller(
        vectordb_client   = app.vectordb_client,
        generation_client = app.generation_client,
        embedding_client  = app.embedding_client,
        template_parser   = app.template_parser,
        collection_name   = f"collection_{embedding_size}_{project_id}",
        cohere_api_key    = settings.COHERE_API_KEY,
    )
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def build_nlp_controller(
    vectordb_client,
    generation_client,
    embedding_client,
    template_parser      = None,
    collection_name: str = "",
    # Re-ranker options
    reranker_model:  str = "BAAI/bge-reranker-v2-m3",
    reranker_device: str = "cpu",
    cohere_api_key:  Optional[str] = None,
    prefer_cohere:   bool = True,
    # Multi-query options
    n_query_variants: int = 3,
    # BM25 options
    bm25_k1: float = 1.5,
    bm25_b:  float = 0.75,
    # RRF options
    rrf_k: int = 60,
    # Feature flags
    enable_hybrid_search: bool = True,
    enable_reranking:     bool = True,
    enable_multi_query:   bool = True,
):
    from controllers.NLPController import NLPController
    from retrieval.hybrid_search import HybridSearchEngine
    from retrieval.rrf import RRFFusion
    from retrieval.reranker import Reranker
    from retrieval.multi_query import MultiQueryExpander

    # ── Hybrid search ──────────────────────────────────────────
    hybrid_engine = None
    if enable_hybrid_search and collection_name:
        try:
            hybrid_engine = HybridSearchEngine(
                vectordb_client  = vectordb_client,
                embedding_client = embedding_client,
                collection_name  = collection_name,
                bm25_k1          = bm25_k1,
                bm25_b           = bm25_b,
            )
            logger.info("[Factory] HybridSearchEngine initialized")
        except Exception as exc:
            logger.warning(f"[Factory] HybridSearchEngine failed to init: {exc}")

    # ── RRF ───────────────────────────────────────────────────
    rrf = None
    try:
        rrf = RRFFusion(k=rrf_k)
        logger.info(f"[Factory] RRFFusion initialized (k={rrf_k})")
    except Exception as exc:
        logger.warning(f"[Factory] RRFFusion failed: {exc}")

    # ── Reranker ──────────────────────────────────────────────
    reranker = None
    if enable_reranking:
        try:
            reranker = Reranker(
                model_name     = reranker_model,
                device         = reranker_device,
                cohere_api_key = cohere_api_key or os.getenv("COHERE_API_KEY"),
                prefer_cohere  = prefer_cohere,
            )
            logger.info(f"[Factory] Reranker initialized (model={reranker_model})")
        except Exception as exc:
            logger.warning(f"[Factory] Reranker failed to init: {exc}")

    # ── Multi-query expander ───────────────────────────────────
    expander = None
    if enable_multi_query:
        try:
            expander = MultiQueryExpander(
                generation_client = generation_client,
                n_variants        = n_query_variants,
                include_original  = True,
            )
            logger.info(f"[Factory] MultiQueryExpander initialized (n={n_query_variants})")
        except Exception as exc:
            logger.warning(f"[Factory] MultiQueryExpander failed to init: {exc}")

    # ── Assemble controller ────────────────────────────────────
    controller = NLPController(
        vectordb_client      = vectordb_client,
        generation_client    = generation_client,
        embedding_client     = embedding_client,
        template_parser      = template_parser,
        hybrid_search_engine = hybrid_engine,
        reranker             = reranker,
        multi_query_expander = expander,
        rrf_fusion           = rrf,
    )

    logger.info("[Factory] NLPController ready")
    return controller