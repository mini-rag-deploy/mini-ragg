"""
Factory that wires all retrieval components together.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger('uvicorn.error')


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
    cohere_backup_key: Optional[str] = None,  # NEW: Backup keys for reranker
    cohere_backup_key2: Optional[str] = None,
    cohere_backup_key3: Optional[str] = None,
    prefer_cohere:   bool = True,
    # Multi-query options
    n_query_variants: int = 3,
    # BM25 options
    bm25_k1: float = 1.5,
    bm25_b:  float = 0.75,
    # RRF options
    rrf_k: int = 60,
    enable_hyde: bool = True,
    hyde_max_chars: int = 800,
    # Feature flags
    enable_hybrid_search: bool = True,
    enable_reranking:     bool = True,
    enable_multi_query:   bool = True,
    enable_agentic_rag:   bool = False,  # NEW: Enable agentic source selection
    tavily_api_key:       Optional[str] = None,  # NEW: Tavily API key for internet search
):
    from controllers.NLPController import NLPController
    from retrieval.hybrid_search import HybridSearchEngine
    from retrieval.rrf import RRFFusion
    from retrieval.reranker import Reranker
    from retrieval.multi_query import MultiQueryExpander

    # HyDE
    hyde_engine = None
    if enable_hyde:
        try:
            from retrieval.hyde import HyDEEngine
            hyde_engine = HyDEEngine(
                generation_client=generation_client,
                embedding_client=embedding_client,
                max_hypo_chars=hyde_max_chars,
            )
            logger.info("[Factory] HyDEEngine initialized")
        except Exception as exc:
            logger.warning(f"[Factory] HyDEEngine failed: {exc}")

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
                hyde_engine      =hyde_engine,
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
                cohere_backup_key = cohere_backup_key or os.getenv("COHERE_API_KEY_BACKUP"),
                cohere_backup_key2 = cohere_backup_key2 or os.getenv("COHERE_API_KEY_BACKUP2"),
                cohere_backup_key3 = cohere_backup_key3 or os.getenv("COHERE_API_KEY_BACKUP3"),
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

    # ── Source Router (Agentic RAG) ────────────────────────────
    source_router = None
    if enable_agentic_rag:
        try:
            from agent.source_router import SourceRouter
            from agent.tools_registry import ToolsRegistry
            from agent.internet_retriever import InternetRetriever
            
            # Initialize tools registry
            tools_registry = ToolsRegistry()
            
            # Initialize internet retriever
            internet_retriever = InternetRetriever(
                tavily_api_key=tavily_api_key or os.getenv("TAVILY_API_KEY"),
            )
            
            # Initialize source router
            source_router = SourceRouter(
                generation_client=generation_client,
                tools_registry=tools_registry,
                internet_retriever=internet_retriever,
            )
            logger.info("[Factory] SourceRouter initialized (Agentic RAG enabled)")
        except Exception as exc:
            logger.warning(f"[Factory] SourceRouter failed to init: {exc}")

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
        source_router        = source_router,  # NEW: Add source router
    )

    logger.info("[Factory] NLPController ready")
    return controller

def build_contextualizer(
    generation_client,
    doc_excerpt_chars: int = 3000,
    min_chunk_chars:   int = 80,
    max_concurrency:   int = 5,
    ):

    from ingestion.contextualizer import ChunkContextualizer

    ctx = ChunkContextualizer(
        generation_client = generation_client,
        doc_excerpt_chars = doc_excerpt_chars,
        min_chunk_chars   = min_chunk_chars,
        max_concurrency   = max_concurrency,
    )

    logger.info("[Factory] ChunkContextualizer initialized")
    
    return ctx




