# src/controllers/NLPController.py
"""
NLPController — updated to integrate:
  - Multi-query expansion
  - Hybrid search (semantic + BM25)
  - Reciprocal Rank Fusion (RRF)
  - Cross-encoder re-ranking
  - Self-correcting RAG graph (existing)

Backward compatible: answer_rag_question() signature unchanged.
New behaviour is opt-in via use_advanced_retrieval flag.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Any

from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnums

logger = logging.getLogger("uvicorn.error")


class NLPController(BaseController):

    def __init__(
        self,
        vectordb_client,
        generation_client,
        embedding_client,
        template_parser=None,
        # ── New optional components ──────────────
        hybrid_search_engine=None,   # HybridSearchEngine instance
        reranker=None,               # Reranker instance
        multi_query_expander=None,   # MultiQueryExpander instance
        rrf_fusion=None,             # RRFFusion instance
        source_router=None,          # SourceRouter instance for agentic RAG
    ):
        super().__init__()
        self.vectordb_client      = vectordb_client
        self.generation_client    = generation_client
        self.embedding_client     = embedding_client
        self.template_parser      = template_parser

        # Advanced retrieval (None = not configured → falls back to basic)
        self.hybrid_search_engine = hybrid_search_engine
        self.reranker             = reranker
        self.multi_query_expander = multi_query_expander
        self.rrf_fusion           = rrf_fusion
        
        # Agentic RAG (None = not configured → uses classic RAG)
        self.source_router        = source_router

    # Helpers
    def create_collection_name(self, project_id: str) -> str:
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()

    async def reset_vector_db_collection(self, project: Project):
        return await self.vectordb_client.delete_collection(
            collection_name=self.create_collection_name(project.project_id)
        )

    async def get_vector_db_collection_info(self, project: Project):
        return await self.vectordb_client.get_collection_info(
            collection_name=self.create_collection_name(project.project_id)
        )

    # BM25 Index Management
    async def rebuild_bm25_index(self, project: Project) -> bool:
        """
        Rebuild BM25 index from all documents in the vector database.
        Call this after indexing is complete.
        """
        if not self.hybrid_search_engine:
            logger.warning("[NLPController] No hybrid search engine configured")
            return False
        
        try:
            collection_name = self.create_collection_name(project.project_id)
            
            # Fetch ALL documents from vector database
            # Note: This assumes your vectordb_client has a method to get all documents
            # If not, we'll need to implement pagination
            
            logger.info(f"[NLPController] Rebuilding BM25 index for {collection_name}")
            
            # Get collection info to know how many documents
            collection_info = await self.vectordb_client.get_collection_info(collection_name)
            if not collection_info:
                logger.warning(f"[NLPController] Collection {collection_name} not found")
                return False
            
            # Fetch all documents (you may need to implement this in your vectordb_client)
            # For now, we'll use a large limit
            all_docs = await self.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=[0.0] * self.embedding_client.embedding_size,  # Dummy vector
                limit=10000,  # Large limit to get all docs
            )
            
            if not all_docs:
                logger.warning(f"[NLPController] No documents found in {collection_name}")
                return False
            
            texts = [doc.text for doc in all_docs]
            metadatas = [getattr(doc, "metadata", {}) for doc in all_docs]
            
            self.hybrid_search_engine.build_bm25_index(texts, metadatas)
            logger.info(f"[NLPController] BM25 index rebuilt with {len(texts)} documents")
            return True
            
        except Exception as exc:
            logger.error(f"[NLPController] Failed to rebuild BM25 index: {exc}")
            return False

    # Indexing
    async def index_into_vector_db(
        self,
        project:     Project,
        chunks:      List[DataChunk],
        chunks_ids:  List[int],
        do_reset:    bool = False,
    ) -> bool:
        collection_name = self.create_collection_name(project.project_id)
        texts     = [chunk.chunk_text     for chunk in chunks]
        metadatas = [chunk.chunk_metadata for chunk in chunks]

        vectors = self.embedding_client.embed_text(
            text=texts,
            document_type=DocumentTypeEnums.DOCUMENT.value,
        )

        await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
        )
        await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadatas=metadatas,
            record_ids=chunks_ids,
        )

        # Rebuild BM25 index if hybrid search is configured
        if self.hybrid_search_engine:
            try:
                self.hybrid_search_engine.build_bm25_index(texts, metadatas)
                logger.info("[NLPController] BM25 index rebuilt after indexing")
            except Exception as exc:
                logger.warning(f"[NLPController] BM25 rebuild failed (non-fatal): {exc}")

        return True

    # Basic search
    async def search_vector_db_collection(
        self,
        project: Project,
        text:    str,
        limit:   int = 10,
    ) -> List[Any]:
        collection_name = self.create_collection_name(project.project_id)

        vector = self.embedding_client.embed_text(
            text=text,
            document_type=DocumentTypeEnums.QUERY.value,
        )

        if not vector or len(vector) == 0:
            logger.error(f"[NLPController] Embedding failed for: {text[:80]}")
            return []

        if isinstance(vector[0], list):
            vector = vector[0]

        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit,
        )

        return results or []

    # Advanced retrieval pipeline
    async def advanced_retrieve(
        self,
        project:  Project,
        query:    str,
        top_k:    int = 5,
    ) -> List[Any]:
        """
        Full pipeline:
        1. Multi-query expansion
        2. Hybrid search (semantic + BM25) for each query variant
        3. RRF fusion of all result lists
        4. Cross-encoder re-ranking

        Falls back gracefully at each step.
        """
        from retrieval.hybrid_search import SearchResult
        from retrieval.rrf import RRFFusion

        # ── Step 1: Multi-query expansion ─────────────────────
        if self.multi_query_expander:
            queries = self.multi_query_expander.expand(query)
        else:
            queries = [query]

        logger.info(f"[NLPController] Queries after expansion: {len(queries)}")

        # ── Step 2: Hybrid search for each query ──────────────
        all_dense_lists:  List[List[SearchResult]] = []
        all_sparse_lists: List[List[SearchResult]] = []

        for q in queries:
            if self.hybrid_search_engine:
                try:
                    dense, sparse = await self.hybrid_search_engine.search(
                        query=q,
                        top_k=top_k * 6,    # Increased from 3x to 6x for better recall
                    )
                    all_dense_lists.append(dense)
                    all_sparse_lists.append(sparse)
                except Exception as exc:
                    logger.warning(f"[NLPController] Hybrid search failed for '{q}': {exc}")
                    # Fallback to basic dense search
                    basic = await self.search_vector_db_collection(project, q, top_k * 6)
                    converted = [
                        SearchResult(
                            text=doc.text,
                            score=float(getattr(doc, "score", 0.0)),
                            metadata=getattr(doc, "metadata", {}),
                            source="semantic_fallback",
                        )
                        for doc in basic
                    ]
                    all_dense_lists.append(converted)
            else:
                # No hybrid engine — basic dense search
                logger.info("[NLPController] No hybrid engine configured, using basic search")
                basic = await self.search_vector_db_collection(project, q, top_k * 6)
                converted = [
                    SearchResult(
                        text=doc.text,
                        score=float(getattr(doc, "score", 0.0)),
                        metadata=getattr(doc, "metadata", {}),
                        source="semantic",
                    )
                    for doc in basic
                ]
                all_dense_lists.append(converted)
        
        logger.info(f"[NLPController] Retrieved {sum(len(lst) for lst in all_dense_lists)} dense + "
                    f"{sum(len(lst) for lst in all_sparse_lists)} sparse results total")

        # ── Step 3: RRF fusion ────────────────────────────────
        fuser = self.rrf_fusion if self.rrf_fusion else RRFFusion(k=60)

        all_lists = all_dense_lists + all_sparse_lists
        fused = fuser.fuse(*all_lists, top_k=top_k * 4)  # Increased from 2x to 4x

        if not fused:
            logger.warning("[NLPController] RRF returned no results")
            return []

        # ── Step 4: Cross-encoder re-ranking ──────────────────
        if self.reranker and fused:
            # Increase candidates for reranking to improve recall
            # Rerank more candidates than requested to ensure good results surface
            rerank_candidates = min(len(fused), top_k * 5)  # Increased from 3x to 5x
            logger.info(f"[NLPController] Re-ranking top {rerank_candidates} fused results")
            try:
                reranked = self.reranker.rerank(query, fused[:rerank_candidates], top_k=top_k)
                logger.info(f"[NLPController] After re-ranking: {len(reranked)} docs")
                return reranked
            except Exception as exc:
                logger.warning(f"[NLPController] Re-ranking failed (non-fatal): {exc}")

        # No re-ranker: return fused results
        return fused[:top_k]

    # Main Q&A entry point (backward compatible)
    async def answer_rag_question(
        self,
        project:                Project,
        query:                  str,
        limit:                  int  = 10,
        use_self_correction:    bool = True,
        use_advanced_retrieval: bool = True,
        question_type:          str  = None,  # NEW: question type for prompt selection
        enable_source_selection: bool = False,  # NEW: enable agentic source selection
    ) -> Tuple[Optional[str], Any, list]:
        """
        Parameters
        ----------
        use_self_correction    : run the LangGraph self-correcting loop
        use_advanced_retrieval : use hybrid + RRF + reranker pipeline
        question_type          : type of question (factual, analytical, comparative, summarization, hallucination)
        enable_source_selection: enable dynamic source selection (requires source_router)

        Returns
        -------
        (answer, metadata_or_full_prompt, chat_history)
        """
        
        if use_self_correction:
            logger.info("[NLPController] Using self-correcting RAG graph for question answering")
            return await self._answer_with_graph(
                project=project,
                query=query,
                use_advanced_retrieval=use_advanced_retrieval,
                question_type=question_type,  # Pass question type
                enable_source_selection=enable_source_selection,  # Pass agentic flag
            )
        else:
            print("answer with basic")
            return await self._answer_basic(project, query, limit, use_advanced_retrieval, question_type)

    # ── Self-correcting path ───────────────────
    async def _answer_with_graph(
        self,
        project:                Project,
        query:                  str,
        use_advanced_retrieval: bool = True,
        question_type:          str = None,  # NEW: question type for prompt selection
        enable_source_selection: bool = False,  # NEW: enable agentic source selection
    ):
        from graph.rag_graph import build_rag_graph

        graph = build_rag_graph(
            nlp_controller=self,
            project=project,
            source_router=self.source_router,  # Pass source router
            use_advanced_retrieval=use_advanced_retrieval,
            question_type=question_type,  # Pass question type to graph
            enable_source_selection=enable_source_selection,  # Pass agentic flag
        )

        initial_state = {
            "question":      query,
            "documents":     [],
            "answer":        None,
            "iterations":    0,
            "grade_reason":  None,
            "question_type": question_type,  # Add to state
            # Agentic RAG fields
            "need_more_details": None,
            "selected_source":   None,
            "source_reason":     None,
            "sources_tried":     [],
            "external_data":     None,
            "audit_decision":    None,
        }

        result = await graph.ainvoke(initial_state)

        answer   = result.get("answer")
        documents = result.get("documents", [])  # Get the documents used
        
        # Convert SearchResult objects to dictionaries for JSON serialization
        serializable_documents = []
        for doc in documents:
            if hasattr(doc, '__dict__'):
                # Convert SearchResult to dict
                doc_dict = {
                    "text": doc.text,
                    "score": doc.score,
                    "rank": doc.rank,
                    "metadata": doc.metadata,
                    "source": getattr(doc, 'source', 'unknown')
                }
                serializable_documents.append(doc_dict)
            else:
                # Already a dict or simple object
                serializable_documents.append(doc)
        
        metadata = {
            "iterations":             result.get("iterations", 0),
            "docs_used":              len(documents),
            "mode":                   "agentic" if enable_source_selection else "self_correcting",
            "advanced_retrieval":     use_advanced_retrieval,
            "documents":              serializable_documents,  # Use serializable version
            # Agentic metadata
            "sources_tried":          result.get("sources_tried", []),
            "selected_source":        result.get("selected_source"),
            "source_reason":          result.get("source_reason"),
        }
        return answer, metadata, []

    # ── Basic path (backward compat) ──────────
    async def _answer_basic(
        self,
        project: Project,
        query:   str,
        limit:   int = 10,
        use_advanced_retrieval: bool = False,
        question_type: str = None,  # NEW: question type for prompt selection
    ):
        answer, chat_history = None, None

        # Use advanced retrieval if available and requested
        if use_advanced_retrieval and self.hybrid_search_engine:
            retrieved = await self.advanced_retrieve(
                project=project, query=query, top_k=limit
            )
        else:
            retrieved = await self.search_vector_db_collection(
                project=project, text=query, limit=limit
            )

        # Convert SearchResult objects to dictionaries for JSON serialization
        serializable_retrieved = []
        for doc in retrieved:
            if hasattr(doc, '__dict__'):
                # Convert SearchResult to dict
                doc_dict = {
                    "text": doc.text,
                    "score": doc.score,
                    "rank": getattr(doc, 'rank', 0),
                    "metadata": getattr(doc, 'metadata', {}),
                    "source": getattr(doc, 'source', 'unknown')
                }
                serializable_retrieved.append(doc_dict)
            else:
                # Already a dict or simple object
                serializable_retrieved.append(doc)

        metadata = {
            "iterations": 0,
            "docs_used": len(retrieved),
            "mode": "basic",
            "advanced_retrieval": use_advanced_retrieval,
            "documents": serializable_retrieved,  # Use serializable version
        }

        if not retrieved:
            return answer, metadata, chat_history

        # ── Hallucination Detection: Check relevance scores ──
        # If all retrieved documents have low scores, return "I don't know"
        if retrieved:
            max_score = max(getattr(doc, 'score', 0.0) for doc in retrieved)
            avg_score = sum(getattr(doc, 'score', 0.0) for doc in retrieved) / len(retrieved)
            
            # Threshold: if max score < 0.5 or avg score < 0.3, documents are not relevant
            if max_score < 0.5 or avg_score < 0.3:
                logger.warning(f"[NLPController] Low relevance scores detected (max={max_score:.3f}, avg={avg_score:.3f}). Returning 'I don't know' response.")
                answer = "I don't have enough information in the provided documents to answer this question accurately. The available information does not seem to be directly relevant to your query."
                # Update metadata with serializable documents
                metadata["documents"] = serializable_retrieved
                return answer, metadata, chat_history

        system_prompt = self.template_parser.get("rag", "system_prompt")
        document_prompt = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num":    i + 1,
                "chunk_text": doc.text,
            })
            for i, doc in enumerate(retrieved)
        ])
        
        # Select footer based on question type
        footer_key = "footer_default"
        if question_type:
            # Map question type to footer key
            type_to_footer = {
                "factual": "footer_factual",
                "analytical": "footer_analytical",
                "comparative": "footer_comparative",
                "summarization": "footer_summarization",
                "hallucination": "footer_hallucination",
            }
            footer_key = type_to_footer.get(question_type.lower(), "footer_default")
        
        footer_prompt = self.template_parser.get("rag", footer_key, {"query": query})

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]
        full_prompt = "\n\n".join([document_prompt, footer_prompt])
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history,
        )

        return answer, metadata, chat_history