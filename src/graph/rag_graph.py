"""
Self-correcting RAG graph with Dynamic Source Selection (LangGraph).

This enhanced version includes:
- Dynamic decision on whether more information is needed
- Intelligent source selection (Vector DB, Tools, Internet)
- Multi-source information retrieval
"""

from __future__ import annotations
import uvicorn
import logging
from typing import Any, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .prompts import (
    ANSWER_GRADE_PROMPT,
    HALLUCINATION_PROMPT,
    NO_CONTEXT_PROMPT,
    RAG_PROMPT,
    RETRIEVAL_GRADE_PROMPT,
    REWRITE_PROMPT,
)

logger = logging.getLogger("uvicorn.error")


# Graph state
class RAGState(TypedDict):
    question:      str
    documents:     List[Any]
    answer:        Optional[str]
    iterations:    int  # Classic RAG iterations (retrieve/rewrite)
    grade_reason:  Optional[str]
    question_type: Optional[str]
    # Agentic RAG fields
    need_more_details: Optional[bool]
    selected_source:   Optional[str]
    source_reason:     Optional[str]
    sources_tried:     List[str]
    external_data:     Optional[Any]
    audit_decision:    Optional[str]  # "end" or "rewrite"
    agentic_iterations: int  # Agentic source selection iterations


# Builder
def build_rag_graph(
    nlp_controller,
    project,
    source_router=None,
    use_advanced_retrieval: bool = True,
    max_iterations:         int  = 1,  # Classic RAG iterations
    max_agentic_iterations: int  = 2,  # Agentic source selection iterations
    retrieval_top_k:        int  = 5,
    question_type:          str  = None,
    enable_source_selection: bool = False,
):
    """
    Build RAG graph with optional dynamic source selection.
    
    Parameters
    ----------
    nlp_controller         : NLPController instance
    project                : Project ORM object
    source_router          : SourceRouter instance (optional, for agentic features)
    use_advanced_retrieval : use hybrid+RRF+reranker pipeline if available
    max_iterations         : max classic RAG loops (retrieve/rewrite) before giving up
    max_agentic_iterations : max agentic source selection attempts
    retrieval_top_k        : number of docs to retrieve per iteration
    question_type          : question type for prompt selection
    enable_source_selection: enable dynamic source selection (requires source_router)
    """
    gen = nlp_controller.generation_client

    # ── Node: Retrieve ────────────────────────────────────────
    async def retrieve(state: RAGState) -> RAGState:
        query = state["question"]
        logger.info(f"[Graph] Retrieve | query: {query[:80]}")

        try:
            if use_advanced_retrieval and hasattr(nlp_controller, "advanced_retrieve"):
                docs = await nlp_controller.advanced_retrieve(
                    project=project,
                    query=query,
                    top_k=retrieval_top_k,
                )
            else:
                docs = await nlp_controller.search_vector_db_collection(
                    project=project,
                    text=query,
                    limit=retrieval_top_k,
                )
        except Exception as exc:
            logger.error(f"[Graph] Retrieve failed: {exc}")
            docs = []

        # Track sources tried
        sources_tried = state.get("sources_tried", [])
        if "vector_db" not in sources_tried:
            sources_tried.append("vector_db")

        return {
            **state,
            "documents": docs or [],
            "sources_tried": sources_tried,
        }

    # ── Node: Grade documents ─────────────────────────────────
    async def grade_documents(state: RAGState) -> RAGState:
        logger.info(f"[Graph] Grading {len(state['documents'])} documents")
        filtered = []

        for doc in state["documents"]:
            text = doc.text if hasattr(doc, "text") else str(doc)
            prompt = RETRIEVAL_GRADE_PROMPT.format(
                document=text, 
                question=state["question"],
            )
            try:
                result = gen.generate_json(prompt=prompt)
                if result.get("score") == "yes":
                    filtered.append(doc)
                else:
                    logger.debug(f"[Graph] Doc rejected: {result.get('reason', '')}")
            except Exception as exc:
                logger.warning(f"[Graph] Grading error (keeping doc): {exc}")
                filtered.append(doc)  

        logger.info(f"[Graph] Kept {len(filtered)}/{len(state['documents'])} docs")
        return {**state, "documents": filtered}

    # ── Node: Generate ────────────────────────────────────────
    async def generate(state: RAGState) -> RAGState:
        logger.info("[Graph] Generating answer")

        if not state["documents"]:
            prompt = NO_CONTEXT_PROMPT.format(
                iterations=state["iterations"],
                question=state["question"],
            )
            answer = gen.generate_text(prompt=prompt, chat_history=[])
            logger.info("[Graph] No context answer: " + answer)
            return {**state, "answer": answer}

        # Use template parser for question-type-specific prompts
        template_parser = nlp_controller.template_parser
        
        # Get system prompt
        system_prompt = template_parser.get("rag", "system_prompt")
        
        # Build document context
        document_prompt = "\n".join([
            template_parser.get("rag", "document_prompt", {
                "doc_num":    i + 1,
                "chunk_text": doc.text if hasattr(doc, "text") else str(doc),
            })
            for i, doc in enumerate(state["documents"])
        ])
        
        # Select footer based on question type
        footer_key = "footer_default"
        q_type = state.get("question_type")
        if q_type:
            type_to_footer = {
                "factual": "footer_factual",
                "analytical": "footer_analytical",
                "comparative": "footer_comparative",
                "summarization": "footer_summarization",
                "hallucination": "footer_hallucination",
            }
            footer_key = type_to_footer.get(q_type.lower(), "footer_default")
        
        footer_prompt = template_parser.get("rag", footer_key, {"query": state["question"]})
        
        # Build chat history
        chat_history = [
            gen.construct_prompt(
                prompt=system_prompt,
                role=gen.enums.SYSTEM.value,
            )
        ]
        
        # Combine prompts
        full_prompt = "\n\n".join([document_prompt, footer_prompt])
        answer = gen.generate_text(prompt=full_prompt, chat_history=chat_history)
        
        logger.info(f"[Graph] Generated answer using {footer_key}")
        return {**state, "answer": answer}

    # ── Node: Rewrite query ───────────────────────────────────
    async def rewrite_query(state: RAGState) -> RAGState:
        new_iter = state["iterations"] + 1
        logger.info(f"[Graph] Rewriting query (iteration {new_iter})")

        prompt = REWRITE_PROMPT.format(query=state["question"])
        try:
            new_query = gen.generate_text(prompt=prompt, chat_history=[])
            new_query = (new_query or state["question"]).strip()
        except Exception as exc:
            logger.warning(f"[Graph] Query rewrite failed: {exc}")
            new_query = state["question"]

        return {
            **state,
            "question":   new_query,
            "iterations": new_iter,
            "documents":  [],
            "need_more_details": None,
            "selected_source": None,
        }

    # ── Node: Evaluate context (Agentic) ─────────────────────
    async def evaluate_context(state: RAGState) -> RAGState:
        """Evaluate if current context is sufficient."""
        if not enable_source_selection or not source_router:
            return {**state, "need_more_details": False}
        
        logger.info("[Graph] Evaluating if more details are needed")

        context = "\n\n".join([
            doc.text if hasattr(doc, "text") else str(doc)
            for doc in state.get("documents", [])
        ])

        if not context:
            context = "No context available"

        # Always ask LLM to decide, even if no documents
        # LLM will determine if question is in-scope or out-of-scope
        need_more, reason = await source_router.decide_need_more_details(
            question=state["question"],
            context=context,
            answer=state.get("answer"),
        )

        logger.info(f"[Graph] Need more: {need_more} - {reason}")

        return {
            **state,
            "need_more_details": need_more,
            "source_reason": reason,
        }

    # ── Node: Route source (Agentic) ─────────────────────────
    async def route_source(state: RAGState) -> RAGState:
        """Select the best source for additional information."""
        if not enable_source_selection or not source_router:
            return state
        
        logger.info("[Graph] Selecting information source")

        context = "\n\n".join([
            doc.text if hasattr(doc, "text") else str(doc)
            for doc in state.get("documents", [])
        ])

        selection = await source_router.select_source(
            question=state["question"],
            context=context,
            previous_sources=state.get("sources_tried", []),
        )

        logger.info(f"[Graph] Selected: {selection['source']}")

        return {
            **state,
            "selected_source": selection["source"],
            "source_reason": selection.get("reason", ""),
        }

    # ── Node: Fetch data (Agentic) ───────────────────────────
    async def fetch_data(state: RAGState) -> RAGState:
        """Fetch information from the selected source (internet only)."""
        if not enable_source_selection or not source_router:
            return state
        
        source = state.get("selected_source", "internet")
        logger.info(f"[Graph] Fetching from source: {source}")

        sources_tried = state.get("sources_tried", [])
        if source not in sources_tried:
            sources_tried.append(source)

        try:
            # Only internet is available
            if source == "internet":
                # Search internet
                logger.info("[Graph] Searching internet")
                internet_result = await source_router._fetch_from_internet({
                    "query": state["question"],
                })
                external_data = internet_result
            else:
                # Fallback to internet if other source was selected
                logger.warning(f"[Graph] Source '{source}' not available, using internet")
                internet_result = await source_router._fetch_from_internet({
                    "query": state["question"],
                })
                external_data = internet_result

            # Convert external data to document format
            new_docs = []
            if external_data:
                # Check if it's an InternetResult
                if hasattr(external_data, 'success') and hasattr(external_data, 'content'):
                    if external_data.success:
                        from retrieval.hybrid_search import SearchResult
                        new_docs = [SearchResult(
                            text=external_data.content,
                            score=1.0,
                            metadata={
                                "source": "internet",
                                "backend": external_data.backend,
                                "urls": external_data.sources,
                            },
                            source="internet",
                        )]

            logger.info(f"[Graph] Fetched {len(new_docs)} items from internet")

            all_docs = state.get("documents", []) + new_docs
            
            # Increment agentic iterations
            agentic_iters = state.get("agentic_iterations", 0) + 1

            return {
                **state,
                "documents": all_docs,
                "external_data": external_data,
                "sources_tried": sources_tried,
                "agentic_iterations": agentic_iters,
            }

        except Exception as exc:
            logger.info(f"[Graph] Fetch from internet failed: {exc}")
            return {
                **state,
                "sources_tried": sources_tried,
            }

    # ── Conditional: after grade_documents ───────────────────
    def decide_after_grading(state: RAGState) -> str:
        if not state["documents"]:
            if state["iterations"] >= max_iterations:
                return "generate"    # emit "no info" message
            return "rewrite_query"
        return "generate"

    # ── Conditional: after generate ──────────────────────────
    def decide_after_generation(state: RAGState) -> str:
        """
        After generation, decide whether to check for more details or audit.
        """
        answer = state.get("answer") or ""

        if not answer.strip():
            return END

        # If source selection is enabled, check if we need more details
        if enable_source_selection:
            # If we already fetched external data, skip evaluation and go to audit
            # This prevents infinite loops
            sources_tried = state.get("sources_tried", [])
            if len(sources_tried) > 1:  # More than just vector_db
                logger.info("[Graph] Already fetched external data, going to audit")
                return "audit_answer"
            
            # First time: evaluate if we need more
            return "evaluate_context"
        else:
            # Classic flow: go to audit
            return "audit_answer"

    # ── Conditional: after evaluate_context ──────────────────
    def decide_after_evaluation(state: RAGState) -> str:
        """
        Route based on whether more details are needed.
        """
        if not enable_source_selection:
            return "audit_answer"

        need_more = state.get("need_more_details", False)

        if need_more:
            # Check agentic iterations (separate from classic iterations)
            agentic_iters = state.get("agentic_iterations", 0)
            if agentic_iters >= max_agentic_iterations:
                logger.info(f"[Graph] Max agentic iterations ({max_agentic_iterations}) reached, proceeding to audit")
                return "audit_answer"
            
            # Check if we've tried too many sources
            sources_tried = state.get("sources_tried", [])
            if len(sources_tried) >= 3:  # Max 3 sources
                logger.info("[Graph] Max sources tried, proceeding to audit")
                return "audit_answer"
            
            return "route_source"
        else:
            return "audit_answer"

    # ── Conditional: after audit_answer ──────────────────────
    def decide_after_audit(state: RAGState) -> str:
        """
        Decide whether to rewrite query or end based on audit results.
        """
        audit_decision = state.get("audit_decision", "end")
        
        if audit_decision == "rewrite":
            return "rewrite_query"
        else:
            return END

    # ── Conditional: audit answer (hallucination + quality) ──
    async def audit_answer(state: RAGState) -> RAGState:
        """
        Audit the answer for hallucination and quality.
        Returns updated state with decision to continue or rewrite.
        """
        answer = state.get("answer") or ""

        if not answer.strip():
            return {**state, "audit_decision": "end"}

        # No documents used → skip grounding check
        if not state["documents"]:
            return {**state, "audit_decision": "end"}

        context = "\n\n".join([
            doc.text if hasattr(doc, "text") else str(doc)
            for doc in state["documents"]
        ])

        # ── Hallucination check ───────────────────────────────
        try:
            hall = gen.generate_json(
                prompt=HALLUCINATION_PROMPT.format(
                    documents=context,
                    answer=answer,
                )
            )
            if hall.get("score") == "no":
                reason = hall.get("reason", "")
                logger.warning(f"[Graph] Hallucination detected: {reason}")
                if state["iterations"] >= max_iterations:
                    return {**state, "audit_decision": "end"}
                return {**state, "audit_decision": "rewrite"}
        except Exception as exc:
            logger.warning(f"[Graph] Hallucination check failed (skipping): {exc}")

        # ── Answer quality check ──────────────────────────────
        try:
            ans = gen.generate_json(
                prompt=ANSWER_GRADE_PROMPT.format(
                    question=state["question"],
                    answer=answer,
                )
            )
            
            if ans.get("score") == "yes":
                logger.info("[Graph] Answer accepted")
                return {**state, "audit_decision": "end"}

            if state["iterations"] >= max_iterations:
                logger.info("[Graph] Max iterations reached")
                return {**state, "audit_decision": "end"}

            return {**state, "audit_decision": "rewrite"}

        except Exception as exc:
            logger.warning(f"[Graph] Answer grading failed (accepting answer): {exc}")
            return {**state, "audit_decision": "end"}

    # ── Build graph ───────────────────────────────────────────
    graph = StateGraph(RAGState)

    # Add all nodes
    graph.add_node("retrieve",        retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate",        generate)
    graph.add_node("rewrite_query",   rewrite_query)
    graph.add_node("audit_answer",    audit_answer)  # Always add audit node

    # Add agentic nodes (only used when enable_source_selection=True)
    if enable_source_selection:
        graph.add_node("evaluate_context", evaluate_context)
        graph.add_node("route_source",     route_source)
        graph.add_node("fetch_data",       fetch_data)

    # Set entry point
    graph.set_entry_point("retrieve")

    # Build flow
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    if enable_source_selection:
        # Agentic flow: generate → evaluate_context → route_source → fetch_data → generate
        graph.add_conditional_edges(
            "generate",
            decide_after_generation,
            {
                "evaluate_context": "evaluate_context",
                "audit_answer": "audit_answer",
            },
        )

        graph.add_conditional_edges(
            "evaluate_context",
            decide_after_evaluation,
            {
                "route_source": "route_source",
                "audit_answer": "audit_answer",
            },
        )

        graph.add_edge("route_source", "fetch_data")
        graph.add_edge("fetch_data", "generate")  # Re-generate with new info
        
        # After re-generation with new data, go directly to audit (skip evaluate_context)
        # This prevents infinite loops
    else:
        # Classic flow: generate → audit → END or rewrite
        graph.add_conditional_edges(
            "generate",
            decide_after_generation,
            {
                "audit_answer": "audit_answer",
            },
        )

    # Audit answer edges (used by both flows)
    graph.add_conditional_edges(
        "audit_answer",
        decide_after_audit,
        {
            "rewrite_query": "rewrite_query",
            END: END,
        },
    )

    graph.add_edge("rewrite_query", "retrieve")

    return graph.compile()