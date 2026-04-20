# src/graph/rag_graph.py
"""
Self-correcting RAG graph (LangGraph).
Updated to use advanced_retrieve() when available.
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


# ─────────────────────────────────────────────
# Graph state
# ─────────────────────────────────────────────
class RAGState(TypedDict):
    question:    str
    documents:   List[Any]
    answer:      Optional[str]
    iterations:  int
    grade_reason: Optional[str]


# ─────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────
def build_rag_graph(
    nlp_controller,
    project,
    use_advanced_retrieval: bool = True,
    max_iterations:         int  = 1,
    retrieval_top_k:        int  = 5,
):
    """
    Parameters
    ----------
    nlp_controller         : NLPController instance
    project                : Project ORM object
    use_advanced_retrieval : use hybrid+RRF+reranker pipeline if available
    max_iterations         : max self-correction loops before giving up
    retrieval_top_k        : number of docs to retrieve per iteration
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

        return {**state, "documents": docs or []}

    # ── Node: Grade documents ─────────────────────────────────
    async def grade_documents(state: RAGState) -> RAGState:
        logger.info(f"[Graph] Grading {len(state['documents'])} documents")
        filtered = []

        for doc in state["documents"]:
            text = doc.text if hasattr(doc, "text") else str(doc)
            prompt = RETRIEVAL_GRADE_PROMPT.format(
                document=text[:2000],   # cap to avoid token overflow
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
                filtered.append(doc)   # keep on error — safer than discarding

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

        context = "\n\n---\n\n".join([
            doc.text if hasattr(doc, "text") else str(doc)
            for doc in state["documents"]
        ])

        system_prompt = (
            "You are an expert assistant. "
            "Answer only from the provided context."
        )
        chat_history = [
            gen.construct_prompt(
                prompt=system_prompt,
                role=gen.enums.SYSTEM.value,
            )
        ]

        prompt = RAG_PROMPT.format(
            context=context[:6000],   # guard against huge contexts
            question=state["question"],
        )
        answer = gen.generate_text(prompt=prompt, chat_history=chat_history)
        logger.info("[Graph] No context answer: " + answer)
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
        }

    # ── Conditional: after grade_documents ───────────────────
    def decide_after_grading(state: RAGState) -> str:
        if not state["documents"]:
            if state["iterations"] >= max_iterations:
                return "generate"    # emit "no info" message
            return "rewrite_query"
        return "generate"

    # ── Conditional: after generate ──────────────────────────
    async def decide_after_generation(state: RAGState) -> str:
        answer = state.get("answer") or ""

        if not answer.strip():
            return END

        # No documents used → skip grounding check
        if not state["documents"]:
            return END

        context = "\n\n".join([
            doc.text if hasattr(doc, "text") else str(doc)
            for doc in state["documents"]
        ])

        # ── Hallucination check ───────────────────────────────
        try:
            hall = gen.generate_json(
                prompt=HALLUCINATION_PROMPT.format(
                    documents=context[:4000],
                    answer=answer[:2000],
                )
            )
            if hall.get("score") == "no":
                reason = hall.get("reason", "")
                logger.warning(f"[Graph] Hallucination detected: {reason}")
                if state["iterations"] >= max_iterations:
                    return END
                return "rewrite_query"
        except Exception as exc:
            logger.warning(f"[Graph] Hallucination check failed (skipping): {exc}")

        # ── Answer quality check ──────────────────────────────
        try:
            ans = gen.generate_json(
                prompt=ANSWER_GRADE_PROMPT.format(
                    question=state["question"],
                    answer=answer[:2000],
                )
            )
            if ans.get("score") == "yes":
                logger.info("[Graph] Answer accepted")
                return END

            if state["iterations"] >= max_iterations:
                logger.info("[Graph] Max iterations reached")
                return END

            return "rewrite_query"

        except Exception as exc:
            logger.warning(f"[Graph] Answer grading failed (accepting answer): {exc}")
            return END

    # ── Build graph ───────────────────────────────────────────
    graph = StateGraph(RAGState)

    graph.add_node("retrieve",        retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate",        generate)
    graph.add_node("rewrite_query",   rewrite_query)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_conditional_edges(
        "generate",
        decide_after_generation,
        {"rewrite_query": "rewrite_query", END: END},
    )
    graph.add_edge("rewrite_query", "retrieve")

    return graph.compile()