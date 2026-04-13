# src/graph/rag_graph.py

from typing import List, Optional, Any
from langgraph.graph import StateGraph, END
import logging

from .state import RAGState
from .prompts import (
    RETRIEVAL_GRADE_PROMPT,
    HALLUCINATION_PROMPT,
    ANSWER_GRADE_PROMPT,
    REWRITE_PROMPT,
    RAG_PROMPT,
    NO_CONTEXT_PROMPT
)

logger = logging.getLogger("uvicorn.error")

# ============================================
# Graph Builder
# ============================================
def build_rag_graph(nlp_controller, project, **kwargs):
    """
    nlp_controller: the existing NLPController
    project: True Project object
    """

    generation_client = nlp_controller.generation_client

    # ── Node 1: Retrieve ──────────────────────────────────────
    async def retrieve(state: RAGState) -> RAGState:
        logger.info(f"Retrieving: {state['question']}")
        docs = await nlp_controller.search_vector_db_collection(
            project=project,
            text=state["question"],
            limit=5
        )
        return {**state, "documents": docs or []}

    # ── Node 2: Grade Documents ───────────────────────────────
    async def grade_documents(state: RAGState) -> RAGState:
        logger.info("Grading documents...")
        filtered = []

        for doc in state["documents"]:
            prompt = RETRIEVAL_GRADE_PROMPT.format(
                document=doc.text,
                question=state["question"]
            )
            result = generation_client.generate_json(prompt=prompt)
            if result.get("score") == "yes":
                filtered.append(doc)

        logger.info(f"   Kept {len(filtered)}/{len(state['documents'])} docs")
        return {**state, "documents": filtered}

    # ── Node 3: Generate ──────────────────────────────────────
    async def generate(state: RAGState) -> RAGState:
        logger.info("Generating answer...")


        if not state["documents"]:

            prompt = NO_CONTEXT_PROMPT.format(
                iterations=state["iterations"],
                question=state["question"]
            )
            answer = generation_client.generate_text(
                prompt=prompt,
                chat_history=[]
            )
            return {**state, "answer": answer}
    
        context = "\n\n".join([doc.text for doc in state["documents"]])

        system_prompt = (
            "You are an expert legal assistant supporting Arabic and English documents. "
            "Answer only from the provided context."
        )
        chat_history = [
            generation_client.construct_prompt(
                prompt=system_prompt,
                role=generation_client.enums.SYSTEM.value
            )
        ]

        full_prompt = RAG_PROMPT.format(
            context=context,
            question=state["question"]
        )

        answer = generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return {**state, "answer": answer}

    # ── Node 4: Rewrite Query ─────────────────────────────────
    async def rewrite_query(state: RAGState) -> RAGState:
        new_iter = state["iterations"] + 1
        logger.info(f"Rewriting query (iteration {new_iter})...")

        prompt = REWRITE_PROMPT.format(query=state["question"])
        new_query = generation_client.generate_text(prompt=prompt, chat_history=[])

        return {
            **state,
            "question": new_query.strip() if new_query else state["question"],
            "iterations": new_iter,
            "documents": [],
        }

    # ── Conditional: after grade_documents ─────────────────────
    async def decide_after_grading(state: RAGState) -> str:
        if not state["documents"]:
            # If max iterations reached, just generate anyway (with empty docs)
            if state["iterations"] >= 3:
                return "generate"
                
            return "rewrite_query"
        return "generate"

    # ── Conditional: after generate ─────────────────────────────
    async def decide_after_generation(state: RAGState) -> str:
        if not state["answer"]:
            return END

        # 1. Hallucination check
        context = "\n\n".join([doc.text for doc in state["documents"]])
        hall_result = generation_client.generate_json(
            prompt=HALLUCINATION_PROMPT.format(
                documents=context,
                answer=state["answer"]
            )
        )

        if hall_result.get("score") == "no":
            logger.warning(f"Hallucination: {hall_result.get('reason')}")
            if state["iterations"] >= 3:
                return END
            return "rewrite_query"

        # 2. Answer quality check
        ans_result = generation_client.generate_json(
            prompt=ANSWER_GRADE_PROMPT.format(
                question=state["question"],
                answer=state["answer"]
            )
        )

        if ans_result.get("score") == "yes":
            logger.info("Answer accepted!")
            return END

        if state["iterations"] >= 3:
            logger.warning("Max iterations reached.")
            return END

        return "rewrite_query"

    # ── Build Graph ───────────────────────────────────────────
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
        {
            "generate":      "generate",
            "rewrite_query": "rewrite_query",
        }
    )

    graph.add_conditional_edges(
        "generate",
        decide_after_generation,
        {
            "rewrite_query": "rewrite_query",
            END: END,
        }
    )

    graph.add_edge("rewrite_query", "retrieve")

    return graph.compile()