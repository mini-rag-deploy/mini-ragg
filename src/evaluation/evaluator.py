"""
RAG System Evaluator

Comprehensive evaluation of RAG systems including:
- Retrieval accuracy (precision, recall, F1)
- Generation quality (BLEU, ROUGE, semantic similarity)
- End-to-end performance
- Hallucination detection
- Multi-language evaluation support
"""

from __future__ import annotations

import json
import logging
import asyncio
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger("uvicorn.error")


@dataclass
class EvaluationResults:
    """Complete evaluation results"""
    timestamp: str
    overall_score: float
    total_questions: int
    retrieval_metrics: Dict[str, float]
    generation_metrics: Dict[str, float]
    end_to_end_metrics: Dict[str, float]
    question_type_breakdown: Dict[str, Dict[str, Any]]
    language_breakdown: Dict[str, Dict[str, Any]]
    detailed_results: List[Dict[str, Any]]
    config: Dict[str, Any]


class RAGEvaluator:
    """
    Comprehensive RAG system evaluator
    """
    
    def __init__(
        self,
        nlp_controller,
        project,
        generation_client,
        
        output_dir: str = "evaluation_results",
        judge_client  = None,
    ):
        self.nlp_controller = nlp_controller
        self.project = project
        self.generation_client = generation_client

        self.judge_client      = judge_client or generation_client

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create project-specific directory
        self.project_dir = self.output_dir / f"project_{project.project_id}"
        self.project_dir.mkdir(exist_ok=True)

        # generation metrics evaluator
        from evaluation.generation_metrics import GenerationMetricsEvaluator
        self._gen_metrics = GenerationMetricsEvaluator(
            generation_client = generation_client,
            judge_client      = self.judge_client,
            embedding_client  = getattr(nlp_controller, "embedding_client", None),
        )
        logger.info(
            "[RAGEvaluator] GenerationMetricsEvaluator wired "
            f"(judge={'same as gen' if judge_client is None else 'separate model'})"
        )

    async def evaluate_dataset(
        self,
        dataset,
        config: Dict[str, Any]
    ) -> EvaluationResults:
        """
        Evaluate RAG system using the provided dataset
        """
        logger.info(f"[RAGEvaluator] Starting evaluation with {len(dataset.questions)} questions")
        
        # Limit questions if specified
        questions_to_evaluate = dataset.questions
        if config.get("max_questions"):
            questions_to_evaluate = questions_to_evaluate[:config["max_questions"]]
        
        # Initialize results storage
        detailed_results = []
        retrieval_scores = []
        generation_scores = []
        end_to_end_scores = []
        
        # Evaluate each question
        for i, question in enumerate(questions_to_evaluate):
            logger.info(f"[RAGEvaluator] Evaluating question {i+1}/{len(questions_to_evaluate)}")
            
            try:
                # Evaluate single question
                result = await self._evaluate_single_question(question, config)
                detailed_results.append(result)
                
                # Collect scores
                if result.get("retrieval_score") is not None:
                    retrieval_scores.append(result["retrieval_score"])
                if result.get("generation_score") is not None:
                    generation_scores.append(result["generation_score"])
                if result.get("end_to_end_score") is not None:
                    end_to_end_scores.append(result["end_to_end_score"])
                    
                # Calculate aggregate metrics
                retrieval_metrics = self._calculate_retrieval_metrics(detailed_results, dataset.retrieval_examples)
                generation_metrics = self._calculate_generation_metrics(detailed_results)
                end_to_end_metrics = self._calculate_end_to_end_metrics(detailed_results)
                
                # Calculate overall score
                overall_score = self._calculate_overall_score(
                    retrieval_metrics, generation_metrics, end_to_end_metrics
                )
                
                # Create results object
                results = EvaluationResults(
                    timestamp=datetime.now().isoformat(),
                    overall_score=overall_score,
                    total_questions=len(detailed_results),
                    retrieval_metrics=retrieval_metrics,
                    generation_metrics=generation_metrics,
                    end_to_end_metrics=end_to_end_metrics,
                    question_type_breakdown={},
                    language_breakdown={},
                    detailed_results=detailed_results,
                    config=config
                )

                # Save results
                results_path = self.save_results(results)
                
                # Serialize results to ensure all objects are JSON-serializable
                serialized_results = self._serialize_results(results)

                logger.info(f"[RAGEvaluator] evaluating question {question.id} completed. Overall score: {overall_score:.3f}")
            
            except Exception as e:
                logger.error(f"[RAGEvaluator] Error evaluating question {question.id}: {e}")
                continue
                
        logger.info(f"[RAGEvaluator] Evaluation completed. Overall score: {overall_score:.3f}")
        
        return results

    async def _evaluate_single_question(
        self,
        question,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a single question across all metrics
        """
        result = {
            "question_id": question.id,
            "question": question.question,
            "reference_answer": question.reference_answer,
            "question_type": question.question_type.value,
            "difficulty": question.difficulty.value,
            "language": question.language,
            "source_chunks": question.source_chunks,
            "source_documents": question.source_documents
        }
        
        try:
            # # 1. Evaluate Retrieval (if enabled)
            if config.get("evaluate_retrieval", True):
                retrieval_result = await self._evaluate_retrieval(question, config)
                result.update(retrieval_result)
            
            # 2. Evaluate Generation (DISABLED FOR NOW)
            if config.get("evaluate_generation", True):
                generation_result = await self._evaluate_generation(question, config)
                result.update(generation_result)
            
            # 3. Evaluate End-to-End (DISABLED FOR NOW)
            if config.get("evaluate_end_to_end", True):
                e2e_result = await self._evaluate_end_to_end(question, config)
                result.update(e2e_result)
            
        except Exception as e:
            logger.error(f"[RAGEvaluator] Error in single question evaluation: {e}")
            result["error"] = str(e)
        
        return result

    async def _evaluate_retrieval(
        self,
        question,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval performance for a question
        """
        try:
            # Perform retrieval using the configured method
            flag = 1
            retrieved_docs = None
            pre_rerank_docs = None
            rerank_time_ms = 0.0
            
            if flag and config.get("use_advanced_retrieval", True) and hasattr(self.nlp_controller, "advanced_retrieve"):
                # For reranker evaluation, we need both pre and post rerank results
                import time
                
                # First: Get pre-rerank results (without reranker)
                if hasattr(self.nlp_controller, 'reranker') and self.nlp_controller.reranker:
                    # Temporarily disable reranker to get pre-rerank results
                    original_reranker = self.nlp_controller.reranker
                    self.nlp_controller.reranker = None
                    
                    pre_rerank_docs = await self.nlp_controller.advanced_retrieve(
                        project=self.project,
                        query=question.question,
                        top_k=10
                    )
                    
                    # Restore reranker and measure only reranking time
                    self.nlp_controller.reranker = original_reranker
                    
                    # Now measure reranking time only
                    rerank_start = time.time()
                    retrieved_docs = self.nlp_controller.reranker.rerank(
                        query=question.question,
                        candidates=pre_rerank_docs,
                        top_k=10
                    )
                    rerank_time_ms = (time.time() - rerank_start) * 1000
                else:
                    # No reranker, just do normal retrieval
                    retrieved_docs = await self.nlp_controller.advanced_retrieve(
                        project=self.project,
                        query=question.question,
                        top_k=10
                    )
                    rerank_time_ms = 0.0
                
            else:
                retrieved_docs = await self.nlp_controller.search_vector_db_collection(
                    project=self.project,
                    text=question.question,
                    limit=10
                )
            
            # Extract retrieved chunk IDs (maintain order for ranking metrics)
            retrieved_chunk_ids = []
            for doc in retrieved_docs:
                chunk_id = None
                
                # Handle SearchResult objects (from advanced_retrieve)
                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                    chunk_id = doc.metadata.get('chunk_id')
                    if not chunk_id:
                        logger.warning(f"[RAGEvaluator] SearchResult has no chunk_id in metadata: {doc.metadata.keys()}")
                # Handle RetrievedDocument objects
                elif hasattr(doc, 'chunk_id') and doc.chunk_id:
                    chunk_id = doc.chunk_id
                elif hasattr(doc, 'id'):
                    chunk_id = doc.id
                # Handle dict objects
                elif isinstance(doc, dict):
                    chunk_id = doc.get('chunk_id') or doc.get('id') or doc.get('metadata', {}).get('chunk_id')
                else:
                    logger.warning(f"[RAGEvaluator] Unknown doc type: {type(doc)}, attributes: {dir(doc) if hasattr(doc, '__dict__') else 'N/A'}")
                
                if chunk_id:
                    retrieved_chunk_ids.append(str(chunk_id))
            
            # Extract pre-rerank chunk IDs if available
            pre_rerank_chunk_ids = []
            if pre_rerank_docs:
                for doc in pre_rerank_docs:
                    chunk_id = None
                    if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                        chunk_id = doc.metadata.get('chunk_id')
                    elif hasattr(doc, 'chunk_id') and doc.chunk_id:
                        chunk_id = doc.chunk_id
                    elif hasattr(doc, 'id'):
                        chunk_id = doc.id
                    elif isinstance(doc, dict):
                        chunk_id = doc.get('chunk_id') or doc.get('id') or doc.get('metadata', {}).get('chunk_id')
                    
                    if chunk_id:
                        pre_rerank_chunk_ids.append(str(chunk_id))
            
            logger.info(f"[RAGEvaluator] Extracted {len(retrieved_chunk_ids)} chunk IDs from {len(retrieved_docs)} docs")
            
            # Calculate retrieval metrics
            relevant_chunks = set(question.source_chunks)
            retrieved_chunks = set(retrieved_chunk_ids)
            
            # Precision: How many retrieved docs are relevant?
            if retrieved_chunks:
                precision = len(relevant_chunks.intersection(retrieved_chunks)) / len(retrieved_chunks)
            else:
                precision = 0.0
            
            # Recall: How many relevant docs were retrieved?
            if relevant_chunks:
                recall = len(relevant_chunks.intersection(retrieved_chunks)) / len(relevant_chunks)
            else:
                recall = 1.0 if not retrieved_chunks else 0.0
            
            # F1 Score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            # Calculate retrieval score (weighted average)
            retrieval_score = 0.3 * precision + 0.4 * recall + 0.3 * f1
            
            # ── Calculate ranking metrics @K=5 ──────────────
            k = 5
            top_k_retrieved = retrieved_chunk_ids[:k]  # First K results
            top_k_retrieved_set = set(top_k_retrieved)
            
            # Precision@K: How many of top-K retrieved docs are relevant?
            if top_k_retrieved_set:
                precision_at_k = len(relevant_chunks.intersection(top_k_retrieved_set)) / len(top_k_retrieved_set)
            else:
                precision_at_k = 0.0
            
            # Recall@K: How many relevant docs were retrieved in top-K?
            if relevant_chunks:
                recall_at_k = len(relevant_chunks.intersection(top_k_retrieved_set)) / len(relevant_chunks)
            else:
                recall_at_k = 1.0 if not top_k_retrieved_set else 0.0
            
            # Hit Rate@K: Did at least one relevant doc appear in top-K?
            hit_rate_at_k = 1.0 if any(chunk_id in relevant_chunks for chunk_id in top_k_retrieved) else 0.0
            
            # MRR@K: Mean Reciprocal Rank - position of first relevant doc
            mrr_at_k = 0.0
            for rank, chunk_id in enumerate(top_k_retrieved, start=1):
                if chunk_id in relevant_chunks:
                    mrr_at_k = 1.0 / rank
                    break
            
            # NDCG@K: Normalized Discounted Cumulative Gain
            # DCG@K = sum(rel_i / log2(i+1)) for i in 1..K
            # IDCG@K = DCG for perfect ranking
            dcg_at_k = 0.0
            for rank, chunk_id in enumerate(top_k_retrieved, start=1):
                relevance = 1.0 if chunk_id in relevant_chunks else 0.0
                dcg_at_k += relevance / np.log2(rank + 1)
            
            # Ideal DCG: all relevant docs at top positions
            num_relevant_in_topk = min(len(relevant_chunks), k)
            idcg_at_k = sum(1.0 / np.log2(i + 1) for i in range(1, num_relevant_in_topk + 1))
            
            ndcg_at_k = dcg_at_k / idcg_at_k if idcg_at_k > 0 else 0.0
            
            # ── Reranker Evaluation Metrics ──────────────
            reranker_metrics = {}
            
            if pre_rerank_chunk_ids and len(pre_rerank_chunk_ids) > 0:
                # Calculate metrics before reranking
                pre_rerank_top_k = pre_rerank_chunk_ids[:k]
                
                # MRR before reranking
                mrr_before = 0.0
                for rank, chunk_id in enumerate(pre_rerank_top_k, start=1):
                    if chunk_id in relevant_chunks:
                        mrr_before = 1.0 / rank
                        break
                
                # NDCG before reranking
                dcg_before = 0.0
                for rank, chunk_id in enumerate(pre_rerank_top_k, start=1):
                    relevance = 1.0 if chunk_id in relevant_chunks else 0.0
                    dcg_before += relevance / np.log2(rank + 1)
                
                ndcg_before = dcg_before / idcg_at_k if idcg_at_k > 0 else 0.0
                
                # Precision@K before reranking
                pre_rerank_top_k_set = set(pre_rerank_top_k)
                precision_before = len(relevant_chunks.intersection(pre_rerank_top_k_set)) / len(pre_rerank_top_k_set) if pre_rerank_top_k_set else 0.0
                
                # Calculate lifts
                reranker_metrics = {
                    "reranker_mrr_before": mrr_before,
                    "reranker_mrr_after": mrr_at_k,
                    "reranker_mrr_lift": mrr_at_k - mrr_before,
                    "reranker_ndcg_before": ndcg_before,
                    "reranker_ndcg_after": ndcg_at_k,
                    "reranker_ndcg_lift": ndcg_at_k - ndcg_before,
                    "reranker_precision_before": precision_before,
                    "reranker_precision_after": precision_at_k,
                    "reranker_precision_lift": precision_at_k - precision_before,
                    "reranker_latency_ms": rerank_time_ms,
                }
                
                # Position bias audit: check if reranker promoted results from lower positions
                # Find relevant docs and their position changes
                position_changes = []
                for chunk_id in relevant_chunks:
                    if chunk_id in pre_rerank_chunk_ids and chunk_id in retrieved_chunk_ids:
                        old_pos = pre_rerank_chunk_ids.index(chunk_id) + 1
                        new_pos = retrieved_chunk_ids.index(chunk_id) + 1
                        position_changes.append({
                            "chunk_id": chunk_id,
                            "old_position": old_pos,
                            "new_position": new_pos,
                            "position_change": old_pos - new_pos  # Positive = promoted
                        })
                
                if position_changes:
                    avg_position_change = sum(pc["position_change"] for pc in position_changes) / len(position_changes)
                    promoted_from_deep = sum(1 for pc in position_changes if pc["old_position"] >= 10 and pc["new_position"] <= 5)
                    
                    reranker_metrics.update({
                        "reranker_avg_position_change": avg_position_change,
                        "reranker_promoted_from_deep": promoted_from_deep,
                        "reranker_position_changes": position_changes,
                    })
                
                # Reranker ROI: quality lift per unit of latency
                if rerank_time_ms > 0:
                    # ROI = (MRR lift + NDCG lift) / (latency in seconds)
                    total_quality_lift = reranker_metrics["reranker_mrr_lift"] + reranker_metrics["reranker_ndcg_lift"]
                    latency_seconds = rerank_time_ms / 1000.0
                    reranker_roi = total_quality_lift / latency_seconds if latency_seconds > 0 else 0.0
                    
                    reranker_metrics["reranker_roi"] = reranker_roi
            
            result = {
                "retrieved_chunks": retrieved_chunk_ids,
                "retrieval_precision": precision,
                "retrieval_recall": recall,
                "retrieval_f1": f1,
                "retrieval_score": retrieval_score,
                "num_retrieved": len(retrieved_chunks),
                "num_relevant": len(relevant_chunks),
                # Ranking metrics @K=5
                "precision_at_5": precision_at_k,
                "recall_at_5": recall_at_k,
                "hit_rate_at_5": hit_rate_at_k,
                "mrr_at_5": mrr_at_k,
                "ndcg_at_5": ndcg_at_k,
            }
            
            # Add reranker metrics if available
            result.update(reranker_metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"[RAGEvaluator] Error in retrieval evaluation: {e}")
            return {
                "retrieval_score": 0.0,
                "retrieval_error": str(e)
            }

    async def _evaluate_generation(
        self,
        question,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate generation quality for a question
        """
        try:
            print("##################################")
            print(f"kind of this question is: {question.question_type.value}")
            print("###################################")
            # Generate answer using the RAG system
            flag=1
            retrieved_docs = []  # Store retrieved documents
            metadata = {}
            
            if flag and hasattr(self.nlp_controller, "answer_rag_question"):
                rag_response = await self.nlp_controller.answer_rag_question(
                    project=self.project,
                    query=question.question,
                    use_advanced_retrieval=config.get("use_advanced_retrieval", True),
                    question_type=question.question_type.value  # Pass question type
                )
                # Handle tuple response (answer, metadata, chat_history)
                if isinstance(rag_response, tuple):
                    generated_answer = rag_response[0] if rag_response[0] else ""
                    metadata = rag_response[1] if len(rag_response) > 1 else {}
                    retrieved_docs = metadata.get("documents", [])
                elif isinstance(rag_response, dict):
                    generated_answer = rag_response.get("answer", "")
                    retrieved_docs = rag_response.get("documents", [])
                else:
                    generated_answer = str(rag_response)
            else:
                # Fallback: use generation client directly
                generated_answer, metadata, _ = await self.nlp_controller._answer_basic(
                    project=self.project,
                    query=question.question,
                    use_advanced_retrieval=config.get("use_advanced_retrieval", True),
                    question_type=question.question_type.value  # Pass question type
                )
                retrieved_docs = metadata.get("documents", [])
            
            if not generated_answer:
                return {
                    "generated_answer": "",
                    "generation_score": 0.0,
                    "generation_error": "No answer generated"
                }
            
            # Calculate generation metrics
            reference_answer = question.reference_answer
            
            # 1. BLEU Score (n-gram overlap)
            bleu_score = self._calculate_bleu_score(generated_answer, reference_answer)
            
            # 2. ROUGE Score (recall-oriented)
            rouge_score = self._calculate_rouge_score(generated_answer, reference_answer)
            
            # 3. Semantic Similarity (using embeddings)
            semantic_similarity = await self._calculate_semantic_similarity(
                generated_answer, reference_answer
            )
            
            # 4. Length Ratio (penalize too short/long answers)
            length_ratio = min(len(generated_answer), len(reference_answer)) / max(len(generated_answer), len(reference_answer))
            
            # 5. Hallucination Detection (for hallucination questions)
            hallucination_score = 1.0
            if question.question_type.value == "hallucination":
                # Check if the model correctly identified no answer
                # Expanded list of phrases that indicate proper hallucination detection
                no_answer_phrases = [
                    "don't have", "don't know", "cannot answer", "not available", 
                    "insufficient information", "not found in", "no information", 
                    "unable to answer", "not enough information", "cannot provide",
                    "not directly relevant", "does not seem to be", "lack sufficient",
                    "cannot determine", "unclear from", "not specified",
                    "i don't have", "i cannot", "i'm unable"
                ]
                # Check if answer contains any of these phrases
                answer_lower = generated_answer.lower()
                if any(phrase in answer_lower for phrase in no_answer_phrases):
                    hallucination_score = 1.0
                    logger.info(f"[RAGEvaluator] Hallucination correctly detected - model refused to answer")
                else:
                    hallucination_score = 0.0
                    logger.warning(f"[RAGEvaluator] Hallucination NOT detected - model answered when it shouldn't")
            
            # Calculate overall generation score
            if question.question_type.value == "hallucination":
                generation_score = hallucination_score
            else:
                generation_score = (
                    0.25 * bleu_score +
                    0.25 * rouge_score +
                    0.35 * semantic_similarity +
                    0.15 * length_ratio
                )
            
           # ── NEW: 3.1 Core Generation Metrics + 3.4 Hallucination ──────
            try:
                print("Start Core Generation Metrics:....")
                # Use the SAME documents that were used for generation
                contexts: List[str] = []
                if retrieved_docs:
                    contexts = [
                        d.text if hasattr(d, "text") else str(d)
                        for d in retrieved_docs
                    ]
                    logger.info(f"[RAGEvaluator] Using {len(contexts)} documents from generation for metrics evaluation")
                else:
                    logger.warning(f"[RAGEvaluator] No documents found in metadata, metrics may be inaccurate")

                gen_metrics = await self._gen_metrics.evaluate(
                    question        = question.question,
                    answer          = generated_answer,
                    contexts        = contexts,
                    ground_truth    = reference_answer,
                    is_unanswerable = (question.question_type.value == "hallucination"),
                )
                # Merge into return dict — overrides simple hallucination_score
                # with the full taxonomy result
                return {
                    "generated_answer":          generated_answer,
                    "semantic_similarity":       semantic_similarity,
                    "length_ratio":              length_ratio,
                    # ── 3.1 Core Generation Metrics ───────────────
                    "faithfulness":              gen_metrics.get("faithfulness", 0.0),
                    "answer_relevance":          gen_metrics.get("answer_relevance", 0.0),
                    "completeness":              gen_metrics.get("completeness", 0.0),
                    "abstention_quality":        gen_metrics.get("abstention_quality", 0.0),
                    # ── Composite ─────────────────────────────────
                    "generation_score":          gen_metrics.get("generation_score_v2", generation_score),
                }
            except Exception as gm_exc:
                logger.warning(f"[RAGEvaluator] GenerationMetrics failed (using legacy): {gm_exc}")
                # Fall through to original return below

            return {
                "generated_answer": generated_answer,
                "bleu_score": bleu_score,
                "rouge_score": rouge_score,
                "semantic_similarity": semantic_similarity,
                "length_ratio": length_ratio,
                "hallucination_score": hallucination_score,
                "generation_score": generation_score
            }
            
        except Exception as e:
            logger.error(f"[RAGEvaluator] Error in generation evaluation: {e}")
            return {
                "generated_answer": "",
                "generation_score": 0.0,
                "generation_error": str(e)
            }

    async def _evaluate_end_to_end(
        self,
        question,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate end-to-end RAG performance
        """
        try:
            # Use the full RAG pipeline
            if hasattr(self.nlp_controller, "answer_rag_question"):
                rag_response = await self.nlp_controller.answer_rag_question(
                    project=self.project,
                    query=question.question,  # Changed from 'question' to 'query'
                    use_advanced_retrieval=config.get("use_advanced_retrieval", True),
                    question_type=question.question_type.value  # Pass question type
                )
                
                # Handle tuple response (answer, metadata, chat_history)
                if isinstance(rag_response, tuple):
                    generated_answer = rag_response[0] if rag_response[0] else ""
                    metadata = rag_response[1] if len(rag_response) > 1 else {}
                    retrieved_docs = []
                elif isinstance(rag_response, dict):
                    generated_answer = rag_response.get("answer", "")
                    retrieved_docs = rag_response.get("documents", [])
                    metadata = rag_response.get("metadata", {})
                else:
                    generated_answer = str(rag_response)
                    retrieved_docs = []
                    metadata = {}
                
                # Calculate end-to-end score based on answer quality and retrieval
                if generated_answer:
                    # Semantic similarity with reference answer
                    semantic_score = await self._calculate_semantic_similarity(
                        generated_answer, question.reference_answer
                    )
                    
                    # Factual consistency (simple keyword overlap)
                    factual_score = self._calculate_factual_consistency(
                        generated_answer, question.reference_answer
                    )
                    
                    # Relevance to question
                    relevance_score = await self._calculate_semantic_similarity(
                        generated_answer, question.question
                    )
                    
                    # Combine scores
                    end_to_end_score = (
                        0.5 * semantic_score +
                        0.3 * factual_score +
                        0.2 * relevance_score
                    )
                else:
                    end_to_end_score = 0.0
                
                return {
                    "e2e_generated_answer": generated_answer,
                    "e2e_retrieved_docs": len(retrieved_docs),
                    "e2e_metadata": metadata,
                    "e2e_semantic_score": semantic_score if 'semantic_score' in locals() else 0.0,
                    "e2e_factual_score": factual_score if 'factual_score' in locals() else 0.0,
                    "e2e_relevance_score": relevance_score if 'relevance_score' in locals() else 0.0,
                    "end_to_end_score": end_to_end_score
                }
            else:
                return {
                    "end_to_end_score": 0.0,
                    "e2e_error": "RAG pipeline not available"
                }
                
        except Exception as e:
            logger.error(f"[RAGEvaluator] Error in end-to-end evaluation: {e}")
            return {
                "end_to_end_score": 0.0,
                "e2e_error": str(e)
            }

    def _calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate BLEU score (simplified n-gram overlap)"""
        try:
            # Simple BLEU-like score based on word overlap
            gen_words = set(generated.lower().split())
            ref_words = set(reference.lower().split())
            
            if not ref_words:
                return 1.0 if not gen_words else 0.0
            
            overlap = len(gen_words.intersection(ref_words))
            return overlap / len(ref_words)
            
        except Exception:
            return 0.0

    def _calculate_rouge_score(self, generated: str, reference: str) -> float:
        """Calculate ROUGE-like score (recall-oriented)"""
        try:
            # Simple ROUGE-like score
            gen_words = generated.lower().split()
            ref_words = reference.lower().split()
            
            if not ref_words:
                return 1.0 if not gen_words else 0.0
            
            # Count overlapping words
            overlap_count = 0
            for word in ref_words:
                if word in gen_words:
                    overlap_count += 1
            
            return overlap_count / len(ref_words)
            
        except Exception:
            return 0.0

    async def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Get embeddings for both texts
            from stores.llm.LLMEnums import DocumentTypeEnums
            
            embedding1 = self.nlp_controller.embedding_client.embed_text(
                text=text1,
                document_type=DocumentTypeEnums.QUERY.value
            )
            embedding2 = self.nlp_controller.embedding_client.embed_text(
                text=text2,
                document_type=DocumentTypeEnums.QUERY.value
            )
            
            if not embedding1 or not embedding2:
                return 0.0
            
            # Handle nested lists
            if isinstance(embedding1[0], list):
                embedding1 = embedding1[0]
            if isinstance(embedding2[0], list):
                embedding2 = embedding2[0]
            
            # Calculate cosine similarity
            import numpy as np
            
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            return (similarity + 1) / 2
            
        except Exception as e:
            logger.warning(f"[RAGEvaluator] Error calculating semantic similarity: {e}")
            return 0.0

    def _calculate_factual_consistency(self, generated: str, reference: str) -> float:
        """Calculate factual consistency (simple keyword-based)"""
        try:
            # Extract key facts (numbers, proper nouns, etc.)
            import re
            
            # Extract numbers
            gen_numbers = set(re.findall(r'\d+(?:\.\d+)?', generated))
            ref_numbers = set(re.findall(r'\d+(?:\.\d+)?', reference))
            
            # Extract capitalized words (potential proper nouns)
            gen_proper = set(re.findall(r'\b[A-Z][a-z]+\b', generated))
            ref_proper = set(re.findall(r'\b[A-Z][a-z]+\b', reference))
            
            # Calculate overlap
            number_overlap = len(gen_numbers.intersection(ref_numbers))
            proper_overlap = len(gen_proper.intersection(ref_proper))
            
            total_ref_facts = len(ref_numbers) + len(ref_proper)
            total_overlap = number_overlap + proper_overlap
            
            if total_ref_facts == 0:
                return 1.0
            
            return total_overlap / total_ref_facts
            
        except Exception:
            return 0.0

    def _calculate_retrieval_metrics(
        self,
        detailed_results: List[Dict[str, Any]],
        retrieval_examples: List
    ) -> Dict[str, float]:
        """Calculate aggregate retrieval metrics"""
        retrieval_scores = [r.get("retrieval_score", 0) for r in detailed_results if r.get("retrieval_score") is not None]
        precisions = [r.get("retrieval_precision", 0) for r in detailed_results if r.get("retrieval_precision") is not None]
        recalls = [r.get("retrieval_recall", 0) for r in detailed_results if r.get("retrieval_recall") is not None]
        f1s = [r.get("retrieval_f1", 0) for r in detailed_results if r.get("retrieval_f1") is not None]
        
        # Ranking metrics @K=5
        precisions_at_k = [r.get("precision_at_5", 0) for r in detailed_results if r.get("precision_at_5") is not None]
        recalls_at_k = [r.get("recall_at_5", 0) for r in detailed_results if r.get("recall_at_5") is not None]
        hit_rates = [r.get("hit_rate_at_5", 0) for r in detailed_results if r.get("hit_rate_at_5") is not None]
        mrrs = [r.get("mrr_at_5", 0) for r in detailed_results if r.get("mrr_at_5") is not None]
        ndcgs = [r.get("ndcg_at_5", 0) for r in detailed_results if r.get("ndcg_at_5") is not None]
        
        # Reranker metrics
        reranker_mrr_lifts = [r.get("reranker_mrr_lift", 0) for r in detailed_results if r.get("reranker_mrr_lift") is not None]
        reranker_ndcg_lifts = [r.get("reranker_ndcg_lift", 0) for r in detailed_results if r.get("reranker_ndcg_lift") is not None]
        reranker_precision_lifts = [r.get("reranker_precision_lift", 0) for r in detailed_results if r.get("reranker_precision_lift") is not None]
        reranker_latencies = [r.get("reranker_latency_ms", 0) for r in detailed_results if r.get("reranker_latency_ms") is not None]
        reranker_rois = [r.get("reranker_roi", 0) for r in detailed_results if r.get("reranker_roi") is not None]
        reranker_position_changes = [r.get("reranker_avg_position_change", 0) for r in detailed_results if r.get("reranker_avg_position_change") is not None]
        reranker_deep_promotions = [r.get("reranker_promoted_from_deep", 0) for r in detailed_results if r.get("reranker_promoted_from_deep") is not None]
        
        metrics = {
            "average_retrieval_score": statistics.mean(retrieval_scores) if retrieval_scores else 0.0,
            "precision": statistics.mean(precisions) if precisions else 0.0,
            "recall": statistics.mean(recalls) if recalls else 0.0,
            "f1_score": statistics.mean(f1s) if f1s else 0.0,
            "num_evaluated": len(retrieval_scores),
            # Ranking metrics @K=5
            "precision_at_5": statistics.mean(precisions_at_k) if precisions_at_k else 0.0,
            "recall_at_5": statistics.mean(recalls_at_k) if recalls_at_k else 0.0,
            "hit_rate_at_5": statistics.mean(hit_rates) if hit_rates else 0.0,
            "mrr_at_5": statistics.mean(mrrs) if mrrs else 0.0,
            "ndcg_at_5": statistics.mean(ndcgs) if ndcgs else 0.0,
        }
        
        # Add reranker metrics if available
        if reranker_mrr_lifts:
            # Calculate P99 latency
            p99_latency = np.percentile(reranker_latencies, 99) if reranker_latencies else 0.0
            
            metrics.update({
                "reranker_avg_mrr_lift": statistics.mean(reranker_mrr_lifts),
                "reranker_avg_ndcg_lift": statistics.mean(reranker_ndcg_lifts),
                "reranker_avg_precision_lift": statistics.mean(reranker_precision_lifts),
                "reranker_avg_latency_ms": statistics.mean(reranker_latencies) if reranker_latencies else 0.0,
                "reranker_p99_latency_ms": p99_latency,
                "reranker_avg_roi": statistics.mean(reranker_rois) if reranker_rois else 0.0,
                "reranker_avg_position_change": statistics.mean(reranker_position_changes) if reranker_position_changes else 0.0,
                "reranker_total_deep_promotions": sum(reranker_deep_promotions),
                "reranker_num_evaluated": len(reranker_mrr_lifts),
                # Quality assessment
                "reranker_meets_mrr_target": statistics.mean(reranker_mrr_lifts) > 0.05 if reranker_mrr_lifts else False,
                "reranker_meets_ndcg_target": statistics.mean(reranker_ndcg_lifts) > 0.04 if reranker_ndcg_lifts else False,
                "reranker_latency_acceptable": p99_latency < 500.0,  # Target: < 500ms for K=50
            })
        
        return metrics

    def _calculate_generation_metrics(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate generation metrics — includes 3.1 Core Generation Metrics."""
        def _mean(key):
            vals = [r.get(key) for r in detailed_results if r.get(key) is not None]
            return statistics.mean(vals) if vals else 0.0

        def _count(key, val):
            return sum(1 for r in detailed_results if r.get(key) == val)

        # Legacy metrics (kept for backward compat)
        generation_scores = [r.get("generation_score", 0) for r in detailed_results if r.get("generation_score") is not None]
        semantic_scores   = [r.get("semantic_similarity", 0) for r in detailed_results if r.get("semantic_similarity") is not None]

        return {
            # Legacy
            "average_generation_score": statistics.mean(generation_scores) if generation_scores else 0.0,
            "semantic_similarity":      statistics.mean(semantic_scores)   if semantic_scores   else 0.0,
            "num_evaluated":            len(generation_scores),
            # ── 3.1 Core Generation Metrics ───────────────────────
            "faithfulness":             _mean("faithfulness"),
            "answer_relevance":         _mean("answer_relevance"),
            "completeness":             _mean("completeness"),
            "abstention_quality":       _mean("abstention_quality"),
        }

    def _calculate_end_to_end_metrics(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate aggregate end-to-end metrics"""
        e2e_scores = [r.get("end_to_end_score", 0) for r in detailed_results if r.get("end_to_end_score") is not None]
        
        return {
            "average_e2e_score": statistics.mean(e2e_scores) if e2e_scores else 0.0,
            "num_evaluated": len(e2e_scores)
        }

    def _calculate_question_type_breakdown(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics breakdown by question type"""
        breakdown = {}
        
        # Group by question type
        by_type = {}
        for result in detailed_results:
            q_type = result.get("question_type", "unknown")
            if q_type not in by_type:
                by_type[q_type] = []
            by_type[q_type].append(result)
        
        # Calculate metrics for each type
        for q_type, results in by_type.items():
            scores = [r.get("end_to_end_score", r.get("generation_score", 0)) for r in results]
            breakdown[q_type] = {
                "count": len(results),
                "average_score": statistics.mean(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0
            }
        
        return breakdown

    def _calculate_language_breakdown(self, detailed_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate metrics breakdown by language"""
        breakdown = {}
        
        # Group by language
        by_language = {}
        for result in detailed_results:
            language = result.get("language", "unknown")
            if language not in by_language:
                by_language[language] = []
            by_language[language].append(result)
        
        # Calculate metrics for each language
        for language, results in by_language.items():
            scores = [r.get("end_to_end_score", r.get("generation_score", 0)) for r in results]
            breakdown[language] = {
                "count": len(results),
                "average_score": statistics.mean(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0
            }
        
        return breakdown

    def _calculate_overall_score(
        self,
        retrieval_metrics: Dict[str, float],
        generation_metrics: Dict[str, float],
        end_to_end_metrics: Dict[str, float]
    ) -> float:
        """
        Overall score — incorporates 3.1 Core Generation Metrics.
        Faithfulness is the most critical metric (hallucination prevention).
        """
        retrieval_score  = retrieval_metrics.get("average_retrieval_score", 0.0)
        generation_score = generation_metrics.get("average_generation_score", 0.0)
        e2e_score        = end_to_end_metrics.get("average_e2e_score", 0.0)

        # ── Use richer 3.1 metrics when available ─────────────────
        faithfulness   = generation_metrics.get("faithfulness", 0.0)
        ans_relevance  = generation_metrics.get("answer_relevance", 0.0)
        completeness   = generation_metrics.get("completeness", 0.0)

        has_new_metrics = faithfulness > 0 or ans_relevance > 0

        if has_new_metrics:
            # 3.1-weighted formula (without answer_correctness and hallucination)
            gen_composite = (
                0.40 * faithfulness  +   # most critical
                0.30 * ans_relevance +
                0.30 * completeness
            )
            overall = (
                0.30 * retrieval_score +
                0.40 * gen_composite   +
                0.30 * e2e_score
            )
        else:
            # Legacy formula (unchanged)
            overall = (
                0.25 * retrieval_score  +
                0.25 * generation_score +
                0.50 * e2e_score
            )

        return round(overall, 4)

    def save_results(self, results: EvaluationResults, filename: Optional[str] = None) -> str:
        """Save evaluation results to JSON file"""
        if filename is None:
            filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.project_dir / filename
        
        # Convert to dict for JSON serialization with custom handling
        results_dict = self._serialize_results(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"[RAGEvaluator] Results saved to {filepath}")
        return str(filepath)
    
    def _serialize_results(self, results: EvaluationResults) -> dict:
        """Convert EvaluationResults to JSON-serializable dict"""
        def serialize_value(obj):
            """Recursively serialize objects"""
            if hasattr(obj, 'dict'):
                # Pydantic model
                return obj.dict()
            elif hasattr(obj, '__dict__'):
                # Regular object with __dict__
                return {k: serialize_value(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            elif isinstance(obj, dict):
                return {k: serialize_value(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize_value(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # Fallback to string representation
                return str(obj)
        
        return {
            'timestamp': results.timestamp,
            'overall_score': results.overall_score,
            'total_questions': results.total_questions,
            'retrieval_metrics': serialize_value(results.retrieval_metrics),
            'generation_metrics': serialize_value(results.generation_metrics),
            'end_to_end_metrics': serialize_value(results.end_to_end_metrics),
            'question_type_breakdown': serialize_value(results.question_type_breakdown),
            'language_breakdown': serialize_value(results.language_breakdown),
            'detailed_results': serialize_value(results.detailed_results),
            'config': serialize_value(results.config)
        }

    @classmethod
    def load_results(cls, filepath: str) -> EvaluationResults:
        """Load evaluation results from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return EvaluationResults(
            timestamp=data['timestamp'],
            overall_score=data['overall_score'],
            total_questions=data['total_questions'],
            retrieval_metrics=data['retrieval_metrics'],
            generation_metrics=data['generation_metrics'],
            end_to_end_metrics=data['end_to_end_metrics'],
            question_type_breakdown=data['question_type_breakdown'],
            language_breakdown=data['language_breakdown'],
            detailed_results=data['detailed_results'],
            config=data['config']
        )