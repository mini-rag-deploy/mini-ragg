from __future__ import annotations

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from .BaseController import BaseController
from models.db_schemes import Project
from evaluation.dataset_generator import DatasetGenerator, QuestionType
from evaluation.evaluator import RAGEvaluator

logger = logging.getLogger("uvicorn.error")


class EvaluationController(BaseController):
    """
    Controller for RAG system evaluation and benchmarking
    """
    
    def __init__(
        self,
        nlp_controller,
        generation_client,
        embedding_client,
        judge_client=None,  # Optional separate judge client
        evaluation_dir: str = "evaluation_results"
    ):
        super().__init__()
        self.nlp_controller = nlp_controller
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.judge_client = judge_client  # Will be passed to evaluator
        self.evaluation_dir = Path(evaluation_dir)
        self.evaluation_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.dataset_generator = None
        self.evaluator = None

    def _initialize_for_project(self, project: Project):
        """Initialize evaluation components for a specific project"""
        self.dataset_generator = DatasetGenerator(
            nlp_controller=self.nlp_controller,
            project=project,
            generation_client=self.generation_client,
            output_dir=str(self.evaluation_dir / "datasets")
        )
        
        self.evaluator = RAGEvaluator(
            nlp_controller=self.nlp_controller,
            project=project,
            generation_client=self.generation_client,
            judge_client=self.judge_client,  # Pass judge client
            output_dir=str(self.evaluation_dir / "results")
        )

    async def generate_evaluation_dataset(
        self,
        project: Project,
        dataset_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate evaluation dataset for a project
        
        Args:
            project: Project to generate dataset for
            dataset_config: Configuration for dataset generation
            
        Returns:
            Dictionary with dataset info and file path
        """
        try:
            self._initialize_for_project(project)
            
            # Default configuration
            default_config = {
                "num_questions_per_type": {
                    QuestionType.FACTUAL: 50,
                    QuestionType.ANALYTICAL: 30,
                    QuestionType.COMPARATIVE: 20,
                    QuestionType.SUMMARIZATION: 20,
                    QuestionType.HALLUCINATION: 10
                },
                "include_retrieval_examples": True,
                "languages": ["en", "ar"]
            }
            
            if dataset_config:
                default_config.update(dataset_config)
            
            logger.info(f"[EvaluationController] Generating dataset for project {project.project_id}")
            
            # Generate dataset
            dataset = await self.dataset_generator.generate_dataset(
                num_questions_per_type=default_config["num_questions_per_type"],
                include_retrieval_examples=default_config["include_retrieval_examples"],
                languages=default_config["languages"]
            )
            
            # Save dataset
            filepath = self.dataset_generator.save_dataset(dataset)
            
            return {
                "signal": "success",
                "dataset_info": {
                    "name": dataset.name,
                    "version": dataset.version,
                    "created_at": dataset.created_at,
                    "total_questions": len(dataset.questions),
                    "total_retrieval_examples": len(dataset.retrieval_examples),
                    "statistics": dataset.statistics,
                    "filepath": filepath
                },
                "message": f"Successfully generated evaluation dataset with {len(dataset.questions)} questions"
            }
            
        except Exception as e:
            logger.error(f"[EvaluationController] Error generating dataset: {e}")
            return {
                "signal": "error",
                "message": f"Failed to generate evaluation dataset: {str(e)}"
            }

    async def evaluate_rag_system(
        self,
        project: Project,
        dataset_path: Optional[str] = None,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system performance using a dataset
        
        Args:
            project: Project to evaluate
            dataset_path: Path to evaluation dataset (if None, generates new one)
            evaluation_config: Configuration for evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            self._initialize_for_project(project)
            
            # Load or generate dataset
            if dataset_path:
                logger.info(f"[EvaluationController] Loading dataset from {dataset_path}")
                dataset = DatasetGenerator.load_dataset(dataset_path)
            else:
                logger.info("[EvaluationController] Generating new dataset for evaluation")
                dataset_result = await self.generate_evaluation_dataset(project)
                if dataset_result["signal"] != "success":
                    return dataset_result
                dataset = DatasetGenerator.load_dataset(dataset_result["dataset_info"]["filepath"])
            
            # Default evaluation configuration
            default_config = {
                "evaluate_retrieval": True,
                "evaluate_generation": True,
                "evaluate_end_to_end": True,
                "max_questions": None,  # Evaluate all questions
                "use_advanced_retrieval": True
            }
            
            if evaluation_config:
                default_config.update(evaluation_config)
            
            logger.info(f"[EvaluationController] Starting evaluation with {len(dataset.questions)} questions")
            
            # Run evaluation
            results = await self.evaluator.evaluate_dataset(dataset, default_config)
            
            # Save results
            results_path = self.evaluator.save_results(results)
            
            # Serialize results to ensure all objects are JSON-serializable
            serialized_results = self.evaluator._serialize_results(results)
            
            return {
                "signal": "success",
                "evaluation_results": {
                    "overall_score": serialized_results["overall_score"],
                    "retrieval_metrics": serialized_results["retrieval_metrics"],
                    "generation_metrics": serialized_results["generation_metrics"],
                    "end_to_end_metrics": serialized_results["end_to_end_metrics"],
                    "question_type_breakdown": serialized_results["question_type_breakdown"],
                    "language_breakdown": serialized_results["language_breakdown"],
                    "detailed_results": serialized_results["detailed_results"][:10],  # First 10 for preview
                    "results_filepath": results_path,
                    "evaluation_config": default_config,
                    "dataset_info": {
                        "name": dataset.name,
                        "total_questions": len(dataset.questions),
                        "statistics": dataset.statistics
                    }
                },
                "message": f"Evaluation completed. Overall score: {results.overall_score:.2f}"
            }
            
        except Exception as e:
            logger.error(f"[EvaluationController] Error during evaluation: {e}")
            return {
                "signal": "error",
                "message": f"Failed to evaluate RAG system: {str(e)}"
            }

