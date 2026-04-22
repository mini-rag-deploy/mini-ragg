"""
RAG Evaluation Module

Comprehensive evaluation framework for RAG systems including:
- Dataset generation from indexed documents
- Multi-metric evaluation (retrieval, generation, end-to-end)
- Benchmarking and comparison tools
- Report generation and export

Components:
- dataset_generator: Generate evaluation datasets
- evaluator: Run comprehensive evaluations
- cli: Command-line interface for evaluations
"""

from .dataset_generator import (
    DatasetGenerator,
    EvaluationDataset,
    EvaluationQuestion,
    RetrievalExample,
    QuestionType,
    DifficultyLevel
)

from .evaluator import (
    RAGEvaluator,
    EvaluationResults
)

__all__ = [
    "DatasetGenerator",
    "EvaluationDataset", 
    "EvaluationQuestion",
    "RetrievalExample",
    "QuestionType",
    "DifficultyLevel",
    "RAGEvaluator",
    "EvaluationResults"
]