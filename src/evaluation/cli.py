#!/usr/bin/env python3
"""
RAG Evaluation CLI Tool

Command-line interface for running RAG evaluations:
- Generate evaluation datasets
- Run performance evaluations
- Benchmark retrieval methods
- Export reports

Usage:
    python -m evaluation.cli generate-dataset --project-id 3 --output dataset.json
    python -m evaluation.cli evaluate --project-id 3 --dataset dataset.json
    python -m evaluation.cli benchmark --project-id 3 --methods basic_vector,hybrid_search
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.config import Settings
from controllers.EvaluationController import EvaluationController
from controllers.NLPController import NLPController
from models.db_schemes import Project
from evaluation.dataset_generator import QuestionType
from stores.llm import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EvaluationCLI:
    """Command-line interface for RAG evaluation"""
    
    def __init__(self):
        self.settings = Settings()
        self.db_engine = None
        self.db_client = None
        self.nlp_controller = None
        self.evaluation_controller = None
    
    async def initialize(self):
        """Initialize database and controllers"""
        try:
            # Database setup
            postgres_conn = (
                f"postgresql+asyncpg://{self.settings.POSTGRES_USERNAME}:"
                f"{self.settings.POSTGRES_PASSWORD}@{self.settings.POSTGRES_HOST}:"
                f"{self.settings.POSTGRES_PORT}/{self.settings.POSTGRES_MAIN_DATABASE}"
            )
            self.db_engine = create_async_engine(postgres_conn)
            self.db_client = sessionmaker(
                self.db_engine, expire_on_commit=False, class_=AsyncSession
            )
            
            # LLM and Vector DB setup
            llm_provider_factory = LLMProviderFactory(self.settings)
            vector_db_provider_factory = VectorDBProviderFactory(self.settings, db_client=self.db_client)
            
            # Generation client (Cohere for generation)
            generation_client = llm_provider_factory.create(provider=self.settings.GENERATION_BACKEND)
            generation_client.set_generation_model(model_id=self.settings.GENERATION_MODEL_ID)
            
            # Judge client (Ollama for evaluation - more capable/faster)
            judge_client = llm_provider_factory.create(provider="OPENAI")
            judge_client.set_generation_model(model_id="llama3.2:latest")
            
            # Embedding client
            embedding_client = llm_provider_factory.create(provider=self.settings.EMBEDDING_BACKEND)
            embedding_client.set_embedding_model(
                model_id=self.settings.EMBEDDING_MODEL_ID,
                embedding_size=self.settings.EMBEDDING_MODEL_SIZE
            )
            
            # Vector database client
            vectordb_client = vector_db_provider_factory.create(provider=self.settings.VECTORD_DB_BACKEND)
            await vectordb_client.connect()
            
            # Initialize NLP controller using factory
            from factories.nlp_factory import build_nlp_controller
            from stores.llm.templates.template_parser import template_parser
            
            # Template parser
            template_parser_instance = template_parser(
                language=self.settings.PRIMARY_LANG,
                default_language=self.settings.DEFAULT_LANG
            )
            
            # For CLI, we'll create a basic NLP controller without project-specific collection
            # The collection name will be set dynamically when needed
            self.nlp_controller = build_nlp_controller(
                vectordb_client=vectordb_client,
                generation_client=generation_client,
                embedding_client=embedding_client,
                template_parser=template_parser_instance,
                collection_name="",  # Will be set per project
                cohere_api_key=getattr(self.settings, 'COHERE_API_KEY', None),
                cohere_backup_key=getattr(self.settings, 'COHERE_API_KEY_BACKUP', None),
                cohere_backup_key2=getattr(self.settings, 'COHERE_API_KEY_BACKUP2', None),
                cohere_backup_key3=getattr(self.settings, 'COHERE_API_KEY_BACKUP3', None),
                # enable_hybrid_search=True,
                # enable_reranking=True,
                # enable_multi_query=True,
                enable_hybrid_search=False,
                enable_reranking=False,
                enable_multi_query=False,
            )
            
            # Initialize evaluation controller
            self.evaluation_controller = EvaluationController(
                nlp_controller=self.nlp_controller,
                generation_client=generation_client,
                embedding_client=embedding_client,
                judge_client=judge_client  # Use Ollama as judge
            )
            
            logger.info("CLI initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLI: {e}")
            raise
    
    async def get_project_nlp_controller(self, project: Project):
        """Get NLP controller configured for a specific project"""
        from factories.nlp_factory import build_nlp_controller
        from stores.llm.templates.template_parser import template_parser
        
        # Template parser
        template_parser_instance = template_parser(
            language=self.settings.PRIMARY_LANG,
            default_language=self.settings.DEFAULT_LANG
        )
        
        # Create collection name for this project
        collection_name = f"collection_{self.nlp_controller.embedding_client.embedding_size}_{project.project_id}"
        
        # Build project-specific NLP controller
        project_nlp_controller = build_nlp_controller(
            vectordb_client=self.nlp_controller.vectordb_client,
            generation_client=self.nlp_controller.generation_client,
            embedding_client=self.nlp_controller.embedding_client,
            template_parser=template_parser_instance,
            collection_name=collection_name,
            cohere_api_key=getattr(self.settings, 'COHERE_API_KEY', None),
            cohere_backup_key=getattr(self.settings, 'COHERE_API_KEY_BACKUP', None),
            cohere_backup_key2=getattr(self.settings, 'COHERE_API_KEY_BACKUP2', None),
            cohere_backup_key3=getattr(self.settings, 'COHERE_API_KEY_BACKUP3', None),
            enable_hybrid_search=True,
            enable_reranking=True,
            enable_multi_query=True,
        )
        
        return project_nlp_controller
    
    async def get_project(self, project_id: str) -> Project:
        """Get project by ID"""
        try:
            # Convert string to integer since project_id is an Integer column
            project_id_int = int(project_id)
        except ValueError:
            raise ValueError(f"Invalid project ID: {project_id}. Must be a number.")
        
        async with self.db_client() as session:
            project = await session.get(Project, project_id_int)
            if not project:
                raise ValueError(f"Project {project_id} not found")
            return project
    
    async def generate_dataset(self, args):
        """Generate evaluation dataset"""
        try:
            logger.info(f"Generating evaluation dataset for project {args.project_id}")
            
            project = await self.get_project(args.project_id)
            project_nlp_controller = await self.get_project_nlp_controller(project)
            
            # Update evaluation controller with project-specific NLP controller
            self.evaluation_controller.nlp_controller = project_nlp_controller
            
            # Configure dataset generation
            dataset_config = {
                "num_questions_per_type": {
                    QuestionType.FACTUAL: args.factual,
                    QuestionType.ANALYTICAL: args.analytical,
                    QuestionType.COMPARATIVE: args.comparative,
                    QuestionType.SUMMARIZATION: args.summarization,
                    QuestionType.HALLUCINATION: args.hallucination
                },
                "include_retrieval_examples": args.include_retrieval,
                "languages": args.languages.split(",") if args.languages else ["en", "ar"]
            }
            
            # Generate dataset
            result = await self.evaluation_controller.generate_evaluation_dataset(
                project=project,
                dataset_config=dataset_config
            )
            
            if result["signal"] == "success":
                dataset_info = result["dataset_info"]
                logger.info(f"Dataset generated successfully:")
                logger.info(f"  - Total questions: {dataset_info['total_questions']}")
                logger.info(f"  - Retrieval examples: {dataset_info['total_retrieval_examples']}")
                logger.info(f"  - File: {dataset_info['filepath']}")
                
                # Copy to output file if specified
                if args.output:
                    import shutil
                    shutil.copy2(dataset_info['filepath'], args.output)
                    logger.info(f"Dataset copied to: {args.output}")
                
                print(json.dumps(result, indent=2))
            else:
                logger.error(f"Dataset generation failed: {result['message']}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Error generating dataset: {e}")
            sys.exit(1)
    
    async def evaluate(self, args):
        """Run RAG evaluation"""
        try:
            logger.info(f"Evaluating RAG system for project {args.project_id}")
            
            project = await self.get_project(args.project_id)
            project_nlp_controller = await self.get_project_nlp_controller(project)
            
            # Update evaluation controller with project-specific NLP controller
            self.evaluation_controller.nlp_controller = project_nlp_controller
            
            # Configure evaluation
            evaluation_config = {
                "evaluate_retrieval": args.retrieval,
                "evaluate_generation": args.generation,
                "evaluate_end_to_end": args.end_to_end,
                "max_questions": args.max_questions,
                "use_advanced_retrieval": args.advanced_retrieval
            }
            
            # Run evaluation
            result = await self.evaluation_controller.evaluate_rag_system(
                project=project,
                dataset_path=args.dataset,
                evaluation_config=evaluation_config
            )
            
            if result["signal"] == "success":
                eval_results = result["evaluation_results"]
                logger.info(f"Evaluation completed successfully:")
                logger.info(f"  - Overall score: {eval_results['overall_score']:.3f}")
                logger.info(f"  - Questions evaluated: {eval_results['dataset_info']['total_questions']}")
                logger.info(f"  - Results file: {eval_results['results_filepath']}")
                
                # Copy to output file if specified
                if args.output:
                    import shutil
                    shutil.copy2(eval_results['results_filepath'], args.output)
                    logger.info(f"Results copied to: {args.output}")
                
                print(json.dumps(result, indent=2, default=str))
            else:
                logger.error(f"Evaluation failed: {result['message']}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            sys.exit(1)
    



def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="RAG Evaluation CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate dataset command
    dataset_parser = subparsers.add_parser("generate-dataset", help="Generate evaluation dataset")
    dataset_parser.add_argument("--project-id", required=True, help="Project ID")
    dataset_parser.add_argument("--factual", type=int, default=1, help="Number of factual questions")
    dataset_parser.add_argument("--analytical", type=int, default=1, help="Number of analytical questions")
    dataset_parser.add_argument("--comparative", type=int, default=1, help="Number of comparative questions")
    dataset_parser.add_argument("--summarization", type=int, default=1, help="Number of summarization questions")
    dataset_parser.add_argument("--hallucination", type=int, default=1, help="Number of hallucination questions")
    dataset_parser.add_argument("--include-retrieval", action="store_true", default=True, help="Include retrieval examples")
    dataset_parser.add_argument("--languages", default="en,ar", help="Comma-separated list of languages")
    dataset_parser.add_argument("--output", help="Output file path")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run RAG evaluation")
    eval_parser.add_argument("--project-id", required=True, help="Project ID")
    eval_parser.add_argument("--dataset", help="Path to evaluation dataset")
    eval_parser.add_argument("--max-questions", type=int, help="Maximum number of questions to evaluate")
    eval_parser.add_argument("--no-retrieval", dest="retrieval", action="store_false", default=True, help="Skip retrieval evaluation")
    eval_parser.add_argument("--no-generation", dest="generation", action="store_false", default=True, help="Skip generation evaluation")
    eval_parser.add_argument("--no-end-to-end", dest="end_to_end", action="store_false", default=True, help="Skip end-to-end evaluation")
    eval_parser.add_argument("--no-advanced-retrieval", dest="advanced_retrieval", action="store_false", default=True, help="Use basic retrieval")
    eval_parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize and run CLI
    cli = EvaluationCLI()
    
    async def run_command():
        await cli.initialize()
        
        if args.command == "generate-dataset":
            await cli.generate_dataset(args)
        elif args.command == "evaluate":
            await cli.evaluate(args)
    
    # Run the async command
    asyncio.run(run_command())


if __name__ == "__main__":
    main()