"""
RAG Evaluation Dataset Generator

Generates comprehensive evaluation datasets for RAG systems including:
- Question-Answer pairs from document chunks
- Retrieval evaluation data
- Generation quality benchmarks
- Multi-language support (Arabic + English)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import random
import re
import unicodedata
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Any, Dict
from urllib.parse import urlparse
from enum import Enum

import numpy as np

logger = logging.getLogger("uvicorn.error")


class QuestionType(Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    SUMMARIZATION = "summarization"
    HALLUCINATION = "hallucination"  # Questions with no answer in corpus


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class EvaluationQuestion:
    """Single evaluation question with metadata"""
    id: str
    question: str
    reference_answer: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    language: str
    source_chunks: List[str]  # Chunk IDs that contain the answer
    source_documents: List[str]  # Document names
    metadata: Dict[str, Any]
    created_at: str


@dataclass
class RetrievalExample:
    """Retrieval evaluation example"""
    id: str
    query: str
    relevant_chunks: List[str]  # Ground truth relevant chunk IDs
    irrelevant_chunks: List[str]  # Known irrelevant chunks
    language: str
    difficulty: DifficultyLevel
    metadata: Dict[str, Any]


@dataclass
class EvaluationDataset:
    """Complete evaluation dataset"""
    name: str
    description: str
    version: str
    created_at: str
    questions: List[EvaluationQuestion]
    retrieval_examples: List[RetrievalExample]
    statistics: Dict[str, Any]


class DatasetGenerator:
    """
    Generates evaluation datasets from indexed document chunks
    """
    
    def __init__(
        self,
        nlp_controller,
        project,
        generation_client,
        output_dir: str = "evaluation_datasets"
    ):
        self.nlp_controller = nlp_controller
        self.project = project
        self.generation_client = generation_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Question generation prompts
        self.question_prompts = {
            QuestionType.FACTUAL: """
Based on the following text chunk, generate a factual question that can be answered directly from the content.
The question should be clear, specific, and have a definitive answer in the text.

Text: {chunk_text}

Generate a JSON response with:
{{
    "question": "your question here",
    "answer": "the answer from the text",
    "difficulty": "easy|medium|hard"
}}
""",
            
            QuestionType.ANALYTICAL: """
Based on the following text chunk, generate an analytical question that requires reasoning or interpretation.
The question should ask about implications, causes, effects, or deeper meaning.

Text: {chunk_text}

Generate a JSON response with:
{{
    "question": "your analytical question here",
    "answer": "reasoned answer based on the text",
    "difficulty": "easy|medium|hard"
}}
""",
            
            QuestionType.COMPARATIVE: """
Based on the following text chunks from different sources, generate a comparative question.
The question should ask about similarities, differences, or relationships between the information.

Text 1: {chunk_text_1}
Source 1: {source_1}

Text 2: {chunk_text_2}
Source 2: {source_2}

Generate a JSON response with:
{{
    "question": "your comparative question here",
    "answer": "answer comparing both sources",
    "difficulty": "easy|medium|hard"
}}
""",
            
            QuestionType.SUMMARIZATION: """
Based on the following text chunks, generate a question that requires summarizing or synthesizing information.

Text chunks:
{chunk_texts}

Generate a JSON response with:
{{
    "question": "your summarization question here",
    "answer": "synthesized answer from all chunks",
    "difficulty": "easy|medium|hard"
}}
""",
            
            QuestionType.HALLUCINATION: """
Generate a plausible question related to the domain of the following text, but that CANNOT be answered from the text.
This is for testing hallucination detection.

Text: {chunk_text}
Domain: {domain}

Generate a JSON response with:
{{
    "question": "question that cannot be answered from the text",
    "answer": "No answer available in the provided context",
    "difficulty": "medium"
}}
"""
        }

    async def generate_dataset(
        self,
        num_questions_per_type: Dict[QuestionType, int] = None,
        include_retrieval_examples: bool = True,
        languages: List[str] = ["en", "ar"]
    ) -> EvaluationDataset:
        """
        Generate complete evaluation dataset
        """
        if num_questions_per_type is None:
            num_questions_per_type = {
                QuestionType.FACTUAL: 50,
                QuestionType.ANALYTICAL: 30,
                QuestionType.COMPARATIVE: 20,
                QuestionType.SUMMARIZATION: 20,
                QuestionType.HALLUCINATION: 10
            }
        
        logger.info(f"[DatasetGenerator] Starting dataset generation for project {self.project.project_id}")
        
        # Get all chunks from the project
        chunks = await self._get_project_chunks()
        if not chunks:
            raise ValueError("No chunks found for the project")
        
        logger.info(f"[DatasetGenerator] Found {len(chunks)} chunks to work with")
        
        questions = []
        retrieval_examples = []
        
        # Generate questions by type
        for question_type, count in num_questions_per_type.items():
            logger.info(f"[DatasetGenerator] Generating {count} {question_type.value} questions")
            
            if question_type == QuestionType.COMPARATIVE:
                type_questions = await self._generate_comparative_questions(chunks, count)
            elif question_type == QuestionType.SUMMARIZATION:
                type_questions = await self._generate_summarization_questions(chunks, count)
            elif question_type == QuestionType.HALLUCINATION:
                type_questions = await self._generate_hallucination_questions(chunks, count)
            else:
                type_questions = await self._generate_single_chunk_questions(
                    chunks, question_type, count
                )
            
            questions.extend(type_questions)
        
        # Generate retrieval examples
        if include_retrieval_examples:
            logger.info("[DatasetGenerator] Generating retrieval evaluation examples")
            retrieval_examples = await self._generate_retrieval_examples(chunks, questions)
        
        # Create dataset
        dataset = EvaluationDataset(
            name=f"rag_evaluation_{self.project.project_id}",
            description=f"Evaluation dataset for project {self.project.project_id}",
            version="1.0.0",
            created_at=datetime.now().isoformat(),
            questions=questions,
            retrieval_examples=retrieval_examples,
            statistics=self._calculate_statistics(questions, retrieval_examples)
        )
        
        logger.info(f"[DatasetGenerator] Generated dataset with {len(questions)} questions and {len(retrieval_examples)} retrieval examples")
        
        return dataset

    async def _get_project_chunks(self) -> List[Dict[str, Any]]:
        """Get all chunks for the project from database"""
        try:
            # Get chunks directly from PostgreSQL database instead of vector database
            # This ensures we get the actual text content
            
            from models.db_schemes.minirag.schemes.datachunk import DataChunk
            from models.db_schemes.minirag.schemes.assest import Asset
            from sqlalchemy import select
            
            # Use the correct database client from the vector database provider
            async with self.nlp_controller.vectordb_client.client() as session:
                # Query chunks for this project with asset information
                stmt = select(DataChunk, Asset).join(
                    Asset, DataChunk.chunk_asset_id == Asset.asset_id
                ).where(DataChunk.chunk_project_id == self.project.project_id)
                
                result = await session.execute(stmt)
                chunk_asset_pairs = result.all()
                
                if not chunk_asset_pairs:
                    logger.warning(f"[DatasetGenerator] No chunks found in database for project {self.project.project_id}")
                    return []
                
                chunks = []
                for db_chunk, asset in chunk_asset_pairs:
                    chunk_dict = {
                        'id': str(db_chunk.chunk_id),
                        'payload': {
                            'text': db_chunk.chunk_text,
                            'source': asset.asset_name if asset else 'unknown',
                            'metadata': db_chunk.chunk_metadata or {},
                            'chunk_order': db_chunk.chunk_order
                        },
                        'score': 1.0  # Default score since we're not doing similarity search
                    }
                    
                    # Debug: Log chunk content
                    if db_chunk.chunk_text:
                        logger.info(f"[DatasetGenerator] Found chunk with text: {db_chunk.chunk_text[:100]}...")
                    else:
                        logger.warning(f"[DatasetGenerator] Chunk {db_chunk.chunk_id} has no text content")
                    
                    chunks.append(chunk_dict)
                
                logger.info(f"[DatasetGenerator] Retrieved {len(chunks)} chunks from database")
                return chunks
            
        except Exception as e:
            logger.error(f"[DatasetGenerator] Error getting project chunks from database: {e}")
            # Fallback to vector database method
            return await self._get_project_chunks_from_vector_db()

    async def _get_project_chunks_from_vector_db(self) -> List[Dict[str, Any]]:
        """Fallback method to get chunks from vector database"""
        try:
            collection_name = self.nlp_controller.create_collection_name(self.project.project_id)
            
            # Get collection info to know how many chunks we have
            collection_info = await self.nlp_controller.vectordb_client.get_collection_info(collection_name)
            if not collection_info:
                return []
            
            # Retrieve all chunks (using a dummy vector to get all results)
            dummy_vector = [0.0] * self.nlp_controller.embedding_client.embedding_size
            results = await self.nlp_controller.vectordb_client.search_by_vector(
                collection_name=collection_name,
                vector=dummy_vector,
                limit=1000  # Adjust based on your needs
            )
            
            # Convert RetrievedDocument objects to dictionaries
            chunks = []
            if results:
                for result in results:
                    # Handle both RetrievedDocument objects and dictionaries
                    if hasattr(result, 'id') and hasattr(result, 'payload'):
                        # RetrievedDocument object
                        chunk_dict = {
                            'id': str(result.id),
                            'payload': result.payload if result.payload else {},
                            'score': getattr(result, 'score', 0.0)
                        }
                    elif isinstance(result, dict):
                        # Already a dictionary
                        chunk_dict = result
                    else:
                        # Try to convert to dict
                        chunk_dict = {
                            'id': str(getattr(result, 'id', '')),
                            'payload': getattr(result, 'payload', {}),
                            'score': getattr(result, 'score', 0.0)
                        }
                    
                    # Debug: Log chunk content to see what we're getting
                    if chunk_dict.get('payload', {}).get('text'):
                        logger.info(f"[DatasetGenerator] Found chunk with text: {chunk_dict['payload']['text'][:100]}...")
                    else:
                        logger.warning(f"[DatasetGenerator] Chunk {chunk_dict.get('id', 'unknown')} has no text content")
                        logger.warning(f"[DatasetGenerator] Chunk payload keys: {list(chunk_dict.get('payload', {}).keys())}")
                    
                    chunks.append(chunk_dict)
            
            return chunks
            
        except Exception as e:
            logger.error(f"[DatasetGenerator] Error getting project chunks from vector database: {e}")
            return []

    async def _generate_single_chunk_questions(
        self,
        chunks: List[Dict[str, Any]],
        question_type: QuestionType,
        count: int
    ) -> List[EvaluationQuestion]:
        """Generate questions from single chunks"""
        questions = []
        selected_chunks = random.sample(chunks, min(count * 2, len(chunks)))  # Sample more than needed
        
        for chunk in selected_chunks[:count]:
            try:
                # Try different ways to extract text from chunk
                chunk_text = ''
                payload = chunk.get('payload', {})
                
                # Try different possible text field names
                if 'text' in payload:
                    chunk_text = payload['text']
                elif 'content' in payload:
                    chunk_text = payload['content']
                elif 'chunk_text' in payload:
                    chunk_text = payload['chunk_text']
                else:
                    # Log available keys for debugging
                    logger.warning(f"[DatasetGenerator] No text found in chunk {chunk.get('id', 'unknown')}. Available keys: {list(payload.keys())}")
                    continue
                
                if not chunk_text or len(chunk_text.strip()) < 50:
                    continue
                
                # Generate question using LLM
                prompt = self.question_prompts[question_type].format(
                    chunk_text=chunk_text[:2000]  # Limit text length
                )
                
                response = await self._call_llm(prompt)
                if not response:
                    continue
                
                # Parse response
                question_data = self._parse_llm_response(response)
                if not question_data:
                    continue
                
                # Create evaluation question
                question = EvaluationQuestion(
                    id=str(uuid.uuid4()),
                    question=question_data['question'],
                    reference_answer=question_data['answer'],
                    question_type=question_type,
                    difficulty=DifficultyLevel(question_data.get('difficulty', 'medium')),
                    language=self._detect_language(question_data['question']),
                    source_chunks=[str(chunk.get('id', ''))],
                    source_documents=[chunk.get('payload', {}).get('source', 'unknown')],
                    metadata={
                        'chunk_score': chunk.get('score', 0.0),
                        'chunk_length': len(chunk_text),
                        'generation_method': 'llm_single_chunk'
                    },
                    created_at=datetime.now().isoformat()
                )
                
                questions.append(question)
                
                if len(questions) >= count:
                    break
                    
            except Exception as e:
                logger.warning(f"[DatasetGenerator] Error generating question from chunk: {e}")
                continue
        
        return questions

    async def _generate_comparative_questions(
        self,
        chunks: List[Dict[str, Any]],
        count: int
    ) -> List[EvaluationQuestion]:
        """Generate comparative questions using pairs of chunks"""
        questions = []
        
        for _ in range(count):
            try:
                # Select two random chunks from different sources if possible
                chunk_pair = self._select_chunk_pair(chunks)
                if not chunk_pair:
                    continue
                
                chunk1, chunk2 = chunk_pair
                
                prompt = self.question_prompts[QuestionType.COMPARATIVE].format(
                    chunk_text_1=chunk1.get('payload', {}).get('text', '')[:1000],
                    source_1=chunk1.get('payload', {}).get('source', 'unknown'),
                    chunk_text_2=chunk2.get('payload', {}).get('text', '')[:1000],
                    source_2=chunk2.get('payload', {}).get('source', 'unknown')
                )
                
                response = await self._call_llm(prompt)
                if not response:
                    continue
                
                question_data = self._parse_llm_response(response)
                if not question_data:
                    continue
                
                question = EvaluationQuestion(
                    id=str(uuid.uuid4()),
                    question=question_data['question'],
                    reference_answer=question_data['answer'],
                    question_type=QuestionType.COMPARATIVE,
                    difficulty=DifficultyLevel(question_data.get('difficulty', 'medium')),
                    language=self._detect_language(question_data['question']),
                    source_chunks=[str(chunk1.get('id', '')), str(chunk2.get('id', ''))],
                    source_documents=[
                        chunk1.get('payload', {}).get('source', 'unknown'),
                        chunk2.get('payload', {}).get('source', 'unknown')
                    ],
                    metadata={
                        'generation_method': 'llm_comparative',
                        'chunk_count': 2
                    },
                    created_at=datetime.now().isoformat()
                )
                
                questions.append(question)
                
            except Exception as e:
                logger.warning(f"[DatasetGenerator] Error generating comparative question: {e}")
                continue
        
        return questions

    async def _generate_summarization_questions(
        self,
        chunks: List[Dict[str, Any]],
        count: int
    ) -> List[EvaluationQuestion]:
        """Generate summarization questions using multiple chunks"""
        questions = []
        
        for _ in range(count):
            try:
                # Select 3-5 related chunks
                selected_chunks = random.sample(chunks, min(5, len(chunks)))
                
                chunk_texts = []
                chunk_ids = []
                source_docs = []
                
                for chunk in selected_chunks:
                    text = chunk.get('payload', {}).get('text', '')
                    if text and len(text.strip()) > 30:
                        chunk_texts.append(text[:500])  # Limit length
                        chunk_ids.append(str(chunk.get('id', '')))
                        source_docs.append(chunk.get('payload', {}).get('source', 'unknown'))
                
                if len(chunk_texts) < 2:
                    continue
                
                prompt = self.question_prompts[QuestionType.SUMMARIZATION].format(
                    chunk_texts='\n\n---\n\n'.join(chunk_texts)
                )
                
                response = await self._call_llm(prompt)
                if not response:
                    continue
                
                question_data = self._parse_llm_response(response)
                if not question_data:
                    continue
                
                question = EvaluationQuestion(
                    id=str(uuid.uuid4()),
                    question=question_data['question'],
                    reference_answer=question_data['answer'],
                    question_type=QuestionType.SUMMARIZATION,
                    difficulty=DifficultyLevel(question_data.get('difficulty', 'hard')),
                    language=self._detect_language(question_data['question']),
                    source_chunks=chunk_ids,
                    source_documents=list(set(source_docs)),
                    metadata={
                        'generation_method': 'llm_summarization',
                        'chunk_count': len(chunk_texts)
                    },
                    created_at=datetime.now().isoformat()
                )
                
                questions.append(question)
                
            except Exception as e:
                logger.warning(f"[DatasetGenerator] Error generating summarization question: {e}")
                continue
        
        return questions

    async def _generate_hallucination_questions(
        self,
        chunks: List[Dict[str, Any]],
        count: int
    ) -> List[EvaluationQuestion]:
        """Generate questions that cannot be answered from the corpus (for hallucination testing)"""
        questions = []
        selected_chunks = random.sample(chunks, min(count, len(chunks)))
        
        for chunk in selected_chunks:
            try:
                chunk_text = chunk.get('payload', {}).get('text', '')
                if not chunk_text:
                    continue
                
                # Infer domain from chunk content
                domain = self._infer_domain(chunk_text)
                
                prompt = self.question_prompts[QuestionType.HALLUCINATION].format(
                    chunk_text=chunk_text[:1000],
                    domain=domain
                )
                
                response = await self._call_llm(prompt)
                if not response:
                    continue
                
                question_data = self._parse_llm_response(response)
                if not question_data:
                    continue
                
                question = EvaluationQuestion(
                    id=str(uuid.uuid4()),
                    question=question_data['question'],
                    reference_answer="No answer available in the provided context",
                    question_type=QuestionType.HALLUCINATION,
                    difficulty=DifficultyLevel.MEDIUM,
                    language=self._detect_language(question_data['question']),
                    source_chunks=[],  # No source chunks for hallucination questions
                    source_documents=[],
                    metadata={
                        'generation_method': 'llm_hallucination',
                        'domain': domain,
                        'expected_answer': 'no_answer'
                    },
                    created_at=datetime.now().isoformat()
                )
                
                questions.append(question)
                
            except Exception as e:
                logger.warning(f"[DatasetGenerator] Error generating hallucination question: {e}")
                continue
        
        return questions

    async def _generate_retrieval_examples(
        self,
        chunks: List[Dict[str, Any]],
        questions: List[EvaluationQuestion]
    ) -> List[RetrievalExample]:
        """Generate retrieval evaluation examples"""
        retrieval_examples = []
        
        # Use existing questions as retrieval queries
        for question in questions[:50]:  # Limit to first 50 questions
            try:
                # Get relevant chunks (from question metadata)
                relevant_chunks = question.source_chunks
                
                # Get some irrelevant chunks (random selection)
                all_chunk_ids = [str(chunk.get('id', '')) for chunk in chunks]
                irrelevant_chunks = [
                    chunk_id for chunk_id in random.sample(all_chunk_ids, min(10, len(all_chunk_ids)))
                    if chunk_id not in relevant_chunks
                ]
                
                retrieval_example = RetrievalExample(
                    id=str(uuid.uuid4()),
                    query=question.question,
                    relevant_chunks=relevant_chunks,
                    irrelevant_chunks=irrelevant_chunks[:5],  # Limit to 5 irrelevant
                    language=question.language,
                    difficulty=question.difficulty,
                    metadata={
                        'question_type': question.question_type.value,
                        'source_question_id': question.id
                    }
                )
                
                retrieval_examples.append(retrieval_example)
                
            except Exception as e:
                logger.warning(f"[DatasetGenerator] Error generating retrieval example: {e}")
                continue
        
        return retrieval_examples

    def _select_chunk_pair(self, chunks: List[Dict[str, Any]]) -> Optional[Tuple[Dict, Dict]]:
        """Select a pair of chunks, preferably from different sources"""
        if len(chunks) < 2:
            return None
        
        # Try to find chunks from different sources
        sources = {}
        for chunk in chunks:
            source = chunk.get('payload', {}).get('source', 'unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(chunk)
        
        if len(sources) >= 2:
            # Pick from different sources
            source_names = list(sources.keys())
            source1, source2 = random.sample(source_names, 2)
            chunk1 = random.choice(sources[source1])
            chunk2 = random.choice(sources[source2])
            return chunk1, chunk2
        else:
            # Pick any two chunks
            return tuple(random.sample(chunks, 2))

    async def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM with prompt and return response"""
        try:
            # Check if this is a CoHere provider (uses different parameter names)
            if hasattr(self.generation_client, 'client') and hasattr(self.generation_client.client, 'chat'):
                # CoHere provider
                response = await asyncio.to_thread(
                    self.generation_client.generate_text,
                    prompt=prompt,
                    max_output_tokens=500,
                    temperature=0.7
                )
            else:
                # Other providers (OpenAI, etc.)
                response = await self.generation_client.generate_text(
                    prompt=prompt,
                    max_tokens=500,
                    temperature=0.7
                )
            return response
        except Exception as e:
            logger.error(f"[DatasetGenerator] LLM call failed: {e}")
            return None

    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            return None
        except Exception as e:
            logger.warning(f"[DatasetGenerator] Failed to parse LLM response: {e}")
            return None

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        import re
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        latin_chars = len(re.findall(r'[A-Za-z]', text))
        
        if arabic_chars > latin_chars:
            return "ar"
        elif latin_chars > 0:
            return "en"
        else:
            return "unknown"

    def _infer_domain(self, text: str) -> str:
        """Infer domain from text content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['iot', 'internet of things', 'sensor', 'device']):
            return "IoT and Technology"
        elif any(word in text_lower for word in ['legal', 'law', 'contract', 'agreement']):
            return "Legal"
        elif any(word in text_lower for word in ['medical', 'health', 'patient', 'clinical']):
            return "Medical"
        elif any(word in text_lower for word in ['business', 'management', 'strategy']):
            return "Business"
        else:
            return "General"

    def _calculate_statistics(
        self,
        questions: List[EvaluationQuestion],
        retrieval_examples: List[RetrievalExample]
    ) -> Dict[str, Any]:
        """Calculate dataset statistics"""
        stats = {
            'total_questions': len(questions),
            'total_retrieval_examples': len(retrieval_examples),
            'questions_by_type': {},
            'questions_by_difficulty': {},
            'questions_by_language': {},
            'average_answer_length': 0,
            'unique_source_documents': set(),
            'unique_chunks_used': set()
        }
        
        # Count by type
        for q_type in QuestionType:
            stats['questions_by_type'][q_type.value] = len([
                q for q in questions if q.question_type == q_type
            ])
        
        # Count by difficulty
        for difficulty in DifficultyLevel:
            stats['questions_by_difficulty'][difficulty.value] = len([
                q for q in questions if q.difficulty == difficulty
            ])
        
        # Count by language
        for question in questions:
            lang = question.language
            if lang not in stats['questions_by_language']:
                stats['questions_by_language'][lang] = 0
            stats['questions_by_language'][lang] += 1
        
        # Calculate average answer length
        if questions:
            total_length = sum(len(q.reference_answer) for q in questions)
            stats['average_answer_length'] = total_length / len(questions)
        
        # Count unique sources and chunks
        for question in questions:
            stats['unique_source_documents'].update(question.source_documents)
            stats['unique_chunks_used'].update(question.source_chunks)
        
        # Convert sets to counts
        stats['unique_source_documents'] = len(stats['unique_source_documents'])
        stats['unique_chunks_used'] = len(stats['unique_chunks_used'])
        
        return stats

    def save_dataset(self, dataset: EvaluationDataset, filename: Optional[str] = None) -> str:
        """Save dataset to JSON file"""
        if filename is None:
            filename = f"{dataset.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        
        # Convert dataclasses to dict for JSON serialization
        dataset_dict = {
            'name': dataset.name,
            'description': dataset.description,
            'version': dataset.version,
            'created_at': dataset.created_at,
            'questions': [
                {
                    **asdict(q),
                    'question_type': q.question_type.value,  # Convert enum to string
                    'difficulty': q.difficulty.value  # Convert enum to string
                }
                for q in dataset.questions
            ],
            'retrieval_examples': [
                {
                    **asdict(r),
                    'difficulty': r.difficulty.value  # Convert enum to string
                }
                for r in dataset.retrieval_examples
            ],
            'statistics': dataset.statistics
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[DatasetGenerator] Dataset saved to {filepath}")
        return str(filepath)

    @classmethod
    def load_dataset(cls, filepath: str) -> EvaluationDataset:
        """Load dataset from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert back to dataclasses
        questions = [
            EvaluationQuestion(
                id=q['id'],
                question=q['question'],
                reference_answer=q['reference_answer'],
                question_type=QuestionType(q['question_type']),
                difficulty=DifficultyLevel(q['difficulty']),
                language=q['language'],
                source_chunks=q['source_chunks'],
                source_documents=q['source_documents'],
                metadata=q['metadata'],
                created_at=q['created_at']
            )
            for q in data['questions']
        ]
        
        retrieval_examples = [
            RetrievalExample(
                id=r['id'],
                query=r['query'],
                relevant_chunks=r['relevant_chunks'],
                irrelevant_chunks=r['irrelevant_chunks'],
                language=r['language'],
                difficulty=DifficultyLevel(r['difficulty']),
                metadata=r['metadata']
            )
            for r in data['retrieval_examples']
        ]
        
        return EvaluationDataset(
            name=data['name'],
            description=data['description'],
            version=data['version'],
            created_at=data['created_at'],
            questions=questions,
            retrieval_examples=retrieval_examples,
            statistics=data['statistics']
        )