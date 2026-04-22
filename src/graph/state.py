# src/graph/state.py

from typing import TypedDict, List, Optional, Any

class RAGState(TypedDict):
    question: str
    documents: List[Any]
    answer: Optional[str]
    iterations: int
    grade_reason: Optional[str]
