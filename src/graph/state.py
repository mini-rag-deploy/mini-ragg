# src/graph/state.py

from typing import TypedDict, List, Optional, Any

class RAGState(TypedDict):
    question: str
    documents: List[Any]
    answer: Optional[str]
    iterations: int
    grade_reason: Optional[str]
    question_type: Optional[str]
    # New fields for dynamic source selection
    need_more_details: Optional[bool]
    selected_source: Optional[str]
    source_reason: Optional[str]
    sources_tried: List[str]
    external_data: Optional[Any]
