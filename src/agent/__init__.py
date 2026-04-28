"""
Agent module for dynamic source selection and tool execution.

This module extends the RAG system with:
- Dynamic decision-making on whether more information is needed
- Intelligent source selection (Vector DB, Tools, Internet)
- Tool registry for external APIs and functions
- Internet retrieval capabilities
"""

from .source_router import SourceRouter
from .tools_registry import ToolsRegistry
from .internet_retriever import InternetRetriever

__all__ = [
    "SourceRouter",
    "ToolsRegistry", 
    "InternetRetriever",
]
