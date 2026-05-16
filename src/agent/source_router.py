"""
Source Router for Dynamic Source Selection.

This module implements the decision-making logic for determining:
1. Whether more information is needed
2. Which source to use (Vector DB, Tools, Internet)

Design principles:
- LLM-powered decision making
- Fallback mechanisms for robustness
- Clean integration with existing RAG pipeline
- Extensible for new sources
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from .prompts import (
    NEED_MORE_DETAILS_PROMPT,
    SOURCE_SELECTION_PROMPT,
    INTERNET_QUERY_OPTIMIZATION_PROMPT,
)

logger = logging.getLogger("uvicorn.error")


class SourceType(str, Enum):
    """Available information sources."""
    VECTOR_DB = "vector_db"
    INTERNET = "internet"


class SourceRouter:
    """
    Intelligent router that decides whether more information is needed
    and selects the best source to retrieve it from.
    """
    
    def __init__(
        self,
        generation_client,
        internet_retriever=None,
    ):
        """
        Initialize the source router.
        
        Parameters
        ----------
        generation_client
            LLM client for decision making
        internet_retriever : InternetRetriever, optional
            Internet search client
        """
        self.generation_client = generation_client
        self.internet_retriever = internet_retriever
        
        logger.info("[SourceRouter] Initialized")

    async def decide_need_more_details(
        self,
        question: str,
        context: str,
        answer: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Decide if more information is needed to answer the question.
        
        Parameters
        ----------
        question : str
            User's question
        context : str
            Current context (retrieved documents)
        answer : Optional[str]
            Current answer (if any)
            
        Returns
        -------
        Tuple[bool, str]
            (need_more, reason)
        """
        logger.info("[SourceRouter] Evaluating if more details are needed")
        
        # Format context
        if not context or context.strip() == "":
            context = "No context available"
        
        if not answer:
            answer = "No answer generated yet"
        
        # Build prompt
        prompt = NEED_MORE_DETAILS_PROMPT.format(
            question=question,
            context=context[:2000],  # Limit context length
            answer=answer[:1000],     # Limit answer length
        )
        
        try:
            # Get LLM decision
            result = self.generation_client.generate_json(prompt=prompt)
            
            need_more = result.get("need_more", False)
            reason = result.get("reason", "No reason provided")
            
            logger.info(f"[SourceRouter] Need more details: {need_more} - {reason}")
            return need_more, reason
            
        except Exception as exc:
            logger.error(f"[SourceRouter] Decision failed: {exc}")
            # Conservative fallback: assume we don't need more if decision fails
            return False, f"Decision error: {exc}"

    async def select_source(
        self,
        question: str,
        context: str,
        previous_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Select the best source for additional information.
        
        Parameters
        ----------
        question : str
            User's question
        context : str
            Current context
        previous_sources : Optional[List[str]]
            Sources already tried (to avoid loops)
            
        Returns
        -------
        Dict[str, Any]
            {
                "source": SourceType,
                "reason": str,
                "query": str (for vector_db or internet),
                "tool_name": str (for tools),
                "tool_params": dict (for tools),
            }
        """
        logger.info("[SourceRouter] Selecting information source")
        
        if previous_sources is None:
            previous_sources = []
        
        # Get available tools summary
        available_tools = "None"
        if self.tools_registry:
            available_tools = self.tools_registry.get_tools_summary()
        
        # Build prompt
        prompt = SOURCE_SELECTION_PROMPT.format(
            question=question,
            context=f"Previous sources tried: {', '.join(previous_sources) if previous_sources else 'None'}\n\nContext: {context[:1500]}",
            available_tools=available_tools,
        )
        
        try:
            # Get LLM decision
            result = self.generation_client.generate_json(prompt=prompt)
            
            source = result.get("source", "internet")  # Default to internet
            reason = result.get("reason", "No reason provided")
            
            # Force internet as the only available source
            if source != "internet":
                logger.info(f"[SourceRouter] Forcing source from '{source}' to 'internet' (only available source)")
                source = "internet"
                reason = "Internet is the only external source available"
            
            # CRITICAL: Check if source was already tried
            if source in previous_sources:
                logger.warning(f"[SourceRouter] Source '{source}' was already tried")
                # Since internet is the only source, if it was tried, we're done
                return {
                    "source": source,
                    "reason": "Internet was already tried, no more sources available",
                    "query": result.get("query", question),
                }
            
            # Build response for internet
            response = {
                "source": source,
                "reason": reason,
                "query": result.get("query", question),
            }
            
            logger.info(f"[SourceRouter] Selected source: {source} - {reason}")
            return response
            
        except Exception as exc:
            logger.error(f"[SourceRouter] Source selection failed: {exc}")
            # Fallback to internet
            return {
                "source": "internet",
                "reason": f"Selection error, defaulting to internet: {exc}",
                "query": question,
            }

    async def optimize_internet_query(self, question: str) -> str:
        """
        Optimize a question for web search.
        
        Parameters
        ----------
        question : str
            Original question
            
        Returns
        -------
        str
            Optimized search query
        """
        logger.info("[SourceRouter] Optimizing query for internet search")
        
        prompt = INTERNET_QUERY_OPTIMIZATION_PROMPT.format(question=question)
        
        try:
            # Get optimized query
            optimized = self.generation_client.generate_text(
                prompt=prompt,
                chat_history=[],
            )
            
            optimized = optimized.strip().strip('"').strip("'")
            
            logger.info(f"[SourceRouter] Optimized query: {optimized}")
            return optimized
            
        except Exception as exc:
            logger.error(f"[SourceRouter] Query optimization failed: {exc}")
            return question  # Fallback to original

    async def route_and_fetch(
        self,
        question: str,
        context: str,
        answer: Optional[str] = None,
        previous_sources: Optional[List[str]] = None,
        nlp_controller=None,
        project=None,
    ) -> Dict[str, Any]:
        """
        Complete routing workflow: decide, select, and fetch.
        
        Parameters
        ----------
        question : str
            User's question
        context : str
            Current context
        answer : Optional[str]
            Current answer
        previous_sources : Optional[List[str]]
            Sources already tried
        nlp_controller
            NLP controller for vector DB access
        project
            Project object for vector DB
            
        Returns
        -------
        Dict[str, Any]
            {
                "need_more": bool,
                "source": str,
                "reason": str,
                "data": Any (fetched data),
                "success": bool,
            }
        """
        # Step 1: Decide if more details are needed
        need_more, reason = await self.decide_need_more_details(
            question=question,
            context=context,
            answer=answer,
        )
        
        if not need_more:
            return {
                "need_more": False,
                "reason": reason,
                "success": True,
            }
        
        # Step 2: Select source
        selection = await self.select_source(
            question=question,
            context=context,
            previous_sources=previous_sources,
        )
        
        source = selection["source"]
        
        # Step 3: Fetch from selected source
        try:
            if source == SourceType.VECTOR_DB.value:
                data = await self._fetch_from_vector_db(
                    selection=selection,
                    nlp_controller=nlp_controller,
                    project=project,
                )
            elif source == SourceType.TOOLS.value:
                data = await self._fetch_from_tools(selection)
            elif source == SourceType.INTERNET.value:
                data = await self._fetch_from_internet(selection)
            else:
                data = None
            
            return {
                "need_more": True,
                "source": source,
                "reason": selection["reason"],
                "data": data,
                "success": data is not None,
            }
            
        except Exception as exc:
            logger.error(f"[SourceRouter] Fetch failed: {exc}")
            return {
                "need_more": True,
                "source": source,
                "reason": selection["reason"],
                "data": None,
                "success": False,
                "error": str(exc),
            }

    async def _fetch_from_vector_db(
        self,
        selection: Dict[str, Any],
        nlp_controller,
        project,
    ) -> List[Any]:
        """Fetch documents from vector database."""
        logger.info("[SourceRouter] Fetching from vector DB")
        
        if not nlp_controller or not project:
            logger.error("[SourceRouter] NLP controller or project not provided")
            return []
        
        query = selection.get("query", "")
        
        # Use advanced retrieval if available
        if hasattr(nlp_controller, "advanced_retrieve"):
            docs = await nlp_controller.advanced_retrieve(
                project=project,
                query=query,
                top_k=5,
            )
        else:
            docs = await nlp_controller.search_vector_db_collection(
                project=project,
                text=query,
                limit=5,
            )
        
        logger.info(f"[SourceRouter] Retrieved {len(docs)} documents from vector DB")
        return docs

    async def _fetch_from_internet(self, selection: Dict[str, Any]) -> Any:
        """Search the internet and return results."""
        logger.info("[SourceRouter] Fetching from internet")
        
        if not self.internet_retriever:
            logger.error("[SourceRouter] No internet retriever available")
            return None
        
        query = selection.get("query", "")
        
        # InternetRetriever.search() returns InternetResult
        result = await self.internet_retriever.search(query)
        
        if result.success:
            logger.info(f"[SourceRouter] Retrieved {len(result.sources)} results from internet ({result.backend})")
        else:
            logger.warning(f"[SourceRouter] Internet search failed: {result.error}")
        
        return result
