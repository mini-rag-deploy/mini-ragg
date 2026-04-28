# src/agent/internet_retriever.py
"""
Internet Retriever — live web search for the Dynamic Source Selection layer.

Backends (tried in order, first available wins):
  1. Tavily Search API  — best quality, designed for AI agents
  2. DuckDuckGo         — free, no API key required (ddgs package)
  3. Mock               — always available, returns a clear "not available" message

Edge cases:
  - API key missing  → skip to next backend
  - Network timeout  → skip to next backend
  - Rate limit       → skip to next backend
  - Empty results    → returns InternetResult with success=False
  - All backends fail → returns mock result with clear message
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Result contract
# ─────────────────────────────────────────────
@dataclass
class InternetResult:
    success:  bool
    content:  str                     # formatted text, ready to merge into RAG context
    sources:  List[str] = field(default_factory=list)   # URLs
    backend:  str = "unknown"
    error:    str = ""


# ─────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────
async def _search_tavily(query: str, api_key: str, max_results: int) -> Optional[InternetResult]:
    """Tavily Search — highest quality for AI agents."""
    if not api_key:
        return None
    try:
        from tavily import TavilyClient
        client  = TavilyClient(api_key=api_key)
        results = client.search(query=query, max_results=max_results)

        snippets = []
        urls     = []
        for r in results.get("results", []):
            title   = r.get("title", "")
            snippet = r.get("content", "")
            url     = r.get("url", "")
            if snippet:
                snippets.append(f"**{title}**\n{snippet}")
                urls.append(url)

        if not snippets:
            return None

        content = "\n\n---\n\n".join(snippets)
        return InternetResult(
            success=True,
            content=f"[Web Search Results — Tavily]\n\n{content}",
            sources=urls,
            backend="tavily",
        )
    except ImportError:
        logger.debug("[InternetRetriever] tavily-python not installed")
        return None
    except Exception as exc:
        logger.warning(f"[InternetRetriever] Tavily failed: {exc}")
        return None


async def _search_duckduckgo(query: str, max_results: int) -> Optional[InternetResult]:
    """DuckDuckGo — free, no API key."""
    try:
        from duckduckgo_search import DDGS

        snippets = []
        urls     = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title   = r.get("title", "")
                body    = r.get("body", "")
                url     = r.get("href", "")
                if body:
                    snippets.append(f"**{title}**\n{body}")
                    urls.append(url)

        if not snippets:
            return None

        content = "\n\n---\n\n".join(snippets)
        return InternetResult(
            success=True,
            content=f"[Web Search Results — DuckDuckGo]\n\n{content}",
            sources=urls,
            backend="duckduckgo",
        )
    except ImportError:
        logger.debug("[InternetRetriever] duckduckgo-search not installed")
        return None
    except Exception as exc:
        logger.warning(f"[InternetRetriever] DuckDuckGo failed: {exc}")
        return None


def _mock_result(query: str) -> InternetResult:
    """Always-available mock — used when all real backends fail."""
    return InternetResult(
        success=False,
        content=(
            f"[Web Search — Unavailable]\n"
            f"Could not retrieve live web results for: '{query}'\n"
            f"All search backends are currently unavailable. "
            f"Please install 'tavily-python' or 'duckduckgo-search', "
            f"or provide a TAVILY_API_KEY."
        ),
        sources=[],
        backend="mock",
        error="All backends failed or unavailable",
    )


# ─────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────
class InternetRetriever:
    """
    Web search retriever with automatic backend fallback.

    Parameters
    ----------
    tavily_api_key : str — optional, enables Tavily as primary backend
    max_results    : int — number of search results to return per query
    """

    def __init__(
        self,
        tavily_api_key: str = "",
        max_results:    int = 3,
    ):
        self.tavily_api_key = tavily_api_key
        self.max_results    = max_results

    async def search(self, query: str) -> InternetResult:
        """
        Search the web for the given query.
        Tries Tavily → DuckDuckGo → Mock.
        """
        if not query or not query.strip():
            return InternetResult(
                success=False,
                content="Empty search query.",
                backend="none",
                error="Empty query",
            )

        # ── Try Tavily ────────────────────────────────────────────
        result = await _search_tavily(
            query, self.tavily_api_key, self.max_results
        )
        if result:
            logger.info(f"[InternetRetriever] Tavily: {len(result.sources)} results")
            return result

        # ── Try DuckDuckGo ────────────────────────────────────────
        result = await _search_duckduckgo(query, self.max_results)
        if result:
            logger.info(f"[InternetRetriever] DuckDuckGo: {len(result.sources)} results")
            return result

        # ── Mock fallback ─────────────────────────────────────────
        logger.warning("[InternetRetriever] All backends failed — returning mock")
        return _mock_result(query)