"""
tools.py — Search Tool Abstraction Layer for ReAct Agent

Design:
    - Abstract base interface (SearchTool) for any search provider
    - DuckDuckGoSearch: free, no API key required (default)
    - TavilySearch: cleaner results, requires API key (optional)
    - Unified return format: plain string (summary or error message)
    - All exceptions caught internally — never crashes the agent loop

Usage:
    from tools import create_search_tool
    search = create_search_tool()              # defaults to DuckDuckGo
    search = create_search_tool("tavily")      # uses Tavily (needs TAVILY_API_KEY in .env)
    result = search("Japan population 2025")
"""

import os
from abc import ABC, abstractmethod


class SearchTool(ABC):
    """Abstract interface for any search provider."""

    @abstractmethod
    def __call__(self, query: str) -> str:
        """
        Execute a search query and return results as a plain text string.

        Args:
            query: The search query string.

        Returns:
            A string containing search result summaries,
            or an error/no-result message.
        """
        ...


class DuckDuckGoSearch(SearchTool):
    """
    Search using DuckDuckGo via the `duckduckgo-search` package.
    Free, no API key required.
    """

    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        try:
            from duckduckgo_search import DDGS
            self._ddgs_cls = DDGS
        except ImportError:
            raise ImportError(
                "duckduckgo-search package is required. "
                "Install it with: pip install duckduckgo-search"
            )

    def __call__(self, query: str) -> str:
        try:
            with self._ddgs_cls() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))

            if not results:
                return f"No results found for: '{query}'"

            # Format each result as a concise summary block
            summaries = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                body = r.get("body", "No snippet available")
                href = r.get("href", "")
                summaries.append(f"[{i}] {title}\n    {body}\n    URL: {href}")

            return "\n\n".join(summaries)

        except Exception as e:
            return f"Search error: {type(e).__name__} — {e}"


class TavilySearch(SearchTool):
    """
    Search using the Tavily API.
    Requires TAVILY_API_KEY in environment variables.
    """

    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TAVILY_API_KEY not found in environment variables. "
                "Set it in your .env file."
            )
        try:
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "tavily-python package is required. "
                "Install it with: pip install tavily-python"
            )

    def __call__(self, query: str) -> str:
        try:
            response = self._client.search(
                query=query,
                max_results=self.max_results,
                search_depth="basic",
            )

            results = response.get("results", [])
            if not results:
                return f"No results found for: '{query}'"

            summaries = []
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                content = r.get("content", "No snippet available")
                url = r.get("url", "")
                summaries.append(f"[{i}] {title}\n    {content}\n    URL: {url}")

            return "\n\n".join(summaries)

        except Exception as e:
            return f"Search error: {type(e).__name__} — {e}"


# ----- Factory Function -----

_PROVIDERS = {
    "duckduckgo": DuckDuckGoSearch,
    "ddg": DuckDuckGoSearch,
    "tavily": TavilySearch,
}


def create_search_tool(provider: str = "duckduckgo", **kwargs) -> SearchTool:
    """
    Factory: create a search tool instance by provider name.

    Args:
        provider: One of "duckduckgo" (or "ddg"), "tavily".
        **kwargs: Passed to the provider constructor (e.g., max_results=5).

    Returns:
        A SearchTool instance ready to be called.

    Example:
        search = create_search_tool("tavily", max_results=5)
        result = search("latest AI news")
    """
    provider_key = provider.lower().strip()
    if provider_key not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys())
        raise ValueError(
            f"Unknown search provider '{provider}'. Available: {available}"
        )

    return _PROVIDERS[provider_key](**kwargs)
