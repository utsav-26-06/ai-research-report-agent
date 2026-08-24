"""
Abstract base class for all web search providers.

Concrete implementations:
    TavilySearchProvider  (app/tools/search/tavily.py)  -- TASK-006

Design contract:
    - search() is always async to avoid blocking the event loop.
    - Returns an ordered list of SearchResult; order = provider relevance rank.
    - Raises SearchProviderError on non-retryable failures.
    - sub_question_id is threaded through so provenance is never lost.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import SearchResult


class SearchProviderError(Exception):
    """Raised when a search provider returns an error or exhausts retries."""


class SearchProvider(ABC):
    """
    Abstract interface for web search providers.

    All concrete implementations MUST override `search`.
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        sub_question_id: str,
        *,
        max_results: int = 5,
    ) -> list[SearchResult]:
        """
        Execute a web search and return ranked results.

        Args:
            query:           The search query string.
            sub_question_id: Provenance ID - attached to every returned SearchResult.
            max_results:     Maximum number of results to return.

        Returns:
            Ordered list of SearchResult objects (may be empty if no results found).

        Raises:
            SearchProviderError: On unrecoverable API errors.
        """

    @property
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'tavily')."""
        return self.__class__.__name__.replace("SearchProvider", "").lower()
