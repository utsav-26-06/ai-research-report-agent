"""
Tavily web search provider implementation (TASK-006).
"""

import logging
from typing import Any

from pydantic import ValidationError
from tavily import AsyncTavilyClient

from app.models import SearchResult
from app.tools.search.base import SearchProvider, SearchProviderError

logger = logging.getLogger(__name__)


class TavilySearchProvider(SearchProvider):
    """
    Search provider using the Tavily API (tavily.com).
    Optimised for LLM research workflows.
    """

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        search_depth: str = "basic",
    ):
        """
        Initialize the Tavily search provider.

        Args:
            api_key:      Tavily API key.
            max_results:  Default maximum results per query.
            search_depth: 'basic' or 'advanced'.
        """
        if not api_key:
            raise ValueError("Tavily API key cannot be empty")
            
        self.api_key = api_key
        self._default_max_results = max_results
        self._default_search_depth = search_depth
        self.client = AsyncTavilyClient(api_key=api_key)

    async def search(
        self,
        query: str,
        sub_question_id: str,
        *,
        max_results: int | None = None,
    ) -> list[SearchResult]:
        """
        Execute a search using Tavily API.

        Args:
            query:           The search query string.
            sub_question_id: Provenance ID.
            max_results:     Override for max results (defaults to instance config).

        Returns:
            Ordered list of SearchResult objects, deduplicated by URL.

        Raises:
            SearchProviderError: On Tavily API failures.
        """
        limit = max_results if max_results is not None else self._default_max_results
        
        try:
            logger.info(f"Tavily search: '{query}' (limit={limit})")
            
            # Using basic search for faster response; advanced is for deeper RAG
            # but usually basic is sufficient for broad sweeps.
            response = await self.client.search(
                query=query,
                search_depth=self._default_search_depth,
                max_results=limit,
                include_raw_content=False,
            )
            
            results = []
            seen_urls = set()
            
            for item in response.get("results", []):
                url = item.get("url", "").strip()
                
                # Filter out bad URLs or empty ones
                if not url or not url.startswith(("http://", "https://")):
                    continue
                    
                # Basic normalization (remove trailing slash)
                norm_url = url.rstrip("/")
                if norm_url in seen_urls:
                    continue
                    
                seen_urls.add(norm_url)
                
                try:
                    search_result = SearchResult(
                        url=url,
                        title=item.get("title", ""),
                        snippet=item.get("content", ""),
                        query=query,
                        sub_question_id=sub_question_id,
                    )
                    results.append(search_result)
                except ValidationError as ve:
                    logger.warning(f"Skipping malformed search result: {ve}")
                    continue
                    
            return results

        except Exception as e:
            logger.error(f"Tavily API error for query '{query}': {e}")
            raise SearchProviderError(f"Tavily search failed: {str(e)}") from e
