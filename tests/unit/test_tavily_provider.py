"""
Unit tests for the TavilySearchProvider (TASK-006).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from app.tools.search.tavily_provider import TavilySearchProvider
from app.tools.search.base import SearchProviderError


@pytest.fixture
def mock_tavily_client():
    with patch("app.tools.search.tavily_provider.AsyncTavilyClient") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def provider(mock_tavily_client):
    return TavilySearchProvider(api_key="test-key", max_results=5)


def test_provider_name(provider):
    assert provider.provider_name == "tavily"


def test_missing_api_key_raises():
    with pytest.raises(ValueError):
        TavilySearchProvider(api_key="")


@pytest.mark.asyncio
async def test_search_success(provider, mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [
            {"url": "https://example.com/1", "title": "Title 1", "content": "Snippet 1"},
            {"url": "https://example.com/2", "title": "Title 2", "content": "Snippet 2"},
        ]
    }

    results = await provider.search("AI research", sub_question_id="sq1")

    assert len(results) == 2
    assert results[0].url == "https://example.com/1"
    assert results[0].title == "Title 1"
    assert results[0].snippet == "Snippet 1"
    assert results[0].query == "AI research"
    assert results[0].sub_question_id == "sq1"

    mock_tavily_client.search.assert_awaited_once_with(
        query="AI research",
        search_depth="basic",
        max_results=5,
        include_raw_content=False,
    )


@pytest.mark.asyncio
async def test_search_max_results_limit(provider, mock_tavily_client):
    mock_tavily_client.search.return_value = {"results": []}

    # Override default max_results
    await provider.search("test", sub_question_id="sq1", max_results=10)

    mock_tavily_client.search.assert_awaited_once_with(
        query="test",
        search_depth="basic",
        max_results=10,
        include_raw_content=False,
    )


@pytest.mark.asyncio
async def test_search_api_error(provider, mock_tavily_client):
    mock_tavily_client.search.side_effect = Exception("API Timeout")

    with pytest.raises(SearchProviderError, match="Tavily search failed"):
        await provider.search("test", sub_question_id="sq1")


@pytest.mark.asyncio
async def test_search_empty_results(provider, mock_tavily_client):
    mock_tavily_client.search.return_value = {}  # Missing 'results' key

    results = await provider.search("test", sub_question_id="sq1")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_filters_bad_urls(provider, mock_tavily_client):
    mock_tavily_client.search.return_value = {
        "results": [
            {"url": "https://example.com/good", "title": "Good"},
            {"url": "ftp://example.com/bad", "title": "Bad Scheme"},
            {"url": "", "title": "Empty URL"},
            {"url": "https://example.com/good/", "title": "Duplicate With Slash"},
            {"url": "https://example.com/good", "title": "Exact Duplicate"},
        ]
    }

    results = await provider.search("test", sub_question_id="sq1")

    # Should only keep the first "good" URL and filter the rest
    assert len(results) == 1
    assert results[0].url == "https://example.com/good"
