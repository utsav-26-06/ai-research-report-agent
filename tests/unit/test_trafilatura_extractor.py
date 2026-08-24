"""
Unit tests for the TrafilaturaExtractor (TASK-007).
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

from app.tools.extraction.trafilatura_extractor import TrafilaturaExtractor
from app.models import SourceDocument


@pytest.fixture
def extractor():
    # Set min_length to 20 for easier testing
    return TrafilaturaExtractor(min_length=20)


def test_extractor_name(extractor):
    assert extractor.extractor_name == "trafilatura"


@pytest.mark.asyncio
@patch("app.tools.extraction.trafilatura_extractor.trafilatura.fetch_url")
@patch("app.tools.extraction.trafilatura_extractor.trafilatura.extract")
async def test_extract_success(mock_extract, mock_fetch, extractor):
    # Mock download success
    mock_fetch.return_value = "<html><body>Fake downloaded content</body></html>"
    
    # Mock trafilatura extraction success (returns JSON string)
    mock_extract.return_value = json.dumps({
        "title": "Test Title",
        "author": "John Doe",
        "date": "2024-01-01",
        "text": "This is a sufficiently long text content for the test to pass the min length check."
    })

    doc = await extractor.extract("https://example.com/article", "sq1", "test query")

    assert doc is not None
    assert doc.url == "https://example.com/article"
    assert doc.title == "Test Title"
    assert doc.author == "John Doe"
    assert doc.published_date == "2024-01-01"
    assert "sufficiently long text" in doc.text
    assert doc.sub_question_id == "sq1"
    assert doc.query == "test query"
    
    mock_fetch.assert_called_once_with("https://example.com/article")
    mock_extract.assert_called_once()


@pytest.mark.asyncio
@patch("app.tools.extraction.trafilatura_extractor.trafilatura.fetch_url")
async def test_extract_handles_download_failure(mock_fetch, extractor):
    # Mock download returning None
    mock_fetch.return_value = None

    doc = await extractor.extract("https://example.com/dead", "sq1")
    
    assert doc is None
    mock_fetch.assert_called_once_with("https://example.com/dead")


@pytest.mark.asyncio
@patch("app.tools.extraction.trafilatura_extractor.trafilatura.fetch_url")
@patch("app.tools.extraction.trafilatura_extractor.trafilatura.extract")
async def test_extract_handles_extract_failure(mock_extract, mock_fetch, extractor):
    mock_fetch.return_value = "<html><body></body></html>"
    mock_extract.return_value = None  # Trafilatura extract fails

    # The fallback BeautifulSoup will also find nothing, resulting in text length 0
    doc = await extractor.extract("https://example.com/empty", "sq1")
    
    assert doc is None
    mock_extract.assert_called_once()


@pytest.mark.asyncio
@patch("app.tools.extraction.trafilatura_extractor.trafilatura.fetch_url")
@patch("app.tools.extraction.trafilatura_extractor.trafilatura.extract")
async def test_beautifulsoup_fallback(mock_extract, mock_fetch, extractor):
    # Mock download success with some text in tags
    mock_fetch.return_value = "<html><head><title>BS4 Title</title></head><body><p>This is the fallback text content extracted by beautiful soup. It needs to be longer than 20 chars.</p></body></html>"
    
    # Mock trafilatura failing
    mock_extract.return_value = None

    doc = await extractor.extract("https://example.com/bs4", "sq1")
    
    assert doc is not None
    assert doc.title == "BS4 Title"
    assert "fallback text content" in doc.text


@pytest.mark.asyncio
@patch("app.tools.extraction.trafilatura_extractor.trafilatura.fetch_url")
@patch("app.tools.extraction.trafilatura_extractor.trafilatura.extract")
async def test_min_length_rejection(mock_extract, mock_fetch, extractor):
    mock_fetch.return_value = "<html><body>Short text</body></html>"
    # Provide JSON with < 20 chars of text
    mock_extract.return_value = json.dumps({"text": "Too short"})

    doc = await extractor.extract("https://example.com/short", "sq1")
    
    assert doc is None
