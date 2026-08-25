"""
Unit tests for SemanticRetriever (TASK-012).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.models import ContentChunk, EmbeddedChunk
from app.rag.retriever import SemanticRetriever, RetrieverError
from app.rag.base import VectorStoreError


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    # async query method
    store.query = AsyncMock()
    return store


@pytest.fixture
def retriever(mock_vector_store):
    return SemanticRetriever(vector_store=mock_vector_store, n_results=2)


def _make_embedded(idx: int = 0, text: str = "test", sq_id: str = "sq-1") -> EmbeddedChunk:
    chunk = ContentChunk(
        source_id="s1",
        url="http://test.com",
        sub_question_id=sq_id,
        chunk_index=idx,
        total_chunks=1,
        text=text,
    )
    return EmbeddedChunk(chunk=chunk, embedding=[0.1], model="test")


@pytest.mark.asyncio
async def test_retrieval_returns_correct_chunks(retriever, mock_vector_store):
    embedded_results = [
        _make_embedded(0, "first result"),
        _make_embedded(1, "second result"),
    ]
    mock_vector_store.query.return_value = embedded_results

    chunks = await retriever.retrieve("search query")

    assert len(chunks) == 2
    assert chunks[0].text == "first result"
    assert chunks[1].text == "second result"
    
    mock_vector_store.query.assert_called_once_with(
        text="search query",
        n=2,
        sub_question_id=None
    )


@pytest.mark.asyncio
async def test_filtering_by_metadata_field(retriever, mock_vector_store):
    mock_vector_store.query.return_value = [_make_embedded(0, "filtered", "sq-42")]

    chunks = await retriever.retrieve("search query", n=5, sub_question_id="sq-42")

    assert len(chunks) == 1
    assert chunks[0].sub_question_id == "sq-42"
    
    mock_vector_store.query.assert_called_once_with(
        text="search query",
        n=5,
        sub_question_id="sq-42"
    )


@pytest.mark.asyncio
async def test_empty_store_returns_empty_list(retriever, mock_vector_store):
    mock_vector_store.query.return_value = []

    chunks = await retriever.retrieve("search query")

    assert chunks == []


@pytest.mark.asyncio
async def test_empty_query_string(retriever, mock_vector_store):
    chunks = await retriever.retrieve("   ")

    assert chunks == []
    mock_vector_store.query.assert_not_called()


@pytest.mark.asyncio
async def test_vector_store_error_raises_retriever_error(retriever, mock_vector_store):
    mock_vector_store.query.side_effect = VectorStoreError("DB failure")

    with pytest.raises(RetrieverError, match="DB failure"):
        await retriever.retrieve("search query")
