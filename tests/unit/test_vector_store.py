"""
Unit tests for ChromaVectorStore (TASK-011).

Uses ChromaDB EphemeralClient (no disk I/O) for fast, isolated tests.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models import ContentChunk, EmbeddedChunk
from app.rag.base import VectorStoreError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(idx: int = 0, sub_question_id: str = "sq-1") -> ContentChunk:
    return ContentChunk(
        source_id="src-1",
        url="https://example.com/doc",
        source_title="Test Doc",
        domain="example.com",
        sub_question_id=sub_question_id,
        chunk_index=idx,
        total_chunks=3,
        text=f"This is chunk number {idx} with some content for testing.",
    )


def _make_embedded(idx: int = 0, sub_question_id: str = "sq-1") -> EmbeddedChunk:
    chunk = _make_chunk(idx, sub_question_id)
    return EmbeddedChunk(
        chunk=chunk,
        embedding=[float(i) / 10 for i in range(768)],
        model="models/text-embedding-004",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedding_provider():
    provider = MagicMock()
    provider.model_name = "models/text-embedding-004"
    provider.embed = AsyncMock(return_value=[[float(i) / 10 for i in range(768)]])
    return provider


@pytest.fixture
def vector_store(mock_embedding_provider):
    """Create an in-memory ChromaVectorStore for testing."""
    from app.rag.vector_store import ChromaVectorStore
    return ChromaVectorStore(
        embedding_provider=mock_embedding_provider,
        persist_directory=None,  # ephemeral
        collection_name="test_collection",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_and_query(vector_store, mock_embedding_provider):
    """Add chunks and verify they are retrievable."""
    embedded = [_make_embedded(i) for i in range(3)]

    await vector_store.add(embedded)

    results = await vector_store.query("chunk content", n=3)

    assert len(results) == 3
    assert all(isinstance(r, EmbeddedChunk) for r in results)
    assert all(r.chunk.source_id == "src-1" for r in results)


@pytest.mark.asyncio
async def test_metadata_preserved(vector_store):
    """Metadata fields must survive the round-trip through ChromaDB."""
    ec = _make_embedded(0, sub_question_id="sq-meta")
    await vector_store.add([ec])

    results = await vector_store.query("chunk content", n=1)

    assert len(results) == 1
    chunk = results[0].chunk
    assert chunk.source_id == "src-1"
    assert chunk.url == "https://example.com/doc"
    assert chunk.source_title == "Test Doc"
    assert chunk.domain == "example.com"
    assert chunk.sub_question_id == "sq-meta"


@pytest.mark.asyncio
async def test_clear_empties_collection(vector_store):
    """clear() should remove all stored chunks."""
    embedded = [_make_embedded(i) for i in range(5)]
    await vector_store.add(embedded)

    await vector_store.clear()

    # After clear, query returns nothing
    results = await vector_store.query("chunk", n=5)
    assert results == []


@pytest.mark.asyncio
async def test_query_with_sub_question_filter(vector_store):
    """Filtering by sub_question_id should restrict results."""
    sq1_chunks = [_make_embedded(i, sub_question_id="sq-1") for i in range(2)]
    sq2_chunks = [_make_embedded(i, sub_question_id="sq-2") for i in range(2, 4)]

    await vector_store.add(sq1_chunks + sq2_chunks)

    results = await vector_store.query("chunk", n=10, sub_question_id="sq-1")
    assert all(r.chunk.sub_question_id == "sq-1" for r in results)


@pytest.mark.asyncio
async def test_add_empty_raises(vector_store):
    with pytest.raises(ValueError):
        await vector_store.add([])


@pytest.mark.asyncio
async def test_store_name(vector_store):
    assert vector_store.store_name == "chroma"
