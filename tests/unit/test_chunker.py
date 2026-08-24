"""
Unit tests for DocumentChunker (TASK-009).
"""

from __future__ import annotations

import pytest

from app.rag.chunker import DocumentChunker
from app.models import SourceDocument


@pytest.fixture
def chunker():
    # Small chunks for testing
    return DocumentChunker(chunk_size=50, chunk_overlap=10)


@pytest.fixture
def source_doc():
    return SourceDocument(
        url="https://example.com/doc",
        title="Test Doc",
        domain="example.com",
        sub_question_id="sq-123",
        text="A" * 100  # 100 characters of 'A'
    )


def test_chunker_initialization():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        DocumentChunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap must be >= 0 and < chunk_size"):
        DocumentChunker(chunk_size=100, chunk_overlap=100)
        
    with pytest.raises(ValueError, match="chunk_overlap must be >= 0 and < chunk_size"):
        DocumentChunker(chunk_size=100, chunk_overlap=-1)


def test_chunk_document_basic(chunker, source_doc):
    # 100 chars, chunk size 50, overlap 10
    # C1: 0-50 (50)
    # C2: 40-90 (50)
    # C3: 80-100 (20)
    # Total chunks should be 3.
    chunks = chunker.chunk_document(source_doc)
    
    assert len(chunks) == 3
    assert chunks[0].total_chunks == 3
    assert chunks[1].total_chunks == 3
    assert chunks[2].total_chunks == 3
    
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[2].chunk_index == 2
    
    # Check provenance
    for chunk in chunks:
        assert chunk.source_id == source_doc.source_id
        assert chunk.url == source_doc.url
        assert chunk.source_title == source_doc.title
        assert chunk.domain == source_doc.domain
        assert chunk.sub_question_id == source_doc.sub_question_id


def test_chunk_empty_document(chunker):
    empty_doc = SourceDocument(
        url="https://example.com/empty",
        text="   \n  ",
        sub_question_id="sq-1"
    )
    chunks = chunker.chunk_document(empty_doc)
    assert len(chunks) == 0


def test_chunk_documents_batch(chunker, source_doc):
    doc2 = SourceDocument(
        url="https://example.com/doc2",
        text="B" * 30,  # Should be 1 chunk
        sub_question_id="sq-2"
    )
    
    # doc1 -> 3 chunks, doc2 -> 1 chunk => 4 chunks total
    all_chunks = chunker.chunk_documents([source_doc, doc2])
    assert len(all_chunks) == 4
    
    # First 3 should belong to doc1
    assert all(c.url == "https://example.com/doc" for c in all_chunks[:3])
    # Last should belong to doc2
    assert all_chunks[3].url == "https://example.com/doc2"
