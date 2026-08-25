"""
Unit tests for RAGSummarizer (TASK-013).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.generation.rag_summarizer import RAGSummarizer, SummarizerError, _FindingResponse
from app.models.research import SubQuestion, ResearchRequest
from app.models.rag import ContentChunk
from app.rag.retriever import RetrieverError


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.structured_complete = AsyncMock()
    return llm


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve = AsyncMock()
    return retriever


@pytest.fixture
def summarizer(mock_llm, mock_retriever):
    return RAGSummarizer(llm=mock_llm, retriever=mock_retriever)


@pytest.fixture
def sub_question():
    request = ResearchRequest(topic="AI", depth="standard")
    return SubQuestion(
        sub_question_id="sq-123",
        text="What are the recent advancements in AI?",
        search_queries=["AI advancements"],
        priority=1,
        request=request,
        research_objective="Understand AI"
    )


@pytest.mark.asyncio
async def test_successful_summarization(summarizer, mock_llm, mock_retriever, sub_question):
    # Mock retrieval
    chunk1 = ContentChunk(source_id="src1", url="http://test.com/1", chunk_index=0, total_chunks=1, text="AI is getting smarter.", sub_question_id="sq-123")
    chunk2 = ContentChunk(source_id="src2", url="http://test.com/2", chunk_index=0, total_chunks=1, text="New models released.", sub_question_id="sq-123")
    mock_retriever.retrieve.return_value = [chunk1, chunk2]

    # Mock LLM response matching _FindingResponse schema
    mock_llm.structured_complete.return_value = {
        "claim": "AI is advancing rapidly.",
        "evidence": "Models are smarter.",
        "used_chunk_indices": [0, 1],
        "confidence": 0.9,
        "uncertain": False,
        "conflict_note": ""
    }

    finding = await summarizer.summarize(sub_question)

    assert finding.sub_question_id == "sq-123"
    assert finding.claim == "AI is advancing rapidly."
    assert len(finding.citations) == 2
    assert finding.citations[0].url == "http://test.com/1"
    assert finding.citations[0].marker == "[1]"
    assert finding.citations[1].url == "http://test.com/2"
    assert finding.citations[1].marker == "[2]"
    assert finding.confidence == 0.9
    assert finding.uncertain is False
    
    mock_retriever.retrieve.assert_called_once_with(
        query="What are the recent advancements in AI?",
        sub_question_id="sq-123"
    )
    mock_llm.structured_complete.assert_called_once()


@pytest.mark.asyncio
async def test_insufficient_evidence(summarizer, mock_llm, mock_retriever, sub_question):
    # Mock retrieval returning empty list
    mock_retriever.retrieve.return_value = []

    finding = await summarizer.summarize(sub_question)

    # Should not call LLM if no chunks
    mock_llm.structured_complete.assert_not_called()
    
    assert finding.uncertain is True
    assert "Unable to answer" in finding.claim
    assert finding.citations == []
    assert finding.confidence == 0.0


@pytest.mark.asyncio
async def test_retriever_error_raises_summarizer_error(summarizer, mock_retriever, sub_question):
    mock_retriever.retrieve.side_effect = RetrieverError("DB down")

    with pytest.raises(SummarizerError, match="Retrieval failed"):
        await summarizer.summarize(sub_question)


@pytest.mark.asyncio
async def test_llm_error_raises_summarizer_error(summarizer, mock_llm, mock_retriever, sub_question):
    mock_retriever.retrieve.return_value = [
        ContentChunk(source_id="s", url="http://u", chunk_index=0, total_chunks=1, text="x", sub_question_id="sq-123")
    ]
    mock_llm.structured_complete.side_effect = Exception("API rate limit")

    with pytest.raises(SummarizerError, match="LLM generation failed"):
        await summarizer.summarize(sub_question)

