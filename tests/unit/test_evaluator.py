"""
Unit tests for the LLMSourceEvaluator (TASK-008).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from app.tools.evaluation.llm_evaluator import LLMSourceEvaluator
from app.generation.base import LLMProvider
from app.models import SourceDocument


class MockLLMProvider(LLMProvider):
    def __init__(self):
        self.mock_structured_complete = AsyncMock()

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def provider_name(self) -> str:
        return "mock"

    async def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        return ""

    async def structured_complete(self, prompt: str, schema: dict, *, temperature: float | None = None) -> dict:
        return await self.mock_structured_complete(prompt, schema=schema, temperature=temperature)


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def evaluator(mock_llm):
    return LLMSourceEvaluator(
        llm=mock_llm,
        min_relevance=0.6,
        min_credibility=0.5,
        max_redundancy=0.8,
        min_overall=0.6,
    )


@pytest.fixture
def source_doc():
    return SourceDocument(
        url="https://example.com/ai-news",
        title="AI News",
        text="Recent advancements in artificial intelligence...",
        sub_question_id="sq1",
        query="AI advancements"
    )


@pytest.mark.asyncio
async def test_high_relevance_source_passes(evaluator, mock_llm, source_doc):
    mock_llm.mock_structured_complete.return_value = {
        "relevance_score": 0.9,
        "credibility_score": 0.8,
        "recency_score": 0.9,
        "redundancy_score": 0.1,
        "reason": "Highly relevant and credible."
    }

    result = await evaluator.evaluate(source_doc, "AI advancements")

    assert result.decision == "include"
    assert result.relevance_score == 0.9
    assert result.credibility_score == 0.8
    assert result.overall_score >= 0.6
    assert result.is_included is True


@pytest.mark.asyncio
async def test_low_relevance_source_is_rejected(evaluator, mock_llm, source_doc):
    mock_llm.mock_structured_complete.return_value = {
        "relevance_score": 0.3,
        "credibility_score": 0.8,
        "recency_score": 0.5,
        "redundancy_score": 0.1,
        "reason": "Not relevant to the query."
    }

    result = await evaluator.evaluate(source_doc, "Quantum mechanics")

    assert result.decision == "exclude"
    assert "Relevance (0.3) below threshold (0.6)" in result.reason
    assert result.is_included is False


@pytest.mark.asyncio
async def test_redundant_source_is_flagged(evaluator, mock_llm, source_doc):
    mock_llm.mock_structured_complete.return_value = {
        "relevance_score": 0.9,
        "credibility_score": 0.8,
        "recency_score": 0.9,
        "redundancy_score": 0.9,  # High redundancy
        "reason": "Very similar to existing source."
    }
    
    # We pass it already_included to simulate the redundancy check context
    already_included = [SourceDocument(
        url="https://example.com/duplicate",
        text="Recent advancements in AI...",
        sub_question_id="sq1",
        query="AI advancements"
    )]

    result = await evaluator.evaluate(source_doc, "AI advancements", already_included=already_included)

    assert result.decision == "exclude"
    assert "Redundancy (0.9) above threshold (0.8)" in result.reason
    assert result.is_included is False


@pytest.mark.asyncio
async def test_evaluator_name(evaluator):
    assert evaluator.evaluator_name == "llmsource"
