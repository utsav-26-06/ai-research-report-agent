"""
Unit tests for the QueryPlanner (TASK-005).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError

from app.agent.query_planner import QueryPlanner, PlannerError
from app.generation.base import LLMProvider, LLMProviderError
from app.models import ResearchRequest


class MockLLMProvider(LLMProvider):
    def __init__(self):
        self.mock_structured_complete = AsyncMock()
        self.mock_complete = AsyncMock()

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def provider_name(self) -> str:
        return "mock-provider"

    async def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        return await self.mock_complete(prompt, temperature=temperature)

    async def structured_complete(self, prompt: str, schema: dict, *, temperature: float | None = None) -> dict:
        return await self.mock_structured_complete(prompt, schema=schema, temperature=temperature)


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def planner(mock_llm):
    return QueryPlanner(llm=mock_llm)


@pytest.fixture
def valid_request():
    return ResearchRequest(topic="Quantum Computing Advances", depth="standard")


@pytest.mark.asyncio
async def test_generate_plan_success(planner, mock_llm, valid_request):
    mock_llm.mock_structured_complete.return_value = {
        "research_objective": "Understand recent advances in quantum computing.",
        "sub_questions": [
            {"text": "What are the latest breakthroughs in qubit stability?", "search_queries": ["qubit stability 2024", "error correction quantum computing"]},
            {"text": "Which companies lead in quantum hardware?", "search_queries": ["quantum hardware leaders", "IBM vs Google quantum"]},
        ]
    }

    plan = await planner.generate_plan(valid_request)

    assert plan.request == valid_request
    assert plan.research_objective == "Understand recent advances in quantum computing."
    assert len(plan.sub_questions) == 2
    assert plan.sub_questions[0].text == "What are the latest breakthroughs in qubit stability?"
    assert plan.sub_questions[0].priority == 1
    assert plan.sub_questions[1].priority == 2


@pytest.mark.asyncio
async def test_generate_plan_deduplication(planner, mock_llm, valid_request):
    # Mock returns two identical sub-questions (differing only by case/whitespace)
    mock_llm.mock_structured_complete.return_value = {
        "research_objective": "Test objective.",
        "sub_questions": [
            {"text": "What is AI?", "search_queries": ["AI definition"]},
            {"text": " WHAT IS AI?  ", "search_queries": ["what is ai"]},
            {"text": "How does it work?", "search_queries": ["ai inner workings"]},
        ]
    }

    plan = await planner.generate_plan(valid_request)

    # Should deduplicate the second question
    assert len(plan.sub_questions) == 2
    assert plan.sub_questions[0].text == "What is AI?"
    assert plan.sub_questions[1].text == "How does it work?"


@pytest.mark.asyncio
async def test_generate_plan_validation_error_retries(planner, mock_llm, valid_request):
    # First call returns invalid schema (missing research_objective), second returns valid
    mock_llm.mock_structured_complete.side_effect = [
        {"sub_questions": []}, # Invalid, missing objective
        {
            "research_objective": "Valid objective.",
            "sub_questions": [
                {"text": "Valid Q1?", "search_queries": ["Q1"]}
            ]
        }
    ]

    plan = await planner.generate_plan(valid_request)
    assert len(plan.sub_questions) == 1
    assert mock_llm.mock_structured_complete.call_count == 2


@pytest.mark.asyncio
async def test_generate_plan_llm_error_retries(planner, mock_llm, valid_request):
    # First call throws LLMProviderError, second succeeds
    mock_llm.mock_structured_complete.side_effect = [
        LLMProviderError("API Rate Limit"),
        {
            "research_objective": "Valid objective.",
            "sub_questions": [
                {"text": "Valid Q1?", "search_queries": ["Q1"]}
            ]
        }
    ]

    plan = await planner.generate_plan(valid_request)
    assert len(plan.sub_questions) == 1
    assert mock_llm.mock_structured_complete.call_count == 2


@pytest.mark.asyncio
async def test_generate_plan_exhausts_retries(planner, mock_llm, valid_request):
    mock_llm.mock_structured_complete.side_effect = LLMProviderError("Fatal API Error")

    with pytest.raises(LLMProviderError):
        await planner.generate_plan(valid_request)

    # 3 attempts configured in tenacity retry
    assert mock_llm.mock_structured_complete.call_count == 3
