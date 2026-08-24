"""
Query Planner Module - Breaks down a research topic into sub-questions.
"""

import logging
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.generation.base import LLMProvider, LLMProviderError
from app.models import ResearchPlan, ResearchRequest, SubQuestion

logger = logging.getLogger(__name__)


class PlannerError(Exception):
    """Raised when the query planner fails to generate a valid plan."""


class PlannedSubQuestion(BaseModel):
    """Schema for LLM structured output for a single sub-question."""
    text: str = Field(description="The focused sub-question text.")
    search_queries: list[str] = Field(description="2-3 specific search engine queries.")


class PlannerOutput(BaseModel):
    """Schema for LLM structured output for the research plan."""
    research_objective: str = Field(description="One-paragraph summary of the research goal.")
    sub_questions: list[PlannedSubQuestion] = Field(description="4-6 focused, non-overlapping sub-questions.")


class QueryPlanner:
    """
    Agent responsible for breaking a research topic into focused sub-questions.
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((LLMProviderError, ValidationError)),
        reraise=True,
    )
    async def generate_plan(self, request: ResearchRequest) -> ResearchPlan:
        """
        Generate a ResearchPlan for the given ResearchRequest.

        Args:
            request: The ResearchRequest containing the topic and parameters.

        Returns:
            A validated ResearchPlan containing sub-questions and search queries.

        Raises:
            PlannerError: If the LLM fails to produce a valid plan after retries.
            LLMProviderError, ValidationError: Bubbled up from retry attempts.
        """
        # Adjust target number of questions based on depth
        if request.depth == "quick":
            target_qs = "2 to 3"
        elif request.depth == "deep":
            target_qs = "6 to 8"
        else:
            target_qs = "4 to 6"

        prompt = (
            f"You are an expert research planner.\n"
            f"Topic: {request.topic}\n\n"
            f"Your task is to break this topic down into {target_qs} focused, non-overlapping sub-questions.\n"
            f"For each sub-question, provide 2-3 specific search engine queries to find relevant information.\n"
            f"Ensure the questions comprehensively cover the topic and avoid redundancy.\n"
            f"Provide a clear, one-paragraph research objective.\n"
        )

        schema = PlannerOutput.model_json_schema()

        try:
            raw_output = await self.llm.structured_complete(
                prompt=prompt, schema=schema, temperature=0.2
            )
            
            # Validate output against our internal model
            planner_output = PlannerOutput.model_validate(raw_output)
            
            # Deduplicate questions (simple lowercasing match)
            seen_texts = set()
            unique_sqs = []
            priority = 1
            
            for sq in planner_output.sub_questions:
                norm_text = sq.text.lower().strip()
                if norm_text not in seen_texts:
                    seen_texts.add(norm_text)
                    unique_sqs.append(
                        SubQuestion(
                            text=sq.text,
                            search_queries=sq.search_queries,
                            priority=priority,
                        )
                    )
                    priority += 1

            if not unique_sqs:
                raise PlannerError("LLM generated no valid sub-questions.")

            # Build final plan
            plan = ResearchPlan(
                request=request,
                research_objective=planner_output.research_objective,
                sub_questions=unique_sqs,
            )
            
            return plan

        except (LLMProviderError, ValidationError) as e:
            logger.warning(f"Failed to generate plan (will retry if applicable): {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in query planner: {e}")
            raise PlannerError(f"Failed to generate research plan: {str(e)}") from e
