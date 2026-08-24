"""
Research pipeline models - user input through to research plan.

Traceability chain:
    ResearchRequest -> ResearchPlan -> SubQuestion (sub_question_id flows downstream)
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, field_validator


def _new_id() -> str:
    return str(uuid.uuid4())


class ResearchRequest(BaseModel):
    """
    Entry point model - represents a user's research request.

    Attributes:
        topic:       Free-text research topic (2-500 chars after stripping whitespace).
        depth:       Research depth controlling number of sub-questions and sources.
        max_sources: Hard cap on total sources kept after evaluation.
    """

    topic: str = Field(
        ...,
        min_length=2,
        max_length=500,
        description="The research topic or question to investigate.",
    )
    depth: str = Field(
        default="standard",
        description="Research depth: quick | standard | deep.",
    )
    max_sources: int = Field(
        default=15,
        ge=3,
        le=50,
        description="Maximum number of sources to retain after evaluation.",
    )

    @field_validator("topic", mode="before")
    @classmethod
    def strip_and_validate_topic(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("topic must be a string")
        v = v.strip()
        if len(v) < 2:
            raise ValueError("topic must have at least 2 characters after stripping whitespace")
        return v

    @field_validator("depth")
    @classmethod
    def validate_depth(cls, v: str) -> str:
        allowed = {"quick", "standard", "deep"}
        v = v.lower().strip()
        if v not in allowed:
            raise ValueError(f"depth must be one of {allowed}, got {v!r}")
        return v


class SubQuestion(BaseModel):
    """
    A focused sub-question derived from the main research topic.

    Attributes:
        sub_question_id: Stable UUID used to trace results back to this question.
        text:            The sub-question text.
        search_queries:  Search engine queries generated for this sub-question.
        priority:        Relative priority (1 = highest).
    """

    sub_question_id: str = Field(
        default_factory=_new_id,
        description="Stable UUID - preserved on all downstream objects for provenance.",
    )
    text: str = Field(..., min_length=3, description="The sub-question text.")
    search_queries: list[str] = Field(
        default_factory=list,
        description="Search queries generated for this sub-question.",
    )
    priority: int = Field(
        default=1,
        ge=1,
        description="Priority rank (1 = highest priority).",
    )

    @field_validator("search_queries")
    @classmethod
    def at_least_one_query(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("sub_question must have at least one search_query")
        return [q.strip() for q in v if q.strip()]


class ResearchPlan(BaseModel):
    """
    Planner output - structured decomposition of a research topic.

    Attributes:
        plan_id:            Stable UUID for this plan instance.
        request:            The original ResearchRequest that triggered this plan.
        research_objective: One-paragraph summary of the research goal.
        sub_questions:      Ordered list of SubQuestion objects.
    """

    plan_id: str = Field(default_factory=_new_id)
    request: ResearchRequest
    research_objective: str = Field(
        ...,
        min_length=10,
        description="Clear statement of what the research aims to answer.",
    )
    sub_questions: list[SubQuestion] = Field(
        ...,
        min_length=1,
        description="Ordered list of sub-questions covering the topic.",
    )

    @property
    def all_queries(self) -> list[tuple[str, str]]:
        """Return [(query, sub_question_id), ...] for every query in the plan."""
        return [
            (q, sq.sub_question_id)
            for sq in self.sub_questions
            for q in sq.search_queries
        ]
