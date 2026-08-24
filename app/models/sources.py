"""
Source pipeline models - search results through source evaluation.

Traceability chain:
    SearchResult -> SourceDocument -> SourceEvaluation
    All carry sub_question_id and url for end-to-end provenance.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SearchResult(BaseModel):
    """
    One result returned by a search engine for a single query.

    Attributes:
        result_id:       Stable UUID for this result.
        url:             Source URL.
        title:           Page title from the search engine.
        snippet:         Short excerpt shown in search results.
        query:           The exact search query that produced this result.
        sub_question_id: ID of the SubQuestion that triggered this query.
        retrieved_at:    UTC timestamp of when the search was performed.
    """

    result_id: str = Field(default_factory=_new_id)
    url: str = Field(..., description="Source URL.")
    title: str = Field(default="", description="Page title.")
    snippet: str = Field(default="", description="Search result snippet/excerpt.")
    query: str = Field(..., description="The search query that produced this result.")
    sub_question_id: str = Field(
        ..., description="ID of the SubQuestion that triggered this query."
    )
    retrieved_at: datetime = Field(default_factory=_now)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"url must start with http:// or https://, got: {v!r}")
        return v


class SourceDocument(BaseModel):
    """
    Full extracted content from a single web page.

    Attributes:
        source_id:       Stable UUID - used in Citations and Findings.
        url:             Original URL (exact copy from SearchResult).
        title:           Page title.
        domain:          Apex domain (e.g. 'example.com').
        author:          Author name if extractable, else None.
        published_date:  ISO-8601 date string if found, else None.
        text:            Clean extracted body text.
        word_count:      Number of whitespace-separated words in text.
        sub_question_id: Provenance link back to the SubQuestion.
        query:           The specific query that led to this document.
        retrieved_at:    UTC timestamp of extraction.
    """

    source_id: str = Field(default_factory=_new_id)
    url: str = Field(..., description="Exact URL of the source.")
    title: str = Field(default="")
    domain: str = Field(default="")
    author: str | None = Field(default=None)
    published_date: str | None = Field(default=None)
    text: str = Field(..., min_length=1, description="Extracted body text.")
    word_count: int = Field(default=0, ge=0)
    sub_question_id: str = Field(..., description="Provenance link to SubQuestion.")
    query: str = Field(default="")
    retrieved_at: datetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def set_word_count(self) -> "SourceDocument":
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())
        return self

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"url must start with http:// or https://")
        return v


class SourceEvaluation(BaseModel):
    """
    AI-assisted heuristic scores for a single SourceDocument.

    DISCLAIMER: These scores are heuristics, not objective truth.
    They are one signal among many and should not be treated as
    verified domain authority data.

    Attributes:
        source_id:         Links back to SourceDocument.source_id.
        url:               Exact URL (for quick lookups).
        relevance_score:   0.0-1.0 - how directly the source addresses the research.
        credibility_score: 0.0-1.0 - estimated domain/author authority heuristic.
        recency_score:     0.0-1.0 - freshness of the content.
        redundancy_score:  0.0-1.0 - overlap with already-accepted sources (0 = unique).
        overall_score:     Weighted composite score.
        decision:          "include" | "exclude".
        reason:            Human-readable explanation of the decision.
    """

    source_id: str = Field(..., description="Matches SourceDocument.source_id.")
    url: str = Field(..., description="Exact URL of the evaluated source.")
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    credibility_score: float = Field(..., ge=0.0, le=1.0)
    recency_score: float = Field(..., ge=0.0, le=1.0)
    redundancy_score: float = Field(..., ge=0.0, le=1.0)
    overall_score: float = Field(..., ge=0.0, le=1.0)
    decision: str = Field(..., description="include | exclude")
    reason: str = Field(default="", description="Explanation for the decision.")

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"include", "exclude"}:
            raise ValueError(f"decision must be 'include' or 'exclude', got {v!r}")
        return v

    @property
    def is_included(self) -> bool:
        """True if this source passed evaluation."""
        return self.decision == "include"
