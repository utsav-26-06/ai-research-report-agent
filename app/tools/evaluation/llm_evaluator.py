"""
LLM-based source evaluation module (TASK-008).
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
from app.models import SourceDocument, SourceEvaluation
from app.tools.evaluation.base import SourceEvaluator, SourceEvaluatorError

logger = logging.getLogger(__name__)


class EvaluationScores(BaseModel):
    """Schema for LLM structured output during evaluation."""
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="How directly the source addresses the research query.")
    credibility_score: float = Field(..., ge=0.0, le=1.0, description="Estimated authority/reliability of the source.")
    recency_score: float = Field(..., ge=0.0, le=1.0, description="Freshness/currency of the information.")
    redundancy_score: float = Field(..., ge=0.0, le=1.0, description="Overlap with already-accepted sources (0 = completely unique, 1 = total duplicate).")
    reason: str = Field(..., description="Brief explanation for these scores.")


class LLMSourceEvaluator(SourceEvaluator):
    """
    Evaluates SourceDocuments using an LLM to heuristic score relevance,
    credibility, recency, and redundancy.
    """

    def __init__(
        self,
        llm: LLMProvider,
        min_relevance: float = 0.6,
        min_credibility: float = 0.5,
        max_redundancy: float = 0.8,
        min_overall: float = 0.6,
    ):
        """
        Initialize the evaluator with thresholds.
        
        Args:
            llm:              The LLM provider for structured scoring.
            min_relevance:    Minimum acceptable relevance score.
            min_credibility:  Minimum acceptable credibility score.
            max_redundancy:   Maximum acceptable redundancy score.
            min_overall:      Minimum acceptable overall composite score.
        """
        self.llm = llm
        self.min_relevance = min_relevance
        self.min_credibility = min_credibility
        self.max_redundancy = max_redundancy
        self.min_overall = min_overall

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((LLMProviderError, ValidationError)),
        reraise=True,
    )
    async def evaluate(
        self,
        doc: SourceDocument,
        query: str,
        already_included: list[SourceDocument] | None = None,
    ) -> SourceEvaluation:
        """
        Evaluate a SourceDocument using the LLM.
        """
        # Prepare context of already included sources for redundancy check
        included_context = ""
        if already_included:
            summaries = []
            for inc in already_included:
                snippet = inc.text[:200].replace('\n', ' ')
                summaries.append(f"- URL: {inc.url} | Title: {inc.title} | Snippet: {snippet}...")
            included_context = "Already Accepted Sources:\n" + "\n".join(summaries)
        else:
            included_context = "Already Accepted Sources: None."

        # Prepare the document text to evaluate (limit to 3000 chars to save tokens/time during evaluation)
        eval_text = doc.text[:3000]

        prompt = (
            f"You are an expert research evaluator. Your task is to score a candidate source document.\n\n"
            f"Research Query: '{query}'\n\n"
            f"Candidate Source:\n"
            f"URL: {doc.url}\n"
            f"Title: {doc.title}\n"
            f"Author: {doc.author}\n"
            f"Date: {doc.published_date}\n"
            f"Content Snippet: {eval_text}...\n\n"
            f"{included_context}\n\n"
            f"Score the candidate source from 0.0 to 1.0 on:\n"
            f"1. Relevance (does it answer the query?)\n"
            f"2. Credibility (is the domain/author reputable?)\n"
            f"3. Recency (is the information up-to-date?)\n"
            f"4. Redundancy (does it heavily overlap with the Already Accepted Sources? 0.0=unique, 1.0=duplicate)\n"
        )

        schema = EvaluationScores.model_json_schema()

        try:
            raw_output = await self.llm.structured_complete(
                prompt=prompt, schema=schema, temperature=0.1
            )
            
            scores = EvaluationScores.model_validate(raw_output)
            
            # Calculate overall score (simple weighted average, can be adjusted)
            # We invert redundancy so that high redundancy lowers the score
            uniqueness = 1.0 - scores.redundancy_score
            overall = (scores.relevance_score * 0.5) + (scores.credibility_score * 0.3) + (uniqueness * 0.2)
            overall = round(overall, 2)
            
            # Decision logic based on thresholds
            decision = "include"
            reasons = [scores.reason]
            
            if scores.relevance_score < self.min_relevance:
                decision = "exclude"
                reasons.append(f"Relevance ({scores.relevance_score}) below threshold ({self.min_relevance})")
            if scores.credibility_score < self.min_credibility:
                decision = "exclude"
                reasons.append(f"Credibility ({scores.credibility_score}) below threshold ({self.min_credibility})")
            if scores.redundancy_score > self.max_redundancy:
                decision = "exclude"
                reasons.append(f"Redundancy ({scores.redundancy_score}) above threshold ({self.max_redundancy})")
            if overall < self.min_overall:
                decision = "exclude"
                reasons.append(f"Overall score ({overall}) below threshold ({self.min_overall})")
                
            final_reason = " | ".join(reasons)
            
            return SourceEvaluation(
                source_id=doc.source_id,
                url=doc.url,
                relevance_score=scores.relevance_score,
                credibility_score=scores.credibility_score,
                recency_score=scores.recency_score,
                redundancy_score=scores.redundancy_score,
                overall_score=overall,
                decision=decision,
                reason=final_reason,
            )

        except (LLMProviderError, ValidationError) as e:
            logger.warning(f"Validation/LLM error during evaluation for {doc.url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error evaluating {doc.url}: {e}")
            raise SourceEvaluatorError(f"Evaluation failed: {str(e)}") from e
