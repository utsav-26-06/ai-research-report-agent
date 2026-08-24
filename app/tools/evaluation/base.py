"""
Abstract base class for source evaluators.

Concrete implementations:
    HeuristicEvaluator  (app/tools/evaluation/heuristic.py)  -- TASK-008

Design contract:
    - evaluate() may be async to allow LLM-assisted scoring in future.
    - MUST return SourceEvaluation with decision in {"include", "exclude"}.
    - Scores are heuristics - document this in any concrete implementation.
    - evaluate_batch() is provided as a convenience default; override for efficiency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import SourceDocument, SourceEvaluation


class SourceEvaluatorError(Exception):
    """Raised when evaluation fails for a reason other than low score."""


class SourceEvaluator(ABC):
    """
    Abstract interface for evaluating and filtering source documents.

    Implementors assess relevance, credibility, recency, and redundancy
    and return a structured SourceEvaluation with an include/exclude decision.

    All concrete implementations MUST override `evaluate`.
    """

    @abstractmethod
    async def evaluate(
        self,
        doc: SourceDocument,
        query: str,
        already_included: list[SourceDocument] | None = None,
    ) -> SourceEvaluation:
        """
        Evaluate a single SourceDocument and return a structured assessment.

        Args:
            doc:               The source document to evaluate.
            query:             The search query that retrieved this document.
            already_included:  Documents already accepted (used for redundancy scoring).

        Returns:
            SourceEvaluation with scores and an include/exclude decision.

        Raises:
            SourceEvaluatorError: On unrecoverable evaluation errors.
        """

    async def evaluate_batch(
        self,
        docs: list[SourceDocument],
        query: str,
    ) -> list[SourceEvaluation]:
        """
        Evaluate a list of documents, accumulating accepted docs for redundancy checks.

        Default implementation calls evaluate() sequentially.
        Override in subclasses for parallelism.

        Args:
            docs:  List of SourceDocuments to evaluate.
            query: The search query that retrieved these documents.

        Returns:
            List of SourceEvaluation objects in the same order as docs.
        """
        accepted: list[SourceDocument] = []
        results: list[SourceEvaluation] = []
        for doc in docs:
            evaluation = await self.evaluate(doc, query, already_included=accepted)
            results.append(evaluation)
            if evaluation.is_included:
                accepted.append(doc)
        return results

    @property
    def evaluator_name(self) -> str:
        """Human-readable evaluator identifier."""
        return self.__class__.__name__.replace("Evaluator", "").lower()
