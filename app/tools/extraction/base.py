"""
Abstract base class for all web content extractors.

Concrete implementations:
    TrafilaturaExtractor  (app/tools/extraction/trafilatura_extractor.py)  -- TASK-007

Design contract:
    - extract() is async; network I/O must never block.
    - Returns None if the page cannot be fetched or parsed (not an error).
    - Raises ContentExtractionError only on programmer errors, not on 404 / empty pages.
    - sub_question_id and query are passed through for provenance on the returned doc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import SourceDocument


class ContentExtractionError(Exception):
    """Raised on unrecoverable extraction failures (e.g. invalid URL scheme)."""


class ContentExtractor(ABC):
    """
    Abstract interface for web page content extractors.

    All concrete implementations MUST override `extract`.
    """

    @abstractmethod
    async def extract(
        self,
        url: str,
        sub_question_id: str,
        query: str = "",
    ) -> SourceDocument | None:
        """
        Fetch and extract clean text content from a URL.

        Args:
            url:             The page URL to fetch and parse.
            sub_question_id: Provenance ID attached to the returned SourceDocument.
            query:           The search query that led to this URL (for provenance).

        Returns:
            SourceDocument with extracted text, or None if the page is
            unreachable / empty / not parseable (not an error condition).

        Raises:
            ContentExtractionError: On invalid input (e.g. non-HTTP URL scheme).
        """

    @property
    def extractor_name(self) -> str:
        """Human-readable extractor identifier."""
        return self.__class__.__name__.replace("Extractor", "").lower()
