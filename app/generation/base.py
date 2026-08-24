"""
Abstract base class for LLM (language model) providers.

Concrete implementations:
    GeminiLLMProvider  (app/generation/gemini_llm.py)  -- used in TASK-005, TASK-013

Design contract:
    - complete() is async; never blocks the event loop.
    - structured_complete() MUST return a dict matching the provided schema.
      It must NOT fabricate data outside the retrieved context.
    - LLMProvider does NOT call any external search or RAG tools directly.
      It only processes the prompt it is given.
    - Concrete implementations must set self.model_name and self.provider_name.

Anti-hallucination note:
    structured_complete() receives a prompt that already contains retrieved
    source context. The LLM must be instructed (in the prompt) to only
    cite sources present in that context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProviderError(Exception):
    """Raised when the LLM API returns an unrecoverable error."""


class LLMProvider(ABC):
    """
    Abstract interface for large language model providers.

    All concrete implementations MUST override `complete` and `structured_complete`.
    """

    @abstractmethod
    async def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        """
        Send a prompt to the LLM and return the generated text response.

        Args:
            prompt:      The full prompt string (system + user context combined).
            temperature: Optional override for sampling temperature.
                         If None, uses the provider's configured default.

        Returns:
            Generated text string. Never empty on success.

        Raises:
            LLMProviderError: On API errors, rate limits, or content filtering.
        """

    @abstractmethod
    async def structured_complete(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """
        Send a prompt and return a structured response matching the given schema.

        The schema is a JSON Schema dict describing the expected output structure.
        Concrete implementations should use JSON mode / function calling to
        guarantee structure.

        Args:
            prompt:      The full prompt including retrieved source context.
            schema:      JSON Schema dict describing the expected response structure.
            temperature: Optional temperature override.

        Returns:
            Dictionary conforming to the provided schema.

        Raises:
            LLMProviderError: On API errors or if the response cannot be parsed
                              into the expected structure.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model (e.g. 'gemini-3.5-flash')."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'gemini', 'openai')."""

    @property
    def supports_structured_output(self) -> bool:
        """
        True if this provider supports native structured/JSON output mode.
        False if structured_complete() uses prompt-based JSON extraction.
        Subclasses should override to reflect actual capability.
        """
        return False
