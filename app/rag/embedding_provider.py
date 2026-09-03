"""
Gemini embedding provider implementation (TASK-010).
Note: Replaces OpenAI requirements from original spec based on project configuration.
"""

import asyncio
import logging
from typing import Any, cast

from google import genai
from google.genai.errors import APIError
from pydantic import ValidationError

from app.models import ContentChunk, EmbeddedChunk
from app.rag.base import EmbeddingProvider, EmbeddingProviderError

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using Google Gemini API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "models/gemini-embedding-001",
        model_name: str | None = None,
    ):
        """
        Initialize the Gemini embedding provider.

        Args:
            api_key:    Gemini API key.
            model:      The embedding model to use.
            model_name: Optional alias for model.
        """
        if not api_key:
            raise ValueError("Gemini API key cannot be empty")
            
        self.api_key = api_key
        self._model = model_name if model_name is not None else model
        self.client = genai.Client(api_key=api_key)
        # 3072 is the default for gemini-embedding-001
        self._embedding_dim = 3072

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @staticmethod
    def _extract_retry_delay(error: Exception, attempt: int) -> float:
        """Extract retry delay requested by Gemini API or fallback to minimal/exponential."""
        import re
        err_str = str(error)
        match = re.search(r"retry in\s+([0-9.]+)\s*s", err_str, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1)) + 1.0
            except ValueError:
                pass

        response_json = getattr(error, "response_json", None)
        if isinstance(response_json, dict):
            details = response_json.get("error", {}).get("details", [])
            for detail in details:
                if isinstance(detail, dict) and "retryDelay" in detail:
                    raw_delay = str(detail["retryDelay"]).rstrip("s")
                    try:
                        return float(raw_delay) + 1.0
                    except ValueError:
                        pass
        return 0.1

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings with automatic backoff retry on 429 rate limits.
        """
        if not texts:
            return []

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                def _do_embed():
                    return self.client.models.embed_content(
                        model=self._model,
                        contents=cast(Any, texts),
                    )

                response = await asyncio.to_thread(_do_embed)

                vectors = []
                for item in response.embeddings:
                    vectors.append(item.values)

                if vectors and len(vectors[0]) > 0:
                    self._embedding_dim = len(vectors[0])

                return vectors

            except APIError as e:
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or getattr(e, "code", None) == 429
                )
                if is_rate_limit and attempt < max_retries:
                    delay = self._extract_retry_delay(e, attempt)
                    logger.warning(
                        f"Gemini embedding API rate limit (429) hit. Pausing for {delay:.1f}s before retry "
                        f"(attempt {attempt}/{max_retries})..."
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning(f"Gemini embedding API error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in Gemini embedding: {e}")
                raise EmbeddingProviderError(f"Embedding failed: {str(e)}") from e

        raise EmbeddingProviderError(f"Embedding failed: max retries ({max_retries}) exceeded.")

    async def embed_chunks(self, chunks: list[ContentChunk]) -> list[EmbeddedChunk]:
        """
        Embed a list of ContentChunk objects, handling batching and pacing.
        """
        if not chunks:
            raise ValueError("Must provide at least one chunk to embed.")

        batch_size = 50  # Gemini typically accepts up to 100 per request, keeping safe
        all_embedded = []

        for i in range(0, len(chunks), batch_size):
            if i > 0:
                await asyncio.sleep(0.5)

            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]

            try:
                vectors = await self.embed(texts)

                if len(vectors) != len(batch):
                    raise EmbeddingProviderError("Mismatch between number of chunks and vectors returned")

                for chunk, vector in zip(batch, vectors):
                    all_embedded.append(
                        EmbeddedChunk(
                            chunk=chunk,
                            embedding=vector,
                            model=self._model
                        )
                    )
            except ValidationError as ve:
                raise EmbeddingProviderError(f"Failed to construct EmbeddedChunk: {ve}") from ve

        return all_embedded

