"""
Gemini embedding provider implementation (TASK-010).
Note: Replaces OpenAI requirements from original spec based on project configuration.
"""

import logging
from typing import Any

from google import genai
from google.genai.errors import APIError
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.models import ContentChunk, EmbeddedChunk
from app.rag.base import EmbeddingProvider, EmbeddingProviderError

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using Google Gemini API.
    """

    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        """
        Initialize the Gemini embedding provider.

        Args:
            api_key: Gemini API key.
            model:   The embedding model to use.
        """
        if not api_key:
            raise ValueError("Gemini API key cannot be empty")
            
        self.api_key = api_key
        self._model = model
        self.client = genai.Client(api_key=api_key)
        # 768 is the default for text-embedding-004
        self._embedding_dim = 768

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(APIError),
        reraise=True,
    )
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings.

        Uses google-genai to generate embeddings in batch.
        
        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors matching the input order.
            
        Raises:
            EmbeddingProviderError: On API failure after retries.
        """
        if not texts:
            return []
            
        try:
            # The synchronous call is used here but since we are in an async method,
            # in a real high-throughput app we might use asyncio.to_thread.
            # But the google-genai SDK has async support, we will just use the sync
            # client inside to_thread to avoid blocking.
            import asyncio
            
            def _do_embed():
                return self.client.models.embed_content(
                    model=self._model,
                    contents=texts
                )
            
            response = await asyncio.to_thread(_do_embed)
            
            vectors = []
            for item in response.embeddings:
                vectors.append(item.values)
                
            return vectors
            
        except APIError as e:
            logger.warning(f"Gemini embedding API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in Gemini embedding: {e}")
            raise EmbeddingProviderError(f"Embedding failed: {str(e)}") from e

    async def embed_chunks(self, chunks: list[ContentChunk]) -> list[EmbeddedChunk]:
        """
        Embed a list of ContentChunk objects, handling batching.
        
        Overrides base to add batching so we don't hit payload size limits
        on large sets of chunks.
        """
        if not chunks:
            raise ValueError("Must provide at least one chunk to embed.")
            
        batch_size = 50  # Gemini typically accepts up to 100 per request, keeping safe
        all_embedded = []
        
        for i in range(0, len(chunks), batch_size):
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

