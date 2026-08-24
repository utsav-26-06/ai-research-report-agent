"""
Abstract base classes for the RAG pipeline:
    - EmbeddingProvider  (embed text -> float vectors)
    - VectorStore        (store, query, clear embedded chunks)

Concrete implementations:
    GeminiEmbeddingProvider   (app/rag/gemini_embeddings.py)   -- TASK-010
    ChromaVectorStore         (app/rag/chroma_store.py)        -- TASK-011

Design contract:
    - All methods are async to avoid blocking the event loop on I/O.
    - EmbeddingProvider.embed() accepts a batch for efficiency.
    - VectorStore.query() returns EmbeddedChunk objects (not raw dicts),
      preserving full provenance metadata.
    - VectorStore.clear() removes all vectors for a fresh research session.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.models import ContentChunk, EmbeddedChunk


# ---------------------------------------------------------------------------
# EmbeddingProvider
# ---------------------------------------------------------------------------


class EmbeddingProviderError(Exception):
    """Raised when the embedding API returns an error or exhausts retries."""


class EmbeddingProvider(ABC):
    """
    Abstract interface for dense text embedding providers.

    All concrete implementations MUST override `embed`.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of text strings into dense float vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors, one per input text, in the same order.
            All vectors MUST have the same dimensionality.

        Raises:
            EmbeddingProviderError: On API errors or empty input.
            ValueError: If texts is empty.
        """

    async def embed_chunks(self, chunks: list[ContentChunk]) -> list[EmbeddedChunk]:
        """
        Convenience: embed a list of ContentChunks and return EmbeddedChunks.

        Subclasses may override for batching optimisation.

        Args:
            chunks: List of ContentChunk objects to embed.

        Returns:
            List of EmbeddedChunk objects preserving full provenance.

        Raises:
            EmbeddingProviderError: On API errors.
            ValueError: If chunks is empty.
        """
        if not chunks:
            raise ValueError("embed_chunks requires at least one chunk")
        texts = [c.text for c in chunks]
        vectors = await self.embed(texts)
        return [
            EmbeddedChunk(chunk=chunk, embedding=vec, model=self.model_name)
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying embedding model (e.g. 'models/text-embedding-004')."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors produced by this provider."""


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class VectorStore(ABC):
    """
    Abstract interface for vector databases storing EmbeddedChunks.

    All concrete implementations MUST override `add`, `query`, and `clear`.
    """

    @abstractmethod
    async def add(self, chunks: list[EmbeddedChunk]) -> None:
        """
        Persist a batch of EmbeddedChunks to the vector store.

        Args:
            chunks: Non-empty list of EmbeddedChunk objects.

        Raises:
            VectorStoreError: On persistence failures.
            ValueError: If chunks is empty.
        """

    @abstractmethod
    async def query(
        self,
        text: str,
        n: int = 5,
        *,
        sub_question_id: str | None = None,
    ) -> list[EmbeddedChunk]:
        """
        Retrieve the top-n most similar EmbeddedChunks for a query text.

        Args:
            text:            Query text to embed and search against.
            n:               Number of results to return.
            sub_question_id: If provided, restrict results to chunks from
                             this sub-question (for focused retrieval).

        Returns:
            Ordered list of EmbeddedChunk objects (most similar first).
            May return fewer than n if the store has fewer entries.

        Raises:
            VectorStoreError: On retrieval failures.
        """

    @abstractmethod
    async def clear(self) -> None:
        """
        Remove all stored chunks from this vector store.

        Used at the start of each new research session to avoid
        cross-contamination between research topics.

        Raises:
            VectorStoreError: On deletion failures.
        """

    @property
    def store_name(self) -> str:
        """Human-readable store identifier (e.g. 'chroma')."""
        return self.__class__.__name__.replace("VectorStore", "").lower()
