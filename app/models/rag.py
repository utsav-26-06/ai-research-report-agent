"""
RAG pipeline models - chunked and embedded document representations.

Traceability chain:
    SourceDocument -> ContentChunk -> EmbeddedChunk
    source_id and sub_question_id are always preserved.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field, field_validator


def _new_id() -> str:
    return str(uuid.uuid4())


class ContentChunk(BaseModel):
    """
    A single text chunk produced by splitting a SourceDocument.

    Attributes:
        chunk_id:        Stable UUID for this chunk.
        source_id:       Links back to SourceDocument.source_id.
        url:             Exact URL of the originating source.
        source_title:    Title of the originating page.
        domain:          Apex domain of the source.
        sub_question_id: Provenance link to the SubQuestion.
        chunk_index:     Zero-based position within the parent document.
        total_chunks:    Total number of chunks from the parent document.
        text:            The chunk text content.
    """

    chunk_id: str = Field(default_factory=_new_id)
    source_id: str = Field(..., description="Matches SourceDocument.source_id.")
    url: str = Field(..., description="Exact URL of the originating source.")
    source_title: str = Field(default="")
    domain: str = Field(default="")
    sub_question_id: str = Field(
        ..., description="Provenance link to SubQuestion.sub_question_id."
    )
    chunk_index: int = Field(..., ge=0, description="Zero-based chunk position.")
    total_chunks: int = Field(..., ge=1, description="Total chunks from parent document.")
    text: str = Field(..., min_length=1, description="Chunk text content.")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v

    @property
    def word_count(self) -> int:
        """Number of words in this chunk."""
        return len(self.text.split())


class EmbeddedChunk(BaseModel):
    """
    A ContentChunk with its dense embedding vector attached.

    Attributes:
        chunk:     The original ContentChunk.
        embedding: Float vector from the embedding model.
        model:     Name of the embedding model used (for auditability).
    """

    chunk: ContentChunk
    embedding: list[float] = Field(
        ...,
        min_length=1,
        description="Dense embedding vector.",
    )
    model: str = Field(
        ...,
        description="Embedding model identifier (e.g. models/text-embedding-004).",
    )

    @field_validator("embedding")
    @classmethod
    def embedding_not_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("embedding vector must not be empty")
        return v

    @property
    def chunk_id(self) -> str:
        """Convenience accessor: same as chunk.chunk_id."""
        return self.chunk.chunk_id

    @property
    def source_id(self) -> str:
        """Convenience accessor: same as chunk.source_id."""
        return self.chunk.source_id

    @property
    def sub_question_id(self) -> str:
        """Convenience accessor: same as chunk.sub_question_id."""
        return self.chunk.sub_question_id
