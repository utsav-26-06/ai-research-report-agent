"""
ChromaDB vector store implementation (TASK-011).
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.models import ContentChunk, EmbeddedChunk
from app.rag.base import EmbeddingProvider, VectorStore, VectorStoreError

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "research_chunks"


def _build_metadata(chunk: ContentChunk) -> dict:
    return {
        "source_id": chunk.source_id,
        "url": chunk.url,
        "source_title": chunk.source_title,
        "domain": chunk.domain,
        "sub_question_id": chunk.sub_question_id,
        "chunk_index": chunk.chunk_index,
        "total_chunks": chunk.total_chunks,
        "text": chunk.text,
    }


def _chunk_from_metadata(meta: dict) -> ContentChunk:
    return ContentChunk(
        source_id=meta["source_id"],
        url=meta["url"],
        source_title=meta.get("source_title", ""),
        domain=meta.get("domain", ""),
        sub_question_id=meta["sub_question_id"],
        chunk_index=meta["chunk_index"],
        total_chunks=meta["total_chunks"],
        text=meta["text"],
    )


class ChromaVectorStore(VectorStore):
    """
    VectorStore backed by ChromaDB.

    Pass persist_directory=None for ephemeral (in-memory) mode used in tests.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        persist_directory: str | None = None,
        collection_name: str = _COLLECTION_NAME,
    ):
        import chromadb

        self._embedding_provider = embedding_provider
        self._collection_name = collection_name

        if persist_directory is not None:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            raise ValueError("Must provide at least one EmbeddedChunk.")
        try:
            await asyncio.to_thread(
                self._collection.upsert,
                ids=[c.chunk_id for c in chunks],
                embeddings=[c.embedding for c in chunks],
                metadatas=[_build_metadata(c.chunk) for c in chunks],
                documents=[c.chunk.text for c in chunks],
            )
        except Exception as e:
            raise VectorStoreError(f"add failed: {e}") from e

    async def query(
        self,
        text: str,
        n: int = 5,
        *,
        sub_question_id: str | None = None,
    ) -> list[EmbeddedChunk]:
        try:
            query_vector = (await self._embedding_provider.embed([text]))[0]
        except Exception as e:
            raise VectorStoreError(f"Embedding query failed: {e}") from e

        where = {"sub_question_id": sub_question_id} if sub_question_id else None

        try:
            results = await asyncio.to_thread(
                self._collection.query,
                query_embeddings=[query_vector],
                n_results=n,
                where=where,
                include=["metadatas", "embeddings", "distances"],
            )
        except Exception as e:
            raise VectorStoreError(f"query failed: {e}") from e

        ids_list = results.get("ids", [[]])[0]
        meta_list = results.get("metadatas", [[]])[0]
        emb_list  = results.get("embeddings", [[]])[0]

        out = []
        for chunk_id, meta, emb in zip(ids_list, meta_list, emb_list):
            try:
                chunk = _chunk_from_metadata(meta)
                # Restore the stored UUID instead of the auto-generated one
                chunk = chunk.model_copy(update={"chunk_id": chunk_id})
                out.append(EmbeddedChunk(chunk=chunk, embedding=list(emb), model=self._embedding_provider.model_name))
            except Exception as exc:
                logger.warning(f"Skipping malformed chunk {chunk_id}: {exc}")
        return out

    async def clear(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorStoreError(f"clear failed: {e}") from e
