"""
Semantic Retriever module (TASK-012).
"""

import logging

from app.models import ContentChunk
from app.rag.base import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """Raised when semantic retrieval fails."""


class SemanticRetriever:
    """
    Given a query string, retrieves the top-N most relevant chunks from a VectorStore.
    """

    def __init__(self, vector_store: VectorStore, n_results: int = 5):
        """
        Initialize the retriever.

        Args:
            vector_store: The underlying vector database implementation.
            n_results:    Default number of chunks to retrieve per query.
        """
        self.vector_store = vector_store
        self.n_results = n_results

    async def retrieve(
        self,
        query: str,
        n: int | None = None,
        *,
        sub_question_id: str | None = None,
    ) -> list[ContentChunk]:
        """
        Retrieve chunks semantically similar to the query.

        Args:
            query:           The search string to embed and look up.
            n:               Optional override for number of results.
            sub_question_id: If provided, filter results to this sub-question only.

        Returns:
            Ranked list of ContentChunks (most relevant first).
            Returns an empty list if the store is empty or no matches exist.
            
        Raises:
            RetrieverError: If retrieval from the vector store fails.
        """
        limit = n if n is not None else self.n_results

        if not query.strip():
            logger.warning("Empty query provided to SemanticRetriever. Returning empty list.")
            return []

        try:
            # Query the vector store (which handles embedding the query string)
            results = await self.vector_store.query(
                text=query,
                n=limit,
                sub_question_id=sub_question_id,
            )
            
            # The VectorStore returns a ranked list of EmbeddedChunk objects.
            # We strip the raw embeddings before passing chunks to the LLM context.
            return [embedded.chunk for embedded in results]

        except VectorStoreError as e:
            logger.error(f"VectorStore retrieval failed for query '{query}': {e}")
            raise RetrieverError(f"Retrieval failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during retrieval: {e}")
            raise RetrieverError(f"Unexpected retrieval error: {e}") from e
