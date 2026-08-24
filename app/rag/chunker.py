"""
Document chunking for RAG (TASK-009).
"""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import ContentChunk, SourceDocument

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Splits accepted SourceDocuments into overlapping text chunks
    suitable for dense vector embedding.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunker.

        Args:
            chunk_size:    Target max character count per chunk.
            chunk_overlap: Number of characters to overlap between adjacent chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use LangChain's recursive splitter for smart paragraph/sentence boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_document(self, doc: SourceDocument) -> list[ContentChunk]:
        """
        Split a single SourceDocument into multiple ContentChunks.

        Args:
            doc: The validated SourceDocument to chunk.

        Returns:
            List of ContentChunks, retaining all provenance fields.
        """
        text = doc.text.strip()
        if not text:
            logger.warning(f"Skipping empty document: {doc.url}")
            return []

        # Split the text
        raw_chunks = self.splitter.split_text(text)
        
        # Filter out empty chunks that might somehow occur
        raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
        
        if not raw_chunks:
            return []

        total = len(raw_chunks)
        content_chunks = []
        
        for idx, chunk_text in enumerate(raw_chunks):
            chunk = ContentChunk(
                source_id=doc.source_id,
                url=doc.url,
                source_title=doc.title,
                domain=doc.domain,
                sub_question_id=doc.sub_question_id,
                chunk_index=idx,
                total_chunks=total,
                text=chunk_text,
            )
            content_chunks.append(chunk)
            
        logger.debug(f"Chunked {doc.url} into {total} chunks.")
        return content_chunks

    def chunk_documents(self, docs: list[SourceDocument]) -> list[ContentChunk]:
        """
        Convenience method to chunk multiple documents.

        Args:
            docs: List of SourceDocuments.

        Returns:
            Flat list of all resulting ContentChunks.
        """
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
