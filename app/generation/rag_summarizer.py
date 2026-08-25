"""
RAG Summarization module (TASK-013).
Retrieves relevant evidence for a sub-question and synthesizes a grounded Finding.
"""

import logging

from pydantic import BaseModel, Field

from app.generation.base import LLMProvider, LLMProviderError
from app.models.report import Citation, Finding
from app.models.research import SubQuestion
from app.rag.retriever import SemanticRetriever, RetrieverError

logger = logging.getLogger(__name__)


class SummarizerError(Exception):
    """Raised when summarization fails."""


class _FindingResponse(BaseModel):
    """Structured response expected from the LLM."""
    claim: str = Field(..., description="The factual statement synthesized from the text.")
    evidence: str = Field(..., description="Supporting evidence, quotes, or close paraphrase.")
    used_chunk_indices: list[int] = Field(..., description="0-based indices of the evidence chunks used.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this finding based on evidence.")
    uncertain: bool = Field(..., description="True if evidence is insufficient, conflicting, or missing.")
    conflict_note: str = Field(..., description="Explanation if evidence conflicts or is missing. Empty otherwise.")


class RAGSummarizer:
    """
    Synthesizes grounded findings from retrieved text chunks.
    """

    def __init__(self, llm: LLMProvider, retriever: SemanticRetriever):
        self.llm = llm
        self.retriever = retriever

    async def summarize(self, sub_question: SubQuestion) -> Finding:
        """
        Retrieve chunks and generate a grounded finding for a sub-question.
        
        Args:
            sub_question: The focused question to answer.
            
        Returns:
            A Finding object with citations mapping to the retrieved chunks.
            
        Raises:
            SummarizerError: If retrieval or generation fails.
        """
        try:
            # 1. Retrieve relevant chunks
            chunks = await self.retriever.retrieve(
                query=sub_question.text,
                sub_question_id=sub_question.sub_question_id,
            )
        except RetrieverError as e:
            logger.error(f"Failed to retrieve chunks for '{sub_question.text}': {e}")
            raise SummarizerError(f"Retrieval failed: {e}") from e

        # If no chunks were retrieved, handle the insufficient evidence path gracefully
        if not chunks:
            logger.warning(f"No chunks retrieved for sub-question: {sub_question.text}")
            return Finding(
                claim=f"Unable to answer: {sub_question.text}",
                evidence="No relevant sources were found to support this question.",
                citations=[],
                confidence=0.0,
                uncertain=True,
                conflict_note="Zero sources retrieved from the vector store.",
                sub_question_id=sub_question.sub_question_id,
            )

        # 2. Build the context for the LLM
        context_parts = []
        for i, chunk in enumerate(chunks):
            # Include an index so the LLM can reference which chunk it used
            context_parts.append(
                f"--- Chunk {i} ---\n"
                f"URL: {chunk.url}\n"
                f"Content: {chunk.text}\n"
            )
        context_str = "\n".join(context_parts)

        # 3. Build the prompt
        prompt = (
            f"You are a strict, objective research assistant.\n"
            f"Answer the following question based ONLY on the provided evidence chunks.\n"
            f"DO NOT invent facts, URLs, or information outside the chunks.\n\n"
            f"Question: {sub_question.text}\n\n"
            f"Evidence Chunks:\n{context_str}\n\n"
            f"Instructions:\n"
            f"- Synthesize a 'claim' answering the question.\n"
            f"- Provide 'evidence' as a direct quote or close paraphrase.\n"
            f"- List the 'used_chunk_indices' (e.g., [0, 2]) that support your claim.\n"
            f"- Rate your 'confidence' from 0.0 to 1.0.\n"
            f"- Set 'uncertain' to true if the evidence is missing, conflicting, or doesn't fully answer the question.\n"
            f"- If 'uncertain' is true, describe why in 'conflict_note'."
        )

        # 4. Generate structured response
        try:
            schema = _FindingResponse.model_json_schema()
            raw_output = await self.llm.structured_complete(
                prompt=prompt,
                schema=schema,
                temperature=0.0  # Zero temperature for maximum factuality
            )
            response = _FindingResponse.model_validate(raw_output)
        except Exception as e:
            logger.error(f"LLM generation failed for sub-question '{sub_question.text}': {e}")
            raise SummarizerError(f"LLM generation failed: {e}") from e

        # 5. Build citations from the indices
        citations = []
        # Filter indices to valid range just in case the LLM hallucinates an index
        valid_indices = [i for i in response.used_chunk_indices if 0 <= i < len(chunks)]
        
        for idx in valid_indices:
            chunk = chunks[idx]
            marker = f"[{len(citations) + 1}]"
            citation = Citation(
                marker=marker,
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
                url=chunk.url,
                title=chunk.source_title,
                domain=chunk.domain,
                excerpt=chunk.text[:100] + "..."  # Short snippet for the citation
            )
            citations.append(citation)

        # 6. Return the mapped Finding
        return Finding(
            claim=response.claim,
            evidence=response.evidence,
            citations=citations,
            confidence=response.confidence,
            uncertain=response.uncertain,
            conflict_note=response.conflict_note,
            sub_question_id=sub_question.sub_question_id,
        )
