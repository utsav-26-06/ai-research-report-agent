"""
Unit tests for provider interfaces (TASK-004).

Tests verify:
- ABCs cannot be instantiated directly (TypeError)
- Concrete subclasses with all abstract methods work correctly
- Concrete subclasses missing abstract methods raise TypeError
- Default implementations (evaluate_batch, embed_chunks) behave correctly
"""

from __future__ import annotations

from typing import Any

import pytest

from app.generation.base import LLMProvider, LLMProviderError
from app.models import (
    ContentChunk,
    EmbeddedChunk,
    SearchResult,
    SourceDocument,
    SourceEvaluation,
)
from app.rag.base import EmbeddingProvider, VectorStore
from app.tools.evaluation.base import SourceEvaluator
from app.tools.extraction.base import ContentExtractor
from app.tools.search.base import SearchProvider


# ---------------------------------------------------------------------------
# Minimal concrete implementations for testing
# ---------------------------------------------------------------------------


class ConcreteSearchProvider(SearchProvider):
    async def search(
        self, query: str, sub_question_id: str, *, max_results: int = 5
    ) -> list[SearchResult]:
        return [
            SearchResult(
                url="https://example.com",
                title="Test",
                snippet="snippet",
                query=query,
                sub_question_id=sub_question_id,
            )
        ]


class ConcreteContentExtractor(ContentExtractor):
    async def extract(
        self, url: str, sub_question_id: str, query: str = ""
    ) -> SourceDocument | None:
        return SourceDocument(
            url=url,
            text="Extracted text content for testing purposes.",
            sub_question_id=sub_question_id,
            query=query,
        )


class ConcreteSourceEvaluator(SourceEvaluator):
    async def evaluate(
        self,
        doc: SourceDocument,
        query: str,
        already_included: list[SourceDocument] | None = None,
    ) -> SourceEvaluation:
        return SourceEvaluation(
            source_id=doc.source_id,
            url=doc.url,
            relevance_score=0.8,
            credibility_score=0.7,
            recency_score=0.9,
            redundancy_score=0.1,
            overall_score=0.8,
            decision="include",
            reason="Test evaluation.",
        )


class ConcreteEmbeddingProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "test-embedding-model"

    @property
    def embedding_dim(self) -> int:
        return 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class ConcreteVectorStore(VectorStore):
    def __init__(self):
        self._store: list[EmbeddedChunk] = []

    async def add(self, chunks: list[EmbeddedChunk]) -> None:
        self._store.extend(chunks)

    async def query(
        self, text: str, n: int = 5, *, sub_question_id: str | None = None
    ) -> list[EmbeddedChunk]:
        results = self._store
        if sub_question_id:
            results = [c for c in results if c.sub_question_id == sub_question_id]
        return results[:n]

    async def clear(self) -> None:
        self._store.clear()


class ConcreteLLMProvider(LLMProvider):
    @property
    def model_name(self) -> str:
        return "test-model"

    @property
    def provider_name(self) -> str:
        return "test-provider"

    async def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        return f"Response to: {prompt[:20]}"

    async def structured_complete(
        self, prompt: str, schema: dict[str, Any], *, temperature: float | None = None
    ) -> dict[str, Any]:
        return {"result": "structured"}


# ---------------------------------------------------------------------------
# Incomplete implementations (missing abstract methods)
# ---------------------------------------------------------------------------


class IncompleteSearchProvider(SearchProvider):
    pass  # Missing: search()


class IncompleteContentExtractor(ContentExtractor):
    pass  # Missing: extract()


class IncompleteSourceEvaluator(SourceEvaluator):
    pass  # Missing: evaluate()


class IncompleteEmbeddingProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "test"

    # Missing: embed() and embedding_dim


class IncompleteVectorStore(VectorStore):
    pass  # Missing: add(), query(), clear()


class IncompleteLLMProvider(LLMProvider):
    pass  # Missing: complete(), structured_complete(), model_name, provider_name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sub_question_id() -> str:
    return "test-sq-id-1234"


@pytest.fixture()
def source_doc(sub_question_id) -> SourceDocument:
    return SourceDocument(
        url="https://example.com/page",
        text="This is test content for source evaluation." * 10,
        sub_question_id=sub_question_id,
        query="test query",
    )


@pytest.fixture()
def content_chunk(source_doc, sub_question_id) -> ContentChunk:
    return ContentChunk(
        source_id=source_doc.source_id,
        url=source_doc.url,
        sub_question_id=sub_question_id,
        chunk_index=0,
        total_chunks=1,
        text="Chunk text for embedding test.",
    )


@pytest.fixture()
def embedded_chunk(content_chunk) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=content_chunk,
        embedding=[0.1, 0.2, 0.3],
        model="test-model",
    )


# ---------------------------------------------------------------------------
# SearchProvider tests
# ---------------------------------------------------------------------------


class TestSearchProvider:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            SearchProvider()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            IncompleteSearchProvider()

    @pytest.mark.asyncio
    async def test_concrete_search_returns_results(self, sub_question_id):
        provider = ConcreteSearchProvider()
        results = await provider.search("AI tools", sub_question_id)
        assert len(results) == 1
        assert results[0].sub_question_id == sub_question_id
        assert results[0].url.startswith("https://")

    def test_provider_name_property(self):
        provider = ConcreteSearchProvider()
        assert isinstance(provider.provider_name, str)


# ---------------------------------------------------------------------------
# ContentExtractor tests
# ---------------------------------------------------------------------------


class TestContentExtractor:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            ContentExtractor()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            IncompleteContentExtractor()

    @pytest.mark.asyncio
    async def test_concrete_extract_returns_source_doc(self, sub_question_id):
        extractor = ConcreteContentExtractor()
        doc = await extractor.extract("https://example.com", sub_question_id, "query")
        assert doc is not None
        assert doc.sub_question_id == sub_question_id
        assert len(doc.text) > 0

    @pytest.mark.asyncio
    async def test_extract_can_return_none(self, sub_question_id):
        """extract() is allowed to return None for unreachable pages."""

        class NullExtractor(ContentExtractor):
            async def extract(self, url, sub_question_id, query=""):
                return None

        extractor = NullExtractor()
        result = await extractor.extract("https://dead-link.example.com", sub_question_id)
        assert result is None

    def test_extractor_name_property(self):
        extractor = ConcreteContentExtractor()
        assert isinstance(extractor.extractor_name, str)


# ---------------------------------------------------------------------------
# SourceEvaluator tests
# ---------------------------------------------------------------------------


class TestSourceEvaluator:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            SourceEvaluator()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            IncompleteSourceEvaluator()

    @pytest.mark.asyncio
    async def test_concrete_evaluate_returns_evaluation(self, source_doc):
        evaluator = ConcreteSourceEvaluator()
        ev = await evaluator.evaluate(source_doc, "AI tools")
        assert ev.source_id == source_doc.source_id
        assert ev.decision in {"include", "exclude"}
        assert 0.0 <= ev.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_batch_uses_sequential_default(self, source_doc):
        evaluator = ConcreteSourceEvaluator()
        results = await evaluator.evaluate_batch([source_doc, source_doc], "test query")
        assert len(results) == 2
        for ev in results:
            assert ev.source_id == source_doc.source_id

    def test_evaluator_name_property(self):
        evaluator = ConcreteSourceEvaluator()
        assert isinstance(evaluator.evaluator_name, str)


# ---------------------------------------------------------------------------
# EmbeddingProvider tests
# ---------------------------------------------------------------------------


class TestEmbeddingProvider:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            IncompleteEmbeddingProvider()

    @pytest.mark.asyncio
    async def test_concrete_embed_returns_vectors(self):
        provider = ConcreteEmbeddingProvider()
        vectors = await provider.embed(["hello", "world"])
        assert len(vectors) == 2
        assert all(len(v) == 3 for v in vectors)

    @pytest.mark.asyncio
    async def test_embed_chunks_default_impl(self, content_chunk):
        provider = ConcreteEmbeddingProvider()
        embedded = await provider.embed_chunks([content_chunk])
        assert len(embedded) == 1
        assert embedded[0].chunk_id == content_chunk.chunk_id
        assert embedded[0].model == provider.model_name

    @pytest.mark.asyncio
    async def test_embed_chunks_empty_raises(self):
        provider = ConcreteEmbeddingProvider()
        with pytest.raises(ValueError, match="at least one chunk"):
            await provider.embed_chunks([])

    def test_model_name_and_dim_properties(self):
        provider = ConcreteEmbeddingProvider()
        assert provider.model_name == "test-embedding-model"
        assert provider.embedding_dim == 3


# ---------------------------------------------------------------------------
# VectorStore tests
# ---------------------------------------------------------------------------


class TestVectorStore:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            VectorStore()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            IncompleteVectorStore()

    @pytest.mark.asyncio
    async def test_add_and_query(self, embedded_chunk):
        store = ConcreteVectorStore()
        await store.add([embedded_chunk])
        results = await store.query("test query", n=5)
        assert len(results) == 1
        assert results[0].chunk_id == embedded_chunk.chunk_id

    @pytest.mark.asyncio
    async def test_clear_empties_store(self, embedded_chunk):
        store = ConcreteVectorStore()
        await store.add([embedded_chunk])
        await store.clear()
        results = await store.query("test query")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_query_with_sub_question_filter(self, embedded_chunk, sub_question_id):
        store = ConcreteVectorStore()
        await store.add([embedded_chunk])
        # Filter matching sub_question_id - should return the chunk
        results = await store.query("test", sub_question_id=sub_question_id)
        assert len(results) == 1
        # Filter non-matching sub_question_id - should return nothing
        results = await store.query("test", sub_question_id="different-id")
        assert len(results) == 0

    def test_store_name_property(self):
        store = ConcreteVectorStore()
        assert isinstance(store.store_name, str)


# ---------------------------------------------------------------------------
# LLMProvider tests
# ---------------------------------------------------------------------------


class TestLLMProvider:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            LLMProvider()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            IncompleteLLMProvider()

    @pytest.mark.asyncio
    async def test_complete_returns_string(self):
        provider = ConcreteLLMProvider()
        result = await provider.complete("What is AI?")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_structured_complete_returns_dict(self):
        provider = ConcreteLLMProvider()
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}
        result = await provider.structured_complete("Summarise AI.", schema)
        assert isinstance(result, dict)
        assert "result" in result

    def test_model_and_provider_name_properties(self):
        provider = ConcreteLLMProvider()
        assert provider.model_name == "test-model"
        assert provider.provider_name == "test-provider"

    def test_supports_structured_output_default_false(self):
        provider = ConcreteLLMProvider()
        assert provider.supports_structured_output is False
