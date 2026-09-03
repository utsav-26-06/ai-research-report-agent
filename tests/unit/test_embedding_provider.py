"""
Unit tests for the GeminiEmbeddingProvider (TASK-010).
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from google.genai.errors import APIError

from app.rag.embedding_provider import GeminiEmbeddingProvider, EmbeddingProviderError
from app.models import ContentChunk


@pytest.fixture
def mock_genai_client():
    with patch("app.rag.embedding_provider.genai.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def provider(mock_genai_client):
    return GeminiEmbeddingProvider(api_key="test-key")


def test_provider_initialization(provider):
    assert provider.model_name == "models/gemini-embedding-001"
    assert provider.embedding_dim == 3072


def test_missing_api_key_raises():
    with pytest.raises(ValueError, match="API key cannot be empty"):
        GeminiEmbeddingProvider(api_key="")


@pytest.mark.asyncio
async def test_embed_success(provider, mock_genai_client):
    # Setup mock response
    mock_response = MagicMock()
    
    mock_emb1 = MagicMock()
    mock_emb1.values = [0.1, 0.2, 0.3]
    
    mock_emb2 = MagicMock()
    mock_emb2.values = [0.4, 0.5, 0.6]
    
    mock_response.embeddings = [mock_emb1, mock_emb2]
    
    mock_genai_client.models.embed_content.return_value = mock_response

    vectors = await provider.embed(["text1", "text2"])

    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    assert vectors[1] == [0.4, 0.5, 0.6]

    mock_genai_client.models.embed_content.assert_called_once_with(
        model="models/gemini-embedding-001",
        contents=["text1", "text2"]
    )


@pytest.mark.asyncio
async def test_embed_empty_list(provider, mock_genai_client):
    vectors = await provider.embed([])
    assert len(vectors) == 0
    mock_genai_client.models.embed_content.assert_not_called()


@pytest.mark.asyncio
async def test_embed_api_error_retry(provider, mock_genai_client):
    # Fail once, then succeed
    mock_response = MagicMock()
    mock_emb = MagicMock()
    mock_emb.values = [0.1]
    mock_response.embeddings = [mock_emb]
    
    # Needs to raise APIError on first call
    # APIError takes (code: int, response_json: Any, ...)
    mock_genai_client.models.embed_content.side_effect = [
        APIError(429, {"message": "Rate Limit"}),
        mock_response
    ]

    vectors = await provider.embed(["test"])
    assert len(vectors) == 1
    assert mock_genai_client.models.embed_content.call_count == 2


@pytest.mark.asyncio
async def test_embed_chunks_batching(provider, mock_genai_client):
    # Create 120 chunks to force batching (batch_size=50)
    chunks = []
    for i in range(120):
        chunks.append(
            ContentChunk(
                source_id="s1",
                url="https://example.com",
                sub_question_id="sq1",
                chunk_index=i,
                total_chunks=120,
                text=f"chunk text {i}"
            )
        )
        
    # Setup mock response to return vectors for however many texts are requested
    def mock_embed_content(model, contents):
        response = MagicMock()
        emb_list = []
        for _ in contents:
            emb = MagicMock()
            emb.values = [0.5] * 3072
            emb_list.append(emb)
        response.embeddings = emb_list
        return response
        
    mock_genai_client.models.embed_content.side_effect = mock_embed_content

    embedded_chunks = await provider.embed_chunks(chunks)

    assert len(embedded_chunks) == 120
    # 120 chunks / 50 per batch = 3 batches (50, 50, 20)
    assert mock_genai_client.models.embed_content.call_count == 3
    
    # Check the embedded chunk contents
    assert embedded_chunks[0].chunk_id == chunks[0].chunk_id
    assert embedded_chunks[0].model == "models/gemini-embedding-001"
    assert len(embedded_chunks[0].embedding) == 3072


@pytest.mark.asyncio
async def test_embed_api_error_exhausts_retries(provider, mock_genai_client):
    # Consistently fail with 429
    mock_genai_client.models.embed_content.side_effect = APIError(429, {"message": "Rate Limit Exhausted"})

    with patch("asyncio.sleep", return_value=None):
        with pytest.raises((APIError, EmbeddingProviderError)):
            await provider.embed(["test"])
    
    assert mock_genai_client.models.embed_content.call_count == 5



