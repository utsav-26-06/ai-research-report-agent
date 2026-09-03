"""
Unit tests for app.config.settings â€” TASK-002 (Gemini stack)

Tests run without real API keys by injecting environment variable overrides.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config.settings import get_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_ENV = {
    "GEMINI_API_KEY": "AIzaTestKey1234567890",
    "TAVILY_API_KEY": "tvly-test-key-1234567890",
    "GEMINI_MODEL": "gemini-3.5-flash",
    "GEMINI_EMBEDDING_MODEL": "models/gemini-embedding-001",
    "LLM_MAX_TOKENS": "4096",
    "LLM_TEMPERATURE": "0.2",
    "TAVILY_MAX_RESULTS": "5",
    "TAVILY_SEARCH_DEPTH": "basic",
    "CHROMA_PERSIST_DIR": "./data/chroma",
    "CHROMA_COLLECTION_NAME": "research_agent",
    "CHUNK_SIZE": "800",
    "CHUNK_OVERLAP": "100",
    "MIN_RELEVANCE_SCORE": "0.5",
    "MIN_CREDIBILITY_SCORE": "0.4",
    "MAX_SOURCES_PER_QUERY": "5",
    "OUTPUT_DIR": "./outputs",
    "LOG_LEVEL": "INFO",
}


def make_settings(**overrides):
    """Create a fresh Settings instance with overrides on top of VALID_ENV."""
    get_settings.cache_clear()
    env = {**VALID_ENV, **overrides}
    with patch.dict(os.environ, env, clear=True):
        from app.config.settings import Settings
        return Settings()


def make_settings_no_file(**env_vars):
    """
    Create Settings using ONLY the provided env_vars â€” no .env file.

    Used for testing missing-required-key behaviour when a real .env exists
    on disk (pydantic-settings would otherwise read it and supply the key).
    """
    from pydantic_settings import SettingsConfigDict
    from app.config.settings import Settings

    class _NoFileSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=None,          # do NOT read .env
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

    get_settings.cache_clear()
    with patch.dict(os.environ, env_vars, clear=True):
        return _NoFileSettings()


# ---------------------------------------------------------------------------
# T01: All fields load correctly
# ---------------------------------------------------------------------------

class TestSettingsLoading:

    def test_loads_gemini_api_key(self):
        s = make_settings()
        assert s.gemini_api_key == "AIzaTestKey1234567890"

    def test_loads_tavily_api_key(self):
        s = make_settings()
        assert s.tavily_api_key == "tvly-test-key-1234567890"

    def test_loads_gemini_model(self):
        s = make_settings()
        assert s.gemini_model == "gemini-3.5-flash"

    def test_loads_gemini_embedding_model(self):
        s = make_settings()
        assert s.gemini_embedding_model == "models/gemini-embedding-001"

    def test_loads_llm_max_tokens(self):
        s = make_settings()
        assert s.llm_max_tokens == 4096

    def test_loads_llm_temperature(self):
        s = make_settings()
        assert s.llm_temperature == 0.2

    def test_loads_tavily_max_results(self):
        s = make_settings()
        assert s.tavily_max_results == 5

    def test_loads_tavily_search_depth(self):
        s = make_settings()
        assert s.tavily_search_depth == "basic"

    def test_loads_chroma_persist_dir_as_path(self):
        s = make_settings()
        assert isinstance(s.chroma_persist_dir, Path)

    def test_loads_chunk_size(self):
        s = make_settings()
        assert s.chunk_size == 800

    def test_loads_chunk_overlap(self):
        s = make_settings()
        assert s.chunk_overlap == 100

    def test_loads_min_relevance_score(self):
        s = make_settings()
        assert s.min_relevance_score == 0.5

    def test_loads_log_level(self):
        s = make_settings()
        assert s.log_level == "INFO"

    def test_log_level_uppercased(self):
        s = make_settings(LOG_LEVEL="debug")
        assert s.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# T02: Missing required fields raise ValidationError
# ---------------------------------------------------------------------------

class TestMissingRequiredFields:

    def test_missing_gemini_api_key_raises(self):
        # Remove GEMINI_API_KEY from env AND bypass .env file so the key
        # is truly absent (pydantic-settings reads .env even with clear=True).
        env = {k: v for k, v in VALID_ENV.items() if k != "GEMINI_API_KEY"}
        with pytest.raises(ValidationError) as exc_info:
            make_settings_no_file(**env)
        assert "gemini_api_key" in str(exc_info.value).lower()

    def test_missing_tavily_api_key_raises(self):
        env = {k: v for k, v in VALID_ENV.items() if k != "TAVILY_API_KEY"}
        with pytest.raises(ValidationError) as exc_info:
            make_settings_no_file(**env)
        assert "tavily_api_key" in str(exc_info.value).lower()

    def test_placeholder_gemini_key_raises(self):
        with pytest.raises(ValidationError, match="placeholder"):
            make_settings(GEMINI_API_KEY="AIza...your-key-here...")

    def test_placeholder_tavily_key_raises(self):
        with pytest.raises(ValidationError, match="placeholder"):
            make_settings(TAVILY_API_KEY="tvly-...your-key-here...")

    def test_empty_gemini_key_raises(self):
        with pytest.raises(ValidationError):
            make_settings(GEMINI_API_KEY="")


# ---------------------------------------------------------------------------
# T03: Field-level validation
# ---------------------------------------------------------------------------

class TestFieldValidation:

    def test_invalid_search_depth_raises(self):
        with pytest.raises(ValidationError, match="search_depth"):
            make_settings(TAVILY_SEARCH_DEPTH="ultra")

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValidationError, match="log_level"):
            make_settings(LOG_LEVEL="VERBOSE")

    def test_chunk_overlap_gte_chunk_size_raises(self):
        with pytest.raises(ValidationError, match="chunk_overlap"):
            make_settings(CHUNK_SIZE="100", CHUNK_OVERLAP="100")

    def test_zero_max_tokens_raises(self):
        with pytest.raises(ValidationError):
            make_settings(LLM_MAX_TOKENS="0")

    def test_temperature_above_2_raises(self):
        with pytest.raises(ValidationError):
            make_settings(LLM_TEMPERATURE="3.0")

    def test_relevance_score_above_1_raises(self):
        with pytest.raises(ValidationError):
            make_settings(MIN_RELEVANCE_SCORE="1.5")

    def test_valid_advanced_search_depth(self):
        s = make_settings(TAVILY_SEARCH_DEPTH="advanced")
        assert s.tavily_search_depth == "advanced"


# ---------------------------------------------------------------------------
# T04: Defaults are correct
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_default_gemini_model(self):
        env = {k: v for k, v in VALID_ENV.items() if k != "GEMINI_MODEL"}
        s = make_settings_no_file(**env)
        assert s.gemini_model == "gemini-3.5-flash"

    def test_default_chunk_size(self):
        env = {k: v for k, v in VALID_ENV.items() if k != "CHUNK_SIZE"}
        s = make_settings_no_file(**env)
        assert s.chunk_size == 800

    def test_default_log_level(self):
        env = {k: v for k, v in VALID_ENV.items() if k != "LOG_LEVEL"}
        s = make_settings_no_file(**env)
        assert s.log_level == "INFO"

    def test_default_embedding_model(self):
        env = {k: v for k, v in VALID_ENV.items() if k != "GEMINI_EMBEDDING_MODEL"}
        s = make_settings_no_file(**env)
        assert s.gemini_embedding_model == "models/gemini-embedding-001"


# ---------------------------------------------------------------------------
# T05: Singleton behavior
# ---------------------------------------------------------------------------

class TestSingleton:

    def test_get_settings_returns_same_instance(self):
        get_settings.cache_clear()
        with patch.dict(os.environ, VALID_ENV, clear=True):
            s1 = get_settings()
            s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_allows_new_instance(self):
        get_settings.cache_clear()
        with patch.dict(os.environ, VALID_ENV, clear=True):
            s1 = get_settings()
        get_settings.cache_clear()
        with patch.dict(os.environ, {**VALID_ENV, "GEMINI_MODEL": "gemini-1.5-pro"}, clear=True):
            s2 = get_settings()
        assert s1 is not s2
        assert s2.gemini_model == "gemini-1.5-pro"


# ---------------------------------------------------------------------------
# T06: Safe repr â€” no secret leakage
# ---------------------------------------------------------------------------

class TestSafeRepr:

    def test_repr_does_not_expose_gemini_key(self):
        s = make_settings()
        r = repr(s)
        assert "AIzaTestKey1234567890" not in r

    def test_repr_does_not_expose_tavily_key(self):
        s = make_settings()
        r = repr(s)
        assert "tvly-test-key-1234567890" not in r

    def test_repr_contains_model_name(self):
        s = make_settings()
        r = repr(s)
        assert "gemini-3.5-flash" in r


# ---------------------------------------------------------------------------
# T07: Helper methods
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_ensure_output_dir_creates_directory(self, tmp_path):
        s = make_settings(OUTPUT_DIR=str(tmp_path / "test_outputs"))
        result = s.ensure_output_dir()
        assert result.exists()
        assert result.is_dir()

    def test_ensure_chroma_dir_creates_directory(self, tmp_path):
        s = make_settings(CHROMA_PERSIST_DIR=str(tmp_path / "test_chroma"))
        result = s.ensure_chroma_dir()
        assert result.exists()
        assert result.is_dir()
