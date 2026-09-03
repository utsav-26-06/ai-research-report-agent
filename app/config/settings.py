"""
Configuration Module — Settings & Environment Loading

All application-wide configuration is defined here.
Loads from .env file automatically via pydantic-settings.

Stack:
  LLM + Embeddings : Google Gemini (free via Google AI Studio)
  Search           : Tavily (free tier — 1000 searches/month)
  Vector DB        : ChromaDB (local, fully free)

Never import API keys directly — always use get_settings().
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central settings object for the Research & Report Agent.

    All values can be overridden via environment variables or a .env file.
    Required fields raise a ValidationError if missing or invalid.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Google Gemini (LLM + Embeddings) ---
    gemini_api_key: str = Field(
        ...,
        description="Google Gemini API key — get free at aistudio.google.com",
    )
    gemini_model: str = Field(
        default="gemini-3.5-flash",
        description="Gemini chat model to use for LLM calls",
    )
    gemini_embedding_model: str = Field(
        default="models/gemini-embedding-001",
        description="Gemini model to use for text embeddings",
    )
    llm_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=32768,
        description="Maximum output tokens for LLM responses",
    )
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (lower = more deterministic)",
    )

    # --- Tavily Search ---
    tavily_api_key: str = Field(
        ...,
        description="Tavily API key — free tier at tavily.com (1000 searches/month)",
    )
    tavily_max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum search results returned per query",
    )
    tavily_search_depth: str = Field(
        default="basic",
        description="Tavily search depth: 'basic' or 'advanced'",
    )

    # --- RAG / ChromaDB ---
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"),
        description="Directory for ChromaDB persistent storage (local, free)",
    )
    chroma_collection_name: str = Field(
        default="research_agent",
        description="ChromaDB collection name for this research session",
    )

    # --- Chunking ---
    chunk_size: int = Field(
        default=800,
        ge=100,
        le=4000,
        description="Character size of each text chunk for embedding",
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Character overlap between consecutive chunks",
    )

    # --- Source Evaluation ---
    min_relevance_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score (0–1) to accept a source",
    )
    min_credibility_score: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum credibility score (0–1) to accept a source",
    )
    max_sources_per_query: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum sources kept per sub-question after filtering",
    )

    # --- Output ---
    output_dir: Path = Field(
        default=Path("./outputs"),
        description="Directory where generated PDF and DOCX reports are saved",
    )

    # --- Logging ---
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #

    @field_validator("gemini_api_key")
    @classmethod
    def validate_gemini_key(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("GEMINI_API_KEY must not be empty")
        if "your-key-here" in v:
            raise ValueError(
                "GEMINI_API_KEY is still set to the example placeholder. "
                "Get a free key at https://aistudio.google.com and add it to your .env"
            )
        return v

    @field_validator("tavily_api_key")
    @classmethod
    def validate_tavily_key(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("TAVILY_API_KEY must not be empty")
        if "your-key-here" in v:
            raise ValueError(
                "TAVILY_API_KEY is still set to the example placeholder. "
                "Get a free key at https://tavily.com and add it to your .env"
            )
        return v

    @field_validator("tavily_search_depth")
    @classmethod
    def validate_search_depth(cls, v: str) -> str:
        allowed = {"basic", "advanced"}
        if v not in allowed:
            raise ValueError(f"tavily_search_depth must be one of {allowed}, got '{v}'")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_chunk_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def ensure_output_dir(self) -> Path:
        """Create output directory if it doesn't exist. Returns the path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def ensure_chroma_dir(self) -> Path:
        """Create ChromaDB persist directory if it doesn't exist. Returns the path."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        return self.chroma_persist_dir

    def __repr__(self) -> str:
        """Safe repr — never exposes API keys."""
        return (
            f"Settings("
            f"gemini_model={self.gemini_model!r}, "
            f"gemini_embedding_model={self.gemini_embedding_model!r}, "
            f"tavily_search_depth={self.tavily_search_depth!r}, "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
            f")"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings singleton.

    Settings are loaded once from .env and cached for the process lifetime.
    In tests, call get_settings.cache_clear() to reset between test cases.

    Raises:
        pydantic_core.ValidationError: If required env vars are missing
                                       or values fail validation.
    """
    return Settings()