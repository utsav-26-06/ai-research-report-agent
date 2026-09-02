"""
Central exception hierarchy for the Research & Report Agent (TASK-021).
All domain exceptions inherit from AgentError for easy top-level handling.
"""


class AgentError(Exception):
    """Base class for all Research Agent errors."""


# --- Search layer ---
class SearchError(AgentError):
    """Raised when a web search provider call fails."""


# --- Extraction layer ---
class ExtractionError(AgentError):
    """Raised when web page content cannot be extracted."""


# --- Evaluation layer ---
class EvaluationError(AgentError):
    """Raised when source quality evaluation fails."""


# --- Embedding layer ---
class EmbeddingError(AgentError):
    """Raised when text embedding generation fails."""


# --- RAG layer ---
class RAGError(AgentError):
    """Raised when vector store operations or retrieval fails."""


# --- Report generation layer ---
class ReportError(AgentError):
    """Raised when report synthesis or assembly fails."""


# --- Export layer ---
class ExportError(AgentError):
    """Raised when DOCX or PDF export fails."""


# --- Config layer ---
class ConfigurationError(AgentError):
    """Raised when required configuration or API keys are missing or invalid."""


# --- Planning layer ---
class PlanningError(AgentError):
    """Raised when research plan generation fails."""
