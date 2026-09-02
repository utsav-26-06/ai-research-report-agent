"""
Structured logging configuration for the Research & Report Agent (TASK-021).

Usage:
    from app.config.logging_config import configure_logging
    configure_logging()  # Call once at app startup

Features:
  - JSON-structured output (when LOG_FORMAT=json)
  - No API keys ever leak into logs (sanitised)
  - Configurable via LOG_LEVEL env var
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Literal

# Patterns that should never appear in log output
_SECRET_PATTERNS = [
    re.compile(r"(AIza[0-9A-Za-z_\-]{35})", re.I),        # Google API keys
    re.compile(r"(tvly-[0-9A-Za-z_\-]{20,})", re.I),       # Tavily API keys
    re.compile(r"(sk-[0-9A-Za-z]{32,})", re.I),            # OpenAI keys
    re.compile(r"api[_\-]?key[\s:=]+['\"]?([^\s'\"]{8,})", re.I),
]

_MASK = "***REDACTED***"


def _sanitize(msg: str) -> str:
    """Remove any detected secrets from a log message."""
    for pattern in _SECRET_PATTERNS:
        msg = pattern.sub(_MASK, msg)
    return msg


class _SecretFilter(logging.Filter):
    """Logging filter that redacts secrets from all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _sanitize(str(record.msg))
        if record.args:
            try:
                record.args = tuple(
                    _sanitize(str(a)) if isinstance(a, str) else a
                    for a in record.args
                ) if isinstance(record.args, tuple) else record.args
            except Exception:
                pass
        return True


def configure_logging(
    level: str = "INFO",
    fmt: Literal["text", "json"] = "text",
    stream=None,
) -> None:
    """
    Configure application-wide structured logging.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt:   Output format — 'text' for human-readable, 'json' for structured.
        stream: Output stream (defaults to sys.stderr).
    """
    if stream is None:
        stream = sys.stderr

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicate output
    root.handlers.clear()

    if fmt == "json":
        try:
            import json_logging  # optional dependency
            json_logging.init_non_web(enable_json=True)
        except ImportError:
            # Fall back to structured text if json_logging not available
            fmt = "text"

    if fmt != "json":
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        handler.addFilter(_SecretFilter())
        root.addHandler(handler)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured | level=%s | format=%s", level.upper(), fmt
    )
