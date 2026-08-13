"""
app.config — Configuration package.

Usage:
    from app.config import get_settings

    settings = get_settings()
    print(settings.openai_model)
"""

from app.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
