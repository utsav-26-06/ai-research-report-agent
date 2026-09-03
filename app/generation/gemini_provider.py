from __future__ import annotations
import asyncio
import json
import logging
import re
from typing import Any
import google.genai as genai
from google.genai import types
from app.generation.base import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)


class GeminiLLMProvider(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.5-flash-lite",
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ):
        self._api_key = api_key
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = genai.Client(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def supports_structured_output(self) -> bool:
        return True

    @staticmethod
    def _extract_retry_delay(error: Exception, attempt: int) -> float:
        """Parse the retry delay suggested by the Gemini API or use exponential backoff."""
        err_str = str(error)

        # 1. Match 'retry in X.Xs'
        match = re.search(r"retry in\s+([0-9.]+)\s*s", err_str, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1)) + 1.0
            except ValueError:
                pass

        # 2. Check response_json structure for retryDelay
        response_json = getattr(error, "response_json", None)
        if isinstance(response_json, dict):
            details = response_json.get("error", {}).get("details", [])
            for detail in details:
                if isinstance(detail, dict) and "retryDelay" in detail:
                    raw_delay = str(detail["retryDelay"]).rstrip("s")
                    try:
                        return float(raw_delay) + 1.0
                    except ValueError:
                        pass

        # 3. Fallback exponential backoff (min 5s, max 60s)
        return min(60.0, 5.0 * (2 ** (attempt - 1)))

    async def _generate_with_retry(
        self,
        contents: Any,
        config: types.GenerateContentConfig,
        max_retries: int = 5,
    ) -> Any:
        """Execute generate_content with automatic backoff retry on 429 / RESOURCE_EXHAUSTED."""
        for attempt in range(1, max_retries + 1):
            try:
                return await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or getattr(e, "code", None) == 429
                )
                if is_rate_limit and attempt < max_retries:
                    delay = self._extract_retry_delay(e, attempt)
                    logger.warning(
                        f"Gemini API rate limit (429) hit. Pausing for {delay:.1f}s before retry "
                        f"(attempt {attempt}/{max_retries})..."
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Safely extract text from response, handling thought tokens or candidates accessor."""
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            parts_text = [
                getattr(p, "text", "")
                for p in parts
                if getattr(p, "text", None) and not getattr(p, "thought", False)
            ]
            if parts_text:
                return "".join(parts_text).strip()
        return ""

    async def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else self._temperature
        try:
            response = await self._generate_with_retry(
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temp, max_output_tokens=self._max_tokens
                ),
            )
            text = self._extract_text(response)
            if not text:
                raise LLMProviderError("Gemini returned an empty response.")
            return text
        except LLMProviderError:
            raise
        except Exception as e:
            raise LLMProviderError(f"Gemini complete() failed: {e}") from e

    async def structured_complete(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        temp = temperature if temperature is not None else self._temperature
        system_instruction = (
            "You are a precise research assistant. "
            "Respond ONLY with valid JSON conforming to the provided schema. "
            "Do not include markdown, explanation, or text outside the JSON object."
        )
        full_prompt = (
            f"{prompt}\n\nRespond with a JSON object matching this schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )
        try:
            response = await self._generate_with_retry(
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=self._max_tokens,
                    response_mime_type="application/json",
                    system_instruction=system_instruction,
                ),
            )
            raw = self._extract_text(response)
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise LLMProviderError(f"Expected JSON object, got {type(parsed).__name__}.")
            return parsed
        except (json.JSONDecodeError, LLMProviderError):
            raise
        except Exception as e:
            raise LLMProviderError(f"Gemini structured_complete() failed: {e}") from e

