from __future__ import annotations
import asyncio, json, logging
from typing import Any
import google.genai as genai
from google.genai import types
from app.generation.base import LLMProvider, LLMProviderError

logger = logging.getLogger(__name__)

class GeminiLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash",
                 max_tokens: int = 8192, temperature: float = 0.2):
        self._api_key = api_key
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = genai.Client(api_key=api_key)

    @property
    def model_name(self) -> str: return self._model_name
    @property
    def provider_name(self) -> str: return "gemini"
    @property
    def supports_structured_output(self) -> bool: return True

    async def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        temp = temperature if temperature is not None else self._temperature
        try:
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model_name, contents=prompt,
                config=types.GenerateContentConfig(temperature=temp, max_output_tokens=self._max_tokens),
            )
            text = response.text
            if not text: raise LLMProviderError("Gemini returned an empty response.")
            return text.strip()
        except LLMProviderError: raise
        except Exception as e: raise LLMProviderError(f"Gemini complete() failed: {e}") from e

    async def structured_complete(self, prompt: str, schema: dict[str, Any], *,
                                   temperature: float | None = None) -> dict[str, Any]:
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
            response = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._model_name, contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=temp, max_output_tokens=self._max_tokens,
                    response_mime_type="application/json",
                    system_instruction=system_instruction,
                ),
            )
            raw = response.text or ""
            raw = raw.strip()
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"): raw = raw[4:]
            raw = raw.strip()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise LLMProviderError(f"Expected JSON object, got {type(parsed).__name__}.")
            return parsed
        except (json.JSONDecodeError, LLMProviderError): raise
        except Exception as e: raise LLMProviderError(f"Gemini structured_complete() failed: {e}") from e
