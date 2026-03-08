"""LLM API client wrapping the OpenAI SDK (drop-in compatible).

Supports both OpenRouter (cloud) and local Ollama backends.
Set FIREEYE_LLM_BACKEND=local and FIREEYE_LOCAL_LLM_URL to use Ollama.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_local_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazy-initialise and return the shared OpenAI client for OpenRouter."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            timeout=120.0,
        )
    return _client


def get_local_client() -> OpenAI:
    """Lazy-initialise and return the shared OpenAI client for local Ollama."""
    global _local_client
    if _local_client is None:
        base_url = settings.local_llm_url.rstrip("/") + "/v1"
        _local_client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama doesn't need a real key
            timeout=300.0,  # local models can be slower
        )
    return _local_client


def _active_client() -> tuple[OpenAI, str]:
    """Return (client, model) for the currently configured backend."""
    if settings.llm_backend == "local" and settings.local_llm_url:
        return get_local_client(), settings.local_llm_model or "qwen2.5vl:7b"
    return get_client(), settings.llm_model


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
    max_retries: int = 3,
) -> str:
    """Send a chat completion request and return the assistant message content.

    Retries on transient errors (rate limits, server errors) with
    exponential backoff.
    """
    client, default_model = _active_client()
    kwargs: dict[str, Any] = {
        "model": model or default_model,
        "messages": messages,
        "temperature": temperature if temperature is not None else settings.llm_temperature,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
        # Only add OpenRouter-specific params when using OpenRouter
        if settings.llm_backend != "local":
            kwargs["extra_body"] = {"provider": {"require_parameters": True}}

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            logger.debug("LLM response: %s", content[:200])
            return content
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Retry on rate limits and server errors
            if any(kw in err_str for kw in ("rate", "429", "500", "502", "503", "timeout")):
                wait = 2 ** attempt
                logger.warning("API error (attempt %d/%d), retrying in %ds: %s",
                               attempt, max_retries, wait, e)
                time.sleep(wait)
            else:
                raise  # Non-transient error, don't retry
    raise last_err


def chat_completion_json(
    messages: list[dict[str, Any]],
    *,
    json_schema: dict,
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int = 2,
) -> dict:
    """Send a chat completion with enforced JSON schema and parse the result."""
    response_format = {
        "type": "json_schema",
        "json_schema": json_schema,
    }
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        raw = chat_completion(
            messages,
            model=model,
            temperature=temperature,
            response_format=response_format,
            max_retries=1,  # API retries handled in chat_completion
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            last_err = e
            logger.warning("JSON parse failed (attempt %d/%d): %s", attempt, max_retries, e)
    raise last_err
