"""OpenRouter API client wrapping the OpenAI SDK (drop-in compatible)."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazy-initialise and return the shared OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
    return _client


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
) -> str:
    """Send a chat completion request and return the assistant message content."""
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": model or settings.llm_model,
        "messages": messages,
        "temperature": temperature if temperature is not None else settings.llm_temperature,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
        kwargs["extra_body"] = {"provider": {"require_parameters": True}}

    completion = client.chat.completions.create(**kwargs)
    content = completion.choices[0].message.content
    logger.debug("LLM response: %s", content[:200])
    return content


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
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            last_err = e
            logger.warning("JSON parse failed (attempt %d/%d): %s", attempt, max_retries, e)
    raise last_err
