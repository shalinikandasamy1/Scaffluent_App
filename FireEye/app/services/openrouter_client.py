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


def _schema_to_description(schema: dict) -> str:
    """Convert a JSON schema to a human-readable field description.

    Used to inject schema info into system prompts for local models
    that don't support strict JSON schema enforcement at decode level.
    """
    s = schema.get("schema", schema)
    lines = []
    props = s.get("properties", {})
    required = set(s.get("required", []))
    for name, prop in props.items():
        ptype = prop.get("type", "any")
        req = " (required)" if name in required else ""
        if "enum" in prop:
            lines.append(f'  "{name}": one of {prop["enum"]}{req}')
        elif ptype == "array":
            items = prop.get("items", {})
            if items.get("type") == "object":
                sub_props = items.get("properties", {})
                sub_fields = ", ".join(
                    f'"{k}": {v.get("type", "any")}'
                    + (f' (one of {v["enum"]})' if "enum" in v else "")
                    for k, v in sub_props.items()
                )
                lines.append(f'  "{name}": array of objects with {{{sub_fields}}}{req}')
            else:
                lines.append(f'  "{name}": array of {items.get("type", "any")}{req}')
        else:
            lines.append(f'  "{name}": {ptype}{req}')
    return "Expected JSON structure:\n{\n" + "\n".join(lines) + "\n}"


def _validate_against_schema(data: dict, schema: dict) -> list[str]:
    """Validate parsed JSON against schema, returning a list of error strings.

    Lightweight validation — checks required fields, types, and enum values.
    """
    errors = []
    s = schema.get("schema", schema)
    required = set(s.get("required", []))
    props = s.get("properties", {})

    for field in required:
        if field not in data:
            errors.append(f"Missing required field: '{field}'")

    for field, value in data.items():
        if field not in props:
            continue
        prop = props[field]
        expected_type = prop.get("type")

        if expected_type == "string" and not isinstance(value, str):
            errors.append(f"Field '{field}' should be string, got {type(value).__name__}")
        elif expected_type == "number" and not isinstance(value, (int, float)):
            errors.append(f"Field '{field}' should be number, got {type(value).__name__}")
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"Field '{field}' should be array, got {type(value).__name__}")

        if "enum" in prop and value not in prop["enum"]:
            errors.append(f"Field '{field}' value '{value}' not in {prop['enum']}")

        # Validate confidence range (semantic check)
        if field == "confidence" and isinstance(value, (int, float)):
            if not 0 <= value <= 1:
                errors.append(f"Field 'confidence' should be 0-1, got {value}")

    return errors


def chat_completion_json(
    messages: list[dict[str, Any]],
    *,
    json_schema: dict,
    model: str | None = None,
    temperature: float | None = None,
    max_retries: int = 2,
) -> dict:
    """Send a chat completion with enforced JSON schema and parse the result.

    For local models, injects schema description into the system prompt
    and validates the response with retry + error feedback for self-correction.
    """
    is_local = settings.llm_backend == "local"

    # For local models, inject schema description into system prompt
    working_messages = list(messages)
    if is_local and working_messages and working_messages[0].get("role") == "system":
        schema_desc = _schema_to_description(json_schema)
        working_messages = [dict(m) for m in working_messages]
        working_messages[0] = {
            **working_messages[0],
            "content": working_messages[0]["content"] + "\n\n" + schema_desc,
        }

    response_format: dict[str, Any]
    if is_local:
        # Ollama uses simpler format: json mode without strict schema
        response_format = {"type": "json_object"}
    else:
        response_format = {"type": "json_schema", "json_schema": json_schema}

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        raw = chat_completion(
            working_messages,
            model=model,
            temperature=temperature,
            response_format=response_format,
            max_retries=1,
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            last_err = e
            logger.warning("JSON parse failed (attempt %d/%d): %s", attempt, max_retries, e)
            if attempt < max_retries:
                # Inject error feedback for retry
                working_messages = list(working_messages) + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"Your response was not valid JSON: {e}. Please respond with ONLY a valid JSON object."},
                ]
            continue

        # Validate against schema
        validation_errors = _validate_against_schema(data, json_schema)
        if not validation_errors:
            return data

        logger.warning("Schema validation errors (attempt %d/%d): %s", attempt, max_retries, validation_errors)
        last_err = ValueError(f"Schema validation failed: {validation_errors}")
        if attempt < max_retries:
            working_messages = list(working_messages) + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Your JSON has errors: {'; '.join(validation_errors)}. Please fix and respond with corrected JSON only."},
            ]

    # If validation failed but we have data, return it with a warning
    if isinstance(last_err, ValueError) and 'data' in locals():
        logger.warning("Returning LLM response despite validation errors: %s", last_err)
        return data
    raise last_err
