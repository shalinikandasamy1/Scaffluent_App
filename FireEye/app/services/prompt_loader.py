"""Load LLM prompts from YAML config files.

Externalises prompts so they can be version-tracked, A/B tested,
and edited without code changes.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


@lru_cache(maxsize=16)
def load_prompt(name: str) -> dict[str, str]:
    """Load a prompt YAML file by name (without extension).

    Returns a dict with at least 'system' and 'user_template' keys.
    Falls back to empty strings if keys are missing.
    """
    # Sanitize name to prevent path traversal (defense in depth)
    safe_name = Path(name).name  # strips any directory components
    path = _PROMPTS_DIR / f"{safe_name}.yaml"
    if not path.resolve().parent == _PROMPTS_DIR.resolve():
        logger.warning("Prompt name %r resolves outside prompts dir", name)
        return {"system": "", "user_template": ""}
    if not path.exists():
        logger.warning("Prompt file not found: %s — using empty prompts", path)
        return {"system": "", "user_template": ""}

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        logger.warning("Prompt file %s has invalid format", path)
        return {"system": "", "user_template": ""}

    return {
        "system": data.get("system", "").strip(),
        "user_template": data.get("user_template", "").strip(),
    }


@lru_cache(maxsize=1)
def _load_model_adapters() -> dict[str, dict]:
    """Load model-specific prompt adapters."""
    path = _PROMPTS_DIR / "model_adapters.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _get_adapter(model_name: str) -> dict:
    """Get the best matching adapter for a model name."""
    adapters = _load_model_adapters()
    # Try exact match, then prefix match, then default
    if model_name in adapters:
        return adapters[model_name]
    for key in adapters:
        if key != "default" and model_name.startswith(key):
            return adapters[key]
    return adapters.get("default", {})


def get_system_prompt(name: str, model_name: str = "") -> str:
    """Get the system prompt for a given agent, with optional model adapter.

    If model_name is provided, appends the model-specific JSON instruction
    to the master system prompt for cross-model output consistency.
    """
    base = load_prompt(name)["system"]
    if model_name:
        adapter = _get_adapter(model_name)
        json_inst = adapter.get("json_instruction", "").strip()
        if json_inst:
            base = base + "\n\n" + json_inst
    return base


def get_user_template(name: str) -> str:
    """Get the user message template for a given agent."""
    return load_prompt(name)["user_template"]


def reload_prompts() -> None:
    """Clear the prompt cache, forcing reload from disk on next access."""
    load_prompt.cache_clear()
    _load_model_adapters.cache_clear()
