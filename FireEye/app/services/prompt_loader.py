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
    path = _PROMPTS_DIR / f"{name}.yaml"
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


def get_system_prompt(name: str) -> str:
    """Get the system prompt for a given agent."""
    return load_prompt(name)["system"]


def get_user_template(name: str) -> str:
    """Get the user message template for a given agent."""
    return load_prompt(name)["user_template"]


def reload_prompts() -> None:
    """Clear the prompt cache, forcing reload from disk on next access."""
    load_prompt.cache_clear()
