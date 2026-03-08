"""Application configuration loaded from environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Paths ---
    base_dir: Path = Path(__file__).resolve().parent.parent
    images_input_dir: Path = base_dir / "images" / "input"
    images_output_dir: Path = base_dir / "images" / "output"

    # --- OpenRouter / LLM ---
    openrouter_api_key: str = ""
    llm_model: str = "google/gemini-3-flash-preview"
    llm_temperature: float = 0.0

    # --- Local LLM (Ollama) ---
    local_llm_url: str = ""  # e.g. "http://localhost:11434" — set to enable local LLM
    local_llm_model: str = ""  # e.g. "qwen2.5vl:7b"
    llm_backend: str = "openrouter"  # "openrouter" or "local"

    # --- YOLO ---
    yolo_model_name: str = "yolo11n.pt"
    yolo_confidence_threshold: float = 0.20
    yolo_device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.

    # --- Risk classifier ---
    risk_confidence_threshold: float = 0.5

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    model_config = {
        "env_file": [
            str(Path(__file__).resolve().parent.parent / ".env"),       # FireEye/.env
            str(Path(__file__).resolve().parent.parent.parent / ".env"),  # repo root .env
        ],
        "env_prefix": "FIREEYE_",
    }


settings = Settings()
