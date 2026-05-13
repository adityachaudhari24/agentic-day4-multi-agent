"""Project root and ``.env`` loading for this package (keep ``.env`` next to ``.env.example``)."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def project_root() -> Path:
    """``agentic-day4-multi-agent/`` (contains ``contracts/``, ``prompts/``, ``.env``)."""
    return Path(__file__).resolve().parent


def load_app_dotenv() -> None:
    """Load ``<project_root>/.env`` into the process environment."""
    load_dotenv(project_root() / ".env")
