"""Load YAML files that expose a top-level ``system`` string for LLM prompts."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_system_prompt(yaml_path: Path) -> str:
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    data = yaml.safe_load(yaml_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {yaml_path}")
    system = data.get("system")
    if not isinstance(system, str) or not system.strip():
        raise ValueError(f"Missing non-empty 'system' string in {yaml_path}")
    return system.strip()
