"""Shared validation/io helpers used across phenospace pipelines."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file into a dict.

    Args:
        path: Config file location (yaml/yml/json).

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError: if the extension is not recognized.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    if p.suffix in (".yaml", ".yml"):
        with p.open() as fh:
            return yaml.safe_load(fh)
    if p.suffix == ".json":
        with p.open() as fh:
            return json.load(fh)
    raise ValueError(f"unsupported config extension: {p.suffix}")


def write_json(data: Any, path: str | Path) -> None:
    """Write JSON with compact, deterministic formatting."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


__all__ = ["load_config", "write_json"]
