"""Loads news_quant/config.yaml. NEVER reads root config.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load() -> dict[str, Any]:
    with _CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    _resolve_env_keys(cfg)
    return cfg


def _resolve_env_keys(cfg: dict[str, Any]) -> None:
    for section in ("llm", "broker"):
        env_var = cfg.get(section, {}).get("api_key_env")
        if env_var:
            cfg[section]["api_key"] = os.environ.get(env_var, "")


if __name__ == "__main__":
    import json
    c = load()
    c.get("llm", {}).pop("api_key", None)
    c.get("broker", {}).pop("api_key", None)
    print(json.dumps(c, indent=2, default=str))
