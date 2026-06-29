"""
config.py
=========

Thin loader for the ``experiment_01_baseline.yaml`` file. The loader returns a
flat ``dict`` whose values can be passed directly into the training pipeline.
"""

from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "experiment_01_baseline.yaml"


def load_config(path: Path | str | None = None) -> dict:
    """Load the experiment configuration from a YAML file."""
    target = Path(path) if path is not None else CONFIG_PATH
    with open(target, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


if __name__ == "__main__":
    cfg = load_config()
    print(f"Loaded {len(cfg)} top-level keys from {CONFIG_PATH.name}")
    for key in sorted(cfg):
        value = cfg[key]
        if isinstance(value, str) and len(value) > 60:
            value = value[:57] + "..."
        print(f"  {key}: {value}")
