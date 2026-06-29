"""
research_pipeline.py
====================

Data preparation utilities shared by all four training scripts:

    * ``build_feature_artifacts`` - re-runs ``prepare_features`` for a list
      of prefix ratios.
    * ``load_feature_channels`` - reads the 108-column feature table and
      returns the chosen channel subset.
    * ``compute_baseline_stats`` - aggregate the BASELINE-window statistics
      that downstream models expect.

The script is intended to be a thin wrapper around the more focused
``prepare_labels.py`` and ``prepare_features.py`` modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prepare_features import build_prefix_feature_table


FEATURE_CHANNELS: tuple[str, ...] = (
    "heartRate",
    "PPInterval",
    "BVPProcessed",
    "acc_mag",
    "gyr_mag",
    "rot_mag",
)


def _json_safe(value: object) -> object:
    """Recursively convert numpy scalars to native Python types for JSON output."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value


def build_feature_artifacts(
    resampled_dir: Path | str,
    label_manifest_path: Path | str,
    artifacts_dir: Path | str,
    prefix_ratios: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> pd.DataFrame:
    """Run the prefix feature builder and write a single wide CSV."""
    label_manifest = pd.read_csv(label_manifest_path)
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    output = artifacts_path / "prefix_feature_table.csv"
    return build_prefix_feature_table(
        resampled_dir=resampled_dir,
        label_manifest=label_manifest,
        output_path=output,
    )


def load_feature_channels(
    feature_table: pd.DataFrame,
    channels: tuple[str, ...] = FEATURE_CHANNELS,
    view: str = "raw",
) -> pd.DataFrame:
    """Return the subset of columns corresponding to ``channels`` x ``view``."""
    keep_columns = ["session_id", "prefix_ratio"]
    keep_columns.extend(
        column
        for column in feature_table.columns
        if any(column.startswith(f"{channel}__{view}__") for channel in channels)
    )
    return feature_table[keep_columns]


def compute_baseline_stats(
    feature_table: pd.DataFrame,
    channels: tuple[str, ...] = FEATURE_CHANNELS,
) -> dict[str, dict[str, float]]:
    """Aggregate raw baseline-window statistics for the dataset."""
    raw = load_feature_channels(feature_table, channels=channels, view="raw")
    out: dict[str, dict[str, float]] = {}
    for channel in channels:
        for summary in ("mean", "std"):
            column = f"{channel}__raw__{summary}"
            if column in raw:
                out.setdefault(channel, {})[summary] = float(raw[column].mean())
    return out
