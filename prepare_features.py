"""
prepare_features.py
===================

Prefix feature extraction. Implements the rules in Section 3.5 of the thesis
(Equations 8-10, Table 6).

For each non-baseline session and each prefix ratio in {0.1, 0.2, 0.3}, the
script:

    1. Loads the six resampled channel families (heartRate, PPInterval,
       BVPProcessed, acc_mag, gyr_mag, rot_mag) at the 10 Hz shared grid.
    2. Truncates the session to T_prefix = rho * T_session.
    3. Computes per-channel summaries (mean, std, range, etc.).
    4. Computes baseline-z and robust-baseline normalized versions of those
       summaries using the participant's BASELINE session.
    5. Emits a 108-dimensional prefix feature row per session per ratio.

Output
------
A ``prefix_feature_table.csv`` with one row per (session, prefix_ratio) pair
and 108 derived features plus metadata columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESAMPLE_HZ = 10.0
STEP_SECONDS = 1.0 / RESAMPLE_HZ
MAX_INTERIOR_GAP_SEC = 2.0
PREFIX_RATIOS = (0.1, 0.2, 0.3)

CHANNELS = (
    "heartRate",
    "PPInterval",
    "BVPProcessed",
    "acc_mag",
    "gyr_mag",
    "rot_mag",
)

SUMMARY_NAMES = (
    "mean",
    "std",
    "range",
    "minimum",
    "maximum",
    "delta",
    "slope",
    "rms",
    "energy",
    "missing_ratio",
)


def _interior_gap_mask(timestamps: np.ndarray, max_gap: float) -> np.ndarray:
    """Return a per-sample boolean mask where ``True`` means the sample is in
    or adjacent to an interior gap larger than ``max_gap`` seconds.
    """
    if len(timestamps) < 2:
        return np.zeros_like(timestamps, dtype=bool)
    diffs = np.diff(timestamps)
    bad_after = diffs > max_gap
    bad = np.zeros_like(timestamps, dtype=bool)
    bad[1:] |= bad_after
    bad[:-1] |= bad_after
    return bad


def _summarize(values: np.ndarray) -> dict[str, float]:
    """Compute the channel-level summary statistics for one prefix window.

    All summaries operate on the non-missing values in ``values``. Returns the
    canonical 10-element summary vector.
    """
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return {name: float("nan") for name in SUMMARY_NAMES}
    result: dict[str, float] = {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "range": float(np.ptp(valid)),
        "missing_ratio": float(np.isnan(values).mean()),
    }
    result["minimum"] = float(np.min(valid))
    result["maximum"] = float(np.max(valid))
    if valid.size >= 2:
        result["delta"] = float(valid[-1] - valid[0])
        t = np.arange(valid.size, dtype=float)
        denom = float(np.sum((t - t.mean()) ** 2))
        slope = (
            0.0
            if denom == 0.0
            else float(np.sum((t - t.mean()) * (valid - valid.mean())) / denom)
        )
        result["slope"] = slope
    else:
        result["delta"] = 0.0
        result["slope"] = 0.0
    result["rms"] = float(np.sqrt(np.mean(valid**2)))
    result["energy"] = float(np.mean(valid**2))
    return result


def _baseline_stats(baseline_values: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(mean, std, median, mad)`` of the baseline window."""
    valid = baseline_values[~np.isnan(baseline_values)]
    if valid.size == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    return mean, std, median, mad


def _build_summary_for_channel(
    values: np.ndarray,
    base_mean: float,
    base_std: float,
    base_median: float,
    base_mad: float,
) -> dict[str, dict[str, float]]:
    """Combine raw, baseline-z, and robust-baseline versions into one row."""
    raw = _summarize(values)
    epsilon = 1e-6
    z = (
        (raw["mean"] - base_mean) / max(base_std, epsilon)
        if not np.isnan(raw["mean"]) and not np.isnan(base_mean)
        else float("nan")
    )
    z_std = (
        raw["std"] / max(base_std, epsilon)
        if not np.isnan(raw["std"]) and not np.isnan(base_std)
        else float("nan")
    )
    robust = (
        (raw["mean"] - base_median) / max(1.4826 * base_mad, epsilon)
        if not np.isnan(raw["mean"]) and not np.isnan(base_median)
        else float("nan")
    )
    return {
        "raw": raw,
        "baseline_z": {
            "mean": z,
            "std": z_std,
            "missing_ratio": raw["missing_ratio"],
        },
        "robust": {
            "mean": robust,
            "missing_ratio": raw["missing_ratio"],
        },
    }


def _to_long_form(session_id: str, prefix_ratio: float, summaries: dict) -> list[dict]:
    """Flatten the nested per-channel summary tree into CSV-ready rows."""
    rows: list[dict] = []
    for channel in CHANNELS:
        per_channel = summaries.get(channel)
        if per_channel is None:
            continue
        for view_name, view in per_channel.items():
            for summary_name, value in view.items():
                column = f"{channel}__{view_name}__{summary_name}"
                rows.append(
                    {
                        "session_id": session_id,
                        "prefix_ratio": prefix_ratio,
                        "column": column,
                        "value": float(value) if value is not None else float("nan"),
                    }
                )
    return rows


def build_prefix_feature_table(
    resampled_dir: Path | str,
    label_manifest: pd.DataFrame,
    output_path: Path | str,
) -> pd.DataFrame:
    """Read the resampled session CSVs and emit the long-form prefix table.

    ``resampled_dir`` must contain one CSV per session with at least the
    columns ``timestamp`` and the six channel families. The session_id is
    derived from the file stem.
    """
    base = Path(resampled_dir)
    label_manifest = label_manifest.copy()
    label_manifest["session_id"] = label_manifest["session_id"].astype(str)

    baseline_rows = label_manifest.loc[
        label_manifest["stimulus_category"] == "BASELINE"
    ]
    if baseline_rows.empty:
        raise ValueError("Label manifest has no BASELINE sessions; cannot normalize.")

    # Aggregate per-participant baseline statistics. In the public release the
    # data are emitted at the session level only, so the participant_id is
    # parsed from the session_id (convention: ``P{participant}_S{session}``).
    label_manifest["participant_id"] = label_manifest["session_id"].str.split(
        "_S", n=1, expand=True
    )[0]

    baseline_by_participant = baseline_rows.merge(
        label_manifest[["session_id", "participant_id"]], on="session_id"
    ).groupby("participant_id")

    all_rows: list[dict] = []
    for csv_path in sorted(base.glob("*.csv")):
        session_id = csv_path.stem
        if session_id not in set(label_manifest["session_id"]):
            continue
        if (
            label_manifest.loc[
                label_manifest["session_id"] == session_id, "final_label"
            ].iat[0]
            == "exclude"
        ):
            continue

        participant_id = label_manifest.loc[
            label_manifest["session_id"] == session_id, "participant_id"
        ].iat[0]
        if participant_id not in baseline_by_participant.groups:
            continue

        baseline_session_id = baseline_by_participant.get_group(participant_id)[
            "session_id"
        ].iat[0]
        baseline_frame = pd.read_csv(base / f"{baseline_session_id}.csv")
        baseline_stats = {
            channel: _baseline_stats(baseline_frame[channel].to_numpy(dtype=float))
            for channel in CHANNELS
        }

        session_frame = pd.read_csv(csv_path)
        timestamps = session_frame["timestamp"].to_numpy(dtype=float)
        bad = _interior_gap_mask(timestamps, MAX_INTERIOR_GAP_SEC)
        valid_mask = ~bad
        n_total = len(timestamps)
        for ratio in PREFIX_RATIOS:
            cutoff = ratio * n_total
            keep = np.arange(n_total)[valid_mask & (np.arange(n_total) < cutoff)]
            session_summaries: dict = {}
            for channel in CHANNELS:
                values = session_frame[channel].to_numpy(dtype=float)
                prefix_values = values[keep]
                base_mean, base_std, base_median, base_mad = baseline_stats[channel]
                session_summaries[channel] = _build_summary_for_channel(
                    prefix_values,
                    base_mean=base_mean,
                    base_std=base_std,
                    base_median=base_median,
                    base_mad=base_mad,
                )
            all_rows.extend(_to_long_form(session_id, ratio, session_summaries))

    long_frame = pd.DataFrame(all_rows)
    wide_frame = long_frame.pivot_table(
        index=["session_id", "prefix_ratio"],
        columns="column",
        values="value",
        aggfunc="first",
    ).reset_index()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wide_frame.to_csv(out_path, index=False)
    return wide_frame


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the prefix feature table for the early-warning study."
    )
    parser.add_argument(
        "--resampled-dir",
        type=Path,
        required=True,
        help="Directory of per-session resampled CSV files (10 Hz).",
    )
    parser.add_argument(
        "--label-manifest",
        type=Path,
        required=True,
        help="Path to session_label_manifest.csv from prepare_labels.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write prefix_feature_table.csv",
    )
    args = parser.parse_args()
    manifest = pd.read_csv(args.label_manifest)
    table = build_prefix_feature_table(args.resampled_dir, manifest, args.output)
    print(
        f"Wrote prefix feature table with {len(table)} rows and "
        f"{table.shape[1] - 2} features to {args.output}"
    )
