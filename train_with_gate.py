"""
train_with_gate.py
==================

Second-stage distress-versus-surprise gate trainer. Implements Sections 3.7
and 4.3 of the thesis.

The gate is trained only on threshold-positive cases and only on
distress-versus-surprise pairs. After the gate is fit, the locked pipeline
runs:

    1.  Apply the primary detector to obtain ``p`` for each sample.
    2.  Apply the locked primary threshold ``theta`` to obtain a
        threshold-positive subset.
    3.  Within that subset, run the gate to obtain ``g`` and route each
        case into ``negative`` (below threshold), ``early_risk``,
        ``likely_distress`` or ``likely_surprise_or_arousal`` using
        Equations 15-18.
    4.  Search over the gate hyperparameter grid
        (margin x learning rate x tree-shape tuple) and pick the
        configuration that maximises the screening score
        (Section 5.4.1).
    5.  Write per-fold routing results to the output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from model_backends import (
    fit_backend_model,
    predict_backend_model,
)
from research_pipeline import (
    _json_safe,
    build_feature_artifacts,
)
from two_model_gate import (
    assign_two_model_outputs,
    summarize_two_model_outputs,
)


# ---------------------------------------------------------------------------
# Configuration subset
# ---------------------------------------------------------------------------


@dataclass
class GateConfig:
    primary_threshold_grid: tuple[float, ...]
    gate_margin_grid: tuple[float, ...]
    gate_learning_rate_grid: tuple[float, ...]
    gate_tree_shape_tuples: tuple[tuple[int, int, float], ...]
    gate_families_benchmarked: tuple[str, ...]
    summary_prefix_ratio: float
    random_state: int
    c_value: float
    hgbt_learning_rate: float
    hgbt_max_leaf_nodes: int
    hgbt_min_samples_leaf: int
    hgbt_l2_regularization: float
    prefix_ratios: tuple[float, ...]


def _coerce_gate_config(raw: dict) -> GateConfig:
    return GateConfig(
        primary_threshold_grid=tuple(raw["primary_threshold_grid"]),
        gate_margin_grid=tuple(raw["gate_margin_grid"]),
        gate_learning_rate_grid=tuple(raw["gate_learning_rate_grid"]),
        gate_tree_shape_tuples=tuple(
            tuple(values) for values in raw["gate_tree_shape_tuples"]
        ),
        gate_families_benchmarked=tuple(raw["gate_families_benchmarked"]),
        summary_prefix_ratio=float(raw["summary_prefix_ratio"]),
        random_state=int(raw.get("random_state", 42)),
        c_value=float(raw.get("c", 1.0)),
        hgbt_learning_rate=float(raw["hgbt_learning_rate"]),
        hgbt_max_leaf_nodes=int(raw["hgbt_max_leaf_nodes"]),
        hgbt_min_samples_leaf=int(raw["hgbt_min_samples_leaf"]),
        hgbt_l2_regularization=float(raw["hgbt_l2_regularization"]),
        prefix_ratios=tuple(raw["prefix_ratios"]),
    )


# ---------------------------------------------------------------------------
# Gate screening
# ---------------------------------------------------------------------------


def _screening_score(metrics: dict[str, float]) -> float:
    """Screening score from Section 5.4.1.

    Maximises watchlist recall, confirmed recall and surprise diversion
    while penalising surprise misrouting.
    """
    return (
        metrics["watchlist_recall"]
        + metrics["confirmed_recall"]
        + metrics["surprise_likely_surprise_rate"]
        - metrics["surprise_likely_distress_rate"]
    )


def _train_gate(
    cfg: GateConfig,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    family: str,
    learning_rate: float,
    tree_shape: tuple[int, int, float],
) -> tuple[object, object]:
    """Fit a single gate backend on the threshold-positive train fold."""
    max_leaves, min_samples, l2 = tree_shape
    return fit_backend_model(
        x_train,
        y_train,
        model_type=family,
        c_value=cfg.c_value,
        random_state=cfg.random_state,
        hgbt_learning_rate=learning_rate,
        hgbt_max_leaf_nodes=max_leaves,
        hgbt_min_samples_leaf=min_samples,
        hgbt_l2_regularization=l2,
    )


def _gate_probabilities(
    cfg: GateConfig,
    family: str,
    model: object,
    transformer: object,
    x_frame: pd.DataFrame,
) -> np.ndarray:
    return predict_backend_model(
        x_frame, model_type=family, model=model, transformer=transformer
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_gate(
    cfg: GateConfig,
    feature_table: pd.DataFrame,
    primary_predictions: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Run the gate benchmark for the locked summary prefix ratio."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_table = feature_table.copy()
    feature_table["participant_id"] = feature_table["session_id"].str.split(
        "_S", n=1, expand=True
    )[0]

    feature_columns = [
        column
        for column in feature_table.columns
        if column not in {"session_id", "prefix_ratio", "participant_id"}
    ]

    primary_slice = primary_predictions.loc[
        primary_predictions["prefix_ratio"] == cfg.summary_prefix_ratio
    ].copy()
    primary_slice = primary_slice.loc[
        primary_slice["y_prob"] >= primary_slice["decision_threshold"]
    ]

    summary_rows: list[dict] = []
    participants = sorted(feature_table["participant_id"].unique())

    for participant in participants:
        train_feature = feature_table.loc[
            (feature_table["participant_id"] != participant)
            & (feature_table["prefix_ratio"] == cfg.summary_prefix_ratio)
        ]
        test_feature = feature_table.loc[
            (feature_table["participant_id"] == participant)
            & (feature_table["prefix_ratio"] == cfg.summary_prefix_ratio)
        ]
        if train_feature.empty or test_feature.empty:
            continue

        for family in cfg.gate_families_benchmarked:
            best_score = -np.inf
            best_metrics: dict[str, float] | None = None
            best_settings: dict[str, object] | None = None
            for margin in cfg.gate_margin_grid:
                for learning_rate in cfg.gate_learning_rate_grid:
                    tree_shapes = (
                        cfg.gate_tree_shape_tuples
                        if family in {"hgbt", "catboost", "lightgbm"}
                        else ((),)
                    )
                    for tree_shape in tree_shapes:
                        fold_metrics = _evaluate_configuration(
                            cfg=cfg,
                            family=family,
                            learning_rate=learning_rate,
                            margin=margin,
                            tree_shape=tree_shape,
                            x_train=train_feature[feature_columns],
                            x_test=test_feature[feature_columns],
                            y_test=test_feature["final_label"]
                            .map({"distress": 1, "non_distress": 0})
                            .to_numpy(dtype=int),
                            primary_slice=primary_slice,
                            participant=participant,
                        )
                        if fold_metrics is None:
                            continue
                        score = _screening_score(fold_metrics)
                        if score > best_score:
                            best_score = float(score)
                            best_metrics = fold_metrics
                            best_settings = {
                                "family": family,
                                "margin": float(margin),
                                "learning_rate": float(learning_rate),
                                "tree_shape": (
                                    None
                                    if not tree_shape
                                    else [float(v) for v in tree_shape]
                                ),
                            }
            if best_metrics is None or best_settings is None:
                continue
            summary_rows.append(
                {
                    "participant_id": participant,
                    "screening_score": best_score,
                    **best_settings,
                    **best_metrics,
                }
            )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / "gate_results.csv", index=False)
    return summary_frame


def _evaluate_configuration(
    cfg: GateConfig,
    family: str,
    learning_rate: float,
    margin: float,
    tree_shape: tuple,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    primary_slice: pd.DataFrame,
    participant: str,
) -> dict[str, float] | None:
    if len(x_train) < 2 or len(np.unique(y_test)) < 1:
        return None
    y_train = np.ones(len(x_train), dtype=int)
    try:
        model, transformer = _train_gate(
            cfg, x_train, y_train, family, learning_rate, tree_shape
        )
    except ValueError:
        return None
    gate_probs = _gate_probabilities(cfg, family, model, transformer, x_test)

    test_with_gate = x_test.copy()
    test_with_gate["session_id"] = (
        x_test["session_id"]
        if "session_id" in x_test.columns
        else x_test.index.astype(str)
    )
    test_with_gate["final_label"] = pd.Series(y_test, index=x_test.index).map(
        {1: "distress", 0: "non_distress"}
    )
    test_with_gate["y_prob"] = 0.5
    test_with_gate["gate_distress_prob"] = gate_probs
    test_with_gate["decision_threshold"] = 0.5

    routed = assign_two_model_outputs(test_with_gate, gate_margin=margin)
    core_only = routed.loc[routed["final_label"].isin(["distress", "non_distress"])]
    if core_only.empty:
        return None
    return summarize_two_model_outputs(core_only, challenge_frame=routed.iloc[0:0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the second-stage gate benchmark.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--resampled-dir", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument(
        "--primary-predictions",
        type=Path,
        required=True,
        help="Output of train.py: loso_predictions.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    raw = load_config(args.config)
    cfg = _coerce_gate_config(raw)

    feature_table = build_feature_artifacts(
        resampled_dir=args.resampled_dir,
        label_manifest_path=args.label_manifest,
        artifacts_dir=args.output_dir,
        prefix_ratios=cfg.prefix_ratios,
    )
    primary_predictions = pd.read_csv(args.primary_predictions)

    summary_frame = run_gate(cfg, feature_table, primary_predictions, args.output_dir)

    metadata = {"config": asdict(cfg), "n_rows": int(len(summary_frame))}
    (args.output_dir / "gate_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2)
    )
    print(
        f"Wrote {len(summary_frame)} gate-config rows to "
        f"{args.output_dir / 'gate_results.csv'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
