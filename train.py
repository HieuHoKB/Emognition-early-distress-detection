"""
train.py
========

End-to-end Leave-One-Subject-Out (LOSO) trainer for the primary
early-warning detector. Implements Sections 3.6, 4.2, 4.3, 4.4 of the
thesis. The second-stage gate lives in ``two_model_gate.py`` and is
trained by ``train_with_gate.py`` so that the primary detector can be
evaluated in isolation.

Workflow per held-out participant
---------------------------------
1.  Split the cohort into train / held-out test frames.
2.  Train the primary detector on the train fold.
3.  Use the primary threshold grid to pick the training-time ``theta``
    following the rule in Section 4.3.
4.  Apply the locked primary detector to the held-out participant and
    report per-row probabilities plus thresholded predictions.
5.  Also compute the legacy two-stage baseline
    ``theta_risk = max(0, theta - m_risk)``,
    ``theta_confirm = min(0.999999, theta + m_confirm)`` and report
    its metrics (Section 3.6, Equations 13-14).

Inputs
------
* ``experiment_01_baseline.yaml`` (or another YAML of the same shape).
* ``session_label_manifest.csv`` from ``prepare_labels.py``.
* The 10 Hz resampled session CSVs referenced by the manifest.

Outputs
-------
* ``<output_dir>/summary.csv``              per (backend, prefix) row.
* ``<output_dir>/loso_predictions.csv``     per-row test predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from config import load_config
from model_backends import (
    SUPPORTED_BACKENDS,
    fit_backend_model,
    predict_backend_model,
)
from research_pipeline import (
    _json_safe,
    build_feature_artifacts,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Subset of the YAML that drives primary-detector training."""

    backends: tuple[str, ...]
    primary_backend: str
    catboost_depth: int
    lightgbm_n_estimators: int
    primary_threshold_grid: tuple[float, ...]
    two_stage_risk_margin: float
    two_stage_confirm_margin: float
    prefix_ratios: tuple[float, ...]
    summary_prefix_ratio: float
    random_state: int
    c_value: float
    hgbt_learning_rate: float
    hgbt_max_leaf_nodes: int
    hgbt_min_samples_leaf: int
    hgbt_l2_regularization: float
    class_weight: str
    calibration_method: str


def _coerce_config(raw: dict) -> ExperimentConfig:
    """Validate the YAML payload and return an ``ExperimentConfig``."""
    supported = tuple(raw.get("supported_backends", SUPPORTED_BACKENDS))
    for backend in supported:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"Backend {backend!r} is not supported. "
                f"Use one of {SUPPORTED_BACKENDS}."
            )
    return ExperimentConfig(
        backends=supported,
        primary_backend=raw.get("primary_backend", "logreg"),
        catboost_depth=int(raw.get("catboost_depth", 6)),
        lightgbm_n_estimators=int(raw.get("lightgbm_n_estimators", 200)),
        primary_threshold_grid=tuple(raw["primary_threshold_grid"]),
        two_stage_risk_margin=float(raw.get("two_stage_risk_margin", 0.05)),
        two_stage_confirm_margin=float(raw.get("two_stage_confirm_margin", 0.45)),
        prefix_ratios=tuple(raw["prefix_ratios"]),
        summary_prefix_ratio=float(raw["summary_prefix_ratio"]),
        random_state=int(raw.get("random_state", 42)),
        c_value=float(raw.get("c", 1.0)),
        hgbt_learning_rate=float(raw["hgbt_learning_rate"]),
        hgbt_max_leaf_nodes=int(raw["hgbt_max_leaf_nodes"]),
        hgbt_min_samples_leaf=int(raw["hgbt_min_samples_leaf"]),
        hgbt_l2_regularization=float(raw["hgbt_l2_regularization"]),
        class_weight=str(raw.get("class_weight", "balanced")),
        calibration_method=str(raw.get("calibration_method", "sigmoid")),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _core_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Core binary metrics at a single fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        auprc = float("nan")
        auroc = float("nan")
    else:
        auprc = float(average_precision_score(y_true, y_prob))
        auroc = float(roc_auc_score(y_true, y_prob))
    return {
        "threshold": float(threshold),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division="warn")
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auprc": auprc,
        "auroc": auroc,
        "distress_recall": float(y_pred[y_true == 1].mean())
        if (y_true == 1).any()
        else 0.0,
        "false_positive_rate": float(
            y_pred[y_true == 0].mean() if (y_true == 0).any() else 0.0
        ),
    }


def _select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid: tuple[float, ...],
    balacc_tolerance: float = 0.04,
) -> float:
    """Pick the primary threshold following the rule in Section 4.3."""
    candidates: list[tuple[float, ...]] = []
    for threshold in grid:
        metrics = _core_metrics(y_true, y_prob, threshold)
        candidates.append(
            (
                metrics["balanced_accuracy"],
                metrics["distress_recall"],
                -metrics["false_positive_rate"],
                metrics["macro_f1"],
                -threshold,
                threshold,
            )
        )
    candidates.sort(reverse=True)
    best_balacc = candidates[0][0]
    for entry in candidates:
        if entry[0] >= best_balacc - balacc_tolerance:
            return float(entry[5])
    return float(candidates[0][5])


def _legacy_two_stage_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    primary_threshold: float,
    risk_margin: float,
    confirm_margin: float,
) -> dict[str, dict[str, float]]:
    """Compute the legacy two-stage baseline metrics (Section 3.6, Eq.13)."""
    theta_risk = max(0.0, primary_threshold - risk_margin)
    theta_confirm = min(0.999999, primary_threshold + confirm_margin)
    return {
        "early_risk_stage": _core_metrics(y_true, y_prob, theta_risk),
        "confirmed_distress_stage": _core_metrics(y_true, y_prob, theta_confirm),
    }


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------


def run_experiment(
    cfg: ExperimentConfig,
    feature_table: pd.DataFrame,
    label_manifest: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Run the full LOSO experiment and write the results to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_table = feature_table.copy()
    feature_table = feature_table.merge(
        label_manifest[["session_id", "final_label"]],
        on="session_id",
        how="inner",
    )
    feature_table["participant_id"] = feature_table["session_id"].str.split(
        "_S", n=1, expand=True
    )[0]

    feature_columns = [
        column
        for column in feature_table.columns
        if column not in {"session_id", "prefix_ratio", "participant_id", "final_label"}
    ]

    participants = sorted(feature_table["participant_id"].unique())
    summary_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for participant in participants:
        train_frame = feature_table.loc[feature_table["participant_id"] != participant]
        test_frame = feature_table.loc[feature_table["participant_id"] == participant]
        if train_frame.empty or test_frame.empty:
            continue
        for prefix_ratio in cfg.prefix_ratios:
            for backend in cfg.backends:
                result = _run_one(
                    cfg=cfg,
                    backend=backend,
                    feature_columns=feature_columns,
                    train_frame=train_frame,
                    test_frame=test_frame,
                    prefix_ratio=prefix_ratio,
                    held_out_participant=participant,
                )
                if result is None:
                    continue
                summary_rows.append(result["summary"])
                prediction_rows.extend(result["predictions"])

    summary_frame = pd.DataFrame(summary_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    prediction_frame.to_csv(output_dir / "loso_predictions.csv", index=False)
    return summary_frame

    summary_frame = pd.DataFrame(summary_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    prediction_frame.to_csv(output_dir / "loso_predictions.csv", index=False)
    return summary_frame


def _run_one(
    cfg: ExperimentConfig,
    backend: str,
    feature_columns: list[str],
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    prefix_ratio: float,
    held_out_participant: str,
) -> dict | None:
    train_slice = train_frame.loc[train_frame["prefix_ratio"] == prefix_ratio]
    test_slice = test_frame.loc[test_frame["prefix_ratio"] == prefix_ratio]
    if train_slice.empty or test_slice.empty:
        return None

    x_train = train_slice[feature_columns].astype(float)
    y_train = (
        train_slice["final_label"]
        .map({"distress": 1, "non_distress": 0})
        .to_numpy(dtype=int)
    )
    if len(np.unique(y_train)) < 2:
        return None

    x_test = test_slice[feature_columns].astype(float)
    y_test = (
        test_slice["final_label"]
        .map({"distress": 1, "non_distress": 0})
        .to_numpy(dtype=int)
    )

    model, transformer = fit_backend_model(
        x_train,
        y_train,
        model_type=backend,
        c_value=cfg.c_value,
        random_state=cfg.random_state,
        hgbt_learning_rate=cfg.hgbt_learning_rate,
        hgbt_max_leaf_nodes=cfg.hgbt_max_leaf_nodes,
        hgbt_min_samples_leaf=cfg.hgbt_min_samples_leaf,
        hgbt_l2_regularization=cfg.hgbt_l2_regularization,
    )

    y_train_prob = predict_backend_model(
        x_train, model_type=backend, model=model, transformer=transformer
    )
    y_test_prob = predict_backend_model(
        x_test, model_type=backend, model=model, transformer=transformer
    )

    threshold = _select_threshold(y_train, y_train_prob, cfg.primary_threshold_grid)
    primary_metrics = _core_metrics(y_test, y_test_prob, threshold)
    legacy_metrics = _legacy_two_stage_metrics(
        y_test,
        y_test_prob,
        primary_threshold=threshold,
        risk_margin=cfg.two_stage_risk_margin,
        confirm_margin=cfg.two_stage_confirm_margin,
    )

    predictions = []
    for session_id, y_true, y_prob in zip(
        test_slice["session_id"], y_test, y_test_prob, strict=False
    ):
        predictions.append(
            {
                "participant_id": held_out_participant,
                "session_id": session_id,
                "prefix_ratio": prefix_ratio,
                "backend": backend,
                "y_true": int(y_true),
                "y_prob": float(y_prob),
                "decision_threshold": float(threshold),
            }
        )

    summary = {
        "participant_id": held_out_participant,
        "backend": backend,
        "prefix_ratio": prefix_ratio,
        "decision_threshold": float(threshold),
        "primary_macro_f1": primary_metrics["macro_f1"],
        "primary_balanced_accuracy": primary_metrics["balanced_accuracy"],
        "primary_auprc": primary_metrics["auprc"],
        "primary_auroc": primary_metrics["auroc"],
        "primary_distress_recall": primary_metrics["distress_recall"],
        "primary_false_positive_rate": primary_metrics["false_positive_rate"],
        "legacy_early_risk_recall": legacy_metrics["early_risk_stage"][
            "distress_recall"
        ],
        "legacy_early_risk_macro_f1": legacy_metrics["early_risk_stage"]["macro_f1"],
        "legacy_confirmed_distress_recall": legacy_metrics["confirmed_distress_stage"][
            "distress_recall"
        ],
        "legacy_confirmed_distress_macro_f1": legacy_metrics[
            "confirmed_distress_stage"
        ]["macro_f1"],
    }
    return {"summary": summary, "predictions": predictions}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LOSO primary-detector experiment."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to experiment_01_baseline.yaml.",
    )
    parser.add_argument(
        "--resampled-dir",
        type=Path,
        required=True,
        help="Directory of per-session resampled CSVs (10 Hz).",
    )
    parser.add_argument(
        "--label-manifest",
        type=Path,
        required=True,
        help="Path to session_label_manifest.csv from prepare_labels.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write summary.csv and loso_predictions.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    raw = load_config(args.config)
    cfg = _coerce_config(raw)

    feature_table = build_feature_artifacts(
        resampled_dir=args.resampled_dir,
        label_manifest_path=args.label_manifest,
        artifacts_dir=args.output_dir,
        prefix_ratios=cfg.prefix_ratios,
    )

    summary_frame = run_experiment(
        cfg, feature_table, pd.read_csv(args.label_manifest), args.output_dir
    )

    metadata = {
        "config": asdict(cfg),
        "n_summary_rows": int(len(summary_frame)),
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2)
    )
    print(
        f"Wrote {len(summary_frame)} summary rows to "
        f"{args.output_dir / 'summary.csv'} and per-row predictions to "
        f"{args.output_dir / 'loso_predictions.csv'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
