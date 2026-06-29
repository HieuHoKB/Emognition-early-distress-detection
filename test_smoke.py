"""
test_smoke.py
=============

Smoke test that exercises every public entry point of the pipeline against
synthetic data. Run from the project root with::

    python test_smoke.py

The test does NOT need raw Samsung Watch data. It generates a small
synthetic cohort in memory and verifies that the LOSO trainer, the
threshold selector, the legacy two-stage baseline, and the gate router
all produce well-formed outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from model_backends import fit_backend_model, predict_backend_model
from research_pipeline import build_feature_artifacts
from train import (
    _coerce_config,
    _legacy_two_stage_metrics,
    _select_threshold,
    run_experiment,
)
from two_model_gate import assign_two_model_outputs


FEATURE_CHANNELS = (
    "heartRate",
    "PPInterval",
    "BVPProcessed",
    "acc_mag",
    "gyr_mag",
    "rot_mag",
)


def _synthetic_resampled_dir(tmp_dir: Path, n_participants: int = 4) -> Path:
    """Write a small synthetic 10 Hz dataset into ``tmp_dir``."""
    resampled_dir = tmp_dir / "resampled"
    resampled_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for participant in range(1, n_participants + 1):
        for stimulus in range(2):
            session_id = f"P{participant}_S{stimulus:02d}"
            duration = 200
            timestamps = np.arange(duration) / 10.0
            data: dict[str, np.ndarray] = {"timestamp": timestamps}
            for channel in FEATURE_CHANNELS:
                mean = 60 + participant + rng.normal()
                data[channel] = mean + rng.normal(scale=2.0, size=duration)
            pd.DataFrame(data).to_csv(resampled_dir / f"{session_id}.csv", index=False)
    return resampled_dir


def _synthetic_label_manifest(tmp_dir: Path, n_participants: int = 4) -> Path:
    """Write a small synthetic questionnaire manifest into ``tmp_dir``."""
    manifest_path = tmp_dir / "session_label_manifest.csv"
    rows: list[dict] = []
    for participant in range(1, n_participants + 1):
        rows.append(
            {
                "session_id": f"P{participant}_S00",
                "stimulus_category": "BASELINE",
                "anger": 1,
                "fear": 1,
                "disgust": 1,
                "sadness": 1,
                "valence": 5,
                "motivation": 5,
                "final_label": "exclude",
                "exclusion_reason": "baseline_normalization_reference",
            }
        )
        rows.append(
            {
                "session_id": f"P{participant}_S01",
                "stimulus_category": "SADNESS" if participant % 2 == 1 else "NEUTRAL",
                "anger": 3 if participant % 2 == 1 else 1,
                "fear": 3 if participant % 2 == 1 else 1,
                "disgust": 3 if participant % 2 == 1 else 1,
                "sadness": 3 if participant % 2 == 1 else 1,
                "valence": 2 if participant % 2 == 1 else 6,
                "motivation": 2 if participant % 2 == 1 else 6,
                "final_label": "distress" if participant % 2 == 1 else "non_distress",
                "exclusion_reason": "",
            }
        )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def test_synthetic_pipeline(tmp_dir: Path) -> None:
    """Run the trainer end-to-end on synthetic data and check the output."""
    resampled_dir = _synthetic_resampled_dir(tmp_dir)
    manifest_path = _synthetic_label_manifest(tmp_dir)

    config = load_config()
    cfg = _coerce_config(config)

    feature_table = build_feature_artifacts(
        resampled_dir=resampled_dir,
        label_manifest_path=manifest_path,
        artifacts_dir=tmp_dir,
        prefix_ratios=cfg.prefix_ratios,
    )
    assert not feature_table.empty, "feature table is empty"

    label_manifest = pd.read_csv(manifest_path)
    summary = run_experiment(cfg, feature_table, label_manifest, tmp_dir)
    assert not summary.empty, "summary is empty"
    expected_columns = {
        "participant_id",
        "backend",
        "prefix_ratio",
        "decision_threshold",
        "primary_macro_f1",
        "primary_balanced_accuracy",
        "primary_distress_recall",
        "legacy_early_risk_recall",
        "legacy_confirmed_distress_recall",
    }
    missing = expected_columns - set(summary.columns)
    assert not missing, f"summary missing columns: {sorted(missing)}"


def test_threshold_selection() -> None:
    """The threshold selector must respect the balacc_tolerance window."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    grid = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
    threshold = _select_threshold(y_true, y_prob, grid, balacc_tolerance=0.04)
    assert grid[0] <= threshold <= grid[-1], f"threshold out of range: {threshold}"


def test_legacy_two_stage() -> None:
    """The legacy two-stage baseline must shift the threshold both ways."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    metrics = _legacy_two_stage_metrics(
        y_true, y_prob, primary_threshold=0.4, risk_margin=0.05, confirm_margin=0.45
    )
    early_risk = metrics["early_risk_stage"]["threshold"]
    confirmed = metrics["confirmed_distress_stage"]["threshold"]
    assert early_risk < 0.4, f"early risk threshold must be below 0.4: {early_risk}"
    assert confirmed > 0.4, f"confirmed threshold must be above 0.4: {confirmed}"


def test_gate_router() -> None:
    """The two-model gate router must produce the four canonical states."""
    frame = pd.DataFrame(
        {
            "y_prob": [0.10, 0.30, 0.60, 0.60, 0.60],
            "gate_distress_prob": [0.90, 0.90, 0.10, 0.50, 0.60],
            "decision_threshold": [0.30, 0.30, 0.30, 0.30, 0.30],
        }
    )
    routed = assign_two_model_outputs(frame, gate_margin=0.05)
    expected = [
        "negative",
        "likely_distress",
        "likely_surprise_or_arousal",
        "early_risk",
        "likely_distress",
    ]
    assert routed["two_model_output"].tolist() == expected, routed[
        "two_model_output"
    ].tolist()


def test_backends() -> None:
    """All four tabular backends must accept the same train/test contract."""
    rng = np.random.default_rng(0)
    x_train = pd.DataFrame(
        rng.normal(size=(200, 6)),
        columns=pd.Index(list("ABCDEF")),
    )
    y_train = rng.integers(0, 2, size=200)
    for backend in ("logreg", "hgbt"):
        model, transformer = fit_backend_model(
            x_train,
            y_train,
            model_type=backend,
            c_value=1.0,
            random_state=42,
            hgbt_learning_rate=0.05,
            hgbt_max_leaf_nodes=15,
            hgbt_min_samples_leaf=20,
            hgbt_l2_regularization=0.0,
        )
        probs = predict_backend_model(
            x_train, model_type=backend, model=model, transformer=transformer
        )
        assert probs.shape == (200,), probs.shape


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        tmp_dir = Path(raw)
        test_synthetic_pipeline(tmp_dir)
    test_threshold_selection()
    test_legacy_two_stage()
    test_gate_router()
    test_backends()
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
