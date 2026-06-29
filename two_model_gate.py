"""
two_model_gate.py
=================

Implements the second-stage distress-versus-surprise gate described in
Section 3.7 of the thesis. The gate is trained on ``distress`` (positive) versus
``surprise`` (negative) and is applied only to cases whose primary distress
probability already crossed the primary threshold.

Routing logic
-------------
For each threshold-positive case the gate returns one of three states based
on the gate distress probability ``g`` and the margin ``delta``:

    g >= 0.5 + delta      -> "likely_distress"
    g <= 0.5 - delta      -> "likely_surprise_or_arousal"
    otherwise             -> "early_risk"

Threshold-negative cases are returned as ``"negative"`` directly.
"""

from __future__ import annotations

import pandas as pd


def assign_two_model_outputs(
    frame: pd.DataFrame,
    gate_margin: float,
) -> pd.DataFrame:
    """Add a ``two_model_output`` column to ``frame`` and return the new frame.

    Parameters
    ----------
    frame:
        Must contain the columns ``y_prob`` (primary distress probability),
        ``gate_distress_prob`` (gate distress probability) and
        ``decision_threshold`` (per-row primary threshold).
    gate_margin:
        Symmetric margin around 0.5 used to separate the three positive
        routing states.
    """
    staged = frame.copy()
    threshold = staged["decision_threshold"].to_numpy(dtype=float)
    distress_probs = staged["y_prob"].to_numpy(dtype=float)
    gate_probs = staged["gate_distress_prob"].to_numpy(dtype=float)

    outputs: list[str] = []
    for distress_prob, threshold_value, gate_prob in zip(
        distress_probs, threshold, gate_probs, strict=False
    ):
        if distress_prob < threshold_value:
            outputs.append("negative")
            continue
        if gate_prob >= 0.5 + gate_margin:
            outputs.append("likely_distress")
        elif gate_prob <= 0.5 - gate_margin:
            outputs.append("likely_surprise_or_arousal")
        else:
            outputs.append("early_risk")

    staged["two_model_output"] = outputs
    return staged


def summarize_two_model_outputs(
    core_frame: pd.DataFrame,
    challenge_frame: pd.DataFrame,
) -> dict[str, float]:
    """Aggregate the routing outcomes for the core and challenge sets."""
    distress_mask = core_frame["y_true"].to_numpy(dtype=int)
    core_frame = core_frame.assign(two_model_output=core_frame["two_model_output"])

    watchlist_recall = float(
        core_frame.loc[distress_mask == 1, "two_model_output"]
        .isin(["early_risk", "likely_distress"])
        .mean()
    )
    confirmed_recall = float(
        core_frame.loc[distress_mask == 1, "two_model_output"]
        .eq("likely_distress")
        .mean()
    )
    confirmed_fpr = float(
        core_frame.loc[distress_mask == 0, "two_model_output"]
        .eq("likely_distress")
        .mean()
    )

    if challenge_frame.empty:
        return {
            "watchlist_recall": watchlist_recall,
            "confirmed_recall": confirmed_recall,
            "confirmed_fpr": confirmed_fpr,
            "surprise_likely_surprise_rate": float("nan"),
            "surprise_likely_distress_rate": float("nan"),
        }

    surprise_mask = challenge_frame["challenge_type"].eq("surprise")
    surprise_likely_surprise_rate = float(
        challenge_frame.loc[surprise_mask, "two_model_output"]
        .eq("likely_surprise_or_arousal")
        .mean()
    )
    surprise_likely_distress_rate = float(
        challenge_frame.loc[surprise_mask, "two_model_output"]
        .eq("likely_distress")
        .mean()
    )

    return {
        "watchlist_recall": watchlist_recall,
        "confirmed_recall": confirmed_recall,
        "confirmed_fpr": confirmed_fpr,
        "surprise_likely_surprise_rate": surprise_likely_surprise_rate,
        "surprise_likely_distress_rate": surprise_likely_distress_rate,
    }
