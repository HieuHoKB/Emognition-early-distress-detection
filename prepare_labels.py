"""
prepare_labels.py
=================

Questionnaire-driven session-level label construction. Implements the rules in
Section 3.4 of the thesis (Equations 4-7, Table 4, Table 5).

Input
-----
A ``session_questionnaire.csv`` with one row per session and at least these
columns:

    session_id
    stimulus_category          # one of BASELINE, SURPRISE, ANGER, FEAR,
                               # DISGUST, SADNESS, NEUTRAL, AWE, ENTHUSIASM,
                               # LIKING, AMUSEMENT
    anger, fear, disgust, sadness    # discrete emotion intensities (1-N)
    valence, motivation               # SAM dimensional ratings (1-N)

Output
------
A ``session_label_manifest.csv`` with the curated binary labels plus named
exclusion reasons so the cohort definition is auditable downstream.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DISTRESS_MEAN_THRESHOLD = 1.25
LOW_VALENCE = 4
HIGH_VALENCE = 5
LOW_MOTIVATION = 4
HIGH_MOTIVATION = 5
DISTRESS_FILMS = {"ANGER", "FEAR", "DISGUST", "SADNESS"}
NON_DISTRESS_FILMS = {"NEUTRAL", "AWE", "ENTHUSIASM", "LIKING", "AMUSEMENT"}


def _distress_intensity(row: pd.Series) -> tuple[float, float]:
    mean = float((row["anger"] + row["fear"] + row["disgust"] + row["sadness"]) / 4.0)
    maximum = float(max(row["anger"], row["fear"], row["disgust"], row["sadness"]))
    return mean, maximum


def _label_session(row: pd.Series) -> tuple[str, str]:
    """Return ``(final_label, exclusion_reason)`` for a single session row."""
    category = str(row["stimulus_category"])
    if category == "BASELINE":
        return "exclude", "baseline_normalization_reference"
    if category == "SURPRISE":
        return "exclude", "surprise_challenge"

    distress_mean, _ = _distress_intensity(row)
    valence_ok_low = row["valence"] <= LOW_VALENCE
    valence_ok_high = row["valence"] >= HIGH_VALENCE
    motivation_ok_low = row["motivation"] <= LOW_MOTIVATION
    motivation_ok_high = row["motivation"] >= HIGH_MOTIVATION

    if category in DISTRESS_FILMS:
        if distress_mean >= DISTRESS_MEAN_THRESHOLD and (
            valence_ok_low or motivation_ok_low
        ):
            return "distress", ""
        if distress_mean < DISTRESS_MEAN_THRESHOLD:
            return "exclude", "distress_low_subjective_response"
        return "exclude", "distress_affect_inconsistent"

    if category in NON_DISTRESS_FILMS:
        if (
            distress_mean <= DISTRESS_MEAN_THRESHOLD
            and valence_ok_high
            and motivation_ok_high
        ):
            return "non_distress", ""
        if distress_mean > DISTRESS_MEAN_THRESHOLD:
            return "exclude", "non_distress_contaminated"
        return "exclude", "non_distress_negative_affect"

    return "exclude", "unknown_stimulus_category"


def build_session_label_manifest(
    questionnaire_path: Path | str,
    output_path: Path | str,
) -> pd.DataFrame:
    """Read the questionnaire CSV and write the curated label manifest."""
    questionnaire = pd.read_csv(questionnaire_path)
    required = {
        "session_id",
        "stimulus_category",
        "anger",
        "fear",
        "disgust",
        "sadness",
        "valence",
        "motivation",
    }
    missing = required - set(questionnaire.columns)
    if missing:
        raise ValueError(
            f"Questionnaire CSV is missing required columns: {sorted(missing)}"
        )

    labels = questionnaire.apply(_label_session, axis=1, result_type="expand")
    labels.columns = ["final_label", "exclusion_reason"]

    manifest = pd.concat(
        [questionnaire[["session_id", "stimulus_category"]], labels],
        axis=1,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Curate session-level labels from the questionnaire."
    )
    parser.add_argument(
        "--questionnaire",
        type=Path,
        required=True,
        help="Path to session_questionnaire.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write session_label_manifest.csv",
    )
    args = parser.parse_args()
    manifest = build_session_label_manifest(args.questionnaire, args.output)
    print(
        f"Wrote {len(manifest)} session rows to {args.output}\n"
        f"  distress:     {(manifest['final_label'] == 'distress').sum()}\n"
        f"  non_distress: {(manifest['final_label'] == 'non_distress').sum()}\n"
        f"  excluded:     {(manifest['final_label'] == 'exclude').sum()}"
    )
