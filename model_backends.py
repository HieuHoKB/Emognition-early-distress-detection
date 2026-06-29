"""
model_backends.py
=================

Single chokepoint for fitting and predicting with the four tabular backends
used in the early-warning pipeline:

    - logreg:    sklearn LogisticRegression with L2 regularization
    - hgbt:      sklearn HistGradientBoostingClassifier
    - catboost:  CatBoostClassifier (auto class-weight Balanced)
    - lightgbm:  LightGBM (LGBMClassifier)

All four backends share the same `random_state` and `class_weight` arguments
so that a single master seed drives the entire LOSO evaluation.

Hyperparameter sharing rules
----------------------------
- logreg uses `c_value` (C in sklearn) and `class_weight`; the `hgbt_*` arguments
  are not consumed by logreg.
- hgbt uses all `hgbt_*` arguments.
- catboost reuses `hgbt_learning_rate` for its own `learning_rate` and uses a
  fixed `depth=6`. Other `hgbt_*` arguments are not consumed.
- lightgbm reuses `hgbt_learning_rate`, `hgbt_max_leaf_nodes`,
  `hgbt_min_samples_leaf`, `hgbt_l2_regularization`; `n_estimators` is fixed
  to 200 inside this module.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


SUPPORTED_BACKENDS = ("logreg", "hgbt", "catboost", "lightgbm")


def _balanced_sample_weight(y_train: np.ndarray) -> np.ndarray | None:
    """Return a per-sample weight vector that mimics ``class_weight="balanced"``."""
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) != 2:
        return None
    total = float(len(y_train))
    weights = {
        int(cls): total / (2.0 * float(count)) for cls, count in zip(classes, counts)
    }
    return np.array([weights[int(label)] for label in y_train], dtype=float)


def fit_backend_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    model_type: str,
    c_value: float,
    random_state: int,
    hgbt_learning_rate: float,
    hgbt_max_leaf_nodes: int,
    hgbt_min_samples_leaf: int,
    hgbt_l2_regularization: float,
) -> tuple[Any, Any]:
    """Fit a single tabular backend and return ``(model, transformer)``.

    ``transformer`` is the StandardScaler for logreg (where feature scaling is
    required) and ``None`` for the tree-based backends.
    """
    if model_type == "logreg":
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)
        model = LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            C=c_value,
            random_state=random_state,
        )
        model.fit(x_scaled, y_train)
        return model, scaler

    if model_type == "hgbt":
        model = HistGradientBoostingClassifier(
            learning_rate=hgbt_learning_rate,
            max_leaf_nodes=hgbt_max_leaf_nodes,
            min_samples_leaf=hgbt_min_samples_leaf,
            l2_regularization=hgbt_l2_regularization,
            random_state=random_state,
        )
        model.fit(x_train.to_numpy(dtype=float), y_train)
        return model, None

    if model_type == "catboost":
        module = importlib.import_module("catboost")
        catboost_cls = module.CatBoostClassifier
        sample_weight = _balanced_sample_weight(y_train)
        model = catboost_cls(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            learning_rate=hgbt_learning_rate,
            depth=6,
            verbose=False,
        )
        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(x_train, y_train, **fit_kwargs)
        return model, None

    if model_type == "lightgbm":
        module = importlib.import_module("lightgbm")
        lgbm_cls = module.LGBMClassifier
        sample_weight = _balanced_sample_weight(y_train)
        model = lgbm_cls(
            learning_rate=hgbt_learning_rate,
            num_leaves=max(7, hgbt_max_leaf_nodes),
            min_child_samples=hgbt_min_samples_leaf,
            reg_lambda=hgbt_l2_regularization,
            random_state=random_state,
            n_estimators=200,
            verbosity=-1,
        )
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        model.fit(x_train, y_train, **fit_kwargs)
        return model, None

    raise ValueError(
        f"Unsupported model_type: {model_type!r}. "
        f"Supported backends: {SUPPORTED_BACKENDS}"
    )


def predict_backend_model(
    x_frame: pd.DataFrame,
    *,
    model_type: str,
    model: Any,
    transformer: Any,
) -> np.ndarray:
    """Return the positive-class probability for a fitted backend."""
    if model_type == "logreg":
        scaler = transformer
        return model.predict_proba(scaler.transform(x_frame))[:, 1]
    if model_type in {"hgbt", "lightgbm"}:
        return model.predict_proba(x_frame.to_numpy(dtype=float))[:, 1]
    if model_type == "catboost":
        return model.predict_proba(x_frame)[:, 1]
    raise ValueError(
        f"Unsupported model_type: {model_type!r}. "
        f"Supported backends: {SUPPORTED_BACKENDS}"
    )
