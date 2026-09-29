"""Downstream learners, OOF predictions and the four-fold final model.

* :func:`make_learner` builds LightGBM, XGBoost or CatBoost with the
  hyper-parameters of ``configs/default.yaml`` (paper Table 3). FORGE never
  changes them.
* :func:`fold_plan` fixes the stratified learner folds ``F1..F4``.
* :func:`fit_oof` returns the OOF predictions ``p_i = f^(-k)(x_i), i in F_k``
  (the residual of Eq. (2) is ``y - p``) and the mean prediction of the four
  fold models on the unlabelled ``apply`` rows. Called on ``I + V`` with
  ``apply = X_T`` it is the CoF core of S2 and S3 (Sec. IV-D, IV-E): its OOF
  predictions are ``p_core^OOF`` and its mean prediction on ``T`` is the
  rebuilt core.
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from .config import learner_params
from .features import Surface, matrix


def make_learner(cfg: dict, **changes):
    """Unfitted learner ``cfg['learner']`` with the configured hyper-parameters (plus ``changes``).

    Built in: ``lightgbm``, ``xgboost``, ``catboost``. Any other learner is given
    as ``"module:function"``; ``function(params, cfg)`` must return an unfitted
    classifier with ``fit(X, y)`` and ``predict_proba(X)`` that accepts a float32
    matrix with NaN for missing values (example: ``examples/my_learner.py``).
    """
    params = {**learner_params(cfg), **changes}
    if ":" in cfg["learner"]:
        module, function = cfg["learner"].split(":")
        return getattr(importlib.import_module(module), function)(params, cfg)
    if cfg["learner"] == "lightgbm":
        return LGBMClassifier(**params, deterministic=True, force_col_wise=True, n_jobs=cfg["threads"],
                              random_state=cfg["seed"], verbosity=-1, objective="binary")
    if cfg["learner"] == "xgboost":
        if params.get("device") == "cpu" and "xgboost" not in sys.modules:
            # A CUDA build of XGBoost talks to the GPU driver even on the CPU and
            # can stall while another process is using the GPU; hide the GPU.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from xgboost import XGBClassifier

        return XGBClassifier(**params, n_jobs=cfg["threads"], random_state=cfg["seed"], verbosity=0)
    from catboost import CatBoostClassifier

    return CatBoostClassifier(**params, thread_count=cfg["threads"], random_seed=cfg["seed"], verbose=False,
                              allow_writing_files=False)


def predict_probability(model, x: np.ndarray) -> np.ndarray:
    """Positive-class probability, one value per row."""
    if isinstance(model, LGBMClassifier):
        return np.asarray(model.booster_.predict(x), float)     # identical to predict_proba, but faster
    return np.asarray(model.predict_proba(x)[:, 1], float)


def fold_plan(y: np.ndarray, cfg: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic stratified learner folds shared by every model on ``I``."""
    splitter = StratifiedKFold(n_splits=cfg["oof_folds"], shuffle=True, random_state=1000 + cfg["seed"])
    return list(splitter.split(np.zeros(len(y)), y))


def fit_oof(surface: Surface, columns: list[str]) -> dict:
    """Four-fold OOF predictions on ``surface.raw`` and the fold models' mean on ``surface.apply``.

    The representation (including its out-of-fold encodings) is computed once
    on all of ``surface.raw`` and sliced by fold. Returns ``oof`` (float32),
    ``apply`` = (1/K) sum_k f^(-k)(x) as float32 and the OOF AP.
    """
    cfg = surface.cfg
    y = surface.raw[cfg["label"]].to_numpy(np.int8)
    x, x_apply = matrix(surface.train, columns), matrix(surface.apply, columns)
    oof, external = np.full(len(y), np.nan, dtype=np.float32), []
    for fit_idx, hold_idx in fold_plan(y, cfg):
        model = make_learner(cfg)
        model.fit(x[fit_idx], y[fit_idx])
        oof[hold_idx] = predict_probability(model, x[hold_idx])
        external.append(predict_probability(model, x_apply))
    return {"oof": oof, "apply": np.mean(external, axis=0).astype("float32"),
            "oof_ap": float(average_precision_score(y, oof))}
