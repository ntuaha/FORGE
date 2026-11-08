from __future__ import annotations

import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from protocol import (
    FeatureSpec,
    apply_feature_specs,
    design_matrix,
    load_outer_partitions,
    load_search_data,
    prepare_e0,
    prepare_e1,
)


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


DEFAULT_GRID = [
    dict(num_leaves=31, learning_rate=0.05, n_estimators=800),
    dict(num_leaves=63, learning_rate=0.05, n_estimators=400),
    dict(num_leaves=63, learning_rate=0.03, n_estimators=1000),
    dict(num_leaves=127, learning_rate=0.03, n_estimators=800),
]


@dataclass(frozen=True)
class FinalResult:
    final_auprc: float
    final_auc: float
    selected_kind: str
    selected_config: dict
    selection_val_auprc: float
    selection_trials: list[dict]
    n_features: int
    n_engineered_features: int
    outer_test_evaluations: int
    outer_train_rows: int
    outer_test_rows: int
    outer_train_positive_rate: float
    outer_test_positive_rate: float
    elapsed_sec: float

    def to_dict(self) -> dict:
        return asdict(self)


def _fixed(seed: int = 42) -> dict:
    return dict(
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=int(os.environ.get("LGBM_THREADS", "8")),
        is_unbalance=True,
        verbosity=-1,
        random_state=seed,
    )


def _xgb_model(config: dict, y: np.ndarray):
    import xgboost as xgb

    prevalence = float(np.mean(y))
    return xgb.XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=8,
        learning_rate=config["learning_rate"],
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=int(os.environ.get("LGBM_THREADS", "8")),
        scale_pos_weight=(1.0 - prevalence) / max(prevalence, 1e-12),
        tree_method="hist",
        random_state=42,
    )


def _candidate_predictions(
    kind: str,
    config: dict,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
) -> np.ndarray:
    model = lgb.LGBMClassifier(**{**_fixed(), **config})
    model.fit(x_fit, y_fit)
    lightgbm_pred = model.predict_proba(x_val)[:, 1]
    if kind == "lgbm":
        return lightgbm_pred
    xgb_model = _xgb_model(config, y_fit)
    xgb_model.fit(x_fit, y_fit)
    return 0.5 * lightgbm_pred + 0.5 * xgb_model.predict_proba(x_val)[:, 1]


def finalize_feature_set(
    cfg: dict,
    specs: list[FeatureSpec],
    *,
    representation: str = "e1",
) -> FinalResult:
    started = time.time()
    selection = load_search_data(cfg, representation=representation)
    engineered = apply_feature_specs(
        selection.train, selection.validation, specs, label=selection.label,
    )
    selection_features = selection.features + engineered
    x_fit = design_matrix(selection.train, selection_features)
    x_val = design_matrix(selection.validation, selection_features)
    y_fit = selection.train[selection.label].to_numpy(dtype=np.int8, copy=False)
    y_val = selection.validation[selection.label].to_numpy(dtype=np.int8, copy=False)

    grid = cfg.get("final_grid") or DEFAULT_GRID
    kinds = ["lgbm"] if cfg.get("final_no_xgb") else ["lgbm", "ensemble"]
    trials: list[dict] = []
    best: tuple[float, str, dict] | None = None
    for config in grid:
        for kind in kinds:
            predictions = _candidate_predictions(kind, config, x_fit, y_fit, x_val)
            score = float(average_precision_score(y_val, predictions))
            trial = dict(kind=kind, config=dict(config), val_auprc=round(score, 7))
            trials.append(trial)
            if best is None or score > best[0]:
                best = (score, kind, dict(config))
    if best is None:
        raise RuntimeError("finalization grid produced no candidates")
    selection_score, selected_kind, selected_config = best
    del selection, x_fit, x_val, y_fit, y_val

    outer_train, outer_test, _split_info = load_outer_partitions(cfg)
    label = str(cfg["label"]).lower()
    if representation == "e1":
        outer_train, outer_test, base_features, _meta = prepare_e1(
            outer_train, outer_test, cfg,
        )
    elif representation == "e0":
        outer_train, outer_test, base_features, _meta = prepare_e0(
            outer_train, outer_test, cfg,
        )
    else:
        raise ValueError("representation must be 'e0' or 'e1'")
    final_engineered = apply_feature_specs(
        outer_train, outer_test, specs, label=label,
    )
    final_features = base_features + final_engineered
    x_train = design_matrix(outer_train, final_features)
    x_test = design_matrix(outer_test, final_features)
    y_train = outer_train[label].to_numpy(dtype=np.int8, copy=False)

    predictions = _candidate_predictions(
        selected_kind, selected_config, x_train, y_train, x_test,
    )
    y_test = outer_test[label].to_numpy(dtype=np.int8, copy=False)
    final_auprc = float(average_precision_score(y_test, predictions))
    final_auc = float(roc_auc_score(y_test, predictions))

    score_dir = os.environ.get("OTHERS_SCORE_DIR")
    if score_dir:
        destination = Path(score_dir)
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "test_pred.npy", np.asarray(predictions, dtype=np.float64))
        np.save(destination / "test_labels.npy", np.asarray(y_test, dtype=np.int8))

    result = FinalResult(
        final_auprc=round(final_auprc, 7),
        final_auc=round(final_auc, 7),
        selected_kind=selected_kind,
        selected_config=selected_config,
        selection_val_auprc=round(selection_score, 7),
        selection_trials=trials,
        n_features=len(final_features),
        n_engineered_features=len(final_engineered),
        outer_test_evaluations=1,
        outer_train_rows=len(outer_train),
        outer_test_rows=len(outer_test),
        outer_train_positive_rate=round(float(outer_train[label].mean()), 10),
        outer_test_positive_rate=round(float(outer_test[label].mean()), 10),
        elapsed_sec=round(time.time() - started, 3),
    )
    return result
