from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from portfolio import (
    BLEND_DEGENERACY_THRESHOLD,
    BLEND_MAX_STEPS,
    STACKER_CONFIG,
    FoldLog,
    _fit_predict,
    _threads,
    caruana_blend,
)

ENGINE_NAME = "native-portfolio"
ENGINE_SEED = 20260717
ENGINE_SEED_CONFIRM = 20270718
FINAL_MEMBER_DEADLINE_FRACTION = 0.85
OOF_BAG_FOLDS = 5


FROZEN_PORTFOLIO: tuple[tuple[str, str, dict], ...] = (
    ("lgbm_default", "lgbm_base", dict(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True)),
    ("xgb_default", "xgboost", dict(
        n_estimators=400, max_depth=6, learning_rate=0.05)),
    ("cat_default", "catboost", dict(
        n_estimators=400, depth=6, learning_rate=0.05)),
    ("lgbm_large", "lgbm_base", dict(
        n_estimators=1000, learning_rate=0.03, num_leaves=128,
        min_child_samples=5, colsample_bytree=0.9, is_unbalance=True)),
    ("xgb_deep", "xgboost", dict(
        n_estimators=600, max_depth=10, learning_rate=0.02)),
    ("rf_default", "random_forest", dict(
        n_estimators=300, max_features=0.8, min_samples_leaf=20)),
    ("lgbm_reg", "lgbm_base", dict(
        n_estimators=400, learning_rate=0.1, num_leaves=31,
        min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=1.0, is_unbalance=True)),
    ("cat_deep", "catboost", dict(
        n_estimators=600, depth=8, learning_rate=0.03)),
    ("xt_default", "extra_trees", dict(
        n_estimators=300, max_features=0.8, min_samples_leaf=20)),
)


def _resolved_params(learner: str, params: dict, seed: int) -> dict:
    if learner == "lgbm_base":
        merged = dict(params)
        merged.setdefault("n_jobs", _threads())
        merged.setdefault("random_state", seed)
        merged.setdefault("verbosity", -1)
        return merged
    return dict(params)


def fit_native_portfolio(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_holdout: np.ndarray,
    *,
    mode: str,
    time_limit: float,
    bag_folds: int = OOF_BAG_FOLDS,
    seed: int = ENGINE_SEED,
    configs: tuple[tuple[str, str, dict], ...] | None = None,
    fold_log: FoldLog | None = None,
    stage: str | None = None,
    blend_max_steps: int = BLEND_MAX_STEPS,
) -> dict:
    if mode not in ("loop", "final"):
        raise ValueError(f"mode must be 'loop' or 'final', got {mode!r}")
    if int(bag_folds) < 2:
        raise ValueError("bag_folds must be >= 2 (OOF bagging is mandatory)")
    configs = tuple(configs) if configs is not None else FROZEN_PORTFOLIO
    if not configs:
        raise ValueError("config portfolio must not be empty")
    y_fit = np.asarray(y_fit)
    x_fit = np.asarray(x_fit, dtype=np.float32)
    x_holdout = np.asarray(x_holdout, dtype=np.float32)
    if x_fit.shape[0] != len(y_fit):
        raise ValueError("x_fit rows must match y_fit length")
    n_rows = len(y_fit)
    log = fold_log or FoldLog()
    stage = stage or f"unified_{mode}"
    started = time.time()
    deadline_fraction = 1.0 if mode == "loop" else FINAL_MEMBER_DEADLINE_FRACTION
    member_deadline = started + float(time_limit) * deadline_fraction

    splitter = StratifiedKFold(
        n_splits=int(bag_folds), shuffle=True, random_state=1000 + int(seed),
    )
    fold_indices = list(splitter.split(np.zeros(n_rows), y_fit))

    member_names: list[str] = []
    member_oof: dict[str, np.ndarray] = {}
    member_holdout: dict[str, np.ndarray] = {}
    fit_seconds: dict[str, float] = {}
    truncated: list[str] = []

    for index, (name, learner, params) in enumerate(configs):
        if index > 0 and time.time() >= member_deadline:
            truncated.append(name)
            continue
        member_started = time.time()
        params_full = _resolved_params(learner, params, seed)
        oof_pred = np.full(n_rows, np.nan, dtype=np.float32)
        holdout_stack: list[np.ndarray] = []
        for fold, (fit_idx, oof_idx) in enumerate(fold_indices):
            log.record(
                stage=stage, member=name, seed=seed, fold=fold,
                n_rows=n_rows, train_idx=fit_idx, oof_idx=oof_idx,
            )
            fold_oof, fold_holdout = _fit_predict(
                learner, params_full, seed,
                x_fit[fit_idx], y_fit[fit_idx],
                [x_fit[oof_idx], x_holdout],
            )
            oof_pred[oof_idx] = fold_oof
            holdout_stack.append(fold_holdout)
        assert not np.isnan(oof_pred).any(), f"OOF coverage incomplete for {name!r}"
        member_names.append(name)
        member_oof[name] = oof_pred
        member_holdout[name] = np.mean(holdout_stack, axis=0).astype("float32")
        fit_seconds[name] = round(time.time() - member_started, 3)
    assert member_names, "config #1 must always complete (admission rule)"

    candidate_names = list(member_names)
    candidate_oof = dict(member_oof)
    candidate_holdout = dict(member_holdout)

    stacker_train_rows = 0
    stacker_input_columns = 0
    if mode == "final":
        import lightgbm as lgb

        stacker_started = time.time()
        oof_matrix = np.column_stack([member_oof[n] for n in member_names])
        stacker_train_rows, stacker_input_columns = oof_matrix.shape
        stacker_params = dict(
            STACKER_CONFIG, n_jobs=_threads(), random_state=seed, verbosity=-1,
        )
        stacker = lgb.LGBMClassifier(**stacker_params)
        stacker.fit(oof_matrix, y_fit)
        holdout_matrix = np.column_stack(
            [member_holdout[n] for n in member_names],
        )
        stacker_holdout = stacker.predict_proba(holdout_matrix)[:, 1].astype("float32")
        stacker_oof = np.full(n_rows, np.nan, dtype=np.float32)
        for fold, (fit_idx, oof_idx) in enumerate(fold_indices):
            log.record(
                stage=stage, member="stacker", seed=seed, fold=fold,
                n_rows=n_rows, train_idx=fit_idx, oof_idx=oof_idx,
            )
            fold_model = lgb.LGBMClassifier(**stacker_params)
            fold_model.fit(oof_matrix[fit_idx], y_fit[fit_idx])
            stacker_oof[oof_idx] = fold_model.predict_proba(
                oof_matrix[oof_idx],
            )[:, 1].astype("float32")
        assert not np.isnan(stacker_oof).any(), "stacker OOF coverage incomplete"
        candidate_names.append("stacker")
        candidate_oof["stacker"] = stacker_oof
        candidate_holdout["stacker"] = stacker_holdout
        fit_seconds["stacker"] = round(time.time() - stacker_started, 3)

    blend_started = time.time()
    oof_matrix = np.column_stack([candidate_oof[n] for n in candidate_names])
    weights = caruana_blend(oof_matrix, y_fit, max_steps=blend_max_steps)
    blend_weights = {
        name: round(float(w), 6) for name, w in zip(candidate_names, weights)
    }
    blend_degenerate = bool(np.max(weights) >= BLEND_DEGENERACY_THRESHOLD)
    pred = np.zeros(len(x_holdout), dtype=np.float64)
    oof_blend = np.zeros(n_rows, dtype=np.float64)
    for name, weight in zip(candidate_names, weights):
        pred += weight * candidate_holdout[name]
        oof_blend += weight * candidate_oof[name]
    pred = pred.astype("float32")
    oof_blend = oof_blend.astype("float32")
    fit_seconds["blend"] = round(time.time() - blend_started, 3)

    disagreement = None
    if len(member_names) >= 2:
        disagreement = np.std(
            np.stack([member_oof[n] for n in member_names], axis=0), axis=0,
        ).astype(np.float32)

    oof_auprc = {
        name: float(average_precision_score(y_fit, candidate_oof[name]))
        for name in candidate_names
    }
    blend_oof_auprc = float(average_precision_score(y_fit, oof_blend))
    member_rows = [
        {
            "model": name,
            "family": ("stacker" if name == "stacker"
                       else dict((n, l) for n, l, _ in configs)[name]),
            "oof_auprc": round(oof_auprc[name], 7),
            "blend_weight": blend_weights[name],
            "folds": int(bag_folds),
            "fit_sec": fit_seconds.get(name),
        }
        for name in candidate_names
    ]
    best_model = max(candidate_names, key=lambda n: blend_weights[n])

    meta = {
        "engine": ENGINE_NAME,
        "mode": mode,
        "presets": ("native_portfolio_medium" if mode == "loop"
                    else "native_portfolio_best"),
        "time_limit_sec": int(time_limit),
        "member_deadline_fraction": deadline_fraction,
        "num_bag_folds": int(bag_folds),
        "engine_seed": int(seed),
        "threads": _threads(),
        "config_list": [name for name, _, _ in configs],
        "n_models": len(candidate_names),
        "best_model": best_model,
        "members_fitted": member_names,
        "truncated": truncated,
        "blend_weights": blend_weights,
        "blend_degenerate": blend_degenerate,
        "blend_max_steps": int(blend_max_steps),
        "blend_oof_auprc": round(blend_oof_auprc, 7),
        "member_oof_auprc": {k: round(v, 7) for k, v in oof_auprc.items()},
        "stacker_train_rows": int(stacker_train_rows),
        "stacker_input_columns": int(stacker_input_columns),
        "fit_seconds": fit_seconds,
        "n_train_rows": int(n_rows),
        "n_predict_rows": int(len(x_holdout)),
        "fit_sec": round(time.time() - started, 3),
    }
    return {
        "pred": pred,
        "oof": oof_blend,
        "disagreement": disagreement,
        "member_rows": member_rows,
        "meta": meta,
    }
