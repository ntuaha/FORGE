from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from native_portfolio import (
    ENGINE_SEED,
    FROZEN_PORTFOLIO,
    _resolved_params,
)
from portfolio import (
    FoldLog,
    _fit_median_imputer,
    _fit_predict,
    _impute,
)
from protocol import design_matrix

ENGINE_NAME = "forge-new2"
MAX_LAYERS = 3
OOF_BAG_FOLDS = 5

BAG_FOLDS_SMALL = 8
BAG_FOLDS_LARGE = 5
BAG_FOLDS_SMALL_MAX_ROWS = 400_000

MODELS_YAML = Path(__file__).resolve().parent / "configs" / "models.yaml"

MLP_CONFIG: dict = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=512,
    learning_rate_init=1e-3,
    max_iter=60,
    early_stopping=False,
    n_iter_no_change=10,
    tol=1e-4,
)
MLP_FIT_CAP = 200_000
RF_FIT_CAP = 1_000_000
CAT_FIT_CAP = 2_000_000
MLP_CLIP = 10.0

_CAPPED_LEARNERS = {"mlp": MLP_FIT_CAP,
                    "random_forest": RF_FIT_CAP,
                    "extra_trees": RF_FIT_CAP,
                    "catboost": CAT_FIT_CAP}


def choose_bag_folds(n_rows: int) -> int:
    return BAG_FOLDS_SMALL if int(n_rows) <= BAG_FOLDS_SMALL_MAX_ROWS \
        else BAG_FOLDS_LARGE


def model_library(models_cfg: dict | None = None) -> tuple[tuple[str, str, dict], ...]:
    if models_cfg is None:
        models_cfg = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
    by_name = {name: (learner, dict(params))
               for name, learner, params in FROZEN_PORTFOLIO}
    tree_names = dict(models_cfg["tree_config_names"])
    library: list[tuple[str, str, dict]] = []
    for key in models_cfg["library"]:
        key = str(key)
        if key == "mlp":
            library.append(("mlp", "mlp", dict(MLP_CONFIG)))
            continue
        config_name = tree_names[key]
        assert config_name in by_name, \
            f"FROZEN_PORTFOLIO missing {config_name!r} (library key {key!r})"
        library.append((key,) + by_name[config_name])
    assert library, "models.yaml library list must not be empty"
    return tuple(library)


def stratified_cap(y: np.ndarray, cap: int, seed: int) -> np.ndarray:
    n = len(y)
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for value in (0, 1):
        rows = np.flatnonzero(y == value)
        take = max(1, int(round(cap * len(rows) / n)))
        take = min(take, len(rows))
        parts.append(rng.choice(rows, size=take, replace=False))
    out = np.concatenate(parts)
    out.sort()
    return out.astype(np.int64, copy=False)


def _fit_predict_mlp(params: dict, seed: int, x_fit: np.ndarray,
                     y_fit: np.ndarray, x_predict_list: list[np.ndarray]) -> list[np.ndarray]:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    merged = dict(MLP_CONFIG)
    merged.update(params or {})
    merged["hidden_layer_sizes"] = tuple(merged["hidden_layer_sizes"])
    medians = _fit_median_imputer(x_fit)
    x = _impute(x_fit, medians)
    scaler = StandardScaler()
    x = np.clip(scaler.fit_transform(x), -MLP_CLIP, MLP_CLIP)
    model = MLPClassifier(random_state=seed, **merged)
    model.fit(x, y_fit)
    outputs: list[np.ndarray] = []
    for matrix in x_predict_list:
        z = np.clip(scaler.transform(_impute(matrix, medians)),
                    -MLP_CLIP, MLP_CLIP)
        outputs.append(model.predict_proba(z)[:, 1].astype("float32"))
    return outputs


def mlp_input_importance(params: dict, seed: int, x_fit: np.ndarray,
                         y_fit: np.ndarray) -> np.ndarray:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    merged = dict(MLP_CONFIG)
    merged.update(params or {})
    merged["hidden_layer_sizes"] = tuple(merged["hidden_layer_sizes"])
    medians = _fit_median_imputer(x_fit)
    x = np.clip(StandardScaler().fit_transform(_impute(x_fit, medians)),
                -MLP_CLIP, MLP_CLIP)
    model = MLPClassifier(random_state=seed, **merged)
    model.fit(x, y_fit)
    return np.abs(model.coefs_[0]).mean(axis=1)


def fit_member(learner: str, params: dict, seed: int, x_fit: np.ndarray,
               y_fit: np.ndarray, x_predict_list: list[np.ndarray],
               *, cap_seed: int = 0) -> list[np.ndarray]:
    cap = _CAPPED_LEARNERS.get(learner)
    if cap is not None and len(y_fit) > cap:
        rows = stratified_cap(np.asarray(y_fit), cap, cap_seed)
        x_fit = x_fit[rows]
        y_fit = np.asarray(y_fit)[rows]
    if learner == "mlp":
        return _fit_predict_mlp(params, seed, x_fit, y_fit, x_predict_list)
    if learner == "lgbm_base":
        params = _resolved_params("lgbm_base", params, seed)
    return _fit_predict(learner, params, seed, x_fit, y_fit, x_predict_list)


def _columns_hash(columns: list[str]) -> str:
    return hashlib.sha256("\x1f".join(columns).encode("utf-8")).hexdigest()[:16]


def fold_plan(y_fit: np.ndarray, *, bag_folds: int = OOF_BAG_FOLDS,
              seed: int = ENGINE_SEED,
              groups: np.ndarray | None = None
              ) -> list[tuple[np.ndarray, np.ndarray]]:
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=int(bag_folds), shuffle=True,
                                        random_state=1000 + int(seed))
        return list(splitter.split(np.zeros(len(y_fit)), y_fit,
                                   np.asarray(groups)))
    splitter = StratifiedKFold(n_splits=int(bag_folds), shuffle=True,
                               random_state=1000 + int(seed))
    return list(splitter.split(np.zeros(len(y_fit)), y_fit))


def fit_hetero_layer(
    train_frame,
    holdout_frame,
    y_fit: np.ndarray,
    members: list[tuple[str, str, str, dict]],
    feature_sets: dict[str, list[str]],
    *,
    layer_index: int,
    forbidden: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    deadline: float | None = None,
    seed: int = ENGINE_SEED,
    fold_log: FoldLog | None = None,
    stage: str = "layer",
    memo: dict | None = None,
) -> dict:
    if not (1 <= int(layer_index) <= MAX_LAYERS):
        raise ValueError(
            f"layer_index {layer_index} outside 1..{MAX_LAYERS} "
            "(MAX_LAYERS hard guard)")
    if not members:
        raise ValueError("member list must not be empty")
    y_fit = np.asarray(y_fit)
    log = fold_log or FoldLog()
    started = time.time()

    member_names: list[str] = []
    member_oof: dict[str, np.ndarray] = {}
    member_holdout: dict[str, np.ndarray] = {}
    fit_seconds: dict[str, float] = {}
    truncated: list[str] = []
    design_meta: dict[str, dict] = {}

    for index, (member_name, model_key, learner, params) in enumerate(members):
        if index > 0 and deadline is not None and time.time() >= deadline:
            truncated.append(member_name)
            continue
        columns = list(dict.fromkeys(feature_sets[member_name]))
        assert columns, f"member {member_name!r} has an empty feature set"
        assert forbidden not in columns, \
            f"label column may never be a model feature ({member_name!r})"
        memo_key = (learner, json.dumps(params, sort_keys=True, default=str),
                    tuple(columns))
        if memo is not None and memo_key in memo:
            cached = memo[memo_key]
            member_names.append(member_name)
            member_oof[member_name] = cached["oof"]
            member_holdout[member_name] = cached["holdout"]
            fit_seconds[member_name] = 0.0
            design_meta[member_name] = dict(cached["design"], memoized=True)
            continue
        member_started = time.time()
        x_fit = design_matrix(train_frame, columns)
        x_holdout = design_matrix(holdout_frame, columns)
        oof_pred = np.full(len(y_fit), np.nan, dtype=np.float32)
        holdout_stack: list[np.ndarray] = []
        for fold, (fit_idx, oof_idx) in enumerate(folds):
            log.record(stage=stage, member=member_name, seed=seed, fold=fold,
                       n_rows=len(y_fit), train_idx=fit_idx, oof_idx=oof_idx)
            fold_oof, fold_holdout = fit_member(
                learner, params, seed,
                x_fit[fit_idx], y_fit[fit_idx],
                [x_fit[oof_idx], x_holdout],
                cap_seed=1000 * fold + seed,
            )
            oof_pred[oof_idx] = fold_oof
            holdout_stack.append(fold_holdout)
        assert not np.isnan(oof_pred).any(), \
            f"OOF coverage incomplete for {member_name!r}"
        del x_fit, x_holdout
        member_names.append(member_name)
        member_oof[member_name] = oof_pred
        member_holdout[member_name] = np.mean(
            holdout_stack, axis=0).astype("float32")
        fit_seconds[member_name] = round(time.time() - member_started, 3)
        design_meta[member_name] = {
            "model": model_key,
            "learner": learner,
            "n_columns": len(columns),
            "columns_sha16": _columns_hash(columns),
            "memoized": False,
        }
        if memo is not None:
            memo[memo_key] = {
                "oof": member_oof[member_name],
                "holdout": member_holdout[member_name],
                "design": {k: v for k, v in design_meta[member_name].items()
                           if k != "memoized"},
            }
    assert member_names, "member #1 must always complete (admission rule)"

    oof_auprc = {
        name: float(average_precision_score(y_fit, member_oof[name]))
        for name in member_names
    }
    return {
        "member_names": member_names,
        "member_oof": member_oof,
        "member_holdout": member_holdout,
        "member_oof_auprc": {k: round(v, 7) for k, v in oof_auprc.items()},
        "fit_seconds": fit_seconds,
        "truncated": truncated,
        "design_meta": design_meta,
        "layer_index": int(layer_index),
        "engine": ENGINE_NAME,
        "engine_seed": int(seed),
        "bag_folds": len(folds),
        "elapsed_sec": round(time.time() - started, 3),
    }
