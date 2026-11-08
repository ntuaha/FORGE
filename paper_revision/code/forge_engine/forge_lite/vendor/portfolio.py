from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold


OOF_FOLDS_ALLOWED = (0, 5)
BLEND_MAX_STEPS = 10
STACKER_CONFIG = dict(
    num_leaves=15, n_estimators=200, learning_rate=0.05,
)
BLEND_DEGENERACY_THRESHOLD = 0.95


def default_state() -> dict:
    return {"members": [], "oof_folds": 0, "stacking": False, "blend": False}


def is_trivial(state: dict | None) -> bool:
    if not state:
        return True
    return (
        not state.get("members")
        and int(state.get("oof_folds") or 0) == 0
        and not state.get("stacking")
        and not state.get("blend")
    )


def state_key(state: dict | None) -> str:
    return json.dumps(state or default_state(), sort_keys=True, default=str)


def apply_portfolio_action(state: dict | None, kind: str, params: dict) -> dict:
    new = copy.deepcopy(state) if state else default_state()
    new.setdefault("members", [])
    new.setdefault("oof_folds", 0)
    new.setdefault("stacking", False)
    new.setdefault("blend", False)
    if kind == "portfolio_learner":
        member = {
            "learner": params["learner"],
            "params": {k: v for k, v in params.items() if k != "learner"},
        }
        member["name"] = f"m{len(new['members']) + 1}_{member['learner']}"
        new["members"].append(member)
    elif kind == "portfolio_bagging":
        new["oof_folds"] = int(params["oof_folds"])
    elif kind == "portfolio_stacking":
        new["stacking"] = bool(params["enabled"])
    elif kind == "portfolio_blend":
        new["blend"] = bool(params["enabled"])
    else:
        raise ValueError(f"unknown portfolio action {kind!r}")
    return new


def check_state_dependencies(state: dict) -> None:
    folds = int(state.get("oof_folds") or 0)
    if folds not in OOF_FOLDS_ALLOWED:
        raise ValueError(f"oof_folds must be one of {OOF_FOLDS_ALLOWED}, got {folds}")
    if state.get("stacking") and folds != 5:
        raise ValueError(
            "portfolio_stacking requires oof_folds=5 (design B.3, mirrors "
            "AG's num_stack_levels>0 => num_bag_folds>=2)"
        )
    if state.get("blend") and folds != 5:
        raise ValueError(
            "portfolio_blend requires oof_folds=5: weights may only be fit "
            "on inner-train OOF predictions (design B.3)"
        )


PORTFOLIO_LEARNERS = ("xgboost", "catboost", "random_forest", "extra_trees")

_LEARNER_IMPORTS = {
    "xgboost": "xgboost",
    "catboost": "catboost",
    "random_forest": "sklearn.ensemble",
    "extra_trees": "sklearn.ensemble",
}

_availability_cache: dict[str, bool] = {}


def learner_available(learner: str) -> bool:
    if learner not in _availability_cache:
        try:
            importlib.import_module(_LEARNER_IMPORTS[learner])
            _availability_cache[learner] = True
        except Exception:
            _availability_cache[learner] = False
    return _availability_cache[learner]


def available_learners() -> tuple[str, ...]:
    return tuple(l for l in PORTFOLIO_LEARNERS if learner_available(l))


def _threads() -> int:
    return int(os.environ.get("LGBM_THREADS", "8"))


_LEARNER_DEFAULTS = {
    "xgboost": dict(n_estimators=400, max_depth=6, learning_rate=0.05),
    "catboost": dict(n_estimators=400, depth=6, learning_rate=0.05),
    "random_forest": dict(n_estimators=300, max_features=0.8, min_samples_leaf=20),
    "extra_trees": dict(n_estimators=300, max_features=0.8, min_samples_leaf=20),
}


def build_member_model(learner: str, params: dict, seed: int):
    merged = dict(_LEARNER_DEFAULTS[learner])
    merged.update(params or {})
    if learner == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            **merged,
            tree_method="hist",
            n_jobs=_threads(),
            random_state=seed,
            verbosity=0,
        )
    if learner == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            **merged,
            thread_count=_threads(),
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
    if learner in ("random_forest", "extra_trees"):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

        cls = RandomForestClassifier if learner == "random_forest" else ExtraTreesClassifier
        return cls(**merged, n_jobs=_threads(), random_state=seed)
    raise ValueError(f"unknown portfolio learner {learner!r}")


_SKLEARN_LEARNERS = ("random_forest", "extra_trees")


def _fit_median_imputer(x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        med = np.nanmedian(x, axis=0)
    return np.where(np.isfinite(med), med, 0.0).astype(np.float32)


def _impute(x: np.ndarray, medians: np.ndarray) -> np.ndarray:
    out = np.array(x, dtype=np.float32, copy=True)
    nan_rows, nan_cols = np.where(np.isnan(out))
    if len(nan_rows):
        out[nan_rows, nan_cols] = medians[nan_cols]
    return out


def _fit_predict(learner: str, params, seed, x_fit, y_fit, x_predict_list):
    if learner == "lgbm_base":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(**params)
        model.fit(x_fit, y_fit)
    elif learner == "catboost" and (params or {}).get("cat_features"):
        import pandas as pd

        cat_idx = list(params["cat_features"])

        def _frame(matrix):
            frame = pd.DataFrame(matrix)
            for index in cat_idx:
                frame[index] = frame[index].fillna(-1).astype("int64")
            return frame

        model = build_member_model(learner, params, seed)
        model.fit(_frame(x_fit), y_fit)
        return [model.predict_proba(_frame(x))[:, 1].astype("float32")
                for x in x_predict_list]
    elif learner in _SKLEARN_LEARNERS:
        medians = _fit_median_imputer(x_fit)
        model = build_member_model(learner, params, seed)
        model.fit(_impute(x_fit, medians), y_fit)
        x_predict_list = [_impute(x, medians) for x in x_predict_list]
    else:
        model = build_member_model(learner, params, seed)
        model.fit(x_fit, y_fit)
    return [model.predict_proba(x)[:, 1].astype("float32") for x in x_predict_list]


def _index_hash(indices: np.ndarray) -> str:
    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


class FoldLog:

    def __init__(self, path=None) -> None:
        self.path = path
        self.entries: list[dict] = []

    def record(self, *, stage: str, member: str, seed: int, fold: int,
               n_rows: int, train_idx: np.ndarray, oof_idx: np.ndarray) -> None:
        train_set = np.sort(np.asarray(train_idx, dtype=np.int64))
        oof_set = np.sort(np.asarray(oof_idx, dtype=np.int64))
        overlap = np.intersect1d(train_set, oof_set, assume_unique=True)
        disjoint = len(overlap) == 0
        entry = {
            "stage": stage,
            "member": member,
            "seed": int(seed),
            "fold": int(fold),
            "n_rows": int(n_rows),
            "n_train": int(len(train_set)),
            "n_oof": int(len(oof_set)),
            "train_idx_sha256": _index_hash(train_set),
            "oof_idx_sha256": _index_hash(oof_set),
            "disjoint": bool(disjoint),
        }
        self.entries.append(entry)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        assert disjoint, (
            f"OOF fold {fold} for member {member!r} (seed {seed}) overlaps its "
            f"own training rows: {len(overlap)} shared indices"
        )


def caruana_blend(
    oof_matrix: np.ndarray,
    y_inner: np.ndarray,
    *,
    max_steps: int = BLEND_MAX_STEPS,
) -> np.ndarray:
    if oof_matrix.ndim != 2:
        raise ValueError("oof_matrix must be 2-D (rows x candidates)")
    n_rows, n_candidates = oof_matrix.shape
    if len(y_inner) != n_rows:
        raise ValueError("y_inner length must match oof_matrix rows")
    counts = np.zeros(n_candidates, dtype=np.int64)
    running = np.zeros(n_rows, dtype=np.float64)
    steps = 0
    for _ in range(int(max_steps)):
        best_candidate, best_score = -1, -np.inf
        for j in range(n_candidates):
            trial = (running * steps + oof_matrix[:, j]) / (steps + 1)
            score = float(average_precision_score(y_inner, trial))
            if score > best_score:
                best_candidate, best_score = j, score
        counts[best_candidate] += 1
        running = (running * steps + oof_matrix[:, best_candidate]) / (steps + 1)
        steps += 1
    weights = counts.astype(np.float64) / float(steps)
    weights /= weights.sum()
    return weights


@dataclass
class PipelineResult:

    holdout_pred: np.ndarray
    member_holdout_preds: dict = field(default_factory=dict)
    member_fit_seconds: dict = field(default_factory=dict)
    blend_weights: dict | None = None
    blend_degenerate: bool = False
    stacker_train_rows: int = 0
    stacker_input_columns: int = 0
    oof_folds: int = 0
    combination: str = "single"


def _member_list(state: dict, base_lgbm_params: dict) -> list[tuple[str, str, dict]]:
    members: list[tuple[str, str, dict]] = [("base_lgbm", "lgbm_base", dict(base_lgbm_params))]
    for member in state.get("members", []):
        members.append((
            str(member.get("name") or member["learner"]),
            str(member["learner"]),
            dict(member.get("params") or {}),
        ))
    return members


def fit_portfolio_pipeline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_holdout: np.ndarray,
    *,
    state: dict,
    base_lgbm_params: dict,
    seed: int,
    fold_log: FoldLog | None = None,
    stage: str = "search",
) -> PipelineResult:
    check_state_dependencies(state)
    y_train = np.asarray(y_train)
    members = _member_list(state, base_lgbm_params)
    oof_folds = int(state.get("oof_folds") or 0)
    log = fold_log or FoldLog()

    member_holdout: dict[str, np.ndarray] = {}
    member_oof: dict[str, np.ndarray] = {}
    fit_seconds: dict[str, float] = {}

    fold_indices: list[tuple[np.ndarray, np.ndarray]] = []
    if oof_folds:
        splitter = StratifiedKFold(
            n_splits=oof_folds, shuffle=True, random_state=1000 + int(seed),
        )
        fold_indices = list(splitter.split(np.zeros(len(y_train)), y_train))

    for name, learner, params in members:
        started = time.time()
        if oof_folds == 0:
            (holdout_pred,) = _fit_predict(
                learner, params, seed, x_train, y_train, [x_holdout],
            )
        else:
            oof_pred = np.full(len(y_train), np.nan, dtype=np.float32)
            holdout_stack = []
            for fold, (fit_idx, oof_idx) in enumerate(fold_indices):
                log.record(
                    stage=stage, member=name, seed=seed, fold=fold,
                    n_rows=len(y_train), train_idx=fit_idx, oof_idx=oof_idx,
                )
                fold_oof, fold_holdout = _fit_predict(
                    learner, params, seed,
                    x_train[fit_idx], y_train[fit_idx],
                    [x_train[oof_idx], x_holdout],
                )
                oof_pred[oof_idx] = fold_oof
                holdout_stack.append(fold_holdout)
            assert not np.isnan(oof_pred).any(), (
                f"OOF coverage incomplete for member {name!r}"
            )
            member_oof[name] = oof_pred
            holdout_pred = np.mean(holdout_stack, axis=0).astype("float32")
        member_holdout[name] = holdout_pred
        fit_seconds[name] = round(time.time() - started, 3)

    candidate_names = [name for name, _, _ in members]
    candidate_holdout = dict(member_holdout)
    candidate_oof = dict(member_oof)

    stacker_train_rows = 0
    stacker_input_columns = 0
    if state.get("stacking"):
        import lightgbm as lgb

        started = time.time()
        oof_matrix = np.column_stack([member_oof[n] for n in candidate_names])
        stacker_train_rows, stacker_input_columns = oof_matrix.shape
        stacker_params = dict(
            STACKER_CONFIG,
            n_jobs=_threads(), random_state=seed, verbosity=-1,
        )
        stacker = lgb.LGBMClassifier(**stacker_params)
        stacker.fit(oof_matrix, y_train)
        holdout_matrix = np.column_stack([member_holdout[n] for n in candidate_names])
        stacker_holdout = stacker.predict_proba(holdout_matrix)[:, 1].astype("float32")
        stacker_oof = np.full(len(y_train), np.nan, dtype=np.float32)
        for fold, (fit_idx, oof_idx) in enumerate(fold_indices):
            log.record(
                stage=stage, member="stacker", seed=seed, fold=fold,
                n_rows=len(y_train), train_idx=fit_idx, oof_idx=oof_idx,
            )
            fold_model = lgb.LGBMClassifier(**stacker_params)
            fold_model.fit(oof_matrix[fit_idx], y_train[fit_idx])
            stacker_oof[oof_idx] = fold_model.predict_proba(
                oof_matrix[oof_idx],
            )[:, 1].astype("float32")
        assert not np.isnan(stacker_oof).any(), "stacker OOF coverage incomplete"
        candidate_names = candidate_names + ["stacker"]
        candidate_holdout["stacker"] = stacker_holdout
        candidate_oof["stacker"] = stacker_oof
        fit_seconds["stacker"] = round(time.time() - started, 3)

    blend_weights = None
    blend_degenerate = False
    if state.get("blend"):
        oof_matrix = np.column_stack([candidate_oof[n] for n in candidate_names])
        weights = caruana_blend(oof_matrix, y_train)
        blend_weights = {
            name: round(float(w), 6) for name, w in zip(candidate_names, weights)
        }
        blend_degenerate = bool(np.max(weights) >= BLEND_DEGENERACY_THRESHOLD)
        holdout_pred = np.zeros(len(x_holdout), dtype=np.float64)
        for name, w in zip(candidate_names, weights):
            holdout_pred += w * candidate_holdout[name]
        final_pred = holdout_pred.astype("float32")
        combination = "blend"
    elif state.get("stacking"):
        final_pred = candidate_holdout["stacker"]
        combination = "stacker"
    elif len(members) > 1:
        final_pred = np.mean(
            [member_holdout[n] for n, _, _ in members], axis=0,
        ).astype("float32")
        combination = "mean"
    else:
        final_pred = member_holdout["base_lgbm"]
        combination = "single"

    return PipelineResult(
        holdout_pred=final_pred,
        member_holdout_preds=member_holdout,
        member_fit_seconds=fit_seconds,
        blend_weights=blend_weights,
        blend_degenerate=blend_degenerate,
        stacker_train_rows=int(stacker_train_rows),
        stacker_input_columns=int(stacker_input_columns),
        oof_folds=oof_folds,
        combination=combination,
    )
