from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from protocol import SearchData, design_matrix


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


@dataclass(frozen=True)
class SearchScore:
    auprc: float
    auc: float
    per_seed_auprc: tuple[float, ...]
    n_features: int


@dataclass(frozen=True)
class GateResult:
    mean_delta: float
    lower_bound: float
    upper_bound: float
    per_seed_delta: tuple[float, ...]
    accepted: bool
    alpha: float
    n_bootstrap: int
    audit_rows: int

    def to_dict(self) -> dict:
        return asdict(self)


def _model_params(meta: dict, seed: int) -> dict:
    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=int(os.environ.get("LGBM_THREADS", "8")),
        is_unbalance=True,
        random_state=seed,
        verbosity=-1,
    )
    params.update(meta.get("lgbm_override", {}))
    params["random_state"] = seed
    return params


class PairedBootstrapEvaluator:

    def __init__(
        self,
        data: SearchData,
        seeds: tuple[int, ...] = (41, 42, 43),
        n_bootstrap: int = 200,
        alpha: float = 0.05,
        bootstrap_seed: int = 20260710,
        max_audit_rows: int = 50_000,
        max_cached_states: int = 5,
    ) -> None:
        self.data = data
        self.seeds = tuple(seeds)
        self.n_bootstrap = int(n_bootstrap)
        self.alpha = float(alpha)
        self.max_cached_states = int(max_cached_states)
        self._prediction_cache: OrderedDict[tuple[tuple[str, ...], int], np.ndarray] = OrderedDict()
        self.model_fits = 0
        self.gate_calls = 0

        y_full = data.validation[data.label].to_numpy(dtype=np.int8, copy=False)
        self._audit_rows = self._stratified_audit_rows(
            y_full, max_rows=max_audit_rows, seed=bootstrap_seed,
        )
        y = y_full[self._audit_rows]
        self._audit_y = y
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError("search-validation must contain both classes")
        rng = np.random.default_rng(bootstrap_seed)
        self._boot_pos = rng.choice(pos, size=(self.n_bootstrap, len(pos)), replace=True)
        self._boot_neg = rng.choice(neg, size=(self.n_bootstrap, len(neg)), replace=True)

    @staticmethod
    def _stratified_audit_rows(y: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
        if len(y) <= max_rows:
            return np.arange(len(y), dtype=np.int64)
        rng = np.random.default_rng(seed)
        rows: list[np.ndarray] = []
        for value in (0, 1):
            candidates = np.flatnonzero(y == value)
            target = max(1, int(round(max_rows * len(candidates) / len(y))))
            target = min(target, len(candidates))
            rows.append(rng.choice(candidates, size=target, replace=False))
        selected = np.concatenate(rows)
        selected.sort()
        return selected.astype(np.int64, copy=False)

    @staticmethod
    def _key(features: list[str], seed: int) -> tuple[tuple[str, ...], int]:
        return tuple(features), int(seed)

    def _predictions(self, features: list[str], seed: int) -> np.ndarray:
        key = self._key(features, seed)
        cached = self._prediction_cache.get(key)
        if cached is not None:
            self._prediction_cache.move_to_end(key)
            return cached

        x_train = design_matrix(self.data.train, features)
        x_val = design_matrix(self.data.validation, features)
        y_train = self.data.train[self.data.label].to_numpy(dtype=np.int8, copy=False)
        model = lgb.LGBMClassifier(**_model_params(self.data.meta, seed))
        model.fit(x_train, y_train)
        predictions = model.predict_proba(x_val)[:, 1].astype("float32")
        self.model_fits += 1
        del x_train, x_val, model

        self._prediction_cache[key] = predictions
        max_entries = self.max_cached_states * len(self.seeds)
        while len(self._prediction_cache) > max_entries:
            self._prediction_cache.popitem(last=False)
        return predictions

    def predictions(self, features: list[str], seed: int | None = None) -> np.ndarray:
        return self._predictions(features, self.seeds[0] if seed is None else seed)

    def score(self, features: list[str]) -> SearchScore:
        y = self.data.validation[self.data.label].to_numpy(dtype=np.int8, copy=False)
        aps: list[float] = []
        aucs: list[float] = []
        for seed in self.seeds:
            pred = self._predictions(features, seed)
            aps.append(float(average_precision_score(y, pred)))
            aucs.append(float(roc_auc_score(y, pred)))
        return SearchScore(
            auprc=round(float(np.mean(aps)), 6),
            auc=round(float(np.mean(aucs)), 6),
            per_seed_auprc=tuple(round(value, 6) for value in aps),
            n_features=len(features),
        )

    def gate(self, base_features: list[str], new_feature: str) -> GateResult:
        self.gate_calls += 1
        candidate_features = list(base_features) + [new_feature]
        y_full = self.data.validation[self.data.label].to_numpy(dtype=np.int8, copy=False)
        full_deltas: list[float] = []
        base_audit: list[np.ndarray] = []
        candidate_audit: list[np.ndarray] = []
        for seed in self.seeds:
            base = self._predictions(base_features, seed)
            candidate = self._predictions(candidate_features, seed)
            full_deltas.append(
                float(average_precision_score(y_full, candidate) - average_precision_score(y_full, base))
            )
            base_audit.append(base[self._audit_rows])
            candidate_audit.append(candidate[self._audit_rows])

        boot_deltas = np.empty(self.n_bootstrap, dtype=np.float64)
        for index in range(self.n_bootstrap):
            rows = np.concatenate((self._boot_pos[index], self._boot_neg[index]))
            y = self._audit_y[rows]
            seed_deltas = [
                average_precision_score(y, candidate_pred[rows]) -
                average_precision_score(y, base_pred[rows])
                for base_pred, candidate_pred in zip(base_audit, candidate_audit)
            ]
            boot_deltas[index] = float(np.mean(seed_deltas))

        lower = float(np.quantile(boot_deltas, self.alpha))
        upper = float(np.quantile(boot_deltas, 1.0 - self.alpha))
        mean_delta = float(np.mean(full_deltas))
        return GateResult(
            mean_delta=round(mean_delta, 7),
            lower_bound=round(lower, 7),
            upper_bound=round(upper, 7),
            per_seed_delta=tuple(round(value, 7) for value in full_deltas),
            accepted=bool(lower > 0.0),
            alpha=self.alpha,
            n_bootstrap=self.n_bootstrap,
            audit_rows=len(self._audit_rows),
        )

    def audit(self) -> dict:
        return dict(
            seeds=list(self.seeds),
            n_bootstrap=self.n_bootstrap,
            alpha=self.alpha,
            audit_rows=len(self._audit_rows),
            gate_calls=self.gate_calls,
            model_fits=self.model_fits,
            cached_prediction_arrays=len(self._prediction_cache),
        )
