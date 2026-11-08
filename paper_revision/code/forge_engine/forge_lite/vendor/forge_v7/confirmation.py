from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class ConfirmationResult:
    baseline_auprc: float
    candidate_auprc: float
    delta: float
    bootstrap_lcb: float
    bootstrap_se: float
    positive_fold_fraction: float
    margin: float
    accepted: bool

    def to_dict(self) -> dict:
        return asdict(self)


def discovery_confirmation_split(y: np.ndarray, *, confirmation_fraction: float = 0.40,
                                 seed: int = 20260719,
                                 chronological: bool = False) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.int8)
    indices = np.arange(len(y))
    if chronological:
        cut = int(np.floor(len(y) * (1.0 - float(confirmation_fraction))))
        if cut <= 0 or cut >= len(y):
            raise ValueError("invalid chronological confirmation fraction")
        discovery, confirmation = indices[:cut], indices[cut:]
        if np.unique(y[discovery]).size < 2 or np.unique(y[confirmation]).size < 2:
            raise ValueError("chronological discovery/confirmation must both contain both classes")
        return discovery, confirmation
    discovery, confirmation = train_test_split(
        indices, test_size=float(confirmation_fraction), random_state=seed,
        stratify=y,
    )
    return np.sort(discovery), np.sort(confirmation)


def _paired_bootstrap(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
                      repeats: int, alpha: float, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(int(repeats), dtype=np.float64)
    for i in range(int(repeats)):
        idx = rng.integers(0, n, size=n)
        if np.unique(y[idx]).size < 2:
            deltas[i] = np.nan
            continue
        deltas[i] = (average_precision_score(y[idx], candidate[idx])
                     - average_precision_score(y[idx], baseline[idx]))
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) < max(50, repeats // 4):
        raise ValueError("too few valid paired bootstrap samples")
    return float(np.quantile(deltas, alpha)), float(np.std(deltas, ddof=1))


def confirmation_gate(
    y: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    *,
    bootstrap_repeats: int = 1000,
    alpha: float = 0.05,
    tau_auprc: float = 0.0005,
    se_multiplier: float = 1.0,
    min_positive_fold_fraction: float = 0.8,
    folds: int = 5,
    seed: int = 20260719,
) -> ConfirmationResult:
    y = np.asarray(y, dtype=np.int8)
    base = np.asarray(baseline_prediction, dtype=np.float64)
    cand = np.asarray(candidate_prediction, dtype=np.float64)
    if len(y) != len(base) or len(y) != len(cand):
        raise ValueError("confirmation arrays have different lengths")
    baseline_score = float(average_precision_score(y, base))
    candidate_score = float(average_precision_score(y, cand))
    delta = candidate_score - baseline_score
    lcb, se = _paired_bootstrap(y, base, cand, bootstrap_repeats, alpha, seed)
    margin = float(se_multiplier) * se

    min_class = int(np.bincount(y).min())
    n_splits = min(int(folds), min_class)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1)
    fold_delta = []
    for _, idx in splitter.split(np.zeros(len(y)), y):
        fold_delta.append(average_precision_score(y[idx], cand[idx])
                          - average_precision_score(y[idx], base[idx]))
    positive_fraction = float(np.mean(np.asarray(fold_delta) > 0))
    accepted = bool(
        delta >= tau_auprc
        and lcb > margin
        and positive_fraction >= min_positive_fold_fraction
    )
    return ConfirmationResult(
        baseline_auprc=baseline_score, candidate_auprc=candidate_score,
        delta=delta, bootstrap_lcb=lcb, bootstrap_se=se,
        positive_fold_fraction=positive_fraction, margin=margin,
        accepted=accepted,
    )
