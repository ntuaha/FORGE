"""Residual score, admission threshold and greedy entry (paper S1.1.1 and S1.1.3, CoF step i).

For a candidate ``g`` the Ridge-residualised value is
``g~_i = g_i - g^(-k)_i, i in G_k`` over five folds ``G_1..G_5``
(Eq. (3)) and the residual score is
``S(g) = corr(g~, r)`` with ``r = y - p_OOF`` (Eqs. (2), (4)).
The admission threshold is ``tau = z_0.95 / sqrt(n_I^+)`` (Eq. (12)).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import matrix


def residualized_scores(candidates: np.ndarray, X: np.ndarray, residual: np.ndarray, cfg: dict) -> np.ndarray:
    """Residual scores ``S(g)`` for every column of ``candidates`` (Eq. (4)).

    One multi-output Ridge (alpha = 1, median imputation, standardisation) per
    fold predicts all candidates from ``X``; the multi-output solution equals
    the per-column solutions. Candidates whose residualised variance is below
    ``1e-6`` of their raw variance are fully explained by ``X`` and score 0.
    """
    candidates = np.asarray(candidates, float)
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    if not candidates.shape[1]:
        return np.array([])
    if not np.isfinite(candidates).all():
        raise ValueError("candidate values must be finite")
    orthogonal = np.full(candidates.shape, np.nan)
    for fit_idx, hold_idx in KFold(cfg["residual_folds"], shuffle=True, random_state=cfg["seed"]).split(X):
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                              StandardScaler(), Ridge(alpha=cfg["ridge_alpha"]))
        model.fit(X[fit_idx], candidates[fit_idx])
        prediction = np.asarray(model.predict(X[hold_idx]))
        if prediction.ndim == 1:
            prediction = prediction[:, None]
        orthogonal[hold_idx] = candidates[hold_idx] - prediction
    centered = orthogonal - orthogonal.mean(axis=0)
    r = np.asarray(residual, float) - np.mean(residual)
    numerator = np.sum(centered * r[:, None], axis=0)
    denominator = np.sqrt(np.sum(centered * centered, axis=0) * np.sum(r * r))
    score = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    score[np.var(orthogonal, axis=0) <= 1e-6 * np.var(candidates, axis=0)] = 0
    return score


def admission_threshold(n_positive: int, alpha: float = 0.05) -> float:
    """``tau = z_{1-alpha} / sqrt(n+)``, Eq. (12) (alpha = 0.05 gives z ~= 1.645)."""
    return float(norm.ppf(1 - alpha) / np.sqrt(n_positive))


def discovery_rows(n_rows: int, cfg: dict) -> np.ndarray:
    """Scoring rows of ``I``: all rows, or a fixed-seed sample of ``discovery_row_cap`` rows."""
    cap = cfg["discovery_row_cap"]
    if n_rows <= cap:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(cfg["seed"]).choice(n_rows, cap, replace=False))


def score_pool(state: dict, registry: dict, cfg: dict) -> list[dict]:
    """Residual scores of every feature in ``registry`` against the state's OOF residual,
    on the scoring rows of ``I`` (the rows that also give ``n_I^+`` of Eq. (12))."""
    if not registry:
        return []
    surface = state["surface"]
    names = list(registry)
    expanded = surface if set(names) <= set(surface.train) else surface.with_registry({**surface.registry, **registry})
    y = surface.raw[cfg["label"]].to_numpy(np.int8)
    rows = discovery_rows(len(y), cfg)
    scores = residualized_scores(matrix(expanded.train.iloc[rows], names),
                                 matrix(surface.train.iloc[rows], state["columns"]),
                                 (y.astype(float) - state["oof"])[rows], cfg)
    return [{"feature": n, "family": registry[n].get("family", ""), "score": float(s)}
            for n, s in zip(names, scores)]


def greedy_entry(train, names: list[str], columns: list[str], residual: np.ndarray, cfg: dict):
    """Greedy residual-score entry (CoF step i, threshold Eq. (12)).

    Repeatedly admit the candidate with the largest ``|S|`` if ``|S| >= tau``,
    add it to the conditioning set and rescore the rest; stop when no
    candidate reaches ``tau`` or ``entry_k_cap`` candidates have entered.
    ``train`` is the materialised representation of ``I``.
    Returns ``(entered, audit_rows)``; ``entered`` holds ``(name, S_I)`` pairs,
    where ``S_I`` is the score from the first pass, before greedy rescoring
    (used by the trailing-window check).
    """
    rows = discovery_rows(len(train), cfg)
    label = cfg["label"]
    n_pos = int(train[label].to_numpy()[rows].sum())
    tau = admission_threshold(n_pos, cfg["entry_alpha"])
    remaining, entered, audit = list(names), [], []
    conditioning = list(columns)
    initial = {}
    while remaining and len(entered) < cfg["entry_k_cap"]:
        scores = residualized_scores(matrix(train.iloc[rows], remaining),
                                     matrix(train.iloc[rows], conditioning), residual[rows], cfg)
        if not initial:
            initial = dict(zip(remaining, map(float, scores)))
        best = int(np.argmax(np.abs(scores)))
        step = {"step": len(entered) + 1, "feature": remaining[best], "score": float(scores[best]),
                "tau": tau, "entered": bool(abs(scores[best]) >= tau)}
        audit.append(step)
        if not step["entered"]:
            break
        name = remaining.pop(best)
        entered.append((name, initial[name]))
        conditioning.append(name)
    return entered, audit
