"""Stability screening (paper S1.1.2) and the CoF trailing-window check (S1.1.3, step ii).

* R0: every candidate must satisfy ``PSI <= 0.25`` and ``|AdvAUC - 0.5| <= 0.15``
  between ``I`` and ``V`` (Eqs. (5), (6), (7), (8)). No labels
  are used.
* CoF: trailing-window consistency check on the last 30% of ``I``.
* :func:`drift_table` applies the screen to the raw columns (numeric) and the
  unseen-category rate (categorical); it feeds the routing rule
  (:func:`forge.pipeline.routing_rule`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import matrix
from .scoring import admission_threshold, residualized_scores


def psi(early: np.ndarray, late: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index, Eq. (5).

    Decile edges come from ``early`` (``I``); bin shares are floored at 1e-4.
    ``PSI = sum_b (q_b - p_b) log(q_b / p_b)``. Returns 0 when either side has
    fewer than 50 finite values or fewer than three distinct edges.
    """
    early = np.asarray(early, float)
    late = np.asarray(late, float)
    early, late = early[np.isfinite(early)], late[np.isfinite(late)]
    if len(early) < 50 or len(late) < 50:
        return 0.0
    edges = np.unique(np.quantile(early, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.clip(np.histogram(early, edges)[0] / len(early), 1e-4, None)
    q = np.clip(np.histogram(late, edges)[0] / len(late), 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def adversarial_auc(early: np.ndarray, late: np.ndarray) -> float:
    """Rank-based adversarial AUC, Eqs. (6), (7).

    ``U = R_V - n_V (n_V + 1) / 2`` with average ranks for ties and
    ``AdvAUC = U / (n_I n_V)``. No classifier is fitted. Returns 0.5 when
    either side has fewer than 50 finite values.
    """
    early = np.asarray(early, float)
    late = np.asarray(late, float)
    early, late = early[np.isfinite(early)], late[np.isfinite(late)]
    if len(early) < 50 or len(late) < 50:
        return 0.5
    ranks = pd.Series(np.concatenate([early, late])).rank(method="average").to_numpy()
    u = float(ranks[len(early):].sum()) - len(late) * (len(late) + 1) / 2
    return float(u / (len(early) * len(late)))


def is_stable(p: float, a: float, cfg: dict) -> bool:
    """Eq. (8): ``PSI <= psi_max`` and ``|AdvAUC - 0.5| <= advauc_band``."""
    return bool(p <= cfg["psi_max"] and abs(a - .5) <= cfg["advauc_band"])


def screen_candidates(surface, names: list[str], cfg: dict):
    """R0 distribution screen of candidate features between ``I`` and ``V`` (S1.1.2).

    Returns ``(kept_names, table)``.
    """
    rows = []
    for name in names:
        early = surface.train[name].to_numpy(float)
        late = surface.apply[name].to_numpy(float)
        p, a = psi(early, late), adversarial_auc(early, late)
        rows.append({"feature": name, "psi": p, "adv_auc": a, "kept": is_stable(p, a, cfg)})
    return [r["feature"] for r in rows if r["kept"]], pd.DataFrame(rows)


def drift_table(I: pd.DataFrame, V: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Shift of every raw column between ``I`` and ``V`` (routing rule, paper S1.2).

    A numeric column is shifted when it fails Eq. (8); a categorical
    column when its unseen-category rate on ``V`` (``novelty``) exceeds 20%.
    PSI and AdvAUC of the ordinal codes of categorical columns are reported
    for information only.
    """
    skip = set(cfg["drop"] + [cfg["label"], cfg["time_col"]])
    rows = []
    for c in [c for c in I if c not in skip]:
        if pd.api.types.is_numeric_dtype(I[c]):
            a = pd.to_numeric(I[c], errors="coerce").to_numpy(float)
            b = pd.to_numeric(V[c], errors="coerce").to_numpy(float)
            novelty = None
        else:
            keys = sorted(I[c].fillna("<missing>").astype(str).unique())
            mapping = {k: i for i, k in enumerate(keys)}
            a = I[c].fillna("<missing>").astype(str).map(mapping).to_numpy(float)
            b = V[c].fillna("<missing>").astype(str).map(mapping).fillna(-1).to_numpy(float)
            novelty = float(np.mean(b == -1))
        p, u = psi(a, b), adversarial_auc(a, b)
        drifted = not is_stable(p, u, cfg) if novelty is None else novelty > cfg["route_unseen_rate"]
        rows.append({"feature": c, "psi": p, "adv_auc": u, "novelty": novelty, "drifted": bool(drifted)})
    if not rows:
        raise ValueError("no predictors available for the drift diagnostic")
    return pd.DataFrame(rows)


def trailing_window_check(train: pd.DataFrame, entered: list[tuple[str, float]], columns: list[str],
                          residual: np.ndarray, cycle: int, cfg: dict):
    """Trailing-window consistency check for CoF candidates (S1.1.3, step ii).

    The residual score is recomputed on the last ``trailing_fraction`` (30%)
    of ``I`` (``S_tw``) with ``tau_tw = z_0.95 / sqrt(n_tw^+)`` and compared
    with ``S_I``, the candidate's score on the scoring rows of ``I`` before
    greedy rescoring.

    * cycle 1: drop a candidate only if ``S_tw * S_I < 0`` **and**
      ``|S_tw| >= tau_tw`` (significant reversal);
    * cycles 2-3: keep a candidate only if ``S_tw * S_I > 0`` **and**
      ``|S_tw| >= tau_tw``.

    ``entered`` holds ``(name, S_I)`` pairs in entry order. Returns
    ``(kept_names, table)``.
    """
    if not entered:
        return [], pd.DataFrame()
    n = len(train)
    rows = np.arange(n - int(round(cfg["trailing_fraction"] * n)), n)
    n_pos = int(train[cfg["label"]].to_numpy()[rows].sum())
    tau_tw = admission_threshold(max(n_pos, 1), cfg["entry_alpha"])
    names = [name for name, _ in entered]
    # Each candidate is residualised against the configuration it entered into.
    table = []
    conditioning = list(columns)
    for name, score in entered:
        s_tw = float(residualized_scores(matrix(train.iloc[rows], [name]),
                                         matrix(train.iloc[rows], conditioning),
                                         residual[rows], cfg)[0])
        strong = abs(s_tw) >= tau_tw
        kept = not (s_tw * score < 0 and strong) if cycle == 1 else bool(s_tw * score > 0 and strong)
        table.append({"feature": name, "score_I": score, "score_tw": s_tw, "tau_tw": tau_tw,
                      "cycle": cycle, "kept": bool(kept)})
        conditioning.append(name)
    frame = pd.DataFrame(table)
    return [n for n, k in zip(names, frame["kept"]) if k], frame
