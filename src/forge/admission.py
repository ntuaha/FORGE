"""Candidate admission (paper S1.1.3).

* R0, :func:`shadow_admission`: each screened candidate gets a shuffled
  "shadow" copy. Permutation importance (PI) is the drop of OOF AP when a
  column is permuted at prediction time with the fitted fold model kept
  fixed. On each contiguous half ``J`` of ``I`` the threshold is the largest
  shadow PI, ``theta^J = max_{c* in S} PI^J(c*)`` (Eq. (9)), and
  a candidate is admitted iff ``PI^{I_a}(c) > theta^{I_a}`` and
  ``PI^{I_b}(c) > theta^{I_b}`` (Eq. (11)).
* CoF cycle 1, :func:`permutation_exit`: features accepted in earlier rounds
  whose OOF PI on ``I`` is not positive are removed (at most two per round).
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .features import matrix
from .learners import fold_plan, make_learner, predict_probability

log = logging.getLogger("forge")

# Fixed strings that seed the deterministic permutations. They are part of the
# random-number derivation used for the published results; changing any of
# them changes the permutations and therefore the reproduced numbers.
PERMUTATION_SALT = "forge_v13_probe"
PERMUTATION_MEMBER = "l1_lgbm_default"
SHADOW_PREFIX = "sh34__"
SHADOW_SEED_OFFSET = 777


def deterministic_permutation(n_rows: int, member: str, column: str, seed: int) -> np.ndarray:
    """Reproducible permutation of ``n_rows`` indices for one (member, column, seed).

    The generator seed is the first 32 bits of
    ``SHA-256("<salt>|<member>|<column>|<seed>")``.
    """
    digest = hashlib.sha256(f"{PERMUTATION_SALT}|{member}|{column}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:4], "big")).permutation(n_rows)


def shadow_admission(surface, names: list[str], columns: list[str], cfg: dict):
    """R0 shadow-reference admission, Eqs. (9), (11).

    ``columns`` are the retained configuration's model columns. The
    candidates and their shadows are added, four OOF fold models are fitted,
    and each probe column is permuted in the held-out fold (with a
    permutation drawn over all rows of ``I``). Returns ``(admitted, table)``.
    """
    if not names:
        return [], pd.DataFrame()
    train, apply = surface.train.copy(), surface.apply.copy()
    rng = np.random.default_rng(cfg["seed"] + SHADOW_SEED_OFFSET)
    shadows = {name: f"{SHADOW_PREFIX}{name}"[:60] for name in names}
    if len(set(shadows.values())) != len(shadows):
        raise ValueError("shadow names collide after truncation")
    for name, shadow in shadows.items():
        train[shadow] = rng.permutation(pd.to_numeric(train[name], errors="coerce").fillna(0).to_numpy())
        # The V-side shadow is not used for admission; it is drawn so that the
        # random stream (and hence the I-side shadows) matches the reference run.
        apply[shadow] = rng.permutation(pd.to_numeric(apply[name], errors="coerce").fillna(0).to_numpy())
    probes = names + list(shadows.values())
    model_columns = list(dict.fromkeys(columns + probes))
    y = surface.raw[cfg["label"]].to_numpy(np.int8)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    probe_oof = {n: np.full(len(y), np.nan, dtype=np.float32) for n in probes}
    global_x = matrix(train, model_columns)
    for fold, (fit_idx, hold_idx) in enumerate(fold_plan(y, cfg), 1):
        model = make_learner(cfg)
        model.fit(matrix(train.iloc[fit_idx], model_columns), y[fit_idx])
        x = matrix(train.iloc[hold_idx], model_columns)
        oof[hold_idx] = predict_probability(model, x)
        batch = []
        for index, name in enumerate(probes):
            z = x.copy()
            col = model_columns.index(name)
            perm = deterministic_permutation(len(y), PERMUTATION_MEMBER, name, cfg["seed"])
            z[:, col] = global_x[perm[hold_idx], col]
            batch.append(z)
            if len(batch) == 4 or index == len(probes) - 1:
                pred = predict_probability(model, np.vstack(batch)).reshape(len(batch), len(hold_idx))
                for j, probe in enumerate(probes[index - len(batch) + 1:index + 1]):
                    probe_oof[probe][hold_idx] = pred[j]
                batch = []
        log.debug("shadow PI fold %d/%d done (%d probes)", fold, cfg["oof_folds"], len(probes))
    halves = np.array_split(np.arange(len(y)), 2)

    def pi(name, rows):
        return float(average_precision_score(y[rows], oof[rows])
                     - average_precision_score(y[rows], probe_oof[name][rows]))

    theta = [max(pi(s, h) for s in shadows.values()) for h in halves]
    rows = []
    for name in names:
        scores = [pi(name, h) for h in halves]
        rows.append({"feature": name, "pi_1": scores[0], "pi_2": scores[1],
                     "theta_1": theta[0], "theta_2": theta[1],
                     "kept": bool(all(s > t for s, t in zip(scores, theta)))})
    return [r["feature"] for r in rows if r["kept"]], pd.DataFrame(rows)


def oof_permutation_importance(surface, columns: list[str], targets: list[str], cfg: dict) -> dict:
    """OOF permutation importance on ``I`` of each column in ``targets``.

    ``PI(c) = AP(p_OOF) - AP(p_OOF with c permuted inside each held-out fold)``.
    """
    y = surface.raw[cfg["label"]].to_numpy(np.int8)
    oof = np.full(len(y), np.nan)
    permuted = {c: np.full(len(y), np.nan) for c in targets}
    for fold, (fit_idx, hold_idx) in enumerate(fold_plan(y, cfg)):
        model = make_learner(cfg)
        model.fit(matrix(surface.train.iloc[fit_idx], columns), y[fit_idx])
        x = matrix(surface.train.iloc[hold_idx], columns)
        oof[hold_idx] = predict_probability(model, x)
        for c in targets:
            z = x.copy()
            col = columns.index(c)
            z[:, col] = x[deterministic_permutation(len(hold_idx), cfg["learner"], c, cfg["seed"] + fold), col]
            permuted[c][hold_idx] = predict_probability(model, z)
    base = average_precision_score(y, oof)
    return {c: float(base - average_precision_score(y, permuted[c])) for c in targets}


def permutation_exit(surface, columns: list[str], earlier_features: list[str], cfg: dict):
    """CoF cycle-1 exit: drop earlier-round features with ``PI <= 0`` (lowest first, at most two).

    Returns ``(removed, table)``.
    """
    targets = [c for c in earlier_features if c in columns]
    if not targets:
        return [], pd.DataFrame()
    importance = oof_permutation_importance(surface, columns, targets, cfg)
    ranked = sorted((c for c in targets if importance[c] <= 0), key=lambda c: importance[c])
    removed = ranked[:cfg["exit_max_drops"]]
    table = pd.DataFrame([{"feature": c, "pi": importance[c], "removed": c in removed} for c in targets])
    return removed, table
