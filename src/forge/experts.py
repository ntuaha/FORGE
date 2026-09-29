"""Optional experts (paper S2).

S2.1 constructs three expert types from the fixed CoF configuration:

* target-encoding experts (S2.1.1, Eq. (15)) for purely categorical columns;
* residual-slice experts (S2.1.2) for subgroups with large OOF residual lift
  ``L(Q) > max(sqrt(n_I^+ / n_Q^+), 1.15)``;
* recency-window experts (S2.1.3, Eqs. (18)-(22)) for
  features that failed the R0 distribution screen (time-ordered data only).

All experts see the CoF core through ``state['oof']`` (rows of ``I``) and
``state['V_prediction']`` (rows of ``V``): the out-of-fold predictions of the
four-fold fit on ``I + V`` that S3 also uses (built by
:func:`forge.pipeline.fit_core`).

S2.2 selects each expert's blend weight ``w*`` on ``I`` with held-out
predictions, fits it once, blends
``p = (1 - w) p_cur + w p_e`` (Eq. (23); slice experts only inside the slice)
and keeps it only if it passes the benefit check on both ``V1`` and ``V2``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import Surface, dependency_registry, expanding_target_encoding, materialize_pair, matrix
from .learners import make_learner, predict_probability
from .screening import adversarial_auc, psi
from .validation import benefit_check


# --------------------------------------------------------------------------- S2.1.1 TE
def target_encoding_candidates(state: dict, V: pd.DataFrame, cfg: dict):
    """TE experts for purely categorical columns (S2.1.1).

    A column qualifies when its unseen-category rate on ``V`` is at most 20%
    and its encoded values pass Eq. (8).
    """
    I, plan = state["surface"].raw, state["surface"].plan
    specs, audit = [], []
    for c in plan["codes"]:
        ti, tv, _, _ = expanding_target_encoding(I, V, c, cfg)
        novelty = float((~V[c].isin(set(I[c]))).mean())
        p, a = psi(ti, tv), adversarial_auc(ti, tv)
        keep = novelty <= cfg["te_max_unseen"] and p <= cfg["psi_max"] and abs(a - .5) <= cfg["advauc_band"]
        audit.append({"column": c, "novelty": novelty, "psi": p, "adv_auc": a, "kept": keep})
        if keep:
            specs.append({"name": "te_" + c, "kind": "te", "source": c, "rows": np.arange(len(I)),
                          "extras": ["te__" + c]})
    return specs, audit


# --------------------------------------------------------------------------- S2.1.2 slices
def apply_slice(frame: pd.DataFrame, spec: dict) -> np.ndarray:
    """Boolean mask of the rows that satisfy a slice rule (interval or single value)."""
    s = frame[spec["column"]]
    if spec["rule"] == "interval":
        return ((s > spec["low"]) & (s <= spec["high"])).fillna(False).to_numpy(bool)
    return s.eq(spec["value"]).fillna(False).to_numpy(bool)


def slice_rules(series: pd.Series, column: str, cfg: dict) -> list[dict]:
    """Candidate subgroups of one column: quintile intervals, or up to eight observed values.

    Code detail: a column with more than 12 distinct values counts as continuous.
    """
    if series.nunique() > 12:
        edges = np.unique(np.quantile(series.dropna(), np.linspace(0, 1, cfg["slice_quintiles"] + 1)))
        if len(edges) < 3:
            return []
        edges[0], edges[-1] = -np.inf, np.inf
        return [{"column": column, "rule": "interval", "low": float(lo), "high": float(hi)}
                for lo, hi in zip(edges[:-1], edges[1:])]
    return [{"column": column, "rule": "value", "value": float(v)}
            for v in series.dropna().unique()[:cfg["slice_max_values"]]]


def slice_candidates(state: dict, V: pd.DataFrame, cfg: dict):
    """Residual-slice experts (S2.1.2).

    Subgroups ``Q`` of the first 25 model columns are kept if they cover 5-60%
    of ``I``, hold >= 30 positives and have residual lift
    ``L(Q) > max(sqrt(n_I^+/n_Q^+), 1.15)``. Ranked by ``L(Q)``, at most one
    slice per column and two columns are used; a selected slice also needs
    >= 800 rows and >= 40 positives in ``I`` and >= 100 rows in ``V``.
    """
    surface = state["surface"]
    y = surface.raw[cfg["label"]].to_numpy(int)
    residual = np.abs(y - state["oof"])
    candidates, audit = [], []
    for column in state["columns"][:cfg["slice_max_columns"]]:
        for rule in slice_rules(surface.train[column], column, cfg):
            mask = apply_slice(surface.train, rule)
            coverage, positives = float(mask.mean()), int(y[mask].sum())
            row = {**rule, "coverage": coverage, "positives": positives, "passed": False}
            audit.append(row)
            if not cfg["slice_min_coverage"] <= coverage <= cfg["slice_max_coverage"] \
                    or positives < cfg["slice_min_positives"]:
                continue
            lift = float(residual[mask].mean() / max(residual.mean(), 1e-12))
            threshold = max(np.sqrt(y.sum() / positives), cfg["slice_min_lift"])
            row.update(lift=lift, lift_threshold=float(threshold), passed=bool(lift > threshold))
            if row["passed"]:
                candidates.append({**rule, "lift": lift, "positives": positives, "rows": np.flatnonzero(mask)})
    selected = []
    for item in sorted(candidates, key=lambda r: -r["lift"]):
        if any(s["column"] == item["column"] for s in selected):
            continue                                      # at most one subgroup per source feature
        v_rows = int(apply_slice(surface.apply, item).sum())
        if (len(item["rows"]) >= cfg["slice_min_rows"] and item["positives"] >= cfg["slice_min_positives_fit"]
                and v_rows >= cfg["slice_min_v_rows"]):
            selected.append({**item, "name": "slice_" + item["column"], "kind": "slice", "extras": []})
        if len(selected) >= cfg["slice_max_experts"]:
            break
    return selected, audit


# --------------------------------------------------------------------------- S2.1.3 recency
def relationship_sign(values: np.ndarray, y: np.ndarray, margin: float) -> int:
    """Direction ``d(A)`` of Eq. (21): +1 if AUC > 0.56, -1 if < 0.44, else 0."""
    if y.sum() == 0 or y.sum() == len(y):
        return 0
    auc = float(roc_auc_score(y, values))
    if auc > 0.5 + margin:
        return 1
    if auc < 0.5 - margin:
        return -1
    return 0


def recency_ladder_level(values: np.ndarray, y: np.ndarray, cfg: dict):
    """Longest stable recent window ``W_k``, k in {0,1,2,3} (Eqs. (18), (22)).

    ``values``/``y`` must be in time order. A window with fewer than 400 rows
    or 20 positives is skipped. Returns ``(k or None, audit)``.
    """
    stable_at, audit = None, []
    n_rows = len(values)
    for level in cfg["recency_levels"]:
        rows = np.arange(n_rows - n_rows // (2 ** level), n_rows, dtype=np.int64)
        positives = int(y[rows].sum())
        if len(rows) < cfg["recency_min_rows"] or positives < cfg["recency_min_positives"]:
            audit.append({"level": level, "rows": len(rows), "positives": positives, "verdict": "too_thin"})
            continue
        middle = len(rows) // 2
        older, newer = rows[:middle], rows[middle:]
        p = psi(values[older], values[newer])
        sign_old = relationship_sign(values[older], y[older], cfg["recency_auc_margin"])
        sign_new = relationship_sign(values[newer], y[newer], cfg["recency_auc_margin"])
        stable = bool(p <= cfg["psi_max"] and sign_old == sign_new)
        audit.append({"level": level, "rows": len(rows), "positives": positives, "psi": p,
                      "sign_old": sign_old, "sign_new": sign_new, "stable": stable})
        if stable and stable_at is None:
            stable_at = level
    return stable_at, audit


def recency_candidates(state: dict, V: pd.DataFrame, dropped: list[str], pool: dict, cfg: dict):
    """Recency-window experts (S2.1.3) for features removed by the R0 distribution screen."""
    if not cfg["time_col"]:
        return [], [{"kept": False, "reason": "no event time; recency experts are not used"}]
    surface = state["surface"]
    expanded = Surface(surface.raw, surface.apply_raw, surface.plan, pool, cfg)
    y = np.r_[surface.raw[cfg["label"]].to_numpy(int), V[cfg["label"]].to_numpy(int)]
    groups, audit, records = {}, [], {}
    for name in dict.fromkeys(dropped):
        values = np.r_[expanded.train[name].to_numpy(float), expanded.apply[name].to_numpy(float)]
        level, detail = recency_ladder_level(values, y, cfg)
        row = {"feature": name, "level": level, "ladder": detail, "kept": False}
        if level is None:
            row["reason"] = "no window passes the local checks"
        elif level == 0:
            row["reason"] = "the full window W0 is stable; no recency expert"
        else:
            row["reason"] = "pending training-size check"
            groups.setdefault(level, []).append(name)
        audit.append(row)
        records[name] = row
    # Paper difference (code detail): features whose first passing window is the
    # same are merged into one expert per window.
    specs = []
    for level, names in sorted(groups.items()):
        start = len(y) - len(y) // (2 ** level)
        rows = np.arange(start, len(surface.raw))
        positives = int(surface.raw.iloc[rows][cfg["label"]].sum())
        keep = len(rows) >= cfg["recency_min_fit_rows"] and positives >= cfg["recency_min_fit_positives"]
        for name in names:
            records[name].update(training_rows=len(rows), training_positives=positives, kept=keep,
                                 reason=f"recency_w{2 ** level} candidate" if keep
                                 else "too few rows or positives of I in the window")
        if keep:
            specs.append({"name": f"recency_w{2 ** level}", "kind": "recency", "rows": rows, "extras": names})
    return specs, audit


# --------------------------------------------------------------------------- S2.2
def expert_pair(fit_raw, apply_raw, state, pool, spec, cfg):
    """Expert representation of (fit, apply) with the CoF features and the expert's extra columns."""
    registry = dependency_registry(state["features"] + spec["extras"], pool)
    f, a = materialize_pair(fit_raw, apply_raw, cfg, state["surface"].plan, registry, expert=True)
    if spec["kind"] == "te":
        ti, ta, _, _ = expanding_target_encoding(fit_raw, apply_raw, spec["source"], cfg)
        f[spec["extras"][0]], a[spec["extras"][0]] = ti, ta
    return f, a


def select_blend_weight(state: dict, pool: dict, spec: dict, cfg: dict) -> dict:
    """S2.2 step 1: ``w* = argmax_w AP((1-w) p_core + w p_e)`` on held-out predictions.

    The eligible rows are cut into five ordered blocks ``B1..B5``; for
    ``b = 2..5`` the expert is trained on ``B1..B(b-1)`` and predicts ``B_b``.
    The core predictions of the same rows are ``state['oof']``, the OOF
    predictions of the four-fold fit on ``I + V`` (not restricted to
    preceding blocks).
    """
    I = state["surface"].raw
    y = I[cfg["label"]].to_numpy(int)
    blocks = np.array_split(spec["rows"], cfg["expert_weight_blocks"])
    eoof, boof = np.full(len(I), np.nan), np.full(len(I), np.nan)
    cols = list(dict.fromkeys(state["columns"] + spec["extras"]))
    for b in range(1, cfg["expert_weight_blocks"]):
        fit_rows, hold = np.concatenate(blocks[:b]), blocks[b]
        if not len(hold) or min(np.bincount(y[fit_rows], minlength=2)) < 5:
            continue
        f, a = expert_pair(I.iloc[fit_rows].reset_index(drop=True), I.iloc[hold].reset_index(drop=True),
                           state, pool, spec, cfg)
        model = make_learner(cfg)
        model.fit(matrix(f, cols), y[fit_rows])
        eoof[hold] = predict_probability(model, matrix(a, cols))
        boof[hold] = state["oof"][hold]
    # Code detail: steps with fewer than 5 samples of a class are skipped, and an
    # expert with fewer than 20 held-out predictions gets weight 0 (rejected).
    valid = np.isfinite(eoof) & np.isfinite(boof)
    if valid.sum() < 20 or np.unique(y[valid]).size < 2:
        return {"weight": 0., "reason": "insufficient held-out predictions", "rows": int(valid.sum())}
    scores = {w: float(average_precision_score(y[valid], (1 - w) * boof[valid] + w * eoof[valid]))
              for w in cfg["blend_weights"]}
    weight = max(scores, key=scores.get)
    return {"weight": weight, "scores": scores, "rows": int(valid.sum()), "reason": "held-out grid"}


@dataclass
class FrozenExpert:
    """An expert fitted once on its eligible ``I`` rows, with a fixed blend weight.

    It is applied to ``V`` in S2 and to ``T`` in S3 without refitting.
    """

    model: object
    fit_raw: pd.DataFrame
    state: dict
    pool: dict
    spec: dict
    columns: list
    weight: float
    cfg: dict

    def predict(self, query: pd.DataFrame):
        """Return ``(expert probability, mask)``; the mask is the slice rule or all rows."""
        _, a = expert_pair(self.fit_raw, query, self.state, self.pool, self.spec, self.cfg)
        p = predict_probability(self.model, matrix(a, self.columns))
        mask = apply_slice(a, self.spec) if self.spec["kind"] == "slice" else np.ones(len(query), bool)
        return p, mask


def blend_prediction(base: np.ndarray, expert: np.ndarray, mask: np.ndarray, weight: float) -> np.ndarray:
    """Eq. (23) inside ``mask``; rows outside the mask keep ``base`` exactly."""
    result = np.asarray(base, float).copy()
    result[mask] = (1 - weight) * result[mask] + weight * np.asarray(expert)[mask]
    return result


def validate_experts(state: dict, V: pd.DataFrame, pool: dict, roster: list[dict], cfg: dict) -> dict:
    """S2.2: weight selection, single fit and sequential benefit check of every expert.

    The stage is disabled when either half of ``V`` has fewer than
    ``expert_min_half_positives`` positives. Returns the accepted experts,
    the audit and the blended ``V`` prediction.
    """
    y = V[cfg["label"]].to_numpy(int)
    reliable = all(h.sum() >= cfg["expert_min_half_positives"] for h in np.array_split(y, 2))
    accepted, audit = [], []
    blend = np.asarray(state["V_prediction"], float).copy()
    Vx = V.drop(columns=cfg["label"])
    for spec in roster:
        if not reliable:
            audit.append({"name": spec["name"], "accepted": False, "reason": "a V half has < 100 positives"})
            continue
        tuning = select_blend_weight(state, pool, spec, cfg)
        if tuning["weight"] <= 0:
            audit.append({"name": spec["name"], "accepted": False, "reason": tuning["reason"]})
            continue
        fit_raw = state["surface"].raw.iloc[spec["rows"]].reset_index(drop=True).copy()
        f, _ = expert_pair(fit_raw, Vx, state, pool, spec, cfg)
        columns = list(dict.fromkeys(state["columns"] + spec["extras"]))
        model = make_learner(cfg)
        model.fit(matrix(f, columns), fit_raw[cfg["label"]].to_numpy(int))
        frozen = FrozenExpert(model, fit_raw, state, copy.deepcopy(pool), copy.deepcopy(spec), columns,
                              tuning["weight"], dict(cfg))
        ep, mask = frozen.predict(Vx)
        candidate = blend_prediction(blend, ep, mask, frozen.weight)
        passed, table = benefit_check(y, blend, candidate, cfg)
        audit.append({"name": spec["name"], "kind": spec["kind"], "accepted": passed, "weight": frozen.weight,
                      "tuning": tuning, "halves": table.to_dict("records")})
        if passed:
            blend = candidate
            accepted.append(frozen)
    return {"experts": accepted, "audit": audit, "reliable": reliable, "V_prediction": blend}
