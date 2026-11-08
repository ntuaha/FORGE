from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def z_quantile(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("quantile level must be in (0, 1)")
    return float(norm.ppf(p))


def entry_noise_floor(n_pos: int, alpha: float = 0.05) -> float:
    if int(n_pos) <= 0:
        raise ValueError("noise floor needs a positive discovery-positive count")
    return z_quantile(1.0 - float(alpha)) / float(np.sqrt(int(n_pos)))


def _corr(x: np.ndarray, r: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    x = x - x.mean()
    r = r - r.mean()
    denominator = float(np.sqrt((x * x).mean() * (r * r).mean()))
    if not np.isfinite(denominator) or denominator <= 1e-300:
        return 0.0
    value = float((x * r).mean() / denominator)
    return value if np.isfinite(value) else 0.0


def residual_correlation(values: np.ndarray, residual: np.ndarray) -> float:
    return _corr(values, residual)


def impute_columns(matrix: np.ndarray) -> np.ndarray:
    values = np.array(matrix, dtype=np.float64, copy=True)
    if values.ndim == 1:
        values = values[:, None]
    for j in range(values.shape[1]):
        column = values[:, j]
        finite = np.isfinite(column)
        if finite.all():
            continue
        fill = float(np.median(column[finite])) if finite.any() else 0.0
        column[~finite] = fill
    return values


def cross_fitted_residualize(
    candidates: np.ndarray,
    conditioning: np.ndarray,
    *,
    folds: int = 5,
    ridge_alpha: float = 1.0,
    seed: int = 20260722,
) -> np.ndarray:
    c = np.asarray(candidates, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    if not np.isfinite(c).all():
        raise ValueError("candidates must be finite (impute before scoring)")
    z = np.asarray(conditioning, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    n = len(c)
    if len(z) != n:
        raise ValueError("candidate and conditioning row counts differ")
    if n < 10:
        raise ValueError("cross-fitting needs at least 10 rows")
    n_splits = int(min(max(2, folds), n // 2))
    residual = np.full_like(c, np.nan, dtype=np.float64)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    for fit_idx, hold_idx in splitter.split(np.zeros(n)):
        if z.shape[1] == 0:
            predicted = np.repeat(c[fit_idx].mean(axis=0, keepdims=True),
                                  len(hold_idx), axis=0)
        else:
            model = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                Ridge(alpha=float(ridge_alpha)),
            )
            model.fit(z[fit_idx], c[fit_idx])
            predicted = np.asarray(model.predict(z[hold_idx]), dtype=np.float64)
            if predicted.ndim == 1:
                predicted = predicted[:, None]
        residual[hold_idx] = c[hold_idx] - predicted
    if not np.isfinite(residual).all():
        median = np.nanmedian(residual, axis=0)
        residual = np.where(np.isfinite(residual), residual, median)
    return residual


MIN_RESIDUAL_VARIANCE_RATIO = 1e-6


def orthogonalized_score(
    candidate: np.ndarray,
    conditioning: np.ndarray,
    residual: np.ndarray,
    *,
    folds: int = 5,
    ridge_alpha: float = 1.0,
    seed: int = 20260722,
) -> float:
    scores = entry_scores(
        np.asarray(candidate, dtype=np.float64).reshape(len(residual), 1),
        ["candidate"], conditioning, residual, folds=folds,
        ridge_alpha=ridge_alpha, seed=seed)
    return scores["candidate"]


def entry_sample_weights(y: np.ndarray, mode: str) -> np.ndarray | None:
    y = np.asarray(y, dtype=np.float64)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if mode == "balanced" and n_pos > 0 and n_neg > 0:
        return np.where(y > 0, 0.5 / n_pos, 0.5 / n_neg) * len(y)
    if mode == "positive" and n_pos > 0:
        return (y > 0).astype(np.float64)
    return None


def _weighted_corr(a: np.ndarray, b: np.ndarray,
                   weights: np.ndarray | None) -> float:
    if weights is None:
        return _corr(a, b)
    w = np.asarray(weights, dtype=np.float64)
    total = w.sum()
    if total <= 0:
        return 0.0
    am = a - (w * a).sum() / total
    bm = b - (w * b).sum() / total
    denom = np.sqrt((w * am * am).sum() * (w * bm * bm).sum())
    return 0.0 if denom <= 0 else float((w * am * bm).sum() / denom)


def entry_scores(
    candidates: np.ndarray,
    names: list[str],
    conditioning: np.ndarray,
    residual: np.ndarray,
    *,
    folds: int = 5,
    ridge_alpha: float = 1.0,
    seed: int = 20260722,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    if candidates.ndim != 2 or candidates.shape[1] != len(names):
        raise ValueError("candidate matrix and name list disagree")
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    if not len(names):
        return {}
    residualized = cross_fitted_residualize(
        candidates, conditioning, folds=folds, ridge_alpha=ridge_alpha,
        seed=seed)
    r = np.asarray(residual, dtype=np.float64)
    candidate_var = np.var(np.asarray(candidates, dtype=np.float64), axis=0)
    residual_var = np.var(residualized, axis=0)
    scores: dict[str, float] = {}
    for j, name in enumerate(names):
        explained_away = (candidate_var[j] <= 0.0
                          or residual_var[j] < MIN_RESIDUAL_VARIANCE_RATIO
                          * candidate_var[j])
        scores[name] = (0.0 if explained_away
                        else _weighted_corr(residualized[:, j], r, weights))
    return scores


def bh_stepup_audit(
    abs_scores: dict[str, float],
    n_pos: int,
    *,
    alpha: float = 0.05,
    total_candidates: int | None = None,
) -> dict:
    ordered = sorted(abs_scores.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    P = int(total_candidates) if total_candidates else len(ordered)
    P = max(P, len(ordered), 1)
    sqrt_n = float(np.sqrt(max(int(n_pos), 1)))
    rows = []
    k_bh = 0
    for k, (name, score) in enumerate(ordered, start=1):
        level = 1.0 - float(alpha) * k / (2.0 * P)
        threshold = z_quantile(level) / sqrt_n
        passed = abs(score) >= threshold
        rows.append({"rank": k, "name": name, "abs_s": abs(float(score)),
                     "threshold": float(threshold), "passed": bool(passed)})
        if passed:
            k_bh = k
    return {"rule": "bh_stepup_audit_only", "alpha": float(alpha),
            "p_total": P, "n_pos": int(n_pos), "k_bh": int(k_bh),
            "table": rows}


def lite_v2_entry(
    candidates: np.ndarray,
    names: list[str],
    conditioning: np.ndarray,
    residual: np.ndarray,
    n_pos: int,
    *,
    incumbent: set[str] | None = None,
    alpha: float = 0.05,
    correlation_threshold: float = 0.95,
    folds: int = 5,
    ridge_alpha: float = 1.0,
    seed: int = 20260722,
    total_hypotheses: int | None = None,
    weights: np.ndarray | None = None,
) -> tuple[list[str], dict]:
    matrix = np.asarray(candidates, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[1] != len(names):
        raise ValueError("candidate matrix and name list disagree")
    incumbent = set(incumbent or ())
    coverage = np.isfinite(matrix).mean(axis=0) if names else np.array([])
    ranked = np.column_stack([
        rankdata(np.where(np.isfinite(matrix[:, j]), matrix[:, j],
                          np.nanmedian(matrix[:, j]) if np.isfinite(matrix[:, j]).any() else 0.0))
        for j in range(len(names))
    ]) if names else np.zeros((len(matrix), 0))
    corr = np.corrcoef(ranked, rowvar=False) if len(names) > 1 else np.eye(len(names))
    order = sorted(range(len(names)), key=lambda j: (
        0 if names[j] in incumbent else 1, -float(coverage[j]),
        len(names[j]), names[j]))
    kept: list[int] = []
    redundant: list[dict] = []
    for j in order:
        conflict = next((k for k in kept
                         if abs(float(corr[j, k])) >= correlation_threshold), None)
        if conflict is None:
            kept.append(j)
        else:
            redundant.append({"factor": names[j], "representative": names[conflict],
                              "abs_spearman": abs(float(corr[j, conflict]))})
    kept.sort()
    kept_names = [names[j] for j in kept]
    scores = entry_scores(matrix[:, kept], kept_names, conditioning, residual,
                          folds=folds, ridge_alpha=ridge_alpha, seed=seed,
                          weights=weights)
    pvals = {name: float(2.0 * norm.sf(abs(score) * np.sqrt(max(n_pos, 1))))
             for name, score in scores.items()}
    ordered = sorted(pvals, key=lambda name: (pvals[name], name))
    family = max(int(total_hypotheses or len(ordered)), len(ordered), 1)
    k_bh = 0
    for rank, name in enumerate(ordered, 1):
        if pvals[name] <= alpha * rank / family:
            k_bh = rank
    selected = ordered[:k_bh]
    return selected, {
        "rule": "forge_lite_v2_collinearity_then_bh",
        "correlation": {"method": "absolute_spearman",
                        "threshold": float(correlation_threshold),
                        "representative_rule": "incumbent>coverage>simplicity>name",
                        "representatives": kept_names, "redundant": redundant},
        "bh": {"alpha": float(alpha), "family_size": family, "k_bh": k_bh,
               "p_values": pvals},
        "first_pass_scores": scores, "selected": selected,
        "k_add": len(selected), "stop_reason": "bh",
    }


def label_free_representatives(
    candidates: np.ndarray,
    names: list[str],
    *,
    correlation_threshold: float = 0.95,
) -> tuple[list[str], dict]:
    matrix = np.asarray(candidates, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[1] != len(names):
        raise ValueError("candidate matrix and name list disagree")
    if not names:
        return [], {
            "method": "absolute_spearman",
            "threshold": float(correlation_threshold),
            "representatives": [], "redundant": [],
        }
    coverage = np.isfinite(matrix).mean(axis=0)
    ranked = np.column_stack([
        rankdata(np.where(
            np.isfinite(matrix[:, j]), matrix[:, j],
            np.nanmedian(matrix[:, j])
            if np.isfinite(matrix[:, j]).any() else 0.0))
        for j in range(len(names))
    ])
    corr = (np.corrcoef(ranked, rowvar=False)
            if len(names) > 1 else np.eye(len(names)))
    order = sorted(range(len(names)), key=lambda j: (
        -float(coverage[j]), len(names[j]), names[j]))
    kept: list[int] = []
    redundant: list[dict] = []
    for j in order:
        conflict = next((k for k in kept
                         if np.isfinite(corr[j, k])
                         and abs(float(corr[j, k]))
                         >= float(correlation_threshold)), None)
        if conflict is None:
            kept.append(j)
        else:
            redundant.append({
                "factor": names[j], "representative": names[conflict],
                "abs_spearman": abs(float(corr[j, conflict])),
            })
    kept.sort()
    representatives = [names[j] for j in kept]
    return representatives, {
        "method": "absolute_spearman",
        "threshold": float(correlation_threshold),
        "representative_rule": "coverage>simplicity>name",
        "representatives": representatives,
        "redundant": redundant,
    }


def global_pair_bh(
    pair_scores: dict[tuple[str, str], float],
    n_pos_by_seat: dict[str, int],
    *,
    q: float,
    unique_factor_cap: int | None = None,
) -> tuple[dict[str, list[str]], dict]:
    if not 0.0 < float(q) < 1.0:
        raise ValueError("BH q must be in (0, 1)")
    rows: list[dict] = []
    for (seat, factor), score in pair_scores.items():
        n_pos = max(int(n_pos_by_seat.get(str(seat), 0)), 1)
        p_value = float(2.0 * norm.sf(
            abs(float(score)) * np.sqrt(float(n_pos))))
        rows.append({
            "seat": str(seat), "factor": str(factor),
            "score": float(score), "n_pos": n_pos,
            "p_value": p_value,
        })
    rows.sort(key=lambda row: (
        row["p_value"], row["seat"], row["factor"]))
    family = max(len(rows), 1)
    k_bh = 0
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row["threshold"] = float(q) * rank / family
        row["passed_raw"] = bool(row["p_value"] <= row["threshold"])
        if row["passed_raw"]:
            k_bh = rank

    running = 1.0
    for rank in range(len(rows), 0, -1):
        row = rows[rank - 1]
        running = min(running, float(row["p_value"]) * family / rank)
        row["adjusted_p"] = float(min(running, 1.0))
    accepted = rows[:k_bh]
    factors_before_cap = sorted({row["factor"] for row in accepted})
    cap = (len(factors_before_cap) if unique_factor_cap is None
           else max(int(unique_factor_cap), 0))
    factor_order = sorted(
        factors_before_cap,
        key=lambda factor: (
            min(row["adjusted_p"] for row in accepted
                if row["factor"] == factor),
            factor,
        ))
    kept_factors = set(factor_order[:cap])
    selected_by_seat: dict[str, list[str]] = {}
    for row in accepted:
        row["kept_after_factor_cap"] = row["factor"] in kept_factors
        if row["factor"] in kept_factors:
            selected_by_seat.setdefault(row["seat"], []).append(row["factor"])
    return selected_by_seat, {
        "rule": "approximate_global_pair_bh",
        "q": float(q), "family_size": len(rows), "k_bh": int(k_bh),
        "unique_factor_cap": int(cap),
        "factors_before_cap": factors_before_cap,
        "factors_after_cap": factor_order[:cap],
        "selected_by_seat": selected_by_seat,
        "table": rows,
    }


def s4_per_seat_noise_floor(
    pair_scores: dict[tuple[str, str], float],
    n_pos_by_seat: dict[str, int],
    *,
    alpha: float = 0.05,
    unique_factor_cap: int | None = None,
) -> tuple[dict[str, list[str]], dict]:
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("noise-floor alpha must be in (0, 1)")
    rows: list[dict] = []
    for (seat, factor), score in pair_scores.items():
        n_pos = max(int(n_pos_by_seat.get(str(seat), 0)), 1)
        floor = entry_noise_floor(n_pos, float(alpha))
        margin = float(abs(float(score)) / floor) if floor > 0 else 0.0
        rows.append({
            "seat": str(seat), "factor": str(factor),
            "score": float(score), "n_pos": n_pos, "floor": float(floor),
            "passed_floor": bool(abs(float(score)) >= floor),
            "floor_margin": margin,
        })
    rows.sort(key=lambda row: (-row["floor_margin"], row["seat"],
                               row["factor"]))
    accepted = [row for row in rows if row["passed_floor"]]
    factors_before_cap = sorted({row["factor"] for row in accepted})
    cap = (len(factors_before_cap) if unique_factor_cap is None
           else max(int(unique_factor_cap), 0))
    factor_order = sorted(
        factors_before_cap,
        key=lambda factor: (
            -max(row["floor_margin"] for row in accepted
                 if row["factor"] == factor),
            factor,
        ))
    kept_factors = set(factor_order[:cap])
    selected_by_seat: dict[str, list[str]] = {}
    for row in accepted:
        row["kept_after_factor_cap"] = row["factor"] in kept_factors
        if row["factor"] in kept_factors:
            selected_by_seat.setdefault(row["seat"], []).append(row["factor"])
    return selected_by_seat, {
        "rule": "s4_per_seat_noise_floor",
        "alpha": float(alpha), "family_size": len(rows),
        "k_floor": len(accepted),
        "unique_factor_cap": int(cap),
        "factors_before_cap": factors_before_cap,
        "factors_after_cap": factor_order[:cap],
        "selected_by_seat": selected_by_seat,
        "table": rows,
    }


def greedy_entry(
    candidates: np.ndarray,
    names: list[str],
    conditioning: np.ndarray,
    residual: np.ndarray,
    n_pos: int,
    *,
    alpha: float = 0.05,
    k_cap: int = 12,
    folds: int = 5,
    ridge_alpha: float = 1.0,
    seed: int = 20260722,
    bh_total_candidates: int | None = None,
    weights: np.ndarray | None = None,
) -> tuple[list[str], dict]:
    names = [str(name) for name in names]
    floor = entry_noise_floor(n_pos, alpha)
    audit: dict = {
        "rule": "greedy_orthogonalized_residual_corr",
        "noise_floor": float(floor),
        "n_pos": int(n_pos),
        "alpha": float(alpha),
        "k_cap": int(k_cap),
        "pool_size": len(names),
        "steps": [],
        "stop_reason": None,
        "first_pass_scores": {},
        "bh_audit": None,
    }
    if not names:
        audit["stop_reason"] = "empty_pool"
        audit["bh_audit"] = bh_stepup_audit(
            {}, n_pos, alpha=alpha, total_candidates=bh_total_candidates)
        return [], audit
    matrix = np.asarray(candidates, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[1] != len(names):
        raise ValueError("candidate matrix and name list disagree")
    conditioning = np.asarray(conditioning, dtype=np.float64)
    if conditioning.ndim == 1:
        conditioning = conditioning[:, None]

    remaining = list(range(len(names)))
    z_current = conditioning
    selected: list[str] = []
    while remaining and len(selected) < int(k_cap):
        scores = entry_scores(
            matrix[:, remaining], [names[j] for j in remaining], z_current,
            residual, folds=folds, ridge_alpha=ridge_alpha, seed=seed,
            weights=weights)
        if not audit["first_pass_scores"]:
            audit["first_pass_scores"] = {
                name: float(value) for name, value in scores.items()}
            audit["bh_audit"] = bh_stepup_audit(
                {name: abs(value) for name, value in scores.items()},
                n_pos, alpha=alpha, total_candidates=bh_total_candidates)
        best_name, best_s = max(
            scores.items(), key=lambda kv: (abs(kv[1]), kv[0]))
        step = {
            "step": len(selected) + 1,
            "name": best_name,
            "s": float(best_s),
            "abs_s": abs(float(best_s)),
            "floor": float(floor),
            "n_remaining": len(remaining),
            "selected": bool(abs(best_s) >= floor),
        }
        audit["steps"].append(step)
        if abs(best_s) < floor:
            audit["stop_reason"] = "noise_floor"
            break
        index = names.index(best_name)
        selected.append(best_name)
        z_current = np.column_stack([z_current, matrix[:, index]]) \
            if z_current.shape[1] else matrix[:, index:index + 1]
        remaining.remove(index)
    if audit["stop_reason"] is None:
        audit["stop_reason"] = ("k_cap" if len(selected) >= int(k_cap)
                                else "pool_exhausted")
    audit["selected"] = list(selected)
    audit["k_add"] = len(selected)
    return selected, audit


def permutation_exit(
    y_disc: np.ndarray,
    base_pred_disc: np.ndarray,
    permuted_pred_disc: dict[str, np.ndarray],
    *,
    d_max: int = 2,
    tau_drop: float = 0.0,
) -> dict:
    y = np.asarray(y_disc, dtype=np.int8)
    if np.unique(y).size < 2:
        raise ValueError("permutation exit needs both classes on discovery rows")
    base = np.asarray(base_pred_disc, dtype=np.float64)
    if len(base) != len(y):
        raise ValueError("base prediction misaligned with discovery labels")
    base_auprc = float(average_precision_score(y, base))
    pi: dict[str, float] = {}
    for name, prediction in permuted_pred_disc.items():
        prediction = np.asarray(prediction, dtype=np.float64)
        if len(prediction) != len(y):
            raise ValueError(f"permuted prediction misaligned: {name}")
        pi[str(name)] = base_auprc - float(average_precision_score(y, prediction))
    below = sorted(((name, value) for name, value in pi.items()
                    if value <= float(tau_drop)),
                   key=lambda kv: (kv[1], kv[0]))
    drops = [name for name, _ in below[:max(0, int(d_max))]]
    return {
        "rule": "oof_permutation_importance_discovery",
        "base_auprc": base_auprc,
        "pi": {name: float(value) for name, value in pi.items()},
        "tau_drop": float(tau_drop),
        "d_max": int(d_max),
        "below_floor": [name for name, _ in below],
        "drops": drops,
        "k_drop": len(drops),
    }


def probe_permutation(n_rows: int, member: str, column: str,
                      base_seed: int) -> tuple[np.ndarray, int]:
    digest = hashlib.sha256(
        f"forge_v13_probe|{member}|{column}|{int(base_seed)}".encode("utf-8")
    ).digest()
    derived = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(derived)
    return rng.permutation(int(n_rows)), derived


@dataclass
class Blacklist:

    entries: dict[str, dict[str, dict]] = field(default_factory=dict)

    def add(self, model_key: str, names: list[str], *, round_index: int,
            pi: dict[str, float] | None = None) -> None:
        bucket = self.entries.setdefault(str(model_key), {})
        for name in names:
            bucket.setdefault(str(name), {
                "round": int(round_index),
                "pi": (None if pi is None or name not in pi
                       else float(pi[name])),
            })

    def blocked(self, model_key: str) -> set[str]:
        return set(self.entries.get(str(model_key), {}))

    def filter(self, model_key: str, names: list[str]) -> list[str]:
        blocked = self.blocked(model_key)
        return [name for name in names if name not in blocked]

    def to_dict(self) -> dict:
        return {"policy": "permanent_per_model", "entries": self.entries}

    @classmethod
    def from_dict(cls, payload: dict) -> "Blacklist":
        entries = {
            str(model): {str(name): dict(info)
                         for name, info in names.items()}
            for model, names in dict(payload.get("entries", {})).items()
        }
        return cls(entries=entries)
