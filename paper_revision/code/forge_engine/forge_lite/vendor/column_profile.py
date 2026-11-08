from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_ROW_CAP = 200_000


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "na"
    return f"{value:.{digits}f}"


def build_column_profile(
    train: pd.DataFrame,
    *,
    label: str,
    features: list[str],
    cats: list[str],
    time_col: str | None = None,
    max_cats: int = 12,
    max_levels: int = 5,
    max_numeric: int = 25,
    max_chars: int = 4000,
) -> tuple[str, dict]:
    n = len(train)
    y = train[label].to_numpy(np.int8, copy=False)
    positive_rate = float(y.mean()) if n else float("nan")

    if time_col and time_col in train.columns:
        order = np.argsort(train[time_col].to_numpy(), kind="stable")
    else:
        order = np.arange(n)
    mid = n // 2
    early, late = order[:mid], order[mid:]

    lines = [
        f"Table profile ({n} rows, positive rate {positive_rate:.5f}).",
        "Target statistics below are split into the EARLY and LATE half of the "
        "fit window on purpose: a level whose rate moves between halves is one "
        "whose target encoding will look strong in-period and fail later.",
    ]
    stats: dict = {"rows": int(n), "positive_rate": round(positive_rate, 8),
                   "categorical": [], "numeric": [], "drift_flags": []}

    ranked_cats = sorted(
        (c for c in cats if c in train.columns),
        key=lambda c: -int(train[c].nunique(dropna=True)))[:max_cats]
    for col in ranked_cats:
        series = train[col]
        card = int(series.nunique(dropna=True))
        missing = float(series.isna().mean())
        counts = series.value_counts(dropna=True).head(max_levels)
        levels = []
        for level, support in counts.items():
            mask = (series == level).to_numpy()
            e = y[early][mask[early]]
            l = y[late][mask[late]]
            rate_e = float(e.mean()) if len(e) else float("nan")
            rate_l = float(l.mean()) if len(l) else float("nan")
            drift = (abs(rate_l - rate_e)
                     if np.isfinite(rate_e) and np.isfinite(rate_l)
                     else float("nan"))
            levels.append({"level": str(level)[:40], "support": int(support),
                           "rate_early": None if not np.isfinite(rate_e) else round(rate_e, 5),
                           "rate_late": None if not np.isfinite(rate_l) else round(rate_l, 5),
                           "drift": None if not np.isfinite(drift) else round(drift, 5)})
        worst = max((x["drift"] for x in levels if x["drift"] is not None),
                    default=None)
        stats["categorical"].append(
            {"column": col, "cardinality": card, "missing": round(missing, 4),
             "levels": levels, "max_level_drift": worst})
        shown = ", ".join(
            f"{x['level']}(n={x['support']}, {_fmt(x['rate_early'], 4)}->"
            f"{_fmt(x['rate_late'], 4)})" for x in levels)
        lines.append(
            f"- {col}: categorical, {card} levels, {missing:.1%} missing. "
            f"Top levels early->late positive rate: {shown}")
        if worst is not None and positive_rate > 0 and worst > positive_rate:
            stats["drift_flags"].append(col)

    numeric = [c for c in features
               if c in train.columns and c not in set(cats)
               and pd.api.types.is_numeric_dtype(train[c])][:max_numeric]
    for col in numeric:
        v = pd.to_numeric(train[col], errors="coerce")
        stats["numeric"].append(
            {"column": col, "missing": round(float(v.isna().mean()), 4),
             "min": None if v.isna().all() else round(float(v.min()), 4),
             "median": None if v.isna().all() else round(float(v.median()), 4),
             "max": None if v.isna().all() else round(float(v.max()), 4)})
    if numeric:
        lines.append(
            "- numeric columns (min/median/max): " + "; ".join(
                f"{d['column']} {_fmt(d['min'])}/{_fmt(d['median'])}/{_fmt(d['max'])}"
                for d in stats["numeric"]))

    if stats["drift_flags"]:
        lines.append(
            "- WARNING, levels in these columns move their positive rate "
            "between halves by more than the base rate: "
            + ", ".join(stats["drift_flags"])
            + ". Target-encoding deltas built on them are the known failure "
              "mode; prefer counts, ratios, gaps and within-entity history.")

    return "\n".join(lines)[:max_chars], stats


def temporal_screen(
    values: np.ndarray,
    y: np.ndarray,
    *,
    order: np.ndarray | None = None,
    alpha_z: float = 1.645,
    decay_floor: float = 0.5,
) -> dict:
    v = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(v)
    idx = np.arange(n) if order is None else np.asarray(order)
    mid = n // 2
    early, late = idx[:mid], idx[mid:]

    def corr(rows):
        a, b = v[rows], y[rows]
        finite = np.isfinite(a)
        a, b = a[finite], b[finite]
        if len(a) < 2 or np.std(a) <= 0 or np.std(b) <= 0:
            return float("nan"), 0
        return float(np.corrcoef(a, b)[0, 1]), int(b.sum())

    r_early, _ = corr(early)
    r_late, pos_late = corr(late)
    floor_late = (alpha_z / np.sqrt(pos_late)) if pos_late > 0 else float("inf")

    ok_finite = np.isfinite(r_early) and np.isfinite(r_late)
    same_sign = ok_finite and (r_early == 0 or np.sign(r_early) == np.sign(r_late))
    clears = ok_finite and abs(r_late) >= floor_late
    retains = ok_finite and abs(r_early) > 0 and (
        abs(r_late) >= decay_floor * abs(r_early))
    passed = bool(same_sign and clears and retains)
    return {
        "r_early": None if not np.isfinite(r_early) else round(r_early, 6),
        "r_late": None if not np.isfinite(r_late) else round(r_late, 6),
        "n_pos_late": int(pos_late),
        "floor_late": None if not np.isfinite(floor_late) else round(floor_late, 6),
        "same_sign": bool(same_sign),
        "clears_late_floor": bool(clears),
        "retains_magnitude": bool(retains),
        "decay_floor": decay_floor,
        "passed": passed,
    }
