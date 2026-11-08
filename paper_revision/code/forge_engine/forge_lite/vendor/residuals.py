from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROFILE_SUBSAMPLE = 200_000
PROFILE_SEED = 20260716


def _subsample(n_rows: int, cap: int = PROFILE_SUBSAMPLE) -> np.ndarray:
    if n_rows <= cap:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(PROFILE_SEED)
    rows = rng.choice(n_rows, size=cap, replace=False)
    rows.sort()
    return rows


def _recall_at_top(y: np.ndarray, scores: np.ndarray, fraction: float) -> float:
    k = max(1, int(round(len(scores) * fraction)))
    threshold_index = np.argpartition(scores, -k)[-k:]
    positives = float(y.sum())
    if positives <= 0:
        return float("nan")
    return float(y[threshold_index].sum() / positives)


def build_residual_profile(
    train: pd.DataFrame,
    oof: np.ndarray,
    disagreement: np.ndarray | None,
    *,
    label: str,
    features: list[str],
    cats: list[str],
    max_chars: int = 2600,
) -> tuple[str, dict]:
    y_full = train[label].to_numpy(dtype=np.int8, copy=False)
    oof = np.asarray(oof, dtype=np.float64)
    assert len(oof) == len(train), "OOF must align with inner-train rows"

    oof_auprc = float(average_precision_score(y_full, oof))
    positive_rate = float(y_full.mean())
    top_fraction = min(max(positive_rate, 0.005), 0.10)
    recall_top = _recall_at_top(y_full, oof, top_fraction)

    rows = _subsample(len(train))
    y = y_full[rows].astype(np.float64)
    p = oof[rows]
    residual = y - p
    abs_residual = np.abs(residual)

    stats: dict = {
        "oof_auprc": round(oof_auprc, 6),
        "inner_train_positive_rate": round(positive_rate, 8),
        "recall_at_top_fraction": {
            "fraction": round(top_fraction, 4),
            "recall": None if np.isnan(recall_top) else round(recall_top, 4),
        },
        "profile_rows": int(len(rows)),
        "mean_abs_residual": round(float(abs_residual.mean()), 6),
        "fn_like_share_of_positives": round(
            float((p[y == 1] < np.median(p)).mean()) if (y == 1).any() else float("nan"), 4,
        ),
    }

    lines = [
        "Quantified residual profile of the CURRENT AutoGluon ensemble "
        "(out-of-fold predictions on training rows; residual = label - oof_probability):",
        f"- OOF AUPRC {oof_auprc:.4f}; positive rate {positive_rate:.5f}; "
        f"recall of positives inside the top {top_fraction:.1%} of OOF scores: "
        f"{recall_top:.3f} (deficit {1.0 - recall_top:.3f}).",
        f"- Mean |residual| {abs_residual.mean():.4f}; share of positives scored "
        f"below the median OOF score: {stats['fn_like_share_of_positives']}.",
    ]

    numeric = [
        f for f in features
        if f in train.columns and pd.api.types.is_numeric_dtype(train[f])
    ]
    correlations: list[tuple[str, float]] = []
    for column in numeric:
        values = pd.to_numeric(train[column].iloc[rows], errors="coerce")
        filled = values.fillna(values.median()).to_numpy(dtype=np.float64)
        if np.nanstd(filled) <= 0:
            continue
        with np.errstate(invalid="ignore"):
            r = float(np.corrcoef(filled, residual)[0, 1])
        if np.isfinite(r):
            correlations.append((column, r))
    correlations.sort(key=lambda item: -abs(item[1]))
    top_corr = correlations[:8]
    stats["residual_correlation_top"] = [
        {"column": c, "r": round(r, 4)} for c, r in top_corr
    ]
    if top_corr:
        lines.append("- Columns most correlated with the residual (r): " + ", ".join(
            f"{c}={r:+.3f}" for c, r in top_corr))

    decile_rows: list[str] = []
    decile_stats: list[dict] = []
    for column, _ in top_corr[:6]:
        values = pd.to_numeric(train[column].iloc[rows], errors="coerce")
        try:
            bins = pd.qcut(values, 10, duplicates="drop")
        except (ValueError, IndexError):
            continue
        grouped = pd.DataFrame(
            {"bin": bins, "res": residual, "y": y},
        ).groupby("bin", observed=True).agg(
            res_mean=("res", "mean"), pos_rate=("y", "mean"), n=("res", "size"),
        )
        grouped = grouped[grouped["n"] >= 50]
        if grouped.empty:
            continue
        worst = grouped.reindex(
            grouped["res_mean"].abs().sort_values(ascending=False).index,
        ).head(2)
        for interval, row in worst.iterrows():
            decile_rows.append(
                f"{column} in {interval}: mean residual {row.res_mean:+.4f}, "
                f"positive rate {row.pos_rate:.4f}, n={int(row.n)}"
            )
            decile_stats.append({
                "column": column, "bin": str(interval),
                "res_mean": round(float(row.res_mean), 5),
                "pos_rate": round(float(row.pos_rate), 5), "n": int(row.n),
            })
    stats["worst_numeric_segments"] = decile_stats[:8]
    if decile_rows:
        lines.append("- Most mispredicted numeric segments:\n  " +
                     "\n  ".join(decile_rows[:8]))

    slice_rows: list[tuple[float, str, dict]] = []
    min_support = max(100, int(0.002 * len(rows)))
    for column in cats:
        if column not in train.columns or column == label:
            continue
        series = train[column].iloc[rows].astype("string").fillna("<NA>")
        counts = series.value_counts().head(12)
        for value, count in counts.items():
            if count < min_support:
                continue
            mask = (series == value).to_numpy(dtype=bool)
            res_mean = float(residual[mask].mean())
            slice_rows.append((abs(res_mean),
                               f"{column}={value}: mean residual {res_mean:+.4f}, n={int(count)}",
                               {"column": column, "value": str(value),
                                "res_mean": round(res_mean, 5), "n": int(count)}))
    slice_rows.sort(key=lambda item: -item[0])
    stats["worst_categorical_slices"] = [item[2] for item in slice_rows[:6]]
    if slice_rows:
        lines.append("- Most mispredicted categorical slices:\n  " +
                     "\n  ".join(item[1] for item in slice_rows[:6]))

    if disagreement is not None and len(disagreement) == len(train):
        d = np.asarray(disagreement, dtype=np.float64)[rows]
        with np.errstate(invalid="ignore"):
            r_dis = float(np.corrcoef(d, abs_residual)[0, 1]) if d.std() > 0 else float("nan")
        stats["disagreement"] = {
            "mean": round(float(d.mean()), 6),
            "p90": round(float(np.quantile(d, 0.9)), 6),
            "corr_with_abs_residual": None if not np.isfinite(r_dis) else round(r_dis, 4),
        }
        lines.append(
            f"- Leaderboard-member OOF disagreement: mean std {d.mean():.4f}, "
            f"p90 {np.quantile(d, 0.9):.4f}, corr(|residual|, disagreement) "
            f"{r_dis if np.isfinite(r_dis) else float('nan'):+.3f} — rows where members disagree "
            "are where the ensemble is uncertain."
        )

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 15] + "\n[truncated]"
    return text, stats
