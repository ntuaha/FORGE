from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

LEGACY_RULE = "legacy_ag"
COMPLEXITY_GATE = "complexity_gate_1se"


def paired_bootstrap_se(y_holdout: np.ndarray, pred_a: np.ndarray,
                        pred_b: np.ndarray, repeats: int = 1000,
                        seed: int = 20260802) -> dict:
    y = np.asarray(y_holdout).astype(int)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    n = y.shape[0]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(int(repeats)):
        idx = rng.integers(0, n, size=n)
        y_s = y[idx]
        if y_s.sum() == 0 or y_s.sum() == y_s.shape[0]:
            continue
        deltas.append(average_precision_score(y_s, b[idx])
                      - average_precision_score(y_s, a[idx]))
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size < 2:
        return {"se": float("nan"), "n_effective": int(deltas.size),
                "repeats": int(repeats), "seed": int(seed),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    return {
        "se": float(deltas.std(ddof=1)),
        "n_effective": int(deltas.size),
        "repeats": int(repeats),
        "seed": int(seed),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
    }


def ag_dystack_decision(search_fit: dict, y_holdout: np.ndarray,
                        rule: str = LEGACY_RULE, k: float = 1.0,
                        bootstrap_repeats: int = 1000,
                        bootstrap_seed: int = 20260802) -> dict:
    candidates = []
    predictions = {}
    for record in search_fit["records"]:
        candidates.append({
            "member": record["member"],
            "affected": record["layer"] == "l2",
            "score_val": float(record["oof_auprc"]),
            "score_holdout": float(average_precision_score(
                y_holdout, record["apply"])),
        })
        predictions[record["member"]] = np.asarray(record["apply"], dtype=float)
    for member, affected, oof, pred in (
        ("weighted_ensemble_l2", False, search_fit["we_l2_oof"],
         search_fit["we_l2_prediction"]),
        ("weighted_ensemble_l3", True, search_fit["we_l3_oof"],
         search_fit["we_l3_prediction"]),
    ):
        candidates.append({
            "member": member,
            "affected": affected,
            "score_val": float(average_precision_score(
                search_fit["train_labels"], oof)),
            "score_holdout": float(average_precision_score(y_holdout, pred)),
        })
        predictions[member] = np.asarray(pred, dtype=float)

    non_affected = max(
        (row for row in candidates if not row["affected"]),
        key=lambda row: row["score_val"])
    affected = max(
        (row for row in candidates if row["affected"]),
        key=lambda row: row["score_val"])

    stacked_overfitting = (
        non_affected["score_val"] < affected["score_val"]
        and non_affected["score_holdout"] >= affected["score_holdout"]
    )
    legacy_stack_enabled = not bool(stacked_overfitting)

    delta = affected["score_holdout"] - non_affected["score_holdout"]
    result = {
        "rule": str(rule),
        "best_non_affected": non_affected,
        "best_affected": affected,
        "stacked_overfitting": bool(stacked_overfitting),
        "legacy_stack_enabled": bool(legacy_stack_enabled),
        "delta_holdout": float(delta),
        "candidate_scores": candidates,
    }

    if rule == LEGACY_RULE:
        result["stack_enabled"] = bool(legacy_stack_enabled)
        return result

    if rule != COMPLEXITY_GATE:
        raise ValueError(f"unknown dystack rule: {rule!r}")

    boot = paired_bootstrap_se(
        y_holdout, predictions[non_affected["member"]],
        predictions[affected["member"]],
        repeats=bootstrap_repeats, seed=bootstrap_seed)
    se = boot["se"]
    enabled = bool(np.isfinite(se) and delta > float(k) * se)
    result.update({
        "k": float(k),
        "se_bootstrap": se,
        "bootstrap": boot,
        "decision_margin_ratio": (float(delta / se)
                                  if np.isfinite(se) and se > 0 else None),
        "stack_enabled": enabled,
        "flipped_vs_legacy": bool(enabled != legacy_stack_enabled),
    })
    return result


def holdout_prediction_table(search_fit: dict) -> dict:
    table = {record["member"]: np.asarray(record["apply"], dtype=float)
             for record in search_fit["records"]}
    table["weighted_ensemble_l2"] = np.asarray(
        search_fit["we_l2_prediction"], dtype=float)
    table["weighted_ensemble_l3"] = np.asarray(
        search_fit["we_l3_prediction"], dtype=float)
    return table
