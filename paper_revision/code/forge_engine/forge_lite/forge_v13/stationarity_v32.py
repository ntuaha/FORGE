from __future__ import annotations

import numpy as np
import pandas as pd

import os as _os
PSI_DROP = float(_os.environ.get("FORGE_PSI_MAX", "0.25"))
ADV_AUC_DROP = 0.5 + float(_os.environ.get("FORGE_ADV_BAND", "0.15"))
PSI_WARN = 0.10


def psi(early: np.ndarray, late: np.ndarray, bins: int = 10) -> float:
    early = early[np.isfinite(early)]
    late = late[np.isfinite(late)]
    if len(early) < 50 or len(late) < 50:
        return 0.0
    edges = np.unique(np.quantile(early, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_frac = np.histogram(early, edges)[0] / len(early)
    l_frac = np.histogram(late, edges)[0] / len(late)
    e_frac = np.clip(e_frac, 1e-4, None)
    l_frac = np.clip(l_frac, 1e-4, None)
    return float(np.sum((l_frac - e_frac) * np.log(l_frac / e_frac)))


def adversarial_auc(early: np.ndarray, late: np.ndarray) -> float:
    early = early[np.isfinite(early)]
    late = late[np.isfinite(late)]
    if len(early) < 50 or len(late) < 50:
        return 0.5
    values = np.concatenate([early, late])
    ranks = pd.Series(values).rank(method="average").to_numpy()
    r_late = float(ranks[len(early):].sum())
    n_e, n_l = len(early), len(late)
    u = r_late - n_l * (n_l + 1) / 2.0
    return float(u / (n_e * n_l))


def screen_bundle(*, factors: list[str], train: pd.DataFrame,
                  apply: pd.DataFrame) -> tuple[list[str], dict]:
    kept: list[str] = []
    rows: list[dict] = []
    for name in factors:
        if name not in train.columns or name not in apply.columns:
            rows.append({"factor": name, "verdict": "missing_column"})
            continue
        early = pd.to_numeric(train[name], errors="coerce").to_numpy(
            np.float64)
        late = pd.to_numeric(apply[name], errors="coerce").to_numpy(
            np.float64)
        p = psi(early, late)
        a = adversarial_auc(early, late)
        drifted = bool(p > PSI_DROP or abs(a - 0.5) > (ADV_AUC_DROP - 0.5))
        if _os.environ.get("FORGE_ABL_NO_SCREEN", "") == "1":
            drifted = False
        rows.append({"factor": name, "psi": round(p, 4),
                     "adv_auc": round(a, 4),
                     "warn": bool(PSI_WARN < p <= PSI_DROP),
                     "verdict": "dropped_drift" if drifted else "kept"})
        if not drifted:
            kept.append(name)
    audit = {"rule": {"psi_drop": PSI_DROP, "adv_auc_drop": ADV_AUC_DROP,
                      "frames": "inner_train_vs_search_val"},
             "n_in": len(factors), "n_kept": len(kept),
             "n_dropped": len(factors) - len(kept), "table": rows}
    return kept, audit
