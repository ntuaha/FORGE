#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "baselines"))
from datasets import get_config
from protocol import FeatureSpec, apply_feature_specs, load_outer_partitions, prepare_e0, prepare_e1

METHODS = ["openfe", "caafe", "featllm", "llmfe", "octree"]
SEEDS = [11, 12, 13]
FIELDS = {"name", "expression", "family", "source", "description", "executor"}


def learner(kind, seed, threads=8):
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                                  subsample=0.8, colsample_bytree=0.8, n_jobs=threads, random_state=seed,
                                  verbosity=-1, deterministic=True, force_col_wise=True)
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, min_child_weight=1.0,
                                 colsample_bytree=1.0, tree_method="hist", n_jobs=threads, random_state=seed,
                                 verbosity=0)
    import catboost as cb
    return cb.CatBoostClassifier(n_estimators=400, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
                                 grow_policy="SymmetricTree", thread_count=threads, random_seed=seed,
                                 verbose=False, allow_writing_files=False)


def matrix(frame, cols):
    m = (np.column_stack([pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64) for c in cols])
         if cols else np.zeros((len(frame), 0)))
    m[~np.isfinite(m)] = np.nan
    return m


def materialize(train, test, specs, label):
    applied = []
    for d in specs:
        try:
            applied += apply_feature_specs(train, test, [FeatureSpec(**{k: v for k, v in d.items() if k in FIELDS})],
                                           label=label)
        except Exception:
            pass
    return applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--what", default="baselines,control")
    ap.add_argument("--learners", default="lgbm,xgb,cat")
    ap.add_argument("--baseline-runs", type=Path, default=HERE.parent / "runs" / "baselines")
    ap.add_argument("--forge-runs", type=Path, default=HERE.parent / "runs" / "main")
    ap.add_argument("--out", type=Path, default=HERE.parent / "scores")
    a = ap.parse_args()
    for ds in a.datasets.split(","):
        cfg = get_config(ds)
        label = str(cfg["label"]).lower()
        otr, ote, _ = load_outer_partitions(cfg)
        e0_tr, e0_te, base, _ = prepare_e0(otr, ote, cfg)
        e1_tr, e1_te, _, _ = prepare_e1(otr, ote, cfg)
        y_tr, y_te = e0_tr[label].to_numpy(np.int8), e0_te[label].to_numpy(np.int8)
        arms = [("raw", None, [], "e0")]
        if "baselines" in a.what:
            for m in METHODS:
                for rep in sorted((a.baseline_runs / ds / m).glob("r*")):
                    p = rep / "selected_features.json"
                    if p.exists():
                        arms.append((f"{m}_{rep.name}", None, json.loads(p.read_text()), "e0"))
        if "control" in a.what:
            for lr in a.learners.split(","):
                p = a.forge_runs / ds / lr / "seed11" / "r0_plan" / "model_factor_plan.json"
                if not p.exists():
                    p = a.forge_runs / ds / "frozensplit_seed11" / f"v34_{lr}_proposal" / "model_factor_plan.json"
                if p.exists():
                    specs = list(json.loads(p.read_text()).get("accepted_llm_specs", {}).values())
                    arms.append(("control_s11", lr, specs, "e1"))
        out = a.out / ds
        out.mkdir(parents=True, exist_ok=True)
        with (out / "scores.jsonl").open("a") as fh:
            for name, only_lr, specs, frame in arms:
                tr, te = (e0_tr.copy(), e0_te.copy()) if frame == "e0" else (e1_tr.copy(), e1_te.copy())
                applied = [c for c in materialize(tr, te, specs, label) if c not in base]
                X_tr = np.column_stack([matrix(e0_tr, base), matrix(tr, applied)])
                X_te = np.column_stack([matrix(e0_te, base), matrix(te, applied)])
                for lr in ([only_lr] if only_lr else a.learners.split(",")):
                    for s in SEEDS:
                        m = learner(lr, s)
                        m.fit(X_tr, y_tr)
                        ap_ = float(average_precision_score(y_te, m.predict_proba(X_te)[:, 1]))
                        fh.write(json.dumps(dict(dataset=ds, arm=name, learner=lr, seed=s, ap=ap_,
                                                 n_specs=len(specs), n_applied=len(applied))) + "\n")
                        print(ds, name, lr, s, f"{ap_:.4f}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()
