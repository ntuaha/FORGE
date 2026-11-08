from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from forge_v13 import run_forge_v13 as E
from llm_forgeag import POINTWISE_SCHEMA, ForgeAgLLM


ORDERS = (
    (1, "unary", (
        "single-column transforms: bounded rescalings, guarded logs, "
        "calendar/periodic decompositions of a timestamp, missingness "
        "indicators, or a monotone reshaping of one raw column")),
    (2, "binary", (
        "two-column combinations: guarded ratios, differences, products of "
        "bounded terms, consistency/mismatch comparisons between two fields, "
        "or one field normalized by another")),
    (3, "ternary", (
        "three-column combinations: a ratio whose denominator is itself a "
        "combination, a difference of two ratios, a two-way normalization, "
        "or a three-field consistency check")),
    (4, "row-series shape (any arity)", (
        "first identify SIBLING COLUMN GROUPS from the names -- numbered "
        "sequences such as x1..x6 form a within-row time series or vector; "
        "then assemble whole-shape summaries over one group per feature: "
        "concentration (HHI = sum of squared shares, guarded by total + "
        "1.0), entropy of shares, coefficient of variation, linear trend "
        "slope with symmetric weights ordered most-recent first, momentum, "
        "run-lengths or counts of a condition across the group, and a "
        "peer-normalized magnitude; if the schema has no sibling group, "
        "fall back to the strongest cross-field consistency shapes")),
    (5, "diversity outside the tree's comfort zone (any arity)", (
        "the evidence block lists what the tree ensemble ALREADY exploits; "
        "propose mechanisms strictly outside that list: unused or low-gain "
        "columns, whole-group shapes over sibling columns, and cross-field "
        "consistency checks the listed pairs do not cover")),
)


def interaction_card(train: pd.DataFrame, label: str,
                     max_rows: int = 60000) -> str:
    import os
    from itertools import combinations

    probe_kind = os.environ.get("V34_PROBE_KIND", "lgbm")
    frame = train.iloc[:max_rows]
    y = pd.to_numeric(frame[label], errors="coerce").fillna(0).astype(int)
    X = pd.DataFrame(index=frame.index)
    columns = []
    for column in frame.columns:
        if column == label:
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            X[column] = pd.to_numeric(series, errors="coerce")
        else:
            X[column] = pd.factorize(series.astype("string"))[0]
        columns.append(column)
    if not columns or y.nunique() < 2:
        return "interaction card unavailable (degenerate frame)"

    top_single: list[tuple[str, float]] = []
    top_pairs: list[tuple[str, str]] = []
    if probe_kind == "xgb":
        import xgboost as xgb
        probe = xgb.XGBClassifier(
            n_estimators=150, max_depth=8, learning_rate=0.1,
            tree_method="hist", n_jobs=8, random_state=20260819,
            verbosity=0)
        probe.fit(X[columns].fillna(0.0), y)
        tdf = probe.get_booster().trees_to_dataframe()
        tdf = tdf[tdf["Feature"] != "Leaf"]
        gains = tdf.groupby("Feature")["Gain"].sum().to_dict()
        pair_gain = {}
        for _, tree in tdf.groupby("Tree"):
            used = tree.groupby("Feature")["Gain"].sum()
            names = list(used.sort_values(ascending=False).index[:4])
            for a, b in combinations(sorted(names), 2):
                pair_gain[(a, b)] = pair_gain.get((a, b), 0.0) + min(
                    used[a], used[b])
        total = max(sum(gains.values()), 1e-9)
        top_single = sorted(((k, v / total) for k, v in gains.items()),
                            key=lambda kv: -kv[1])[:10]
        top_pairs = [p for p, _ in sorted(pair_gain.items(),
                                          key=lambda kv: -kv[1])[:12]]
    elif probe_kind == "cat":
        import catboost as cb
        probe = cb.CatBoostClassifier(
            iterations=150, learning_rate=0.1, depth=8, thread_count=8,
            random_seed=20260819, verbose=False,
            allow_writing_files=False)
        pool = cb.Pool(X[columns].fillna(0.0), y)
        probe.fit(pool)
        fi = probe.get_feature_importance(pool)
        total = max(float(sum(fi)), 1e-9)
        top_single = sorted(zip(columns, (v / total for v in fi)),
                            key=lambda kv: -kv[1])[:10]
        inter = probe.get_feature_importance(type="Interaction")
        top_pairs = [(columns[int(i)], columns[int(j)])
                     for i, j, _ in inter[:12]]
    elif probe_kind == "nn":
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        sub = min(len(X), 20000)
        Xs, ys = X[columns].iloc[:sub], y.iloc[:sub]
        probe = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60,
                          random_state=20260819))
        probe.fit(Xs, ys)
        imp = permutation_importance(
            probe, Xs.iloc[-4000:], ys.iloc[-4000:], n_repeats=3,
            random_state=20260819, scoring="average_precision")
        total = max(float(imp.importances_mean.sum()), 1e-9)
        top_single = sorted(zip(columns,
                                (v / total for v in imp.importances_mean)),
                            key=lambda kv: -kv[1])[:10]
        top_pairs = []
    else:
        import lightgbm as lgb
        probe = lgb.LGBMClassifier(
            n_estimators=150, num_leaves=31, learning_rate=0.1,
            min_child_samples=30, n_jobs=8, verbose=-1,
            random_state=20260819)
        probe.fit(X[columns], y)
        booster = probe.booster_
        gains = dict(zip(columns, booster.feature_importance("gain")))
        pair_gain = {}
        for tree in booster.dump_model()["tree_info"]:
            used = {}

            def walk(node):
                if "split_feature" in node:
                    used[node["split_feature"]] = (
                        used.get(node["split_feature"], 0.0)
                        + float(node.get("split_gain", 0.0)))
                    walk(node["left_child"])
                    walk(node["right_child"])

            walk(tree["tree_structure"])
            names = sorted(used, key=used.get, reverse=True)[:4]
            for a, b in combinations(sorted(names), 2):
                key = (columns[a], columns[b])
                pair_gain[key] = pair_gain.get(key, 0.0) + min(
                    used[a], used[b])
        total = max(sum(gains.values()), 1e-9)
        top_single = sorted(((k, v / total) for k, v in gains.items()),
                            key=lambda kv: -kv[1])[:10]
        top_pairs = [p for p, _ in sorted(pair_gain.items(),
                                          key=lambda kv: -kv[1])[:12]]

    lines = [f"Model-native evidence ({probe_kind} probe): the columns "
             "and column pairs below are ALREADY heavily exploited by "
             "this arm's own model family.  Factors that restate them "
             "add nothing.  Spend your entire proposal budget on "
             "mechanisms NOT on this list: unused or low-gain columns, "
             "whole-group shapes (concentration, volatility, trends, "
             "run-lengths across sibling columns), and cross-field "
             "consistency checks the listed pairs do not cover.",
             "Columns the model already uses (importance share):"]
    for name, share in top_single:
        lines.append(f"- {name}: {share:.1%}")
    if top_pairs:
        lines.append("Column pairs the model already combines internally "
                     "(do NOT propose factors that restate these pairs):")
        for a, b in top_pairs:
            lines.append(f"- {a} x {b}")
    return "\n".join(lines)


def _schema_block(train: pd.DataFrame, label: str, limit: int = 160) -> str:
    lines = []
    for column in list(train.columns)[:limit]:
        if str(column) == label:
            continue
        series = train[column]
        kind = ("numeric" if pd.api.types.is_numeric_dtype(series)
                else "categorical/other")
        try:
            n_unique = int(series.nunique(dropna=True))
        except Exception:
            n_unique = -1
        null_frac = float(series.isna().mean())
        detail = f"{kind}, n_unique={n_unique}, null_frac={null_frac:.3f}"
        if pd.api.types.is_numeric_dtype(series):
            finite = pd.to_numeric(series, errors="coerce")
            if finite.notna().any():
                detail += (f", min={float(finite.min()):.4g}"
                           f", median={float(finite.median()):.4g}"
                           f", max={float(finite.max()):.4g}")
        lines.append(f"- {column}: {detail}")
    return "\n".join(lines)


def _prompt(*, order: int, kind: str, guidance: str, n: int,
            dataset_description: str, label: str, schema_block: str,
            already: list[str], model_evidence: str = "") -> str:
    avoid = ("\nNames already proposed (do not repeat or restate): "
             + ", ".join(already[:120])) if already else ""
    return f"""You are proposing engineered features for a tabular \
classification model. You have full autonomy: no search procedure will \
refine your ideas, no statistical screen will rank them, and no feedback \
round will follow. Every factor you return that compiles is handed \
unchanged to all seven models and trained on. Choose accordingly.

Dataset: {dataset_description}
Target column (FORBIDDEN in every expression): {label}

Columns available (from the training partition only):
{schema_block}

{model_evidence}
{avoid}

Return exactly {n} distinct {kind} (order-{order}) features. \
{guidance.capitalize()}.

Every expression must:
- be a single Python expression over df, using only df['column'], np.*, \
pd.to_numeric, and vectorized Series methods;
- be POINTWISE: no groupby, transform, rank, rolling, expanding, merge, \
join, apply, or lambda;
- never reference the target column {label!r};
- guard every division and logarithm (e.g. denominator + 1.0, \
np.log1p(np.maximum(x, 0.0))), and tolerate NaN;
- use a new snake_case name prefixed 'free{order}_' and carry a one-sentence \
reason naming the mechanism you believe it captures.

Output only the requested JSON object."""


def propose_free(dataset: str, out_dir: Path, cfg: dict, *,
                 per_order: int = 12, cap: int = 30,
                 stub_llm: bool = False,
                 dataset_description: str | None = None) -> dict:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    l1_library, _ = E.assert_frozen_graph(cfg)
    dataset_cfg = E.apply_outer_split_override(E.get_config(dataset), cfg)
    label = str(dataset_cfg["label"]).lower()

    raw_inner, raw_search, split = E._load_raw_partitions(
        dataset_cfg, purpose="search")
    train, apply, features, _ = E.prepare_e1(raw_inner, raw_search, dataset_cfg)
    representation_audit = E.select_and_apply_representation(
        raw_inner, raw_search, train, apply, features, dataset_cfg, cfg)
    del raw_inner, raw_search

    time_values = None
    if dataset_cfg.get("time_col"):
        time_col = str(dataset_cfg["time_col"]).lower()
        if time_col in train.columns:
            time_values = pd.to_numeric(
                train[time_col], errors="coerce").to_numpy(np.float64)

    description = dataset_description or str(
        dataset_cfg.get("description") or dataset)
    schema_block = _schema_block(train, label)
    model_evidence = interaction_card(train, label)

    llm = ForgeAgLLM(
        out_dir / "llm_calls.jsonl",
        model=(os.environ.get("FORGE_LLM_MODEL", "").strip()
               or str(cfg["llm"].get("model", "gpt-5.5"))),
        max_calls=int(cfg["llm"].get("max_calls", 40)),
        stub=bool(stub_llm))

    accepted: list[E.FeatureSpec] = []
    proposed_names: list[str] = []
    trace: list[dict] = []
    from concurrent.futures import ThreadPoolExecutor
    def _call(order, kind, guidance):
        return llm.call_json(
            f"free_order{order}",
            _prompt(order=order, kind=kind, guidance=guidance, n=per_order,
                    dataset_description=description, label=label,
                    schema_block=schema_block, already=[],
                    model_evidence=(model_evidence if order == 5 else "")),
            POINTWISE_SCHEMA)
    with ThreadPoolExecutor(max_workers=len(ORDERS)) as _pool:
        _futures = {order: _pool.submit(_call, order, kind, guidance)
                    for order, kind, guidance in ORDERS}
    _parsed_by_order = {order: future.result()
                        for order, future in _futures.items()}
    for order, kind, guidance in ORDERS:
        parsed = _parsed_by_order[order]
        rows = []
        for item in parsed.get("features", []):
            name = str(item.get("name", "")).strip().lower().replace(" ", "_")
            expression = str(item.get("python_expr", "")).strip()
            row = {"order": order, "name": name, "expression": expression,
                   "description": str(item.get("description", "")),
                   "outcome": None}
            rows.append(row)
            if not name or not expression:
                row["outcome"] = "empty"
                continue
            proposed_names.append(name)
            if name in {spec.name for spec in accepted}:
                row["outcome"] = "duplicate_name"
                continue
            if len(accepted) >= int(cap):
                row["outcome"] = "over_cap"
                continue
            spec = E.FeatureSpec(
                name=name, expression=expression,
                family=str(item.get("family", f"free_order{order}")),
                source="v25_0_free_llm",
                description=str(item.get("description", "")),
                executor="pointwise")
            try:
                ok = E.materialize_spec(
                    spec=spec, train=train, apply=apply, label=label,
                    time_values=time_values)
            except Exception as exc:
                row["outcome"] = f"materialize_error: {type(exc).__name__}"
                continue
            if not ok:
                row["outcome"] = "materialize_failed"
                continue
            column = pd.to_numeric(train[spec.name], errors="coerce")
            if not np.isfinite(column.to_numpy(np.float64)).any():
                row["outcome"] = "all_non_finite"
                train.drop(columns=[spec.name], inplace=True, errors="ignore")
                apply.drop(columns=[spec.name], inplace=True, errors="ignore")
                continue
            if float(column.nunique(dropna=True)) < 2:
                row["outcome"] = "constant"
                train.drop(columns=[spec.name], inplace=True, errors="ignore")
                apply.drop(columns=[spec.name], inplace=True, errors="ignore")
                continue
            row["outcome"] = "accepted"
            accepted.append(spec)
        trace.append({"order": order, "requested": per_order,
                      "returned": len(rows), "proposals": rows})
        E.log(f"free-LLM order {order}: {len(rows)} returned, "
              f"{sum(1 for r in rows if r['outcome'] == 'accepted')} accepted "
              f"(running total {len(accepted)}/{cap})")

    names = [spec.name for spec in accepted]
    plan = {
        "l1_factor_surfaces": {key: list(names) for key, _, _ in l1_library},
        "l2_rule": "features_plus_stack",
        "accepted_llm_specs": {spec.name: spec.to_dict() for spec in accepted},
        "stack_graph": None,
        "accepted_rounds": 0,
        "stop_reason": "v25_0_free_llm_no_search",
        "s4_depth3": None,
    }
    E.write_json(out_dir / "model_factor_plan.json", plan)
    audit = {
        "arm": "V25_0_free_llm",
        "dataset": dataset,
        "label": label,
        "representation": str(representation_audit.get("choice", "e1")),
        "split": split.to_dict() if hasattr(split, "to_dict") else None,
        "selection": {
            "ruler_fusion": False, "greedy_entry": False, "noise_floor": False,
            "shadow": False, "permutation_exit": False, "bh": False,
            "round_gate": False, "rounds": 0, "per_seat_targeting": False,
            "note": ("no statistical selection of any kind; the label "
                     "firewall (AST whitelist + forbidden label column + "
                     "shift>=1 aggregates) is retained for correctness"),
        },
        "per_order_request": per_order,
        "factor_cap": int(cap),
        "n_accepted": len(accepted),
        "accepted": names,
        "trace": trace,
        "llm": llm.audit(),
        "wall_sec": round(time.time() - started, 2),
    }
    E.write_json(out_dir / "free_llm_audit.json", audit)
    E.log(f"V25_0 free-LLM proposer: {len(accepted)}/{cap} factors kept in "
          f"{audit['wall_sec']:.1f}s, {llm.audit().get('calls')} LLM calls -> "
          f"{out_dir / 'model_factor_plan.json'}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(
        )
    parser.add_argument("dataset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--outer-split-seed", type=int)
    parser.add_argument("--l1", default=None)
    parser.add_argument("--l2", default=None)
    parser.add_argument("--oof-folds", type=int)
    parser.add_argument("--representation",
                        choices=("raw", "identity", "e1", "union"))
    parser.add_argument("--per-order", type=int, default=12)
    parser.add_argument("--cap", type=int, default=30)
    parser.add_argument("--stub-llm", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(E.CONFIG_PATH.read_text(encoding="utf-8"))
    E.apply_device_override(cfg, args.device)
    E.apply_graph_override(cfg, args.l1, args.l2)
    if args.seed is not None:
        cfg["engine"]["seed"] = int(args.seed)
    if args.outer_split_seed is not None:
        cfg["experiment"]["outer_split_seed_override"] = int(
            args.outer_split_seed)
    if args.oof_folds is not None:
        cfg["engine"]["bag_folds"] = int(args.oof_folds)
    E._LITE.clear()
    if args.representation:
        E._LITE["representation"] = str(args.representation)
    cfg["engine"]["lite"] = dict(E._LITE)
    propose_free(args.dataset, args.out, cfg,
                 per_order=int(args.per_order), cap=int(args.cap),
                 stub_llm=bool(args.stub_llm))


if __name__ == "__main__":
    main()
