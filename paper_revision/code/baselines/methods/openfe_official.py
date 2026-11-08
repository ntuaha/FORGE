from __future__ import annotations

import argparse
import re
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from datasets import get_config
from evaluation import PairedBootstrapEvaluator
from finalization import finalize_feature_set
from protocol import FeatureSpec, apply_feature_spec, load_search_data
from search_utils import RunLedger, write_json


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "OpenFE"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from openfe import OpenFE, tree_to_formula


def _apply_column_policy(input_columns: list[str], cfg: dict, *,
                         policy: str = "all",
                         max_columns: int | None = None) -> tuple[list[str], dict]:
    pattern = cfg.get("anonymous_col_pattern")
    audit = {
        "policy": policy,
        "n_input_before": len(input_columns),
        "anonymous_col_pattern": pattern,
        "max_input_columns": max_columns,
    }
    kept = list(input_columns)

    if policy == "named":
        if not pattern:
            raise SystemExit(
                f"--column-policy named requires anonymous_col_pattern in the "
                f"config for dataset {cfg['name']!r}")
        regex = re.compile(pattern)

        def base_name(column: str) -> str:
            for suffix in ("_oof_te", "_freq", "_log"):
                if column.endswith(suffix):
                    return column[: -len(suffix)]
            return column

        kept = [c for c in kept if not regex.match(base_name(c))]
        if len(kept) < 4:
            raise SystemExit(
                f"named-column policy leaves {len(kept)} column(s) for "
                f"{cfg['name']!r} ({kept}); this dataset is essentially all "
                "anonymized, so the restriction does not apply -- report the "
                "official 'all' run instead")
    elif policy != "all":
        raise SystemExit(f"unknown column policy: {policy!r}")

    if max_columns is not None and len(kept) > int(max_columns):
        kept = kept[: int(max_columns)]
        audit["truncated"] = True

    audit["n_input_after"] = len(kept)
    audit["dropped"] = audit["n_input_before"] - len(kept)
    audit["input_columns"] = kept
    return kept, audit


def run(
    dataset: str,
    output_dir: Path,
    *,
    repeat: int = 1,
    top_k: int = 20,
    n_jobs: int = 8,
    no_finalize: bool = False,
    column_policy: str = "all",
    max_input_columns: int | None = None,
) -> dict:
    if dataset == "talkingdata":
        raise RuntimeError(
            "Full official OpenFE enumeration on 184.9M rows is preregistered as omitted; "
            "run the representative large-data comparison instead."
        )
    cfg = get_config(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger("OpenFE-official", dataset, repeat)
    data = load_search_data(cfg)
    evaluator = PairedBootstrapEvaluator(
        data,
        seeds=(601 + repeat * 10, 602 + repeat * 10, 603 + repeat * 10),
        n_bootstrap=40,
        bootstrap_seed=20266710 + repeat,
    )
    base_score = evaluator.score(data.features)

    input_columns = list(dict.fromkeys(data.features + data.meta["cats"]))
    input_columns, column_policy_audit = _apply_column_policy(
        input_columns, cfg, policy=column_policy,
        max_columns=max_input_columns)
    search_train = data.train[input_columns].copy()
    categorical = [column for column in data.meta["cats"] if column in search_train.columns]
    for column in categorical:
        search_train[column] = search_train[column].astype("category")

    train_index = val_index = None
    if data.meta["time_col"]:
        cut = int(len(search_train) * 0.8)
        train_index = search_train.index[:cut]
        val_index = search_train.index[cut:]

    temporary = output_dir / "openfe_work.feather"
    engine = OpenFE()
    nodes = engine.fit(
        data=search_train,
        label=data.train[[data.label]].copy(),
        task="classification",
        train_index=train_index,
        val_index=val_index,
        categorical_features=categorical,
        metric="auc",
        feature_boosting=True,
        n_data_blocks=8,
        min_candidate_features=2000,
        n_jobs=n_jobs,
        seed=repeat,
        tmp_save_path=str(temporary),
        verbose=True,
    )

    specs: list[FeatureSpec] = []
    for index, node in enumerate(nodes[:top_k]):
        formula = tree_to_formula(node)
        spec = FeatureSpec(
            name=f"openfe_{index}",
            expression=formula,
            family="openfe_23_operator_space",
            source="openfe_official_3e3bdf9",
            description="Official OpenFE-ranked formula; global statistics fit on training only.",
            executor="openfe",
        )
        if apply_feature_spec(data.train, data.validation, spec, label=data.label):
            specs.append(spec)
    ledger.proposals = len(nodes)
    ledger.valid_programs = len(specs)
    ledger.accepted = len(specs)
    selected_score = evaluator.score(data.features + [spec.name for spec in specs])
    final_result = None if no_finalize else finalize_feature_set(cfg, specs).to_dict()

    write_json(output_dir / "split_manifest.json", asdict(data.manifest))
    write_json(output_dir / "selected_features.json", [spec.to_dict() for spec in specs])
    if final_result is not None:
        write_json(output_dir / "final_result.json", final_result)
    summary = {
        "method": ("OpenFE-official" if column_policy == "all"
                   else f"OpenFE-official-{column_policy}-columns"),
        "adapter": "official-3e3bdf9-inductive-global-operators",
        "column_policy": column_policy_audit,
        "dataset": dataset,
        "repeat": repeat,
        "protocol_version": data.manifest.protocol_version,
        "operator_contract": "all 23 official operators",
        "internal_metric": "auc (AUPRC unavailable in upstream API)",
        "feature_boosting": True,
        "base_search_val": asdict(base_score),
        "selected_search_val": asdict(selected_score),
        "n_official_ranked_nodes": len(nodes),
        "n_selected_features": len(specs),
        "search_evaluator": evaluator.audit(),
        "run": ledger.audit(),
        "final": final_result,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument(
        "--column-policy", choices=("all", "named"), default="all")
    parser.add_argument("--max-input-columns", type=int)
    args = parser.parse_args()
    run(
        args.dataset,
        args.out,
        repeat=args.repeat,
        top_k=args.top_k,
        n_jobs=args.n_jobs,
        no_finalize=args.no_finalize,
        column_policy=args.column_policy,
        max_input_columns=args.max_input_columns,
    )


if __name__ == "__main__":
    main()

