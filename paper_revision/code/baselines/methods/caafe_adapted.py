from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from datasets import get_config
from evaluation import PairedBootstrapEvaluator
from finalization import finalize_feature_set
from llm_client import StructuredLLM
from protocol import FeatureSpec, apply_feature_spec, load_search_data
from search_utils import RunLedger, write_json


def run(
    dataset: str,
    output_dir: Path,
    *,
    repeat: int = 1,
    iterations: int = 10,
    features_per_block: int = 3,
    no_finalize: bool = False,
) -> dict:
    cfg = get_config(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger("CAAFE-adapted", dataset, repeat)
    data = load_search_data(cfg)
    evaluator = PairedBootstrapEvaluator(
        data,
        seeds=(301 + repeat * 10, 302 + repeat * 10, 303 + repeat * 10),
        n_bootstrap=40,
        bootstrap_seed=20263710 + repeat,
    )
    client = StructuredLLM(output_dir / "llm_calls.jsonl")
    current_specs: list[FeatureSpec] = []
    current_features = list(data.features)
    current_score = evaluator.score(current_features)
    initial_score = current_score
    tried: set[str] = set()
    history: list[dict] = []

    raw_columns = [column for column in data.train.columns if column != data.label][:80]
    sample = data.train[raw_columns[:12] + [data.label]].head(5).to_dict(orient="records")
    stats = []
    for column in raw_columns[:30]:
        series = data.train[column]
        if pd.api.types.is_numeric_dtype(series):
            stats.append(
                f"{column}: min={series.min(skipna=True):.4g}, "
                f"max={series.max(skipna=True):.4g}, missing={series.isna().mean():.3f}"
            )
        else:
            stats.append(f"{column}: categorical, unique={series.nunique(dropna=True)}")

    for iteration in range(1, iterations + 1):
        feedback = history[-1] if history else {"status": "initial program", "delta": None}
        guidance = f"""Follow the CAAFE context-aware code-block pattern.
Dataset statistics:
{chr(10).join(stats)}
Five example rows: {json.dumps(sample, ensure_ascii=False, default=str)}
Current search-validation AUPRC: {current_score.auprc}
Last block feedback: {json.dumps(feedback, ensure_ascii=False)}

Propose a coherent code block of pointwise features grounded in the dataset context. The
block will be kept only if its combined validation score improves. Do not make cosmetic
variants of previous features."""
        block = client.propose(
            dataset_description=data.meta["desc"],
            label=data.label,
            columns=raw_columns + [spec.name for spec in current_specs],
            existing_features=current_features,
            tried=tried,
            guidance=guidance,
            n=features_per_block,
            source="caafe_adapted",
        )
        ledger.proposals += features_per_block
        ledger.valid_programs += len(block)
        applied: list[FeatureSpec] = []
        for spec in block:
            tried.add(spec.name)
            if apply_feature_spec(data.train, data.validation, spec, label=data.label):
                applied.append(spec)
        if not applied:
            history.append({"iteration": iteration, "status": "invalid", "delta": 0.0})
            continue

        candidate_features = current_features + [spec.name for spec in applied]
        candidate_score = evaluator.score(candidate_features)
        delta = round(candidate_score.auprc - current_score.auprc, 7)
        kept = delta > 0
        if kept:
            current_specs.extend(applied)
            current_features = candidate_features
            current_score = candidate_score
            ledger.accepted += len(applied)
        history.append({
            "iteration": iteration,
            "status": "kept" if kept else "discarded",
            "delta": delta,
            "features": [spec.to_dict() for spec in applied],
            "search_val_auprc": candidate_score.auprc,
        })

    final_result = None if no_finalize else finalize_feature_set(cfg, current_specs).to_dict()
    write_json(output_dir / "split_manifest.json", asdict(data.manifest))
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "selected_features.json", [spec.to_dict() for spec in current_specs])
    if final_result is not None:
        write_json(output_dir / "final_result.json", final_result)
    summary = {
        "method": "CAAFE-adapted",
        "adapter": "official-bb65b4d-context-loop-common-auprc",
        "dataset": dataset,
        "repeat": repeat,
        "protocol_version": data.manifest.protocol_version,
        "iterations": iterations,
        "features_per_block": features_per_block,
        "initial_search_val": asdict(initial_score),
        "best_search_val": asdict(current_score),
        "n_selected_features": len(current_specs),
        "search_evaluator": evaluator.audit(),
        "llm": client.audit(),
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
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--features-per-block", type=int, default=3)
    parser.add_argument("--no-finalize", action="store_true")
    args = parser.parse_args()
    run(
        args.dataset,
        args.out,
        repeat=args.repeat,
        iterations=args.iterations,
        features_per_block=args.features_per_block,
        no_finalize=args.no_finalize,
    )


if __name__ == "__main__":
    main()

