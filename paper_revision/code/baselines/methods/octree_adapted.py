from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from datasets import get_config
from evaluation import PairedBootstrapEvaluator
from finalization import finalize_feature_set
from llm_client import FEATURE_SCHEMA, StructuredLLM
from protocol import FeatureSpec, apply_feature_spec, compile_pointwise_expression, design_matrix, load_search_data
from search_utils import RunLedger, write_json


@dataclass
class RuleTrial:
    spec: FeatureSpec
    column_index: int
    round_index: int
    search_val_auprc: float
    cart_text: str

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "column_index": self.column_index,
            "round_index": self.round_index,
            "search_val_auprc": self.search_val_auprc,
            "cart_text": self.cart_text,
        }


def _parse_one(response: dict, label: str, column_index: int, round_index: int) -> FeatureSpec | None:
    items = response.get("features", [])
    if not items:
        return None
    item = items[0]
    base_name = str(item.get("name", "generated_column")).strip().lower().replace(" ", "_")
    expression = str(item.get("python_expr", "")).strip()
    try:
        compile_pointwise_expression(expression, forbidden_columns=(label,))
    except Exception:
        return None
    return FeatureSpec(
        name=f"octree_c{column_index}_r{round_index}_{base_name}",
        expression=expression,
        family=str(item.get("family", "optimized_column")),
        source="octree_budget_matched",
        description=str(item.get("description", "")),
    )


def _cart_reasoning(data, features: list[str], sample_cap: int = 200_000) -> str:
    if len(data.train) > sample_cap:
        indices = np.linspace(0, len(data.train) - 1, sample_cap, dtype=np.int64)
        frame = data.train.iloc[indices]
    else:
        frame = data.train
    x = design_matrix(frame, features)
    y = frame[data.label].to_numpy(dtype=np.int8, copy=False)
    model = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
    model.fit(x, y)
    return export_text(model, feature_names=features, max_depth=4)


def run(
    dataset: str,
    output_dir: Path,
    *,
    repeat: int = 1,
    max_columns: int = 3,
    refinement_rounds: int = 24,
    paper_faithful: bool = False,
    no_finalize: bool = False,
) -> dict:
    if paper_faithful:
        max_columns = 6
        refinement_rounds = 50
    cfg = get_config(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger(
        "OCTree-paper" if paper_faithful else "OCTree-budget-matched",
        dataset,
        repeat,
    )
    data = load_search_data(cfg)
    evaluator = PairedBootstrapEvaluator(
        data,
        seeds=(401 + repeat * 10, 402 + repeat * 10, 403 + repeat * 10),
        n_bootstrap=40,
        bootstrap_seed=20264710 + repeat,
    )
    client = StructuredLLM(output_dir / "llm_calls.jsonl")
    selected_specs: list[FeatureSpec] = []
    current_features = list(data.features)
    current_score = evaluator.score(current_features)
    initial_score = current_score
    all_trials: list[RuleTrial] = []

    for column_index in range(1, max_columns + 1):
        trajectory: list[RuleTrial] = []
        for round_index in range(refinement_rounds):
            if round_index == 0:
                feedback = "No previous rule: propose a semantically useful new column and its rule."
            else:
                top = sorted(trajectory, key=lambda trial: trial.search_val_auprc)[-7:]
                feedback = json.dumps(
                    [
                        {
                            "rule": trial.spec.expression,
                            "score": trial.search_val_auprc,
                            "cart": trial.cart_text,
                        }
                        for trial in top
                    ],
                    ensure_ascii=False,
                )
            prompt = f"""Implement OCTree rule optimization for one new column.

Dataset: {data.meta['desc']}
Target (forbidden in code): {data.label}
Existing columns for this step: {', '.join(current_features[:160])}
The trajectory below is ordered by validation score. CART text is reasoning extracted
from training data with maximum depth four.
{feedback}

Return exactly one new rule that is materially different and aims to improve the best
score. The rule must be a pointwise numeric expression using df and NumPy/Pandas only;
no target, groupby, transform, rank, rolling, merge, apply, or lambda. Return only JSON."""
            response, _ = client.call_json(prompt, schema=FEATURE_SCHEMA)
            ledger.proposals += 1
            spec = _parse_one(response, data.label, column_index, round_index)
            if spec is None:
                continue
            ledger.valid_programs += 1
            if not apply_feature_spec(data.train, data.validation, spec, label=data.label):
                continue
            trial_features = current_features + [spec.name]
            score = evaluator.score(trial_features)
            cart = _cart_reasoning(data, trial_features)
            trial = RuleTrial(spec, column_index, round_index, score.auprc, cart)
            trajectory.append(trial)
            all_trials.append(trial)

        if not trajectory:
            break
        best = max(trajectory, key=lambda trial: trial.search_val_auprc)
        if best.search_val_auprc <= current_score.auprc:
            break
        selected_specs.append(best.spec)
        current_features.append(best.spec.name)
        current_score = evaluator.score(current_features)
        ledger.accepted += 1

    final_result = None if no_finalize else finalize_feature_set(cfg, selected_specs).to_dict()
    write_json(output_dir / "split_manifest.json", asdict(data.manifest))
    write_json(output_dir / "rule_trials.json", [trial.to_dict() for trial in all_trials])
    write_json(output_dir / "selected_features.json", [spec.to_dict() for spec in selected_specs])
    if final_result is not None:
        write_json(output_dir / "final_result.json", final_result)
    summary = {
        "method": ledger.method,
        "adapter": "official-a11c485-protocol-corrected",
        "dataset": dataset,
        "repeat": repeat,
        "protocol_version": data.manifest.protocol_version,
        "paper_faithful": paper_faithful,
        "max_columns": max_columns,
        "refinement_rounds": refinement_rounds,
        "cart_max_depth": 4,
        "initial_search_val": asdict(initial_score),
        "best_search_val": asdict(current_score),
        "n_selected_features": len(selected_specs),
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
    parser.add_argument("--max-columns", type=int, default=3)
    parser.add_argument("--refinement-rounds", type=int, default=24)
    parser.add_argument("--paper-faithful", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    args = parser.parse_args()
    run(
        args.dataset,
        args.out,
        repeat=args.repeat,
        max_columns=args.max_columns,
        refinement_rounds=args.refinement_rounds,
        paper_faithful=args.paper_faithful,
        no_finalize=args.no_finalize,
    )


if __name__ == "__main__":
    main()

