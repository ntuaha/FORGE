from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, roc_auc_score

from datasets import get_config
from evaluation import PairedBootstrapEvaluator
from finalization import finalize_feature_set
from llm_client import StructuredLLM
from protocol import FeatureSpec, apply_feature_spec, compile_pointwise_expression, load_search_data
from search_utils import RunLedger, write_json


RULE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "classes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "class_label": {"type": "integer"},
                    "rules": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "python_expr": {"type": "string"},
                            },
                            "required": ["name", "description", "python_expr"],
                        },
                    },
                },
                "required": ["class_label", "rules"],
            },
        }
    },
    "required": ["classes"],
}


class NonNegativeRuleModel(nn.Module):
    def __init__(self, widths: list[int]) -> None:
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.ones(width, 1) / max(width, 1)) for width in widths
        ])

    def forward(self, matrices: list[torch.Tensor]) -> torch.Tensor:
        class_scores = [
            matrix @ torch.clamp(weight, min=0)
            for matrix, weight in zip(matrices, self.weights)
        ]
        return torch.cat(class_scores, dim=1)


def _balanced_shot_indices(y: np.ndarray, shot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    selected: list[np.ndarray] = []
    base = shot // len(classes)
    remainder = shot % len(classes)
    for index, value in enumerate(classes):
        candidates = np.flatnonzero(y == value)
        count = min(len(candidates), base + (1 if index < remainder else 0))
        selected.append(rng.choice(candidates, size=count, replace=False))
    rows = np.concatenate(selected)
    rng.shuffle(rows)
    return rows


def _parse_rules(response: dict, label: str, trial: int) -> dict[int, list[FeatureSpec]]:
    parsed: dict[int, list[FeatureSpec]] = {0: [], 1: []}
    for class_item in response.get("classes", []):
        class_label = int(class_item.get("class_label", -1))
        if class_label not in parsed:
            continue
        for index, item in enumerate(class_item.get("rules", [])[:10]):
            expression = str(item.get("python_expr", "")).strip()
            base_name = str(item.get("name", "rule")).strip().lower().replace(" ", "_")
            try:
                compile_pointwise_expression(expression, forbidden_columns=(label,))
            except Exception:
                continue
            parsed[class_label].append(FeatureSpec(
                name=f"featllm_t{trial}_c{class_label}_r{index}_{base_name}",
                expression=expression,
                family="binary_rule",
                source=f"featllm_trial_{trial}_class_{class_label}",
                description=str(item.get("description", "")),
            ))
    return parsed


def _train_rule_model(
    data,
    rules: dict[int, list[FeatureSpec]],
    shot_indices: np.ndarray,
) -> tuple[NonNegativeRuleModel, np.ndarray]:
    train_matrices: list[torch.Tensor] = []
    val_matrices: list[torch.Tensor] = []
    for class_label in (0, 1):
        names = [spec.name for spec in rules[class_label]]
        train_values = data.train.loc[shot_indices, names].to_numpy(dtype=np.float32)
        val_values = data.validation[names].to_numpy(dtype=np.float32)
        train_matrices.append(torch.from_numpy(np.nan_to_num(train_values)))
        val_matrices.append(torch.from_numpy(np.nan_to_num(val_values)))
    labels = torch.from_numpy(
        data.train.loc[shot_indices, data.label].to_numpy(dtype=np.int64),
    )
    model = NonNegativeRuleModel([matrix.shape[1] for matrix in train_matrices])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        optimizer.zero_grad()
        logits = model(train_matrices)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        if float((logits.argmax(dim=1) == labels).float().mean()) >= 1.0:
            break
    with torch.no_grad():
        probabilities = functional.softmax(model(val_matrices), dim=1)[:, 1].cpu().numpy()
    return model, probabilities


def run(
    dataset: str,
    output_dir: Path,
    *,
    repeat: int = 1,
    num_queries: int = 5,
    shot: int = 10,
    no_finalize: bool = False,
) -> dict:
    cfg = get_config(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger("FeatLLM-adapted", dataset, repeat)
    data = load_search_data(cfg)
    evaluator = PairedBootstrapEvaluator(
        data,
        seeds=(501 + repeat * 10, 502 + repeat * 10, 503 + repeat * 10),
        n_bootstrap=40,
        bootstrap_seed=20265710 + repeat,
    )
    client = StructuredLLM(output_dir / "llm_calls.jsonl")
    y_train = data.train[data.label].to_numpy(dtype=np.int8, copy=False)
    shot_indices = _balanced_shot_indices(y_train, shot, seed=20260710 + repeat)
    visible_columns = [column for column in data.train.columns if column != data.label][:80]
    examples = data.train.loc[shot_indices, visible_columns[:15] + [data.label]].to_dict(orient="records")

    trials: list[dict] = []
    trial_predictions: list[np.ndarray] = []
    all_specs: list[FeatureSpec] = []
    for trial in range(num_queries):
        prompt = f"""Implement FeatLLM class-specific binary-rule generation.

Dataset: {data.meta['desc']}
Target: {data.label}
Available columns: {', '.join(visible_columns)}
Balanced few-shot examples: {json.dumps(examples, ensure_ascii=False, default=str)}

For class 0 and class 1, produce ten distinct conditions characteristic of that class.
Each python_expr must be a vectorized boolean or 0/1 pointwise expression using df; it
must not reference the target and must not use groupby, transform, rank, rolling, merge,
apply, or lambda. Return only the requested JSON."""
        response, _ = client.call_json(prompt, schema=RULE_SCHEMA)
        ledger.proposals += 20
        rules = _parse_rules(response, data.label, trial)
        if not rules[0] or not rules[1]:
            continue
        valid = True
        for class_specs in rules.values():
            for spec in class_specs:
                if not apply_feature_spec(data.train, data.validation, spec, label=data.label):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        specs = rules[0] + rules[1]
        ledger.valid_programs += len(specs)
        _, predictions = _train_rule_model(data, rules, shot_indices)
        trial_predictions.append(predictions)
        all_specs.extend(specs)
        trials.append({
            "trial": trial,
            "n_class0_rules": len(rules[0]),
            "n_class1_rules": len(rules[1]),
            "few_shot_val_auprc": round(float(average_precision_score(
                data.validation[data.label], predictions,
            )), 7),
        })

    if trial_predictions:
        ensemble = np.mean(np.stack(trial_predictions), axis=0)
        faithful = {
            "shot": shot,
            "num_ensembles": len(trial_predictions),
            "search_val_auprc": round(float(average_precision_score(
                data.validation[data.label], ensemble,
            )), 7),
            "search_val_auc": round(float(roc_auc_score(
                data.validation[data.label], ensemble,
            )), 7),
        }
    else:
        faithful = None

    adapted_features = data.features + [spec.name for spec in all_specs]
    adapted_score = evaluator.score(adapted_features) if all_specs else evaluator.score(data.features)
    final_result = None if no_finalize else finalize_feature_set(cfg, all_specs).to_dict()
    write_json(output_dir / "split_manifest.json", asdict(data.manifest))
    write_json(output_dir / "trials.json", trials)
    write_json(output_dir / "selected_features.json", [spec.to_dict() for spec in all_specs])
    if final_result is not None:
        write_json(output_dir / "final_result.json", final_result)
    summary = {
        "method": "FeatLLM-adapted",
        "adapter": "official-d60f685-rules-plus-full-data-transfer",
        "dataset": dataset,
        "repeat": repeat,
        "protocol_version": data.manifest.protocol_version,
        "faithful_few_shot_component": faithful,
        "adapted_full_data_search_val": asdict(adapted_score),
        "n_selected_features": len(all_specs),
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
    parser.add_argument("--num-queries", type=int, default=5)
    parser.add_argument("--shot", type=int, default=10)
    parser.add_argument("--no-finalize", action="store_true")
    args = parser.parse_args()
    run(
        args.dataset,
        args.out,
        repeat=args.repeat,
        num_queries=args.num_queries,
        shot=args.shot,
        no_finalize=args.no_finalize,
    )


if __name__ == "__main__":
    main()

