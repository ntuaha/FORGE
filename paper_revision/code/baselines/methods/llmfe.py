from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from datasets import get_config
from evaluation import PairedBootstrapEvaluator, SearchScore
from finalization import finalize_feature_set
from llm_client import StructuredLLM
from protocol import FeatureSpec, apply_feature_spec, compile_pointwise_expression, load_search_data
from search_utils import RunLedger, write_json


PROGRAM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "programs": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reasoning": {"type": "string"},
                    "features": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "family": {"type": "string"},
                                "description": {"type": "string"},
                                "python_expr": {"type": "string"},
                            },
                            "required": ["name", "family", "description", "python_expr"],
                        },
                    },
                },
                "required": ["reasoning", "features"],
            },
        }
    },
    "required": ["programs"],
}


@dataclass
class Program:
    specs: list[FeatureSpec]
    score: SearchScore
    island: int
    evaluation_index: int

    def serializable(self) -> dict:
        return {
            "specs": [spec.to_dict() for spec in self.specs],
            "score": asdict(self.score),
            "island": self.island,
            "evaluation_index": self.evaluation_index,
        }


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = (values - np.max(values)) / max(temperature, 1e-6)
    exp = np.exp(np.clip(scaled, -50, 50))
    return exp / exp.sum()


def _sample_examples(
    island: list[Program],
    rng: np.random.Generator,
    k: int = 2,
) -> list[Program]:
    clusters: dict[float, list[Program]] = {}
    for program in island:
        clusters.setdefault(round(program.score.auprc, 6), []).append(program)
    signatures = np.array(list(clusters), dtype=float)
    temperature = 0.1 * (1.0 - (len(island) % 30_000) / 30_000)
    probabilities = _softmax(signatures, temperature)
    chosen: list[Program] = []
    for _ in range(min(k, len(clusters))):
        signature = float(rng.choice(signatures, p=probabilities))
        candidates = clusters[signature]
        lengths = np.array([max(1, len(item.specs)) for item in candidates], dtype=float)
        length_probs = _softmax(-lengths, 1.0)
        chosen.append(candidates[int(rng.choice(len(candidates), p=length_probs))])
    return sorted(chosen, key=lambda program: program.score.auprc)


def _parse_programs(
    response: dict,
    label: str,
    tried_names: set[str],
    max_programs: int,
) -> list[list[FeatureSpec]]:
    parsed: list[list[FeatureSpec]] = []
    for program_index, item in enumerate(response.get("programs", [])[:max_programs]):
        specs: list[FeatureSpec] = []
        for feature_index, raw in enumerate(item.get("features", [])):
            name = str(raw.get("name", "")).strip().lower().replace(" ", "_")
            expression = str(raw.get("python_expr", "")).strip()
            if not name or not expression:
                continue
            unique_name = f"llmfe_{program_index}_{feature_index}_{name}"
            if unique_name in tried_names:
                continue
            try:
                compile_pointwise_expression(expression, forbidden_columns=(label,))
            except Exception:
                continue
            specs.append(FeatureSpec(
                name=unique_name,
                expression=expression,
                family=str(raw.get("family", "program")),
                source="llmfe_official_adapter",
                description=str(raw.get("description", "")),
            ))
        if specs:
            parsed.append(specs)
    return parsed


def run(
    dataset: str,
    output_dir: Path,
    *,
    repeat: int = 1,
    max_evaluations: int = 20,
    no_finalize: bool = False,
) -> dict:
    cfg = get_config(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = RunLedger("LLM-FE", dataset, repeat)
    data = load_search_data(cfg)
    evaluator = PairedBootstrapEvaluator(
        data,
        seeds=(101 + repeat * 10, 102 + repeat * 10, 103 + repeat * 10),
        n_bootstrap=40,
        bootstrap_seed=20261710 + repeat,
    )
    base_score = evaluator.score(data.features)
    base = Program([], base_score, island=-1, evaluation_index=0)
    islands: list[list[Program]] = [
        [Program([], base_score, island=index, evaluation_index=0)] for index in range(3)
    ]
    all_programs: list[Program] = [base]
    rng = np.random.default_rng(20260710 + repeat)
    py_rng = random.Random(20260710 + repeat)
    client = StructuredLLM(output_dir / "llm_calls.jsonl")
    tried_names: set[str] = set()
    evaluations = 0
    base_visible_columns = [
        column for column in data.train.columns if column != data.label
    ][:120]

    while evaluations < max_evaluations:
        island_id = py_rng.randrange(3)
        examples = _sample_examples(islands[island_id], rng, k=2)
        example_text = json.dumps(
            [
                {
                    "validation_auprc": program.score.auprc,
                    "features": [spec.to_dict() for spec in program.specs],
                }
                for program in examples
            ],
            ensure_ascii=False,
        )
        prompt = f"""You are evolving complete multi-feature transformation programs for
binary tabular prediction.

Dataset: {data.meta['desc']}
Target (forbidden in code): {data.label}
Available columns: {', '.join(base_visible_columns)}

These programs were sampled from one evolutionary island and are ordered from lower to
higher validation AUPRC:
{example_text}

Generate exactly three new, distinct programs. Each program may add several complementary
numeric features. Use only pointwise df['column'] expressions with NumPy/Pandas; no target,
groupby, transform, rank, rolling, merge, apply, or lambda. Every expression must be
standalone over the available columns above; do not reference generated feature names from
an example. Do not merely rename an example.
Return only the requested JSON."""
        response, _ = client.call_json(prompt, schema=PROGRAM_SCHEMA)
        remaining = min(3, max_evaluations - evaluations)
        programs = _parse_programs(response, data.label, tried_names, remaining)
        ledger.proposals += remaining
        ledger.valid_programs += len(programs)
        if not programs:
            break

        for specs in programs:
            applied: list[FeatureSpec] = []
            for spec in specs:
                tried_names.add(spec.name)
                if apply_feature_spec(data.train, data.validation, spec, label=data.label):
                    applied.append(spec)
            if not applied:
                continue
            features = data.features + [spec.name for spec in applied]
            score = evaluator.score(features)
            evaluations += 1
            program = Program(applied, score, island_id, evaluations)
            islands[island_id].append(program)
            all_programs.append(program)
            if evaluations >= max_evaluations:
                break

    best = max(all_programs, key=lambda program: program.score.auprc)
    final_result = None if no_finalize else finalize_feature_set(cfg, best.specs).to_dict()
    write_json(output_dir / "split_manifest.json", asdict(data.manifest))
    write_json(output_dir / "programs.json", [program.serializable() for program in all_programs])
    write_json(output_dir / "selected_features.json", [spec.to_dict() for spec in best.specs])
    if final_result is not None:
        write_json(output_dir / "final_result.json", final_result)

    summary = {
        "method": "LLM-FE",
        "adapter": "official-e9a58ca-protocol-corrected",
        "dataset": dataset,
        "repeat": repeat,
        "protocol_version": data.manifest.protocol_version,
        "official_structure": {
            "num_islands": 3,
            "functions_per_prompt": 2,
            "samples_per_prompt": 3,
            "max_evaluations": max_evaluations,
            "cluster_temperature_initial": 0.1,
            "cluster_temperature_period": 30000,
        },
        "base_search_val": asdict(base_score),
        "best_search_val": asdict(best.score),
        "n_program_evaluations": evaluations,
        "n_selected_features": len(best.specs),
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
    parser.add_argument("--max-evaluations", type=int, default=20)
    parser.add_argument("--no-finalize", action="store_true")
    args = parser.parse_args()
    run(
        args.dataset,
        args.out,
        repeat=args.repeat,
        max_evaluations=args.max_evaluations,
        no_finalize=args.no_finalize,
    )


if __name__ == "__main__":
    main()
