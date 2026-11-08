from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_chronological_blocks(y: np.ndarray, weights: list[int]) -> list[np.ndarray]:
    y = np.asarray(y)
    if not weights or any(int(w) <= 0 for w in weights):
        raise ValueError("block weights must be positive")
    boundaries = np.rint(
        np.cumsum([0, *weights]) / sum(weights) * len(y)).astype(int)
    blocks = [np.arange(boundaries[i], boundaries[i + 1], dtype=np.int64)
              for i in range(len(weights))]
    for index, block in enumerate(blocks, 1):
        if not len(block) or np.unique(y[block]).size != 2:
            raise ValueError(f"gate block {index} must contain both classes")
    joined = np.concatenate(blocks)
    if not np.array_equal(joined, np.arange(len(y))):
        raise AssertionError("gate blocks are not exhaustive and disjoint")
    return blocks


def max_weight_delta(left: dict, right: dict) -> float:
    keys = set(left) | set(right)
    return max((abs(float(left.get(k, 0)) - float(right.get(k, 0)))
                for k in keys), default=0.0)


def compare_v9(candidate: dict, reference: dict, candidate_plan: dict,
               reference_plan: dict, contract: dict) -> dict:
    checks = {
        "auprc": abs(float(candidate["final_auprc"])
                     - float(reference["final_auprc"]))
                 <= float(contract["auprc_tolerance"]),
        "auc": abs(float(candidate["final_auc"])
                   - float(reference["final_auc"]))
               <= float(contract["auc_tolerance"]),
        "l1_count": int(candidate["n_l1"]) == int(contract["l1_models"]),
        "l2_count": int(candidate["n_l2"]) == int(contract["l2_models"]),
        "top6": int(candidate["top_factors_per_model"])
                == int(contract["top_k_per_model"]),
        "factor_bank_14": int(candidate["n_factor_bank"])
                          == int(reference["n_factor_bank"]) == 14,
        "llm_accepts": int(candidate["llm_gate_accepts"])
                       == int(reference["llm_gate_accepts"]) == 0,
        "factor_plan_exact": candidate_plan == reference_plan,
        "stack_graph": candidate_plan.get("stack_graph")
                       == contract["stack_graph"],
        "we_l2_weights": max_weight_delta(
            candidate["final_we_l2_weights"],
            reference["final_we_l2_weights"])
            <= float(contract["weight_tolerance"]),
        "we_l3_weights": max_weight_delta(
            candidate["final_we_l3_weights"],
            reference["final_we_l3_weights"])
            <= float(contract["weight_tolerance"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def validate_receipt(receipt_path: Path, expected_hashes: dict[str, str]) -> dict:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("passed") is not True:
        raise RuntimeError("V12 experiment requires a passing parity receipt")
    actual = receipt.get("locked_hashes", {})
    stale = {key: {"receipt": actual.get(key), "current": value}
             for key, value in expected_hashes.items()
             if actual.get(key) != value}
    if stale:
        raise RuntimeError(f"stale V9 parity receipt: {stale}")
    return receipt
