from __future__ import annotations

import json
import random
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from protocol import FeatureSpec


def error_profile(
    validation: pd.DataFrame,
    predictions: np.ndarray,
    features: list[str],
    label: str,
    top_k: int = 6,
) -> str:
    y = validation[label].to_numpy(dtype=np.int8, copy=False)
    precision, recall, thresholds = precision_recall_curve(y, predictions)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    threshold = float(thresholds[int(np.argmax(f1[:-1]))]) if len(thresholds) else 0.5
    predicted = predictions >= threshold
    masks = {
        "FP": predicted & (y == 0),
        "TP": predicted & (y == 1),
        "FN": (~predicted) & (y == 1),
        "TN": (~predicted) & (y == 0),
    }

    numeric_features = [
        feature for feature in features
        if feature in validation.columns and pd.api.types.is_numeric_dtype(validation[feature])
    ]

    def contrast(left: str, right: str) -> list[tuple[str, float, float, float, float]]:
        rows = []
        for feature in numeric_features:
            values = pd.to_numeric(validation[feature], errors="coerce")
            left_values = values[masks[left]]
            right_values = values[masks[right]]
            left_median = float(left_values.median()) if len(left_values) else 0.0
            right_median = float(right_values.median()) if len(right_values) else 0.0
            missing_gap = float(left_values.isna().mean() - right_values.isna().mean()) if len(left_values) and len(right_values) else 0.0
            scale = float(values.quantile(0.75) - values.quantile(0.25))
            standardized = abs(left_median - right_median) / max(abs(scale), 1e-6)
            rows.append((feature, left_median, right_median, missing_gap, standardized))
        return sorted(rows, key=lambda row: -max(row[4], abs(row[3])))[:top_k]

    fp_rows = contrast("FP", "TP")
    fn_rows = contrast("FN", "TN")

    def render(left: str, right: str, rows) -> str:
        if not rows:
            return "  (no stable numeric contrast available)"
        return "\n".join(
            f"  {feature}: median {left}={left_value:.4g}, {right}={right_value:.4g}, "
            f"missing-rate gap={missing_gap:+.3f}"
            for feature, left_value, right_value, missing_gap, _ in rows
        )

    return (
        f"Validation error profile at F1-selected threshold {threshold:.6g}; "
        f"counts TP={int(masks['TP'].sum())}, FP={int(masks['FP'].sum())}, "
        f"TN={int(masks['TN'].sum())}, FN={int(masks['FN'].sum())}.\n"
        f"False-positive versus true-positive contrasts:\n{render('FP', 'TP', fp_rows)}\n"
        f"False-negative versus true-negative contrasts:\n{render('FN', 'TN', fn_rows)}\n"
        "Treat these as heuristic clues, not causal effects or exact residuals."
    )


def random_grammar_proposals(
    columns: list[str],
    tried: set[str],
    rng: random.Random,
    n: int,
) -> list[FeatureSpec]:
    usable = [column for column in columns if column not in tried]
    if not usable:
        return []
    proposals: list[FeatureSpec] = []
    attempts = 0
    while len(proposals) < n and attempts < n * 20:
        attempts += 1
        left = rng.choice(usable)
        right = rng.choice(usable)
        operation = rng.choice(("ratio", "product", "difference", "sum", "absdiff", "logabs"))
        suffix = rng.randrange(1_000_000)
        name = f"random_{operation}_{suffix:06d}"
        if name in tried:
            continue
        if operation == "ratio":
            expression = f"df['{left}'] / (df['{right}'].abs() + 1e-6)"
        elif operation == "product":
            expression = f"df['{left}'] * df['{right}']"
        elif operation == "difference":
            expression = f"df['{left}'] - df['{right}']"
        elif operation == "sum":
            expression = f"df['{left}'] + df['{right}']"
        elif operation == "absdiff":
            expression = f"(df['{left}'] - df['{right}']).abs()"
        else:
            expression = f"np.log1p(df['{left}'].abs())"
        proposals.append(FeatureSpec(
            name=name,
            expression=expression,
            family=operation,
            source="random_grammar",
            description="typed random-grammar null candidate",
        ))
    return proposals


@dataclass
class RunLedger:
    method: str
    dataset: str
    repeat: int
    started_at: float = 0.0
    proposals: int = 0
    valid_programs: int = 0
    accepted: int = 0

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = time.time()

    def audit(self) -> dict:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        return {
            **asdict(self),
            "elapsed_sec": round(time.time() - self.started_at, 3),
            "max_rss_gb": round(max(usage.ru_maxrss, children.ru_maxrss) / 1e6, 3),
            "cpu_user_sec": round(usage.ru_utime + children.ru_utime, 3),
            "cpu_sys_sec": round(usage.ru_stime + children.ru_stime, 3),
        }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

