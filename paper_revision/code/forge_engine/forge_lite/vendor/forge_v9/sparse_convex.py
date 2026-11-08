from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score


_EPS = 1e-7


@dataclass(frozen=True)
class SparseHullConfig:
    max_iter: int = 25
    support_cap: int = 2
    cost_lambda: float = 0.002
    fw_gap_tolerance: float = 1e-7
    metric_tolerance: float = 5e-4
    weight_floor: float = 1e-6
    line_search_steps: int = 60


def _probabilities(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), _EPS, 1.0 - _EPS)


def binary_cross_entropy(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = _probabilities(prediction)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def normalize_costs(costs: np.ndarray | None, n_candidates: int) -> np.ndarray:
    if costs is None:
        return np.zeros(n_candidates, dtype=np.float64)
    values = np.asarray(costs, dtype=np.float64)
    if values.shape != (n_candidates,):
        raise ValueError("costs must have one value per candidate")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("costs must be finite and non-negative")
    values = np.log1p(values)
    maximum = float(values.max(initial=0.0))
    return values / maximum if maximum > 0 else np.zeros_like(values)


def _objective(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray,
               costs: np.ndarray, cost_lambda: float) -> float:
    return binary_cross_entropy(y, prediction) + float(
        cost_lambda * np.dot(costs, weights))


def _prediction_gradient(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    p = _probabilities(prediction)
    return (p - y) / (p * (1.0 - p) * len(y))


def _line_search(
    y: np.ndarray,
    prediction: np.ndarray,
    atom: np.ndarray,
    current_cost: float,
    atom_cost: float,
    cost_lambda: float,
    steps: int,
) -> float:
    direction = atom - prediction
    cost_direction = atom_cost - current_cost

    def derivative(gamma: float) -> float:
        trial = _probabilities(prediction + gamma * direction)
        loss_derivative = np.mean(
            (trial - y) * direction / (trial * (1.0 - trial)))
        return float(loss_derivative + cost_lambda * cost_direction)

    if derivative(0.0) >= 0:
        return 0.0
    if derivative(1.0) <= 0:
        return 1.0
    lower, upper = 0.0, 1.0
    for _ in range(int(steps)):
        middle = 0.5 * (lower + upper)
        if derivative(middle) <= 0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _snapshot(
    *, iteration: int, names: tuple[str, ...], weights: np.ndarray,
    prediction: np.ndarray, y: np.ndarray, costs: np.ndarray,
    cost_lambda: float, gap: float | None,
) -> dict:
    active = np.flatnonzero(weights > 0)
    return {
        "iteration": int(iteration),
        "weights": weights.copy(),
        "active_members": [names[index] for index in active],
        "support_size": int(len(active)),
        "objective": _objective(
            y, prediction, weights, costs, cost_lambda),
        "auprc": float(average_precision_score(y, prediction)),
        "normalized_cost": float(np.dot(costs, weights)),
        "fw_gap": None if gap is None else float(gap),
    }


def sparse_convex_hull(
    oof_matrix: np.ndarray,
    y: np.ndarray,
    *,
    names: list[str] | tuple[str, ...] | None = None,
    costs: np.ndarray | list[float] | None = None,
    config: SparseHullConfig | None = None,
) -> tuple[np.ndarray, dict]:
    cfg = config or SparseHullConfig()
    matrix = _probabilities(oof_matrix)
    y = np.asarray(y, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(y):
        raise ValueError("oof_matrix must be rows x candidates and match y")
    if matrix.shape[1] == 0:
        raise ValueError("at least one candidate is required")
    if not np.isfinite(matrix).all():
        raise ValueError("oof_matrix must contain finite probabilities")
    if not np.isin(y, (0, 1)).all() or np.unique(y).size < 2:
        raise ValueError("y must be binary and contain both classes")
    if int(cfg.support_cap) < 1:
        raise ValueError("support_cap must be >= 1")
    if int(cfg.max_iter) < 1 or int(cfg.line_search_steps) < 1:
        raise ValueError("max_iter and line_search_steps must be >= 1")

    n_candidates = matrix.shape[1]
    labels = tuple(names or [f"model_{index}" for index in range(n_candidates)])
    if len(labels) != n_candidates or len(set(labels)) != n_candidates:
        raise ValueError("names must be unique and match the candidate count")
    normalized_cost = normalize_costs(
        None if costs is None else np.asarray(costs), n_candidates)

    individual_objectives = np.array([
        binary_cross_entropy(y, matrix[:, index])
        + cfg.cost_lambda * normalized_cost[index]
        for index in range(n_candidates)
    ])
    start = int(np.argmin(individual_objectives))
    weights = np.zeros(n_candidates, dtype=np.float64)
    weights[start] = 1.0
    prediction = matrix[:, start].copy()
    snapshots = [_snapshot(
        iteration=0, names=labels, weights=weights, prediction=prediction,
        y=y, costs=normalized_cost, cost_lambda=cfg.cost_lambda, gap=None)]
    stop_reason = "max_iter"

    for iteration in range(1, int(cfg.max_iter) + 1):
        gradient_p = _prediction_gradient(y, prediction)
        gradient_w = matrix.T @ gradient_p + cfg.cost_lambda * normalized_cost
        vertex = int(np.argmin(gradient_w))
        gap = float(np.dot(weights, gradient_w) - gradient_w[vertex])
        if gap <= cfg.fw_gap_tolerance:
            stop_reason = "fw_gap"
            snapshots[-1]["fw_gap"] = gap
            break
        active = np.flatnonzero(weights > cfg.weight_floor)
        if weights[vertex] <= cfg.weight_floor and len(active) >= cfg.support_cap:
            stop_reason = "support_cap"
            snapshots[-1]["fw_gap"] = gap
            break
        gamma = _line_search(
            y, prediction, matrix[:, vertex],
            float(np.dot(normalized_cost, weights)),
            float(normalized_cost[vertex]), cfg.cost_lambda,
            cfg.line_search_steps)
        if gamma <= np.finfo(float).eps:
            stop_reason = "zero_step"
            snapshots[-1]["fw_gap"] = gap
            break
        weights *= 1.0 - gamma
        weights[vertex] += gamma
        prediction = (1.0 - gamma) * prediction + gamma * matrix[:, vertex]
        snapshots.append(_snapshot(
            iteration=iteration, names=labels, weights=weights,
            prediction=prediction, y=y, costs=normalized_cost,
            cost_lambda=cfg.cost_lambda, gap=gap))

    best_auprc = max(item["auprc"] for item in snapshots)
    eligible = [
        item for item in snapshots
        if item["auprc"] >= best_auprc - cfg.metric_tolerance
    ]
    chosen = min(
        eligible,
        key=lambda item: (
            item["support_size"], item["objective"],
            item["normalized_cost"], item["iteration"]))
    final = np.asarray(chosen["weights"], dtype=np.float64).copy()
    final[final < cfg.weight_floor] = 0.0
    final /= final.sum()

    audit_snapshots = []
    for item in snapshots:
        audit_snapshots.append({
            key: value for key, value in item.items() if key != "weights"
        } | {
            "weights": {
                name: float(value)
                for name, value in zip(labels, item["weights"])
                if value > cfg.weight_floor
            }
        })
    audit = {
        "method": "cost_aware_frank_wolfe_sparse_convex_hull",
        "config": asdict(cfg),
        "candidate_names": list(labels),
        "raw_costs_sec": (
            [float(value) for value in costs] if costs is not None
            else [0.0] * n_candidates),
        "normalized_costs": {
            name: float(value)
            for name, value in zip(labels, normalized_cost)
        },
        "initial_member": labels[start],
        "stop_reason": stop_reason,
        "iterations_run": len(snapshots) - 1,
        "best_path_auprc": float(best_auprc),
        "chosen_iteration": int(chosen["iteration"]),
        "chosen_active_members": [
            name for name, value in zip(labels, final)
            if value > cfg.weight_floor
        ],
        "chosen_weights": {
            name: float(value)
            for name, value in zip(labels, final)
            if value > cfg.weight_floor
        },
        "path": audit_snapshots,
    }
    return final, audit
