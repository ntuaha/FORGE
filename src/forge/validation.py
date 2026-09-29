"""Round validation (paper S1.2), benefit checks (S2.2) and core selection.

* :func:`paired_bootstrap` gives ``SE_h``, the sample standard deviation of
  paired bootstrap AP differences (Eq. (27)); both predictions
  use the same resampled rows in every replicate.
* :func:`harm_only_gate` accepts a round update iff
  ``Delta_h >= -2 SE_h`` on every validation region ``h`` (Eq. (14)).
* :func:`benefit_check` accepts an expert iff ``Delta_h >= +1 SE_h`` on both
  halves ``V1, V2`` (Eq. (28)).
* :class:`ValidationBudget` hands out the 3:1:1 blocks of ``V`` in order; a
  round that does not change the configuration consumes no block.
* :func:`choose_configuration` fixes the CoF core by AP on the complete ``V``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import validation_blocks, validation_halves


def paired_bootstrap(y: np.ndarray, incumbent: np.ndarray, candidate: np.ndarray,
                     repeats: int, seed: int) -> dict:
    """Paired bootstrap of ``AP(candidate) - AP(incumbent)``.

    Replicates in which the resample holds a single class are skipped.
    Returns the SE (``ddof = 1``), the number of valid replicates and a 95%
    percentile interval.
    """
    y = np.asarray(y, int)
    incumbent = np.asarray(incumbent, float)
    candidate = np.asarray(candidate, float)
    rng, deltas = np.random.default_rng(seed), []
    for _ in range(repeats):
        rows = rng.integers(0, len(y), size=len(y))
        if y[rows].sum() in (0, len(rows)):
            continue
        deltas.append(average_precision_score(y[rows], candidate[rows])
                      - average_precision_score(y[rows], incumbent[rows]))
    deltas = np.asarray(deltas, float)
    if len(deltas) < 2:
        raise ValueError("too few valid bootstrap replicates to estimate the SE")
    return {"se": float(deltas.std(ddof=1)), "n_effective": len(deltas),
            "ci_lo": float(np.percentile(deltas, 2.5)), "ci_hi": float(np.percentile(deltas, 97.5)),
            "delta_mean": float(deltas.mean())}


def _region_test(y, base, candidate, regions, repeats, seed, k, rule):
    table = []
    for i, h in enumerate(regions):
        if len(np.unique(y[h])) < 2:
            table.append({"region": i + 1, "accepted": False, "reason": "region has one class"})
            continue
        base_ap = float(average_precision_score(y[h], base[h]))
        cand_ap = float(average_precision_score(y[h], candidate[h]))
        delta = cand_ap - base_ap
        boot = paired_bootstrap(y[h], base[h], candidate[h], repeats, seed + 2 + i)
        threshold = k * boot["se"]
        table.append({"region": i + 1, "rows": len(h), "positives": int(y[h].sum()),
                      "base_ap": base_ap, "candidate_ap": cand_ap, "delta": delta,
                      "se": boot["se"], "threshold": threshold, "n_effective": boot["n_effective"],
                      "accepted": bool(delta >= threshold), "rule": rule})
    return all(r["accepted"] for r in table), pd.DataFrame(table)


def harm_only_gate(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, regions, cfg: dict):
    """S1.2 harm-only gate, Eq. (14): ``Delta_h >= -gate_harm_k * SE_h`` for all ``h``.

    ``regions`` are index arrays into ``V``: the two halves of the round's
    block for R0, the whole block for a CoF round. Returns ``(passed, table)``.
    """
    return _region_test(y, base, candidate, regions, cfg["bootstrap_repeats"], cfg["seed"],
                        -cfg["gate_harm_k"], "harm_only")


def benefit_check(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, cfg: dict):
    """S2.2 benefit check, Eq. (28): ``Delta_h >= +SE_h`` on ``V1`` and ``V2``."""
    return _region_test(y, base, candidate, validation_halves(len(y)), cfg["expert_bootstrap_repeats"],
                        cfg["seed"], cfg["expert_benefit_k"], "benefit")


class ValidationBudget:
    """The three single-use S1 validation blocks of ``V`` (Eq. (1))."""

    def __init__(self, n_rows: int, cfg: dict):
        self.blocks = validation_blocks(n_rows, cfg["validation_block_weights"])
        self.used = 0

    @property
    def exhausted(self) -> bool:
        return self.used >= len(self.blocks)

    def next_regions(self, initial_round: bool) -> tuple[int, list[np.ndarray]]:
        """Consume the next block; R0 validates on its two halves, a CoF round on the whole block."""
        if self.exhausted:
            raise RuntimeError("all validation blocks have been consumed")
        block = self.blocks[self.used]
        self.used += 1
        return self.used, (np.array_split(block, 2) if initial_round else [block])


def choose_configuration(accepted: list[dict], y_V: np.ndarray, cfg: dict):
    """Fix the CoF core: the baseline or accepted configuration with the highest AP on the complete ``V``.

    Ties keep the earlier configuration. Returns ``(state, table)``.
    """
    rows = [{"configuration": i, "features": len(state["features"]),
             "V_ap": float(average_precision_score(y_V, state["V_prediction"]))}
            for i, state in enumerate(accepted)]
    best = max(rows, key=lambda r: (r["V_ap"], -r["configuration"]))["configuration"]
    for r in rows:
        r["selected"] = r["configuration"] == best
    return accepted[best], pd.DataFrame(rows)
