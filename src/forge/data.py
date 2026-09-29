"""Datasets, the I/V/T partition and the sealed test vault (paper Sec. IV-B).

* :func:`validate_dataset` checks the schema of a loaded dataset
  (loaders are in :mod:`forge.datasets`).
* :func:`split_dataset` implements ``D = I (64%) + V (16%) + T (20%)``.
* :func:`validation_blocks` implements Eq. (1): ``V`` split into
  three contiguous blocks with relative sizes 3:1:1 for S1 round validation.
* :func:`validation_halves` gives the two contiguous halves ``V1, V2`` used in S2.
* :class:`TestVault` holds the test labels and releases scores exactly once.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- hashing
def canonical_hash(value) -> str:
    """SHA-256 of a JSON-serialisable value with sorted keys (used for audit records)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    """SHA-256 of a DataFrame's contents and index."""
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def validate_dataset(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Check the schema: unique columns, binary label without missing values, required columns.

    A non-numeric event-time column (e.g. date strings read from a CSV) is
    parsed into datetimes so that sorting and "strictly earlier" comparisons
    are chronological.
    """
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("the dataset must be a DataFrame with unique column names")
    required = [cfg["label"]] + cfg["drop"] + cfg["entity_cols"] + ([cfg["time_col"]] if cfg["time_col"] else [])
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    y = frame[cfg["label"]]
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("the label must contain both 0 and 1 and no missing values")
    if cfg["time_col"] and frame[cfg["time_col"]].isna().any():
        raise ValueError(f"the time column {cfg['time_col']!r} must not contain missing values")
    if len(frame) < 500 or y.value_counts().min() < 25:
        raise ValueError("at least 500 rows and 25 samples per class are required")
    frame = frame.copy()
    time_col = cfg["time_col"]
    if time_col and not pd.api.types.is_numeric_dtype(frame[time_col]):
        try:
            frame[time_col] = pd.to_datetime(frame[time_col])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"time column {time_col!r} must be numeric or parseable as dates") from exc
    return frame


# --------------------------------------------------------------------------- partition
@dataclass
class TestVault:
    """Sealed test labels. :meth:`evaluate` may be called exactly once.

    All prediction vectors must be complete before the vault is opened
    (paper S3 and Algorithm 1: "open y_T once").
    """

    labels: np.ndarray
    used: bool = False

    def evaluate(self, predictions: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        if self.used:
            raise RuntimeError("the test vault has already been opened")
        for p in predictions.values():
            if len(p) != len(self.labels) or not np.isfinite(p).all():
                raise ValueError("all test predictions must be complete before opening the vault")
        self.used = True
        return {name: {"ap": float(average_precision_score(self.labels, p)),
                       "roc_auc": float(roc_auc_score(self.labels, p))}
                for name, p in predictions.items()}


@dataclass
class Partition:
    """Result of :func:`split_dataset`.

    ``outer`` is ``I + V`` (used only by the S3 rebuild); ``X_T`` has no label
    column; the test labels are inside ``vault``.
    """

    I: pd.DataFrame
    V: pd.DataFrame
    outer: pd.DataFrame
    X_T: pd.DataFrame
    vault: TestVault
    ids: dict
    class_balance: list
    outer_ids: list | None = None         # original row ids of ``outer``, in its order


def split_dataset(frame: pd.DataFrame, cfg: dict) -> Partition:
    """Split ``frame`` into I/V/T = 64/16/20 (paper Sec. IV-B).

    Time-ordered data is sorted by event time and cut in order. Otherwise a
    stratified 80/20 split (``split_seed``) is followed by a stratified 80/20
    split of the first part (``split_seed + 1``).
    """
    frame = frame.reset_index(drop=True)
    label = cfg["label"]
    if cfg["time_col"]:
        ordered = frame.sort_values(cfg["time_col"], kind="stable")
        n_outer = int(.8 * len(ordered))
        I_idx, V_idx, T_idx = np.split(np.arange(len(ordered)), [int(n_outer * .8), n_outer])
        outer, inner, valid, test = (ordered.iloc[:len(I_idx) + len(V_idx)], ordered.iloc[I_idx],
                                     ordered.iloc[V_idx], ordered.iloc[T_idx])
    else:
        outer, test = train_test_split(frame, test_size=.2, random_state=cfg["split_seed"],
                                       stratify=frame[label])
        inner, valid = train_test_split(outer, test_size=.2, random_state=cfg["split_seed"] + 1,
                                        stratify=outer[label])
    sets = [set(inner.index), set(valid.index), set(test.index)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for part in (inner, valid, test):
        if part[label].nunique() != 2:
            raise ValueError("each of I, V and T needs both classes")
    ids = {k: v.index.tolist() for k, v in [("I", inner), ("V", valid), ("T", test)]}
    balance = [{"split": name, "rows": len(part), "positives": int(part[label].sum()),
                "positive_rate": float(part[label].mean())}
               for name, part in [("I", inner), ("V", valid), ("T", test)]]
    return Partition(I=inner.reset_index(drop=True), V=valid.reset_index(drop=True),
                     outer=outer.reset_index(drop=True),
                     X_T=test.drop(columns=label).reset_index(drop=True),
                     vault=TestVault(test[label].to_numpy(copy=True)), ids=ids,
                     class_balance=balance, outer_ids=outer.index.tolist())


def validation_blocks(n_rows: int, weights=(3, 1, 1)) -> list[np.ndarray]:
    """Contiguous S1 validation blocks of ``V`` (paper Eq. (1)).

    With the default 3:1:1 weights the blocks hold 60%, 20% and 20% of ``V``.
    Each S1 round that changes the configuration consumes the next unused block.
    """
    rows = np.arange(n_rows)
    first = int(round(weights[0] / sum(weights) * n_rows))
    rest = rows[first:]
    cuts = np.cumsum(weights[1:-1]) / sum(weights[1:]) * len(rest)
    return [rows[:first]] + np.split(rest, [int(round(c)) for c in cuts])


def validation_halves(n_rows: int) -> list[np.ndarray]:
    """Two contiguous halves ``V1, V2`` of ``V`` used by the S2 benefit check."""
    return np.array_split(np.arange(n_rows), 2)
