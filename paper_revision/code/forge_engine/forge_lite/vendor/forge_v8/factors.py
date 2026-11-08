from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .gate_policy import CandidateOrigin, gate_decision


_ENTITY = re.compile(
    r"(^|_)(id|uuid|account|acct|cc|customer|user|device|card|merchant|"
    r"name|address|email|phone|ip|network|session)($|_)"
)
_AMOUNT = re.compile(
    r"(^|_)(amount|amt|value|price|payment|purchase|transaction_value)($|_)"
)
_FORBIDDEN = re.compile(r"(^|_)(label|target|class|fraud|isfraud|is_fraud)($|_)")


def _normal(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _matches(pattern: re.Pattern[str], name: str) -> bool:
    return bool(pattern.search(_normal(name)))


@dataclass(frozen=True)
class SparseEntity:
    name: str
    dtype: str
    n_unique: int
    unique_ratio: float
    repeat_fraction: float
    admission: str


@dataclass(frozen=True)
class FactorAudit:
    name: str
    family: str
    origin: str
    inputs: tuple[str, ...]
    gate_required: bool
    gate_status: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["inputs"] = list(self.inputs)
        return payload


def discover_sparse_entities(
    frame: pd.DataFrame,
    *,
    label: str,
    forbidden: Iterable[str] = (),
    min_repeat_fraction: float = 0.02,
    min_unique_ratio: float = 0.50,
    min_unique: int = 2,
) -> list[SparseEntity]:
    blocked = {str(c) for c in forbidden} | {label}
    n_rows = max(1, len(frame))
    found: list[SparseEntity] = []
    for column in frame.columns:
        if column in blocked or _matches(_FORBIDDEN, column):
            continue
        if not _matches(_ENTITY, column):
            continue
        n_unique = int(frame[column].nunique(dropna=True))
        unique_ratio = n_unique / n_rows
        repeat_fraction = 1.0 - unique_ratio
        if (n_unique >= min_unique
                and unique_ratio > min_unique_ratio
                and repeat_fraction >= min_repeat_fraction):
            found.append(SparseEntity(
                name=str(column), dtype=str(frame[column].dtype),
                n_unique=n_unique, unique_ratio=float(unique_ratio),
                repeat_fraction=float(repeat_fraction),
                admission="id_like_and_repeated_sparse_key",
            ))
    return sorted(found, key=lambda item: (-item.repeat_fraction, item.name))


def discover_amounts(frame: pd.DataFrame, *, label: str) -> list[str]:
    return [str(c) for c in frame.columns
            if c != label and _matches(_AMOUNT, c)
            and pd.to_numeric(frame[c], errors="coerce").notna().any()]


def _assert_chronological(train: pd.DataFrame, apply: pd.DataFrame,
                          time_col: str) -> None:
    if time_col not in train or time_col not in apply:
        raise KeyError(f"time column {time_col!r} missing")
    merged = pd.concat([train[time_col], apply[time_col]], ignore_index=True)
    if pd.api.types.is_datetime64_any_dtype(merged):
        values = merged.astype("int64").to_numpy(dtype=np.float64)
    else:
        values = pd.to_numeric(merged, errors="coerce").to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) > 1 and not np.all(finite[1:] >= finite[:-1]):
        raise ValueError("strict-prior factors require chronological partitions")


def _split(values: np.ndarray, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.array(values, dtype=np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values[:n_train], values[n_train:]


def _forward_target_encoding(
    train: pd.DataFrame,
    apply: pd.DataFrame,
    *,
    key: str,
    label: str,
    smooth: float,
) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(train[label], errors="coerce").fillna(0.0).astype(float)
    grouped = train.groupby(key, dropna=False, sort=False)
    past_count = grouped.cumcount().astype(float)
    past_sum = grouped[label].cumsum().astype(float) - y
    row_number = np.arange(len(train), dtype=np.float64)
    past_global = (y.cumsum().to_numpy() - y.to_numpy() + 1.0) / (row_number + 2.0)
    train_te = ((past_sum.to_numpy() + smooth * past_global)
                / (past_count.to_numpy() + smooth))

    full_mean = float(y.mean())
    stats = train.groupby(key, dropna=False)[label].agg(["sum", "count"])
    mapping = (stats["sum"] + smooth * full_mean) / (stats["count"] + smooth)
    apply_te = apply[key].map(mapping).fillna(full_mean).to_numpy(dtype=float)
    return train_te, apply_te


def _add_factor(
    train_model: pd.DataFrame,
    apply_model: pd.DataFrame,
    audit: list[FactorAudit],
    *,
    name: str,
    family: str,
    inputs: tuple[str, ...],
    train_values: np.ndarray,
    apply_values: np.ndarray,
) -> None:
    decision = gate_decision(CandidateOrigin.MODEL_DERIVED)
    train_model[name] = np.asarray(train_values, dtype=np.float32)
    apply_model[name] = np.asarray(apply_values, dtype=np.float32)
    audit.append(FactorAudit(
        name=name, family=family, origin=decision.origin.value,
        inputs=inputs, gate_required=decision.requires_gate,
        gate_status=decision.reason,
    ))


def materialize_model_factors(
    raw_train: pd.DataFrame,
    raw_apply: pd.DataFrame,
    train_model: pd.DataFrame,
    apply_model: pd.DataFrame,
    *,
    label: str,
    time_col: str | None,
    forbidden: Iterable[str] = (),
    min_repeat_fraction: float = 0.02,
    min_unique_ratio: float = 0.50,
    te_smooth: float = 20.0,
) -> tuple[list[str], dict]:
    entities = discover_sparse_entities(
        raw_train, label=label, forbidden=forbidden,
        min_repeat_fraction=min_repeat_fraction,
        min_unique_ratio=min_unique_ratio,
    )
    amounts = discover_amounts(raw_train, label=label)
    if time_col is not None:
        _assert_chronological(raw_train, raw_apply, time_col)

    audit: list[FactorAudit] = []
    n_train = len(raw_train)
    stream = pd.concat([raw_train, raw_apply], ignore_index=True)
    amount = amounts[0] if amounts else None

    te_columns: list[str] = [str(c) for c in train_model.columns
                             if str(c).endswith("_oof_te")]
    entity_te: dict[str, str] = {}
    for entity in entities:
        key = entity.name
        safe = _normal(key)
        keys = stream[key]
        prior_count = (keys.to_frame("__key")
                       .groupby("__key", dropna=False, sort=False)
                       .cumcount().to_numpy(dtype=np.float64))
        left, right = _split(prior_count, n_train)
        _add_factor(
            train_model, apply_model, audit,
            name=f"v8_{safe}_prior_count", family="sparse_entity_history",
            inputs=(key,), train_values=left, apply_values=right,
        )
        _add_factor(
            train_model, apply_model, audit,
            name=f"v8_{safe}_is_repeated", family="sparse_entity_history",
            inputs=(key,), train_values=(left > 0), apply_values=(right > 0),
        )

        if time_col is not None:
            time_values = pd.to_numeric(stream[time_col], errors="coerce")
            previous = time_values.groupby(keys, dropna=False, sort=False).shift(1)
            gap = (time_values - previous).to_numpy(dtype=np.float64)
            left, right = _split(gap, n_train)
            _add_factor(
                train_model, apply_model, audit,
                name=f"v8_{safe}_time_since_prev", family="sparse_entity_velocity",
                inputs=(key, time_col), train_values=left, apply_values=right,
            )

        if amount is not None:
            values = pd.to_numeric(stream[amount], errors="coerce").astype(float)
            filled = values.fillna(0.0)
            present = values.notna().astype(float)
            prior_sum = (filled.groupby(keys, dropna=False, sort=False).cumsum()
                         - filled)
            prior_n = (present.groupby(keys, dropna=False, sort=False).cumsum()
                       - present)
            prior_mean = prior_sum / prior_n.replace(0.0, np.nan)
            ratio = values / (prior_mean.abs() + 1.0)
            left, right = _split(ratio.to_numpy(dtype=np.float64), n_train)
            _add_factor(
                train_model, apply_model, audit,
                name=f"v8_{safe}_{_normal(amount)}_prior_mean_ratio",
                family="sparse_entity_value_context",
                inputs=(key, amount), train_values=left, apply_values=right,
            )

        existing_te = f"{key}_oof_te"
        if existing_te in train_model.columns:
            entity_te[key] = existing_te
        else:
            train_te, apply_te = _forward_target_encoding(
                raw_train, raw_apply, key=key, label=label, smooth=te_smooth,
            )
            te_name = f"v8_{safe}_forward_te"
            _add_factor(
                train_model, apply_model, audit,
                name=te_name, family="forward_model_prediction",
                inputs=(key,), train_values=train_te, apply_values=apply_te,
            )
            te_columns.append(te_name)
            entity_te[key] = te_name

    te_columns = list(dict.fromkeys(c for c in te_columns
                                    if c in train_model and c in apply_model))
    if len(te_columns) >= 2:
        train_te = train_model[te_columns].apply(pd.to_numeric, errors="coerce")
        apply_te = apply_model[te_columns].apply(pd.to_numeric, errors="coerce")
        summaries = {
            "v8_te_cross_field_std": (train_te.std(axis=1, ddof=0),
                                       apply_te.std(axis=1, ddof=0)),
            "v8_te_cross_field_range": (train_te.max(axis=1) - train_te.min(axis=1),
                                         apply_te.max(axis=1) - apply_te.min(axis=1)),
        }
        for name, (left, right) in summaries.items():
            _add_factor(
                train_model, apply_model, audit,
                name=name, family="prediction_disagreement",
                inputs=tuple(te_columns), train_values=left.to_numpy(),
                apply_values=right.to_numpy(),
            )

        for key, te_name in entity_te.items():
            context = [c for c in te_columns if c != te_name]
            if not context:
                continue
            left = (pd.to_numeric(train_model[te_name], errors="coerce")
                    - train_te[context].mean(axis=1))
            right = (pd.to_numeric(apply_model[te_name], errors="coerce")
                     - apply_te[context].mean(axis=1))
            _add_factor(
                train_model, apply_model, audit,
                name=f"v8_{_normal(key)}_te_context_delta",
                family="prediction_context_delta",
                inputs=(te_name, *context), train_values=left.to_numpy(),
                apply_values=right.to_numpy(),
            )

        if amount is not None and amount in train_model:
            left = np.log1p(pd.to_numeric(train_model[amount], errors="coerce")
                            .clip(lower=0)) * train_te.std(axis=1, ddof=0)
            right = np.log1p(pd.to_numeric(apply_model[amount], errors="coerce")
                             .clip(lower=0)) * apply_te.std(axis=1, ddof=0)
            _add_factor(
                train_model, apply_model, audit,
                name="v8_amount_weighted_te_disagreement",
                family="value_weighted_prediction_disagreement",
                inputs=(amount, *te_columns), train_values=left.to_numpy(),
                apply_values=right.to_numpy(),
            )

    names = [entry.name for entry in audit]
    return names, {
        "sparse_entities": [asdict(entity) for entity in entities],
        "amount_candidates": amounts,
        "te_columns": te_columns,
        "factors": [entry.to_dict() for entry in audit],
        "gate_policy": {
            "model_derived": "bypass",
            "llm_inferred": "consumer_gate_required",
        },
    }
