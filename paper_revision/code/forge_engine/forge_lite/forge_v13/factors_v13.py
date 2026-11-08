from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from forge_v8.factors import (
    _ENTITY,
    _FORBIDDEN,
    _assert_chronological,
    _forward_target_encoding,
    _matches,
    _normal,
    _split,
    discover_amounts,
)


@dataclass(frozen=True)
class UnifiedEntity:
    name: str
    dtype: str
    n_unique: int
    unique_ratio: float
    repeat_fraction: float
    admission: str


@dataclass(frozen=True)
class FactorRecord:
    name: str
    family: str
    inputs: tuple[str, ...]
    origin: str
    uses_label: bool

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["inputs"] = list(self.inputs)
        return payload


def discover_entities(
    frame: pd.DataFrame,
    *,
    label: str,
    forbidden: Iterable[str] = (),
    time_col: str | None = None,
    min_repeat_fraction: float = 0.02,
    min_unique: int = 20,
    top_k: int = 3,
) -> list[UnifiedEntity]:
    blocked = {str(c) for c in forbidden} | {label}
    if time_col:
        blocked.add(str(time_col))
    n_rows = max(1, len(frame))
    found: list[UnifiedEntity] = []
    for column in frame.columns:
        if column in blocked or _matches(_FORBIDDEN, column):
            continue
        if not _matches(_ENTITY, column):
            continue
        n_unique = int(frame[column].nunique(dropna=True))
        unique_ratio = n_unique / n_rows
        repeat_fraction = 1.0 - unique_ratio
        if n_unique >= int(min_unique) and repeat_fraction >= float(min_repeat_fraction):
            found.append(UnifiedEntity(
                name=str(column), dtype=str(frame[column].dtype),
                n_unique=n_unique, unique_ratio=float(unique_ratio),
                repeat_fraction=float(repeat_fraction),
                admission="id_like_and_repeated_unified",
            ))
    found.sort(key=lambda item: (-item.repeat_fraction, item.name))
    return found[:int(top_k)]


def _add_factor(
    train_model: pd.DataFrame,
    apply_model: pd.DataFrame,
    audit: list[FactorRecord],
    *,
    name: str,
    family: str,
    inputs: tuple[str, ...],
    train_values: np.ndarray,
    apply_values: np.ndarray,
    uses_label: bool = False,
) -> None:
    train_model[name] = np.asarray(train_values, dtype=np.float64)
    apply_model[name] = np.asarray(apply_values, dtype=np.float64)
    audit.append(FactorRecord(
        name=name, family=family, inputs=inputs,
        origin="model_derived", uses_label=bool(uses_label),
    ))


def _pair_key(keys: pd.Series, partners: pd.Series) -> pd.Series:
    return keys.astype(str).str.cat(partners.astype(str), sep="\x1f")


def _prior_pair_count(pair_values: pd.Series) -> np.ndarray:
    return (pair_values.to_frame("__pair")
            .groupby("__pair", dropna=False, sort=False)
            .cumcount().to_numpy(dtype=np.float64))


def _prior_partner_nunique(keys: pd.Series,
                           pair_prior: np.ndarray) -> np.ndarray:
    key_values = keys.astype(str)
    pair_first = (np.asarray(pair_prior) == 0).astype(np.float64)
    nunique_incl = (pd.Series(pair_first)
                    .groupby(key_values.to_numpy(), sort=False)
                    .cumsum().to_numpy(dtype=np.float64))
    return nunique_incl - pair_first


def _replay_entity(frame: pd.DataFrame, name: str) -> UnifiedEntity:
    n_rows = max(1, len(frame))
    n_unique = int(frame[name].nunique(dropna=True))
    unique_ratio = n_unique / n_rows
    return UnifiedEntity(
        name=str(name), dtype=str(frame[name].dtype), n_unique=n_unique,
        unique_ratio=float(unique_ratio),
        repeat_fraction=float(1.0 - unique_ratio),
        admission="frozen_spec_replay",
    )


def _cyclical_components(time_values: pd.Series) -> dict[str, np.ndarray]:
    seconds = pd.to_numeric(time_values, errors="coerce").astype(float)
    span = float(np.nanmax(seconds) - np.nanmin(seconds)) if len(seconds) else 0.0
    out: dict[str, np.ndarray] = {}
    if not np.isfinite(span) or span < 7200.0:
        return out
    day = 86400.0
    tod = np.mod(seconds.to_numpy(dtype=np.float64), day) / day
    out["tod_frac"] = tod
    out["tod_sin"] = np.sin(2.0 * np.pi * tod)
    out["tod_cos"] = np.cos(2.0 * np.pi * tod)
    if span >= 4 * day:
        dow = np.mod(np.floor(seconds.to_numpy(dtype=np.float64) / day), 7.0) / 7.0
        out["dow_frac"] = dow
        out["dow_sin"] = np.sin(2.0 * np.pi * dow)
        out["dow_cos"] = np.cos(2.0 * np.pi * dow)
    return out


def materialize_unified_factors(
    raw_train: pd.DataFrame,
    raw_apply: pd.DataFrame,
    train_model: pd.DataFrame,
    apply_model: pd.DataFrame,
    *,
    label: str,
    time_col: str | None,
    forbidden: Iterable[str] = (),
    min_repeat_fraction: float = 0.02,
    min_unique: int = 20,
    top_entities: int = 3,
    te_smooth: float = 20.0,
    frozen_spec: dict | None = None,
    calendar_family: bool = False,
    allow_te: bool = True,
) -> tuple[list[str], dict]:
    if frozen_spec is None:
        entities = discover_entities(
            raw_train, label=label, forbidden=forbidden, time_col=time_col,
            min_repeat_fraction=min_repeat_fraction, min_unique=min_unique,
            top_k=top_entities,
        )
        amounts = discover_amounts(raw_train, label=label)
        amount = amounts[0] if amounts else None
    else:
        entity_names = [str(name) for name in frozen_spec["entities"]]
        missing = [name for name in entity_names
                   if name not in raw_train.columns
                   or name not in raw_apply.columns]
        if missing:
            raise RuntimeError(
                "frozen factor-spec replay: entity columns missing from the "
                f"target frames: {missing}")
        amount = frozen_spec.get("amount_used")
        if amount is not None and (amount not in raw_train.columns
                                   or amount not in raw_apply.columns):
            raise RuntimeError(
                "frozen factor-spec replay: amount column missing from the "
                f"target frames: {[amount]}")
        amounts = [amount] if amount is not None else []
        entities = [_replay_entity(raw_train, name) for name in entity_names]
    if time_col is not None:
        _assert_chronological(raw_train, raw_apply, time_col)

    audit: list[FactorRecord] = []
    n_train = len(raw_train)
    stream = pd.concat([raw_train, raw_apply], ignore_index=True)
    time_values = (pd.to_numeric(stream[time_col], errors="coerce")
                   if time_col is not None else None)

    if calendar_family and time_values is not None:
        for suffix, series in _cyclical_components(time_values).items():
            left, right = _split(np.asarray(series, dtype=np.float64), n_train)
            _add_factor(
                train_model, apply_model, audit,
                name=f"cal_{suffix}", family="calendar_periodicity",
                inputs=(str(time_col),), train_values=left, apply_values=right,
            )

    if not allow_te and frozen_spec is not None and (
            frozen_spec.get("entity_te") or frozen_spec.get("te_columns")):
        raise RuntimeError(
            "frozen factor-spec replay: spec contains TE choices but the "
            "run disables the TE bank (--bank-no-te); refusing a silently "
            "different bank")
    te_columns: list[str] = ([str(c) for c in train_model.columns
                              if str(c).endswith("_oof_te")]
                             if allow_te else [])
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
            name=f"v13_{safe}_prior_count", family="entity_history",
            inputs=(key,), train_values=left, apply_values=right,
        )
        _add_factor(
            train_model, apply_model, audit,
            name=f"v13_{safe}_is_repeated", family="entity_history",
            inputs=(key,), train_values=(left > 0), apply_values=(right > 0),
        )

        if time_values is not None:
            previous = time_values.groupby(keys, dropna=False, sort=False).shift(1)
            gap = (time_values - previous).to_numpy(dtype=np.float64)
            left, right = _split(gap, n_train)
            _add_factor(
                train_model, apply_model, audit,
                name=f"v13_{safe}_time_since_prev", family="entity_velocity",
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
                name=f"v13_{safe}_{_normal(amount)}_prior_mean_ratio",
                family="entity_value_context",
                inputs=(key, amount), train_values=left, apply_values=right,
            )

        if not allow_te:
            continue
        existing_te = f"{key}_oof_te"
        forward_name = f"v13_{safe}_forward_te"
        if frozen_spec is not None:
            te_choice = frozen_spec.get("entity_te", {}).get(key)
            if te_choice not in (existing_te, forward_name):
                raise RuntimeError(
                    "frozen factor-spec replay: unrecognized TE choice "
                    f"{te_choice!r} for entity {key!r}")
            reuse_existing = te_choice == existing_te
            if reuse_existing and (existing_te not in train_model.columns
                                   or existing_te not in apply_model.columns):
                raise RuntimeError(
                    "frozen factor-spec replay: required TE column missing "
                    f"from the target model frames: {[existing_te]}")
        else:
            reuse_existing = existing_te in train_model.columns
        if reuse_existing:
            entity_te[key] = existing_te
        else:
            train_te, apply_te = _forward_target_encoding(
                raw_train, raw_apply, key=key, label=label, smooth=te_smooth,
            )
            te_name = forward_name
            _add_factor(
                train_model, apply_model, audit,
                name=te_name, family="forward_target_encoding",
                inputs=(key,), train_values=train_te, apply_values=apply_te,
                uses_label=True,
            )
            te_columns.append(te_name)
            entity_te[key] = te_name

    if frozen_spec is not None:
        frozen_te = [str(column) for column in frozen_spec.get("te_columns", [])]
        missing = [column for column in frozen_te
                   if column not in train_model.columns
                   or column not in apply_model.columns]
        if missing:
            raise RuntimeError(
                "frozen factor-spec replay: TE columns missing from the "
                f"target model frames: {missing}")
        te_columns = frozen_te
    else:
        te_columns = list(dict.fromkeys(c for c in te_columns
                                        if c in train_model and c in apply_model))
    if len(te_columns) >= 2:
        train_te = train_model[te_columns].apply(pd.to_numeric, errors="coerce")
        apply_te = apply_model[te_columns].apply(pd.to_numeric, errors="coerce")
        summaries = {
            "v13_te_cross_field_std": (train_te.std(axis=1, ddof=0),
                                       apply_te.std(axis=1, ddof=0)),
            "v13_te_cross_field_range": (train_te.max(axis=1) - train_te.min(axis=1),
                                         apply_te.max(axis=1) - apply_te.min(axis=1)),
        }
        for name, (left, right) in summaries.items():
            _add_factor(
                train_model, apply_model, audit,
                name=name, family="te_cross_field",
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
                name=f"v13_{_normal(key)}_te_context_delta",
                family="te_context_delta",
                inputs=(te_name, *context), train_values=left.to_numpy(),
                apply_values=right.to_numpy(),
            )

    pair_meta = None
    if frozen_spec is not None:
        pair_spec = frozen_spec.get("pair")
        pair_keys = ((str(pair_spec["e1"]), str(pair_spec["e2"]))
                     if pair_spec else None)
    else:
        pair_keys = ((entities[0].name, entities[1].name)
                     if len(entities) >= 2 else None)
    if pair_keys is not None:
        first, second = pair_keys
        s1, s2 = _normal(first), _normal(second)
        pair_values = _pair_key(stream[first], stream[second])
        pair_prior = _prior_pair_count(pair_values)
        left, right = _split(pair_prior, n_train)
        _add_factor(
            train_model, apply_model, audit,
            name=f"v13_pair_{s1}__{s2}_prior_count",
            family="entity_pair_history",
            inputs=(first, second), train_values=left, apply_values=right,
        )
        if time_values is not None:
            previous = time_values.groupby(
                pair_values, dropna=False, sort=False).shift(1)
            gap = (time_values - previous).to_numpy(dtype=np.float64)
            left, right = _split(gap, n_train)
            _add_factor(
                train_model, apply_model, audit,
                name=f"v13_pair_{s1}__{s2}_prev_gap",
                family="entity_pair_velocity",
                inputs=(first, second, time_col),
                train_values=left, apply_values=right,
            )
        nunique = _prior_partner_nunique(stream[first], pair_prior)
        left, right = _split(nunique, n_train)
        _add_factor(
            train_model, apply_model, audit,
            name=f"v13_{s1}_prior_{s2}_nunique",
            family="cross_key_prior_nunique",
            inputs=(first, second), train_values=left, apply_values=right,
        )
        pair_meta = {"e1": first, "e2": second}

    names = [entry.name for entry in audit]
    spec = {
        "entities": [entity.name for entity in entities],
        "amount_used": amount,
        "entity_te": dict(entity_te),
        "te_columns": list(te_columns),
        "pair": pair_meta,
        "names": list(names),
    }
    if frozen_spec is not None:
        expected = [str(name) for name in frozen_spec.get("names", [])]
        if names != expected:
            missing = sorted(set(expected) - set(names))
            extra = sorted(set(names) - set(expected))
            raise RuntimeError(
                "frozen factor-spec replay produced a different bank: "
                f"missing={missing} extra={extra}")
    return names, {
        "entities": [asdict(entity) for entity in entities],
        "entity_rule": {
            "id_like_regex": _ENTITY.pattern,
            "min_repeat_fraction": float(min_repeat_fraction),
            "min_unique": int(min_unique),
            "top_k": int(top_entities),
            "unique_ratio_cap": None,
        },
        "amount_candidates": amounts,
        "amount_used": amount,
        "te_columns": te_columns,
        "pair": pair_meta,
        "factors": [entry.to_dict() for entry in audit],
        "label_channel": "forward_te_cross_fitted_only",
        "spec": spec,
        "replayed_from_spec": frozen_spec is not None,
    }
