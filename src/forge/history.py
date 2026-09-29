"""Strictly causal entity-history features.

Used by CoF aggregate templates (S1.1.1) and by the fixed entity-history
library of the routing rule (S1.2): prior event counts, repeat indicators,
time since the previous event and a value's deviation from the entity's
earlier values. A row only sees events of the same entity at a strictly
earlier time; no labels are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HISTORY_AGGREGATIONS = ("count", "repeat", "mean", "sum", "std", "last", "deviation", "time_since")
"""Aggregations of a causal entity-history feature (CoF aggregate templates and the fixed library)."""


def detect_entity_columns(raw: pd.DataFrame, cfg: dict) -> list[str]:
    """Entity columns for entity-history features.

    The configured ``entity_cols`` plus automatically detected entity columns:
    non-numeric predictors with at least 20 distinct values that repeat (at
    most one distinct value per two rows). At most four columns are used.

    Paper difference: the paper detects entity columns automatically only; the
    dataset files in ``configs/datasets/`` also list the entities of the
    paper's Table 1 (e.g. device and IP for Fraudecom), because numeric
    identifiers such as an IP address are not detected by this rule.
    """
    found = list(cfg["entity_cols"])
    skip = set(cfg["drop"]) | set(found) | {cfg["label"], cfg["time_col"]}
    candidates = []
    for column in raw.columns:
        if column in skip or pd.api.types.is_numeric_dtype(raw[column]) or pd.api.types.is_bool_dtype(raw[column]):
            continue
        nunique = int(raw[column].nunique(dropna=True))
        if 20 <= nunique <= 0.5 * len(raw):
            candidates.append((-nunique, column))
    found += [column for _, column in sorted(candidates)]
    return found[:4]


def _event_order(history: pd.DataFrame, queries: pd.DataFrame | None, cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows whose past is visible and their event times.

    ``history`` holds the fitting partition. Query rows of a later partition
    (``V`` during search, ``T`` in S3) are appended: their feature values are
    label-free, so earlier query rows may be part of the history. Without an
    event-time column the partition order is used as time.
    """
    frame = history if queries is None else pd.concat([history, queries], ignore_index=True)
    if cfg["time_col"] and pd.api.types.is_datetime64_any_dtype(frame[cfg["time_col"]]):
        times = frame[cfg["time_col"]].to_numpy("datetime64[ns]").astype("int64") / 1e9   # seconds
    elif cfg["time_col"]:
        times = pd.to_numeric(frame[cfg["time_col"]], errors="raise").to_numpy(float)
    else:
        times = np.arange(len(frame), dtype=float)
    return frame.reset_index(drop=True), times


def entity_history(history: pd.DataFrame, queries: pd.DataFrame | None, spec: dict, cfg: dict) -> np.ndarray:
    """Strictly causal entity-history feature (CoF aggregate templates, library).

    ``spec`` holds ``entity`` (one column or a list of columns forming a
    pair), ``value`` (a numeric column, unused by ``count``/``repeat``/
    ``time_since``), ``agg`` (one of :data:`HISTORY_AGGREGATIONS`), ``window``
    (number of most recent past events; 0 = full history) and ``shift``
    (>= 1; ``shift = s`` also skips the ``s - 1`` most recent past events).
    Only events at a strictly earlier time than the current row contribute,
    so rows at the same timestamp never see each other. Returns the values of
    ``history`` (when ``queries`` is None) or of ``queries``.
    """
    agg, window, shift = spec["agg"], int(spec.get("window", 0)), int(spec.get("shift", 1))
    entity = [spec["entity"]] if isinstance(spec["entity"], str) else list(spec["entity"])
    if agg not in HISTORY_AGGREGATIONS or shift < 1 or window < 0:
        raise ValueError("invalid entity-history specification (aggregation, window >= 0, shift >= 1)")
    forbidden = {cfg["label"], cfg["time_col"]}
    if set(entity) & forbidden or spec.get("value") in forbidden:
        raise ValueError("an entity-history feature cannot use the label or the time column")
    frame, times = _event_order(history, queries, cfg)
    keys = frame.groupby(entity, sort=False, dropna=False).ngroup().to_numpy()
    order = np.lexsort((np.arange(len(frame)), times, keys))
    k, t = keys[order], times[order]
    idx = np.arange(len(frame))
    group_start = np.maximum.accumulate(np.where(np.r_[True, k[1:] != k[:-1]], idx, 0))
    tie_start = np.maximum.accumulate(np.where(np.r_[True, (k[1:] != k[:-1]) | (t[1:] != t[:-1])], idx, 0))
    earlier = tie_start - group_start                          # past events of the same entity
    end = group_start + np.maximum(earlier - (shift - 1), 0)
    begin = np.maximum(end - window, group_start) if window else group_start
    n = (end - begin).astype(float)
    has = n > 0
    if agg in ("count", "repeat", "time_since"):
        values = np.zeros(len(frame))
    else:
        values = pd.to_numeric(frame[spec["value"]], errors="coerce").fillna(0).to_numpy(float)
    v = values[order]
    cs, cs2 = np.r_[0., np.cumsum(v)], np.r_[0., np.cumsum(v * v)]
    total = cs[end] - cs[begin]
    mean = np.divide(total, n, out=np.zeros_like(total), where=has)
    last_index = np.maximum(end - 1, 0)
    if agg == "count":
        out = n
    elif agg == "repeat":
        out = has.astype(float)
    elif agg == "sum":
        out = total
    elif agg == "mean":
        out = mean
    elif agg == "std":
        second = np.divide(cs2[end] - cs2[begin], n, out=np.zeros_like(total), where=has)
        out = np.sqrt(np.maximum(second - mean * mean, 0))
    elif agg == "last":
        out = np.where(has, v[last_index], 0.)
    elif agg == "deviation":
        out = np.where(has, v - mean, 0.)
    else:                                                       # time_since
        out = np.where(has, t - t[last_index], -1.)
    result = np.empty(len(frame), dtype=np.float32)
    result[order] = out
    return result if queries is None else result[len(history):]


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_") or "col"


def history_feature_name(spec: dict) -> str:
    """Canonical name of an entity-history feature."""
    entity = [spec["entity"]] if isinstance(spec["entity"], str) else list(spec["entity"])
    parts = ["hist", spec["agg"], "_".join(_safe_name(e) for e in entity)]
    if spec["agg"] not in ("count", "repeat", "time_since"):
        parts.append(_safe_name(spec["value"]))
    if spec.get("window", 0):
        parts.append(f"w{int(spec['window'])}")
    if spec.get("shift", 1) != 1:
        parts.append(f"s{int(spec['shift'])}")
    return "__".join(parts)


def entity_history_library(raw: pd.DataFrame, cfg: dict, plan: dict) -> dict:
    """Fixed library of causal entity-history features (routing rule and CoF cycle 1).

    Over the detected entity columns and their pairs: prior event counts,
    repeat indicators, time since the previous event (event-time data only)
    and the deviation of a value from the entity's earlier values (first
    three numeric columns). Empty when no entity column is found.
    """
    entities = detect_entity_columns(raw, cfg)
    keys = [[e] for e in entities] + [[a, b] for i, a in enumerate(entities) for b in entities[i + 1:]][:3]
    values = [c for c in plan["nums"] if c not in entities][:3]
    specs = []
    for key in keys:
        specs += [{"entity": key, "agg": "count"}, {"entity": key, "agg": "repeat"}]
        if cfg["time_col"]:
            specs.append({"entity": key, "agg": "time_since"})
        specs += [{"entity": key, "agg": "deviation", "value": v} for v in values]
    library = {}
    for spec in specs:
        spec = {"executor": "entity_history", "window": 0, "shift": 1, "value": "", **spec,
                "family": "history_library", "description": f"causal entity-history {spec['agg']}"}
        library[history_feature_name(spec)] = spec
    return library
