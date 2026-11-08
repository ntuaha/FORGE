from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd


TEMPORAL_OPS = (
    "lag", "rolling_stat", "expanding_stat", "cum_count",
    "time_since_prev", "group_ratio",
)
STATIC_OPS = ("group_stat", "group_count", "nunique")
AGG_WHITELIST = ("mean", "sum", "std", "max", "min")
VALUE_FREE_OPS = ("cum_count", "time_since_prev", "group_count")
AGG_STAT_OPS = ("rolling_stat", "expanding_stat", "group_stat")
WINDOW_MIN, WINDOW_MAX = 1, 10_000
MAX_ENTITY_COLS = 3

PERIODIC_OP = "periodic"
PERIODIC_UNITS = ("hour", "dayofweek")
PERIODIC_ENCODINGS = ("sin", "cos", "raw")
_EPOCH_WEEKDAY = 3


def validate_agg_spec(
    raw: dict,
    *,
    columns: list[str],
    label: str,
    temporal: bool,
) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("aggregation spec must be an object")
    column_set = {str(c).lower() for c in columns}
    label = str(label).lower()

    op = str(raw.get("op", "")).strip().lower()
    if op == PERIODIC_OP:
        if not temporal:
            raise ValueError(
                "periodic ops require a temporal dataset (no time column here)"
            )
        source = str(raw.get("source", "time") or "time").strip().lower()
        if source != "time":
            raise ValueError("periodic source must be 'time'")
        unit = str(raw.get("unit", "") or "").strip().lower()
        if unit not in PERIODIC_UNITS:
            raise ValueError(f"periodic unit {unit!r} not in {PERIODIC_UNITS}")
        encoding = str(raw.get("encoding", "") or "").strip().lower()
        if encoding not in PERIODIC_ENCODINGS:
            raise ValueError(
                f"periodic encoding {encoding!r} not in {PERIODIC_ENCODINGS}"
            )
        return {
            "op": PERIODIC_OP,
            "source": "time",
            "unit": unit,
            "encoding": encoding,
            "entity_cols": [],
            "value_col": None,
            "agg": None,
            "window": None,
            "shift": None,
        }
    allowed_ops = TEMPORAL_OPS if temporal else STATIC_OPS
    if op not in allowed_ops:
        raise ValueError(
            f"op {op!r} is not allowed for a "
            f"{'temporal' if temporal else 'non-temporal'} dataset"
        )

    entity_cols_raw = raw.get("entity_cols")
    if isinstance(entity_cols_raw, str):
        entity_cols_raw = [entity_cols_raw]
    if not isinstance(entity_cols_raw, (list, tuple)) or not entity_cols_raw:
        raise ValueError("entity_cols must be a non-empty list")
    entity_cols = list(dict.fromkeys(str(c).strip().lower() for c in entity_cols_raw))
    if len(entity_cols) > MAX_ENTITY_COLS:
        raise ValueError(f"at most {MAX_ENTITY_COLS} entity columns are allowed")
    for col in entity_cols:
        if col == label:
            raise ValueError("entity column must never be the label")
        if col not in column_set:
            raise ValueError(f"entity column {col!r} does not exist")

    value_col: str | None = None
    if op not in VALUE_FREE_OPS:
        value_col = str(raw.get("value_col", "") or "").strip().lower()
        if not value_col:
            raise ValueError(f"op {op!r} requires a value_col")
        if value_col == label:
            raise ValueError("value column must never be the label")
        if value_col not in column_set:
            raise ValueError(f"value column {value_col!r} does not exist")

    agg: str | None = None
    if op in AGG_STAT_OPS:
        agg = str(raw.get("agg", "") or "").strip().lower()
        if agg not in AGG_WHITELIST:
            raise ValueError(f"agg {agg!r} is not in the whitelist {AGG_WHITELIST}")

    window: int | None = None
    if op == "rolling_stat":
        try:
            window = int(raw.get("window"))
        except (TypeError, ValueError):
            raise ValueError("rolling_stat requires an integer window")
        if not WINDOW_MIN <= window <= WINDOW_MAX:
            raise ValueError(f"window must be in [{WINDOW_MIN}, {WINDOW_MAX}]")

    shift: int | None = None
    if temporal:
        shift_raw = raw.get("shift", 1)
        try:
            shift = int(1 if shift_raw is None else shift_raw)
        except (TypeError, ValueError):
            raise ValueError("shift must be an integer")
        if not WINDOW_MIN <= shift <= WINDOW_MAX:
            raise ValueError(f"shift must be in [{WINDOW_MIN}, {WINDOW_MAX}]")

    return {
        "op": op,
        "entity_cols": entity_cols,
        "value_col": value_col,
        "agg": agg,
        "window": window,
        "shift": shift,
    }


def _key_frame(frames: list[pd.DataFrame], entity_cols: list[str]) -> pd.DataFrame:
    parts = {}
    for index, col in enumerate(entity_cols):
        parts[f"__e{index}"] = np.concatenate(
            [frame[col].to_numpy() for frame in frames],
        )
    return pd.DataFrame(parts)


def _numeric_concat(frames: list[pd.DataFrame], col: str) -> pd.Series:
    values = pd.concat(
        [pd.to_numeric(frame[col], errors="coerce") for frame in frames],
        ignore_index=True,
    )
    return values.astype("float64").replace([np.inf, -np.inf], np.nan)


def _restore_order(grouped_result: pd.Series, n_keys: int) -> pd.Series:
    return grouped_result.reset_index(
        level=list(range(n_keys)), drop=True,
    ).sort_index()


def _temporal_values(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    spec: dict,
    time_values: np.ndarray | None,
) -> np.ndarray:
    frames = [train, validation]
    entity_cols = spec["entity_cols"]
    frame = _key_frame(frames, entity_cols)
    key_cols = [f"__e{i}" for i in range(len(entity_cols))]
    op = spec["op"]
    shift = int(spec["shift"] or 1)

    if spec["value_col"] is not None:
        frame["__v"] = _numeric_concat(frames, spec["value_col"]).to_numpy()
    if op == "time_since_prev":
        if time_values is None:
            raise ValueError("time_since_prev requires the ordered time values")
        if len(time_values) != len(frame):
            raise ValueError("time values do not align with the partitions")
        frame["__t"] = np.asarray(time_values, dtype="float64")

    grouped = frame.groupby(key_cols, dropna=False, sort=False)
    if op == "lag":
        out = grouped["__v"].shift(shift)
    elif op == "rolling_stat":
        shifted = grouped["__v"].shift(shift)
        keys = [frame[c] for c in key_cols]
        rolled = shifted.groupby(keys, dropna=False, sort=False).rolling(
            int(spec["window"]), min_periods=1,
        )
        out = _restore_order(getattr(rolled, spec["agg"])(), len(key_cols))
    elif op == "expanding_stat":
        shifted = grouped["__v"].shift(shift)
        keys = [frame[c] for c in key_cols]
        expanded = shifted.groupby(keys, dropna=False, sort=False).expanding(
            min_periods=1,
        )
        out = _restore_order(getattr(expanded, spec["agg"])(), len(key_cols))
    elif op == "cum_count":
        out = grouped.cumcount().astype("float64")
    elif op == "time_since_prev":
        out = grouped["__t"].diff(shift)
    elif op == "group_ratio":
        shifted = grouped["__v"].shift(1)
        keys = [frame[c] for c in key_cols]
        expanded = shifted.groupby(keys, dropna=False, sort=False).expanding(
            min_periods=1,
        )
        past_mean = _restore_order(expanded.mean(), len(key_cols))
        out = frame["__v"] / past_mean
    else:
        raise ValueError(f"unknown temporal op {op!r}")
    return out.to_numpy(dtype="float64")


def periodic_values(spec: dict, time_values: np.ndarray) -> np.ndarray:
    t = np.asarray(time_values, dtype="float64")
    if spec["unit"] == "hour":
        phase = np.mod(np.floor_divide(t, 3600.0), 24.0)
        period = 24.0
    elif spec["unit"] == "dayofweek":
        phase = np.mod(np.floor_divide(t, 86400.0) + _EPOCH_WEEKDAY, 7.0)
        period = 7.0
    else:
        raise ValueError(f"unknown periodic unit {spec['unit']!r}")
    encoding = spec["encoding"]
    if encoding == "raw":
        return phase
    angle = 2.0 * np.pi * phase / period
    if encoding == "sin":
        return np.sin(angle)
    if encoding == "cos":
        return np.cos(angle)
    raise ValueError(f"unknown periodic encoding {encoding!r}")


def _static_fit_transform(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    spec: dict,
) -> tuple[np.ndarray, np.ndarray]:
    entity_cols = spec["entity_cols"]
    key_cols = [f"__e{i}" for i in range(len(entity_cols))]
    op = spec["op"]

    fit = _key_frame([train], entity_cols)
    if spec["value_col"] is not None:
        fit["__v"] = _numeric_concat([train], spec["value_col"]).to_numpy()

    if op == "group_stat":
        stats = fit.groupby(key_cols, dropna=False)["__v"].agg(spec["agg"])
        fallback = float(getattr(fit["__v"], spec["agg"])())
    elif op == "group_count":
        stats = fit.groupby(key_cols, dropna=False).size() / max(len(train), 1)
        fallback = 0.0
    elif op == "nunique":
        stats = fit.groupby(key_cols, dropna=False)["__v"].nunique().astype("float64")
        fallback = float(stats.mean()) if len(stats) else 0.0
    else:
        raise ValueError(f"unknown static op {op!r}")
    stats_frame = stats.rename("__stat").reset_index()

    def transform(frame: pd.DataFrame) -> np.ndarray:
        probe = _key_frame([frame], entity_cols)
        merged = probe.merge(stats_frame, on=key_cols, how="left")
        if len(merged) != len(probe):
            raise ValueError("static aggregation merge changed row count")
        return merged["__stat"].fillna(fallback).to_numpy(dtype="float64")

    return transform(train), transform(validation)


def apply_agg_spec(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    name: str,
    spec: dict,
    *,
    label: str,
    time_values: np.ndarray | None = None,
) -> bool:
    try:
        label = str(label).lower()
        referenced = list(spec.get("entity_cols") or []) + (
            [spec["value_col"]] if spec.get("value_col") else []
        )
        if any(str(col).lower() == label for col in referenced):
            return False
        if spec["op"] == PERIODIC_OP:
            if time_values is None:
                raise ValueError("periodic ops require the ordered time values")
            if len(time_values) != len(train) + len(validation):
                raise ValueError("time values do not align with the partitions")
            values = periodic_values(spec, np.asarray(time_values, dtype="float64"))
            train_values = values[: len(train)]
            val_values = values[len(train):]
        elif spec["op"] in TEMPORAL_OPS:
            values = _temporal_values(train, validation, spec, time_values)
            train_values = values[: len(train)]
            val_values = values[len(train):]
        else:
            train_values, val_values = _static_fit_transform(train, validation, spec)
        train_series = (
            pd.Series(train_values, index=train.index)
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )
        val_series = (
            pd.Series(val_values, index=validation.index)
            .replace([np.inf, -np.inf], np.nan)
            .astype("float32")
        )
        if train_series.isna().all() or val_series.isna().all():
            return False
        train[name] = train_series.to_numpy()
        validation[name] = val_series.to_numpy()
        return True
    except Exception:
        return False


def entity_candidates(
    train: pd.DataFrame,
    *,
    label: str,
    cats: list[str],
    nums: list[str],
    max_candidates: int = 16,
) -> list[str]:
    n_rows = max(len(train), 1)
    label = str(label).lower()
    object_hits: list[tuple[int, str]] = []
    numeric_hits: list[tuple[int, str]] = []
    for col in dict.fromkeys(list(cats) + list(nums)):
        if col == label or col not in train.columns:
            continue
        series = train[col]
        nunique = int(series.nunique(dropna=True))
        if nunique < 10 or nunique >= n_rows:
            continue
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            if nunique <= 0.98 * n_rows:
                object_hits.append((nunique, col))
            continue
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            continue
        sample = values.iloc[:5000].to_numpy(dtype="float64")
        integral = bool(np.allclose(sample, np.round(sample), rtol=0.0, atol=1e-9))
        if integral and nunique <= 0.5 * n_rows:
            numeric_hits.append((nunique, col))
        elif float(values.min()) >= 0.0 and nunique <= 0.98 * n_rows:
            numeric_hits.append((nunique, col))
    ranked = [col for _, col in sorted(object_hits)] + [
        col for _, col in sorted(numeric_hits)
    ]
    return ranked[:max_candidates]


def value_candidates(
    train: pd.DataFrame,
    *,
    label: str,
    nums: list[str],
    max_candidates: int = 24,
) -> list[str]:
    label = str(label).lower()
    candidates = [
        col for col in dict.fromkeys(nums)
        if col != label and col in train.columns
        and pd.api.types.is_numeric_dtype(train[col])
    ]

    def rank_key(col: str) -> tuple:
        values = pd.to_numeric(train[col], errors="coerce").to_numpy(
            dtype=np.float64)
        finite = values[np.isfinite(values)]
        coverage = float(len(finite)) / len(values) if len(values) else 0.0
        spread = float(finite.std()) if len(finite) > 1 else 0.0
        return (-coverage, -spread, col)

    return sorted(candidates, key=rank_key)[:max_candidates]


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _to_seconds(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(series):
        values = series.dt.as_unit("ns").astype("int64").to_numpy(dtype="float64") / 1e9
        values[series.isna().to_numpy()] = np.nan
        return values
    if pd.api.types.is_timedelta64_dtype(series):
        values = series.dt.total_seconds().to_numpy(dtype="float64")
        return values
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")


def load_split_time_values(cfg: dict, purpose: str) -> tuple[np.ndarray, np.ndarray]:
    time_col = cfg.get("time_col")
    if not time_col:
        raise ValueError("dataset has no time column")
    con = duckdb.connect(str(cfg["db_path"]), read_only=True)
    try:
        con.execute(f"PRAGMA threads={max(1, min(16, int(cfg.get('read_threads', 16))))}")
        quoted = _quote(time_col)
        frame = con.execute(
            f"SELECT {quoted} AS t FROM {_quote(cfg['table'])} ORDER BY {quoted}, rowid"
        ).df()
    finally:
        con.close()
    values = _to_seconds(frame["t"])
    n_total = len(values)
    outer_cut = int(math.floor(n_total * (1.0 - cfg["outer_test_fraction"])))
    search_cut = int(math.floor(outer_cut * (1.0 - cfg["search_val_fraction_of_outer"])))
    if purpose == "search":
        return values[:search_cut], values[search_cut:outer_cut]
    if purpose == "outer":
        return values[:outer_cut], values[outer_cut:]
    raise ValueError(f"unknown partition purpose {purpose!r}")
