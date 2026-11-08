from __future__ import annotations

import ast
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROTOCOL_VERSION = "nested-v2-lcb-2026-07-10"


@dataclass(frozen=True)
class SplitManifest:
    dataset: str
    split_kind: str
    split_seed: int
    n_inner_train: int
    n_search_val: int
    n_outer_train: int
    n_outer_test: int
    inner_train_positive_rate: float
    search_val_positive_rate: float
    outer_train_positive_rate: float
    outer_test_positive_rate: float
    protocol_version: str = PROTOCOL_VERSION


@dataclass
class SearchData:

    train: pd.DataFrame
    validation: pd.DataFrame
    features: list[str]
    label: str
    meta: dict
    manifest: SplitManifest


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    expression: str
    family: str = "llm"
    source: str = "proposal"
    description: str = ""
    executor: str = "pointwise"

    def to_dict(self) -> dict:
        return asdict(self)


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _open(cfg: dict):
    kind = cfg.get("source_kind", "duckdb")
    path = str(cfg.get("source_path", cfg["db_path"]))
    if kind == "duckdb":
        con = duckdb.connect(path, read_only=True)
        relation = _quote(cfg["table"])
    else:
        con = duckdb.connect(":memory:")
        reader = "read_parquet" if kind == "parquet" else "read_csv_auto"
        relation = f"{reader}('{path}')"
    con.execute(
        f"PRAGMA threads={max(1, min(16, int(cfg.get('read_threads', 16))))}")
    return con, relation


def _read_count(cfg: dict) -> int:
    con, relation = _open(cfg)
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM {relation}").fetchone()[0])
    finally:
        con.close()


def _read_temporal_range(cfg: dict, start: int, stop: int) -> pd.DataFrame:
    limit = max(0, int(stop) - int(start))
    con, relation = _open(cfg)
    try:
        time_col = _quote(cfg["time_col"])
        tiebreak = ", rowid" if cfg.get("source_kind", "duckdb") == "duckdb" else ""
        query = (
            f"SELECT * FROM {relation} "
            f"ORDER BY {time_col}{tiebreak} LIMIT {limit} OFFSET {int(start)}"
        )
        return con.execute(query).df()
    finally:
        con.close()


def _read_all(cfg: dict) -> pd.DataFrame:
    con, relation = _open(cfg)
    try:
        return con.execute(f"SELECT * FROM {relation}").df()
    finally:
        con.close()


def _normalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df.columns = [str(c).lower() for c in df.columns]
    if label not in df.columns:
        raise KeyError(f"label {label!r} not found; columns begin {list(df.columns[:8])}")
    df[label] = pd.to_numeric(df[label], errors="coerce").fillna(0).astype("int8")
    return df.reset_index(drop=True)


def _load_raw_partitions(cfg: dict, purpose: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    label = str(cfg["label"]).lower()
    time_col = cfg.get("time_col")
    if time_col:
        n_total = _read_count(cfg)
        outer_cut = int(math.floor(n_total * (1.0 - cfg["outer_test_fraction"])))
        search_cut = int(math.floor(outer_cut * (1.0 - cfg["search_val_fraction_of_outer"])))
        if purpose == "search":
            left = _read_temporal_range(cfg, 0, search_cut)
            right = _read_temporal_range(cfg, search_cut, outer_cut)
        elif purpose == "outer":
            left = _read_temporal_range(cfg, 0, outer_cut)
            right = _read_temporal_range(cfg, outer_cut, n_total)
        else:
            raise ValueError(f"unknown partition purpose {purpose!r}")
        split_info = dict(
            split_kind="chronological",
            n_total=n_total,
            n_outer_train=outer_cut,
            n_outer_test=n_total - outer_cut,
            n_inner_train=search_cut,
            n_search_val=outer_cut - search_cut,
        )
    else:
        full = _normalize(_read_all(cfg), label)
        outer_train, outer_test = train_test_split(
            full,
            test_size=cfg["outer_test_fraction"],
            random_state=cfg["split_seed"],
            stratify=full[label],
        )
        del full
        outer_train = outer_train.reset_index(drop=True)
        outer_test = outer_test.reset_index(drop=True)
        inner_train, search_val = train_test_split(
            outer_train,
            test_size=cfg["search_val_fraction_of_outer"],
            random_state=cfg["split_seed"] + 1,
            stratify=outer_train[label],
        )
        if purpose == "search":
            left, right = inner_train, search_val
        elif purpose == "outer":
            left, right = outer_train, outer_test
        else:
            raise ValueError(f"unknown partition purpose {purpose!r}")
        split_info = dict(
            split_kind="stratified_random",
            n_total=len(outer_train) + len(outer_test),
            n_outer_train=len(outer_train),
            n_outer_test=len(outer_test),
            n_inner_train=len(inner_train),
            n_search_val=len(search_val),
        )
    return _normalize(left, label), _normalize(right, label), split_info


def _classify_columns(
    train: pd.DataFrame,
    label: str,
    drop: Iterable[str],
    time_col: str | None,
) -> tuple[list[str], list[str], list[str]]:
    drop_set = {str(c).lower() for c in drop}
    cats: list[str] = []
    nums: list[str] = []
    bins: list[str] = []
    n_rows = len(train)
    for col in train.columns:
        if col == label or col in drop_set or col == time_col:
            continue
        series = train[col]
        nunique = int(series.nunique(dropna=True))
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            cats.append(col)
        elif nunique <= 2:
            bins.append(col)
        elif pd.api.types.is_integer_dtype(series) and 10 < nunique < min(50_000, 0.5 * n_rows):
            nums.append(col)
            cats.append(col)
        else:
            nums.append(col)
    return cats, nums, bins


def _numeric_array(values, index: pd.Index) -> np.ndarray:
    series = pd.Series(values, index=index)
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .astype("float32")
        .to_numpy()
    )


def _assign_numeric(frame: pd.DataFrame, name: str, values) -> None:
    frame[name] = _numeric_array(values, frame.index)


def _random_oof_encodings(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    col: str,
    label: str,
    seed: int,
    smooth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = train[label].to_numpy(dtype=np.int8, copy=False)
    min_class = int(np.bincount(y).min())
    n_splits = max(2, min(5, min_class))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_te = np.full(len(train), np.nan, dtype=np.float64)
    oof_freq = np.zeros(len(train), dtype=np.float64)
    for fit_idx, hold_idx in splitter.split(np.zeros(len(train)), y):
        fit = train.iloc[fit_idx]
        hold = train.iloc[hold_idx]
        fold_mean = float(fit[label].mean())
        stats = fit.groupby(col, dropna=False)[label].agg(["sum", "count"])
        mapping = (stats["sum"] + smooth * fold_mean) / (stats["count"] + smooth)
        oof_te[hold_idx] = hold[col].map(mapping).fillna(fold_mean).to_numpy()
        freq = fit[col].value_counts(dropna=False, normalize=True)
        oof_freq[hold_idx] = hold[col].map(freq).fillna(0.0).to_numpy()

    full_mean = float(train[label].mean())
    full_stats = train.groupby(col, dropna=False)[label].agg(["sum", "count"])
    full_mapping = (
        (full_stats["sum"] + smooth * full_mean) /
        (full_stats["count"] + smooth)
    )
    val_te = validation[col].map(full_mapping).fillna(full_mean).to_numpy(dtype=float)
    full_freq = train[col].value_counts(dropna=False, normalize=True)
    val_freq = validation[col].map(full_freq).fillna(0.0).to_numpy(dtype=float)
    return oof_te, val_te, oof_freq, val_freq


def _forward_encodings(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    col: str,
    label: str,
    smooth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = train[label].astype("float64")
    grouped = train.groupby(col, dropna=False, sort=False)
    past_count = grouped.cumcount().astype("float64")
    past_sum = grouped[label].cumsum().astype("float64") - y
    row_number = np.arange(len(train), dtype=np.float64)
    past_global = (y.cumsum().to_numpy() - y.to_numpy() + 1.0) / (row_number + 2.0)
    train_te = (past_sum.to_numpy() + smooth * past_global) / (past_count.to_numpy() + smooth)
    train_freq = past_count.to_numpy() / np.maximum(row_number, 1.0)

    full_mean = float(y.mean())
    stats = train.groupby(col, dropna=False)[label].agg(["sum", "count"])
    mapping = (stats["sum"] + smooth * full_mean) / (stats["count"] + smooth)
    val_te = validation[col].map(mapping).fillna(full_mean).to_numpy(dtype=float)
    freq = train[col].value_counts(dropna=False, normalize=True)
    val_freq = validation[col].map(freq).fillna(0.0).to_numpy(dtype=float)
    return train_te, val_te, train_freq, val_freq


def prepare_e1(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    label = str(cfg["label"]).lower()
    time_col = str(cfg["time_col"]).lower() if cfg.get("time_col") else None
    cats, nums, bins = _classify_columns(train, label, cfg.get("drop", []), time_col)
    features = list(dict.fromkeys(nums + bins))
    train_columns: dict[str, np.ndarray] = {
        label: train[label].to_numpy(copy=False),
    }
    validation_columns: dict[str, np.ndarray] = {
        label: validation[label].to_numpy(copy=False),
    }

    for col in features:
        train_columns[col] = _numeric_array(train[col], train.index)
        validation_columns[col] = _numeric_array(validation[col], validation.index)

    for col in cats:
        if time_col:
            encoded = _forward_encodings(train, validation, col, label, smooth=20.0)
        else:
            encoded = _random_oof_encodings(
                train, validation, col, label, cfg["split_seed"], smooth=20.0,
            )
        te_name = f"{col}_oof_te"
        freq_name = f"{col}_freq"
        train_columns[te_name] = _numeric_array(encoded[0], train.index)
        validation_columns[te_name] = _numeric_array(encoded[1], validation.index)
        train_columns[freq_name] = _numeric_array(encoded[2], train.index)
        validation_columns[freq_name] = _numeric_array(encoded[3], validation.index)
        features.extend([te_name, freq_name])

    for col in nums:
        numeric = pd.to_numeric(train[col], errors="coerce")
        if numeric.notna().any() and numeric.min(skipna=True) >= 0 and numeric.skew() > 2:
            name = f"{col}_log"
            train_columns[name] = _numeric_array(
                np.log1p(numeric.clip(lower=0)), train.index,
            )
            val_numeric = pd.to_numeric(validation[col], errors="coerce")
            validation_columns[name] = _numeric_array(
                np.log1p(val_numeric.clip(lower=0)), validation.index,
            )
            features.append(name)

    for col in cats:
        if col not in train_columns:
            train_columns[col] = train[col].to_numpy(copy=False)
            validation_columns[col] = validation[col].to_numpy(copy=False)

    meta = dict(
        dataset=cfg["name"],
        desc=cfg.get("desc", ""),
        cats=cats,
        nums=nums,
        binaries=bins,
        time_col=time_col,
        split_kind="chronological" if time_col else "stratified_random",
        final_grid=cfg.get("final_grid"),
        final_no_xgb=bool(cfg.get("final_no_xgb", False)),
        lgbm_override=dict(cfg.get("lgbm_override", {})),
    )
    features = list(dict.fromkeys(features))
    train_out = pd.DataFrame(train_columns, index=train.index)
    validation_out = pd.DataFrame(validation_columns, index=validation.index)
    return train_out, validation_out, features, meta


def prepare_e0(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict]:
    label = str(cfg["label"]).lower()
    time_col = str(cfg["time_col"]).lower() if cfg.get("time_col") else None
    cats, nums, bins = _classify_columns(train, label, cfg.get("drop", []), time_col)
    features = list(dict.fromkeys(nums + bins + cats))
    train_columns: dict[str, np.ndarray] = {
        label: train[label].to_numpy(copy=False),
    }
    validation_columns: dict[str, np.ndarray] = {
        label: validation[label].to_numpy(copy=False),
    }

    for col in features:
        if col in cats and (
            pd.api.types.is_object_dtype(train[col]) or
            pd.api.types.is_string_dtype(train[col])
        ):
            observed = pd.Index(train[col].dropna().astype(str).unique())
            mapping = pd.Series(np.arange(len(observed), dtype=np.float32), index=observed)
            train_values = train[col].astype("string").map(mapping).fillna(-1.0)
            val_values = validation[col].astype("string").map(mapping).fillna(-1.0)
            train_columns[col] = _numeric_array(train_values, train.index)
            validation_columns[col] = _numeric_array(val_values, validation.index)
        else:
            train_columns[col] = _numeric_array(train[col], train.index)
            validation_columns[col] = _numeric_array(validation[col], validation.index)

    meta = dict(
        dataset=cfg["name"],
        desc=cfg.get("desc", ""),
        cats=cats,
        nums=nums,
        binaries=bins,
        time_col=time_col,
        split_kind="chronological" if time_col else "stratified_random",
        representation="e0",
        final_grid=cfg.get("final_grid"),
        final_no_xgb=bool(cfg.get("final_no_xgb", False)),
        lgbm_override=dict(cfg.get("lgbm_override", {})),
    )
    train_out = pd.DataFrame(train_columns, index=train.index)
    validation_out = pd.DataFrame(validation_columns, index=validation.index)
    return train_out, validation_out, features, meta


def _positive_rate(frame: pd.DataFrame, label: str) -> float:
    return round(float(frame[label].mean()), 10)


def load_search_data(cfg: dict, representation: str | None = None) -> SearchData:
    if representation is None:
        representation = os.environ.get("BASELINE_REPRESENTATION", "e0")
    inner_train, search_val, info = _load_raw_partitions(cfg, "search")
    label = str(cfg["label"]).lower()
    if representation == "e1":
        inner_train, search_val, features, meta = prepare_e1(inner_train, search_val, cfg)
        meta["representation"] = "e1"
    elif representation == "e0":
        inner_train, search_val, features, meta = prepare_e0(inner_train, search_val, cfg)
    else:
        raise ValueError("representation must be 'e0' or 'e1'")

    if cfg.get("time_col"):
        outer_train_rate = float("nan")
        outer_test_rate = float("nan")
    else:
        outer_train_rate = (
            len(inner_train) * _positive_rate(inner_train, label) +
            len(search_val) * _positive_rate(search_val, label)
        ) / max(len(inner_train) + len(search_val), 1)
        outer_test_rate = float("nan")

    manifest = SplitManifest(
        dataset=cfg["name"],
        split_kind=info["split_kind"],
        split_seed=int(cfg["split_seed"]),
        n_inner_train=int(info["n_inner_train"]),
        n_search_val=int(info["n_search_val"]),
        n_outer_train=int(info["n_outer_train"]),
        n_outer_test=int(info["n_outer_test"]),
        inner_train_positive_rate=_positive_rate(inner_train, label),
        search_val_positive_rate=_positive_rate(search_val, label),
        outer_train_positive_rate=round(float(outer_train_rate), 10),
        outer_test_positive_rate=round(float(outer_test_rate), 10),
    )
    return SearchData(inner_train, search_val, features, label, meta, manifest)


def load_outer_partitions(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return _load_raw_partitions(cfg, "outer")


def split_outer_train_for_selection(
    outer_train: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label = str(cfg["label"]).lower()
    if cfg.get("time_col"):
        cut = int(len(outer_train) * (1.0 - cfg["search_val_fraction_of_outer"]))
        return (
            outer_train.iloc[:cut].copy().reset_index(drop=True),
            outer_train.iloc[cut:].copy().reset_index(drop=True),
        )
    fit, validation = train_test_split(
        outer_train,
        test_size=cfg["search_val_fraction_of_outer"],
        random_state=cfg["split_seed"] + 2,
        stratify=outer_train[label],
    )
    return fit.reset_index(drop=True), validation.reset_index(drop=True)


_ALLOWED_NAMES = {"df", "np", "pd"}
_ALLOWED_SERIES_METHODS = {
    "abs", "astype", "between", "clip", "fillna", "isna", "notna",
    "isin", "pow", "replace", "round",
}
_ALLOWED_NUMPY_CALLS = {
    "abs", "clip", "exp", "expm1", "isfinite", "isnan", "log", "log1p",
    "maximum", "minimum", "power", "sqrt", "square", "where",
}
_ALLOWED_PANDAS_CALLS = {"to_numeric", "isna", "notna"}
_BANNED_TEXT = ("groupby", "transform", "rolling", "expanding", "rank", "merge", "join")


class _ExpressionValidator(ast.NodeVisitor):
    def __init__(self, forbidden_columns: Iterable[str] = ()) -> None:
        self.forbidden_columns = {str(value).lower() for value in forbidden_columns}

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in _ALLOWED_NAMES:
            raise ValueError(f"name {node.id!r} is not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            raise ValueError("private attributes are not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if not isinstance(func, ast.Attribute):
            raise ValueError("only qualified calls are allowed")
        if isinstance(func.value, ast.Name) and func.value.id == "np":
            if func.attr not in _ALLOWED_NUMPY_CALLS:
                raise ValueError(f"np.{func.attr} is not allowed")
        elif isinstance(func.value, ast.Name) and func.value.id == "pd":
            if func.attr not in _ALLOWED_PANDAS_CALLS:
                raise ValueError(f"pd.{func.attr} is not allowed")
        elif func.attr not in _ALLOWED_SERIES_METHODS:
            raise ValueError(f"Series method {func.attr!r} is not allowed")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if not isinstance(node.value, ast.Name) or node.value.id != "df":
            raise ValueError("subscripts are restricted to df['column']")
        key = node.slice
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise ValueError("dataframe column must be a string literal")
        if key.value.lower() in self.forbidden_columns:
            raise ValueError(f"forbidden column {key.value!r} in feature expression")
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        raise ValueError("lambda is not allowed")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        raise ValueError("comprehensions are not allowed")


def compile_pointwise_expression(
    expression: str,
    forbidden_columns: Iterable[str] = (),
) -> Callable[[pd.DataFrame], pd.Series]:
    if any(token in expression.lower() for token in _BANNED_TEXT):
        raise ValueError("data-dependent aggregation is not a pointwise feature")
    tree = ast.parse(expression, mode="eval")
    _ExpressionValidator(forbidden_columns).visit(tree)
    code = compile(tree, "<feature-expression>", "eval")

    def execute(df: pd.DataFrame) -> pd.Series:
        value = eval(code, {"__builtins__": {}}, {"df": df, "np": np, "pd": pd})
        if np.isscalar(value):
            value = pd.Series(value, index=df.index)
        return pd.Series(value, index=df.index)

    return execute


def apply_feature_spec(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    spec: FeatureSpec,
    label: str | None = None,
) -> bool:
    try:
        if spec.executor == "openfe":
            from openfe_inductive import apply_openfe_formula

            train_value, val_value = apply_openfe_formula(
                train, validation, spec.expression,
            )
            train[spec.name] = train_value.astype("float32").to_numpy()
            validation[spec.name] = val_value.astype("float32").to_numpy()
            return not train[spec.name].isna().all() and not validation[spec.name].isna().all()
        if spec.executor != "pointwise":
            raise ValueError(f"unknown feature executor {spec.executor!r}")
        fn = compile_pointwise_expression(
            spec.expression,
            forbidden_columns=(() if label is None else (label,)),
        )
        train_value = pd.to_numeric(fn(train), errors="coerce").replace([np.inf, -np.inf], np.nan)
        val_value = pd.to_numeric(fn(validation), errors="coerce").replace([np.inf, -np.inf], np.nan)
        if train_value.isna().all() or val_value.isna().all():
            return False
        train[spec.name] = train_value.astype("float32").to_numpy()
        validation[spec.name] = val_value.astype("float32").to_numpy()
        return True
    except Exception:
        return False


def apply_feature_specs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    specs: Iterable[FeatureSpec],
    label: str | None = None,
) -> list[str]:
    applied: list[str] = []
    for spec in specs:
        if spec.name in train.columns or spec.name in validation.columns:
            continue
        if apply_feature_spec(train, validation, spec, label=label):
            applied.append(spec.name)
        else:
            raise ValueError(f"failed to reproduce feature {spec.name!r}")
    return applied


def design_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    matrix = np.empty((len(frame), len(features)), dtype=np.float32)
    for index, feature in enumerate(features):
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        matrix[:, index] = values
    matrix[np.isinf(matrix)] = np.nan
    return matrix
