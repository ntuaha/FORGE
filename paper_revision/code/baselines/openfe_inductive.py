from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "OpenFE"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from openfe.FeatureGenerator import FNode, Node
from openfe.utils import formula_to_tree


def _keys(left: pd.Series, right: pd.Series) -> pd.Series:
    result = left.astype("string") + "_" + right.astype("string")
    result[left.isna() | right.isna()] = pd.NA
    return result


def _map_stat(
    train_value: pd.Series,
    val_value: pd.Series,
    train_group: pd.Series,
    val_group: pd.Series,
    aggregation: str,
) -> tuple[pd.Series, pd.Series]:
    grouped = train_value.groupby(train_group, dropna=False)
    statistic = getattr(grouped, aggregation)()
    return train_group.map(statistic), val_group.map(statistic)


def _group_rank(
    train_value: pd.Series,
    val_value: pd.Series,
    train_group: pd.Series,
    val_group: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    train_rank = train_value.groupby(train_group, dropna=False).rank(ascending=True, pct=True)
    val_rank = pd.Series(np.nan, index=val_value.index, dtype=float)
    for group_value, positions in val_group.groupby(val_group, dropna=False).groups.items():
        if pd.isna(group_value):
            mask = train_group.isna()
        else:
            mask = train_group == group_value
        reference = np.sort(pd.to_numeric(train_value[mask], errors="coerce").dropna().to_numpy())
        if len(reference) == 0:
            continue
        values = pd.to_numeric(val_value.loc[positions], errors="coerce").to_numpy()
        val_rank.loc[positions] = np.searchsorted(reference, values, side="right") / len(reference)
    return train_rank, val_rank


def _execute(node, train: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if isinstance(node, FNode):
        return train[node.name], validation[node.name]

    children = [_execute(child, train, validation) for child in node.children]
    name = node.name
    if len(children) == 1:
        train_value, val_value = children[0]
        if name == "abs":
            return train_value.abs(), val_value.abs()
        if name == "log":
            return np.log(np.abs(train_value.replace(0, np.nan))), np.log(np.abs(val_value.replace(0, np.nan)))
        if name == "sqrt":
            return np.sqrt(np.abs(train_value)), np.sqrt(np.abs(val_value))
        if name == "square":
            return np.square(train_value), np.square(val_value)
        if name == "sigmoid":
            return 1 / (1 + np.exp(-train_value)), 1 / (1 + np.exp(-val_value))
        if name == "round":
            return np.floor(train_value), np.floor(val_value)
        if name == "residual":
            return train_value - np.floor(train_value), val_value - np.floor(val_value)
        if name == "freq":
            counts = train_value.value_counts(dropna=False)
            return train_value.map(counts), val_value.map(counts)
        raise NotImplementedError(name)

    (train_left, val_left), (train_right, val_right) = children
    if name == "+":
        return train_left + train_right, val_left + val_right
    if name == "-":
        return train_left - train_right, val_left - val_right
    if name == "*":
        return train_left * train_right, val_left * val_right
    if name == "/":
        return train_left / train_right.replace(0, np.nan), val_left / val_right.replace(0, np.nan)
    if name == "min":
        return np.minimum(train_left, train_right), np.minimum(val_left, val_right)
    if name == "max":
        return np.maximum(train_left, train_right), np.maximum(val_left, val_right)
    if name == "Combine":
        train_key, val_key = _keys(train_left, train_right), _keys(val_left, val_right)
        categories = {value: index for index, value in enumerate(train_key.dropna().unique())}
        return train_key.map(categories).astype(float), val_key.map(categories).astype(float)
    if name == "CombineThenFreq":
        train_key, val_key = _keys(train_left, train_right), _keys(val_left, val_right)
        counts = train_key.value_counts(dropna=False)
        return train_key.map(counts), val_key.map(counts)
    if name == "GroupByThenMin":
        return _map_stat(train_left, val_left, train_right, val_right, "min")
    if name == "GroupByThenMax":
        return _map_stat(train_left, val_left, train_right, val_right, "max")
    if name == "GroupByThenMean":
        return _map_stat(train_left, val_left, train_right, val_right, "mean")
    if name == "GroupByThenMedian":
        return _map_stat(train_left, val_left, train_right, val_right, "median")
    if name == "GroupByThenStd":
        return _map_stat(train_left, val_left, train_right, val_right, "std")
    if name == "GroupByThenRank":
        return _group_rank(train_left, val_left, train_right, val_right)
    if name == "GroupByThenNUnique":
        return _map_stat(train_left, val_left, train_right, val_right, "nunique")
    raise NotImplementedError(name)


def apply_openfe_formula(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    formula: str,
) -> tuple[pd.Series, pd.Series]:
    tree = formula_to_tree(formula)
    train_value, val_value = _execute(tree, train, validation)
    train_value = pd.to_numeric(train_value, errors="coerce").replace([np.inf, -np.inf], np.nan)
    val_value = pd.to_numeric(val_value, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return train_value, val_value

