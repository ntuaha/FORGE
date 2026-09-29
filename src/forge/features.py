"""Feature representation, safe candidate expressions and materialisation.

Two representations are used:

* **Search representation** (:func:`encode_search`): numeric and binary raw
  columns, ``log1p`` of non-negative right-skewed columns, and for every
  categorical-like column a smoothed target encoding (``_oof_te``,
  Eq. (15) with alpha = 20, five shuffled out-of-fold folds on the fitting
  rows) and a frequency encoding (``_freq``). It is computed once on ``I``
  during search and once on ``I + V`` in the S3 rebuild. The learner uses the
  raw columns (``plan['base']``); the encodings are available to candidate
  expressions.
* **Expert representation** (:func:`encode_expert`): the same raw columns, but
  target encodings are built from *expanding blocks* so that each block only
  uses labels of preceding blocks (S2.1.1). Used by S2 experts.

Candidate features are pointwise Python expressions over ``df['column']``
(S1.1.1) or strictly causal entity-history aggregates (:mod:`forge.history`).
:func:`validate_pointwise_expression` whitelists the syntax so that a
candidate cannot read the label, perform I/O, aggregate over rows or inspect
objects.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .history import entity_history

# --------------------------------------------------------------------------- column plan
def classify_columns(train: pd.DataFrame, cfg: dict) -> tuple[list[str], list[str], list[str]]:
    """Classify predictors of ``train`` into (categorical, numeric, binary) lists.

    Object/string columns are categorical; columns with at most two values are
    binary; integer columns with 10 < cardinality < min(50,000, n/2) are both
    numeric and categorical (they get raw values and encodings); all other
    columns are numeric. Label, dropped and time columns are skipped; entity
    columns stay predictors (they also feed entity-history features).

    Code detail: the paper only says the Raw representation holds "the
    original columns with learner-required preprocessing"; the thresholds
    below are this implementation's choice.
    """
    skip = {str(c).lower() for c in cfg["drop"]} | {cfg["label"], cfg["time_col"]}
    cats, nums, binaries = [], [], []
    for column in train.columns:
        if column in skip:
            continue
        series = train[column]
        nunique = int(series.nunique(dropna=True))
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            cats.append(column)
        elif nunique <= 2:
            binaries.append(column)
        elif (pd.api.types.is_integer_dtype(series)
              and cfg["category_min_unique"] < nunique < min(cfg["category_max_unique"], 0.5 * len(train))):
            nums.append(column)
            cats.append(column)
        else:
            nums.append(column)
    return cats, nums, binaries


def make_representation_plan(raw: pd.DataFrame, cfg: dict) -> dict:
    """Fix the learner's base columns from ``I`` (the Raw representation).

    ``base`` = numeric + binary columns, ``<col>_log`` for non-negative columns
    with skew > 2, and ``<col>_code`` (label-free ordinal codes) for purely
    categorical columns.

    Paper difference: the paper (Sec. V-B) describes Raw as "the original
    columns with learner-required preprocessing only". This code also adds the
    ``_log`` columns. The baseline trained here is not the Raw column of the
    paper's Table 7, which comes from a separate baseline run (Credit Default
    seed 11: 0.5482592 here, 0.5402590 in the paper).
    """
    cats, nums, binaries = classify_columns(raw, cfg)
    logs = [c for c in nums if raw[c].notna().any() and raw[c].min() >= 0 and raw[c].skew() > cfg["log_skew"]]
    codes = [c for c in cats if c not in nums and c not in binaries]
    return {"cats": cats, "nums": nums, "binaries": binaries, "logs": logs, "codes": codes,
            "base": list(dict.fromkeys(nums + binaries + [c + "_log" for c in logs]
                                       + [c + "_code" for c in codes]))}


# --------------------------------------------------------------------------- encodings
def _smoothed_mapping(frame: pd.DataFrame, column: str, cfg: dict) -> tuple[pd.Series, float]:
    """TE(v) = (n_v^+ + alpha * ybar) / (n_v + alpha), paper Eq. (15), alpha = ``te_alpha``."""
    label, alpha = cfg["label"], cfg["te_alpha"]
    prior = float(frame[label].mean())
    stats = frame.groupby(column, dropna=False)[label].agg(["sum", "count"])
    return (stats["sum"] + alpha * prior) / (stats["count"] + alpha), prior


def oof_target_and_frequency(train: pd.DataFrame, apply: pd.DataFrame, column: str, cfg: dict):
    """Shuffled five-fold OOF target/frequency encodings on ``train``; full-``train`` maps for ``apply``.

    Returns ``(train_te, apply_te, train_freq, apply_freq)``.
    """
    label = cfg["label"]
    y = train[label].to_numpy(np.int8, copy=False)
    splitter = StratifiedKFold(n_splits=cfg["encoding_folds"], shuffle=True, random_state=cfg["split_seed"])
    oof_te = np.full(len(train), np.nan)
    oof_freq = np.zeros(len(train))
    for fit_idx, hold_idx in splitter.split(np.zeros(len(train)), y):
        fit, hold = train.iloc[fit_idx], train.iloc[hold_idx]
        mapping, prior = _smoothed_mapping(fit, column, cfg)
        oof_te[hold_idx] = hold[column].map(mapping).fillna(prior)
        freq = fit[column].value_counts(dropna=False, normalize=True)
        oof_freq[hold_idx] = hold[column].map(freq).fillna(0.0)
    mapping, prior = _smoothed_mapping(train, column, cfg)
    apply_te = apply[column].map(mapping).fillna(prior).to_numpy(float)
    freq = train[column].value_counts(dropna=False, normalize=True)
    apply_freq = apply[column].map(freq).fillna(0.0).to_numpy(float)
    return oof_te, apply_te, oof_freq, apply_freq


def expanding_target_encoding(fit: pd.DataFrame, apply: pd.DataFrame, column: str, cfg: dict):
    """Expanding-block target encoding (S2.1.1).

    ``fit`` is cut into ``te_blocks`` ordered blocks; block ``b`` is encoded with the
    mapping estimated on blocks ``< b`` (code detail: block 1 has no preceding
    labels and receives the neutral value 0.5). ``apply`` rows use the mapping from all of ``fit``.
    Returns ``(fit_values, apply_values, mapping, prior)``.
    """
    values = np.full(len(fit), .5)
    for block in np.array_split(np.arange(len(fit)), cfg["te_blocks"])[1:]:
        past = fit.iloc[:block[0]]
        mapping, prior = _smoothed_mapping(past, column, cfg)
        values[block] = fit.iloc[block][column].map(mapping).fillna(prior).to_numpy(float)
    mapping, prior = _smoothed_mapping(fit, column, cfg)
    return values, apply[column].map(mapping).fillna(prior).to_numpy(float), mapping.to_dict(), prior


def _codes(fit_raw: pd.DataFrame, apply_raw: pd.DataFrame, column: str):
    """Label-free ordinal codes; categories unseen in ``fit_raw`` map to -1."""
    keys = sorted(fit_raw[column].fillna("<missing>").astype(str).unique())
    mapping = {key: i for i, key in enumerate(keys)}
    fit = fit_raw[column].fillna("<missing>").astype(str).map(mapping).astype("float32")
    apply = apply_raw[column].fillna("<missing>").astype(str).map(mapping).fillna(-1).astype("float32")
    return fit, apply


def encode_search(fit_raw: pd.DataFrame, apply_raw: pd.DataFrame, cfg: dict, plan: dict):
    """Search representation of ``fit_raw`` (labelled) and ``apply_raw`` (unlabelled).

    The column decisions come from ``plan`` (fixed on ``I``), so the S3
    rebuild on ``I + V`` has exactly the columns the search used. Encodings of
    ``fit_raw`` are out-of-fold; ``apply_raw`` is encoded with maps fitted on
    all of ``fit_raw``. Returns two DataFrames; only the first contains the label.

    Paper difference (code detail): the target encodings are computed once on
    all of ``fit_raw`` with their own five shuffled folds, not inside each
    learner fold, so the encoding of a training row can use labels of rows the
    learner holds out in another fold (never labels of ``V`` or ``T``).
    """
    label = cfg["label"]
    fit_raw = fit_raw.reset_index(drop=True)
    apply_raw = apply_raw.reset_index(drop=True)
    cats, nums, binaries = plan["cats"], plan["nums"], plan["binaries"]
    fit_cols = {label: fit_raw[label].to_numpy(copy=False)}
    apply_cols = {}
    for column in dict.fromkeys(nums + binaries):
        fit_cols[column] = pd.to_numeric(fit_raw[column], errors="coerce").astype("float32")
        apply_cols[column] = pd.to_numeric(apply_raw[column], errors="coerce").astype("float32")
    for column in cats:
        oof_te, apply_te, oof_freq, apply_freq = oof_target_and_frequency(fit_raw, apply_raw, column, cfg)
        fit_cols[f"{column}_oof_te"] = oof_te.astype("float32")
        apply_cols[f"{column}_oof_te"] = apply_te.astype("float32")
        fit_cols[f"{column}_freq"] = oof_freq.astype("float32")
        apply_cols[f"{column}_freq"] = apply_freq.astype("float32")
    for column in plan["logs"]:
        # x_log = log(1 + max(x, 0)) for non-negative right-skewed columns.
        values = pd.to_numeric(fit_raw[column], errors="coerce")
        fit_cols[f"{column}_log"] = np.log1p(values.clip(lower=0)).astype("float32")
        apply_values = pd.to_numeric(apply_raw[column], errors="coerce")
        apply_cols[f"{column}_log"] = np.log1p(apply_values.clip(lower=0)).astype("float32")
    for column in plan["codes"]:
        fit_cols[f"{column}_code"], apply_cols[f"{column}_code"] = _codes(fit_raw, apply_raw, column)
    return pd.DataFrame(fit_cols), pd.DataFrame(apply_cols, index=pd.RangeIndex(len(apply_raw)))


def encode_expert(fit_raw: pd.DataFrame, apply_raw: pd.DataFrame, cfg: dict, plan: dict):
    """Expert representation (S2): ``plan`` columns plus expanding-block encodings."""
    fit_raw, apply_raw = fit_raw.reset_index(drop=True).copy(), apply_raw.reset_index(drop=True).copy()
    fit, apply = {}, {}
    for c in plan["nums"] + plan["binaries"]:
        fit[c] = pd.to_numeric(fit_raw[c], errors="coerce").astype("float32")
        apply[c] = pd.to_numeric(apply_raw[c], errors="coerce").astype("float32")
    for c in plan["codes"]:
        fit[c + "_code"], apply[c + "_code"] = _codes(fit_raw, apply_raw, c)
    for c in plan["cats"]:
        ti, ta, _, _ = expanding_target_encoding(fit_raw, apply_raw, c, cfg)
        fi = fit_raw[c].map(fit_raw[c].value_counts(normalize=True)).fillna(0).to_numpy()
        fa = apply_raw[c].map(fit_raw[c].value_counts(normalize=True)).fillna(0).to_numpy()
        fit[c + "_oof_te"], apply[c + "_oof_te"] = ti.astype("float32"), ta.astype("float32")
        fit[c + "_freq"], apply[c + "_freq"] = fi.astype("float32"), fa.astype("float32")
    for c in plan["logs"]:
        fit[c + "_log"] = np.log1p(fit_raw[c].clip(lower=0)).astype("float32")
        apply[c + "_log"] = np.log1p(apply_raw[c].clip(lower=0)).astype("float32")
    fit[cfg["label"]] = fit_raw[cfg["label"]].to_numpy()
    return pd.DataFrame(fit), pd.DataFrame(apply, index=pd.RangeIndex(len(apply_raw)))


# --------------------------------------------------------------------------- expressions
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Attribute,
                  ast.Name, ast.Load, ast.Constant, ast.Subscript, ast.keyword, ast.Add, ast.Sub,
                  ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Gt, ast.GtE, ast.Lt,
                  ast.LtE, ast.Eq, ast.NotEq, ast.BitAnd, ast.BitOr, ast.Invert)
_NUMPY_CALLS = {"abs", "clip", "log", "log1p", "sqrt", "maximum", "minimum", "where", "exp", "expm1",
                "square", "sign", "power"}
_SERIES_CALLS = {"abs", "clip", "fillna", "astype", "replace", "to_numpy"}


def validate_pointwise_expression(expression: str, available_columns, label: str) -> ast.Expression:
    """Parse and whitelist a candidate expression (S1.1.1 safe operations).

    Allowed: ``df['col']`` for known non-label columns, arithmetic and
    comparisons, a fixed set of ``np.*`` element-wise functions,
    ``pd.to_numeric`` and a few element-wise Series methods. Anything else
    (other names, private attributes, I/O, aggregations, in-place keywords)
    raises ``ValueError``.
    """
    if len(expression) > 20000:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"syntax not allowed: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in {"df", "np", "pd"}:
                raise ValueError(f"unknown name: {node.id}")
            parent = parents.get(node)
            if node.id == "df" and not (isinstance(parent, ast.Subscript) and parent.value is node):
                raise ValueError("only df['column'] access is allowed")
        if isinstance(node, ast.Subscript):
            if (not isinstance(node.value, ast.Name) or node.value.id != "df"
                    or not isinstance(node.slice, ast.Constant)):
                raise ValueError("only df['column'] subscripts are allowed")
            if node.slice.value == label or node.slice.value not in available_columns:
                raise ValueError("the label and unknown columns cannot be read")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or not isinstance(parents.get(node), ast.Call):
                raise ValueError("attribute access is not allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute):
                raise ValueError("only whitelisted functions are allowed")
            owner, name = node.func.value, node.func.attr
            if isinstance(owner, ast.Name) and owner.id == "np":
                allowed = name in _NUMPY_CALLS
            elif isinstance(owner, ast.Name) and owner.id == "pd":
                allowed = name == "to_numeric"
            else:
                allowed = name in _SERIES_CALLS and not isinstance(owner, ast.Name)
            if not allowed:
                raise ValueError(f"function not allowed: {name}")
            if any(k.arg in {None, "out", "inplace"} for k in node.keywords):
                raise ValueError("in-place modification is not allowed")
    return tree


def materialize_expression(frame: pd.DataFrame, expression: str, label: str) -> np.ndarray:
    """Evaluate a validated expression row-wise and return a float32 vector."""
    validate_pointwise_expression(expression, set(frame) - {label}, label)
    with np.errstate(all="ignore"):
        values = eval(expression, {"np": np, "pd": pd, "__builtins__": {}},  # noqa: S307 (validated AST)
                      {"df": frame.drop(columns=label, errors="ignore")})
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(frame),):
        raise ValueError("an expression must return exactly one value per row")
    return values


def materialize_pair(fit_raw, apply_raw, cfg, plan, registry, expert=False):
    """Build the representation of (fit, apply) and append every feature of ``registry`` in order."""
    encode = encode_expert if expert else encode_search
    fit, apply = encode(fit_raw, apply_raw, cfg, plan)
    for index, (name, spec) in enumerate(registry.items()):
        if index and index % 32 == 0:
            fit, apply = fit.copy(), apply.copy()   # defragment
        if spec.get("executor", "pointwise") == "entity_history":
            fit[name] = entity_history(fit_raw, None, spec, cfg)
            apply[name] = entity_history(fit_raw, apply_raw, spec, cfg)
        else:
            fit[name] = materialize_expression(fit, spec["expression"], cfg["label"])
            apply[name] = materialize_expression(apply, spec["expression"], cfg["label"])
    return fit.copy(), apply.copy()


def dependency_registry(names, pool: dict) -> dict:
    """Return the specs needed to compute ``names``, dependencies first.

    A composed candidate may reference an earlier candidate as ``df['name']``.
    """
    ordered, visiting = {}, set()

    def visit(name):
        if name in ordered or name not in pool:
            return
        if name in visiting:
            raise ValueError("cyclic feature dependency")
        visiting.add(name)
        spec = pool[name]
        if spec.get("executor", "pointwise") == "pointwise":
            for node in ast.walk(ast.parse(spec["expression"], mode="eval")):
                if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                        and node.value.id == "df" and isinstance(node.slice, ast.Constant)):
                    visit(node.slice.value)
        visiting.remove(name)
        ordered[name] = copy.deepcopy(spec)

    for name in names:
        visit(name)
    return ordered


@dataclass
class Surface:
    """Materialised search representation of ``raw`` (labelled) and ``apply_raw`` (unlabelled).

    ``train``/``apply`` hold the base representation plus every feature in
    ``registry``.
    """

    raw: pd.DataFrame
    apply_raw: pd.DataFrame
    plan: dict
    registry: dict
    cfg: dict
    train: pd.DataFrame = field(init=False)
    apply: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.train, self.apply = materialize_pair(self.raw, self.apply_raw, self.cfg, self.plan, self.registry)

    def with_registry(self, registry: dict) -> "Surface":
        return Surface(self.raw, self.apply_raw, self.plan, registry, self.cfg)


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Float32 design matrix of ``columns`` (non-numeric -> NaN, +-inf -> NaN)."""
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    values[np.isinf(values)] = np.nan
    return values
