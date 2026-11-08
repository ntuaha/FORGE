from __future__ import annotations

import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REGISTRY_PATH = Path(os.environ.get("OTHERS_REGISTRY", HERE / "datasets.yaml"))

_DUCKDB_SUFFIXES = {".duckdb", ".db"}
_CSV_SUFFIXES = {".csv", ".tsv"}
_PARQUET_SUFFIXES = {".parquet", ".pq"}


def data_dir() -> Path:
    override = os.environ.get("OTHERS_DATA_DIR")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"OTHERS_DATA_DIR={path} is not a directory")
        return path
    for parent in HERE.parents:
        candidate = parent / "datasets" / "duckdb_v2"
        if candidate.is_dir():
            return candidate
    local = HERE / "data"
    if local.is_dir():
        return local
    raise FileNotFoundError(
        "cannot locate a data directory. Set OTHERS_DATA_DIR to the folder "
        "holding your data files, or create ./data next to datasets.py.")


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"registry not found: {REGISTRY_PATH}")
    doc = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8")) or {}
    datasets = doc.get("datasets") or {}
    if not datasets:
        raise ValueError(f"{REGISTRY_PATH} declares no datasets")
    return {"defaults": doc.get("defaults") or {}, "datasets": datasets}


_REGISTRY = _load_registry()
DATASETS: dict[str, dict] = _REGISTRY["datasets"]
DATASET_ORDER = list(DATASETS)

MECHANISM_SUBSET = [name for name in
                    ("credit_default", "creditcard", "ieeecis", "paysim")
                    if name in DATASETS]


def _source_kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv.gz") or name.endswith(".tsv.gz"):
        return "csv"
    suffix = path.suffix.lower()
    if suffix in _DUCKDB_SUFFIXES:
        return "duckdb"
    if suffix in _CSV_SUFFIXES:
        return "csv"
    if suffix in _PARQUET_SUFFIXES:
        return "parquet"
    if "*" in path.name:
        return "parquet"
    raise ValueError(
        f"cannot tell how to read {path.name!r}; expected .duckdb, .csv, "
        ".csv.gz, .parquet, or a parquet glob")


def get_config(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(
            f"unknown dataset {name!r}; registry {REGISTRY_PATH} defines: "
            f"{', '.join(DATASET_ORDER)}")
    cfg = dict(_REGISTRY["defaults"])
    cfg.update(DATASETS[name])
    cfg["name"] = name

    filename = cfg.get("file") or f"{name}.duckdb"
    path = Path(filename)
    if not path.is_absolute():
        path = data_dir() / filename
    cfg["source_path"] = path
    cfg["source_kind"] = _source_kind(path)
    cfg["db_path"] = path

    cfg.setdefault("table", "raw")
    cfg.setdefault("outer_test_fraction", 0.20)
    cfg.setdefault("search_val_fraction_of_outer", 0.20)
    cfg.setdefault("split_seed", 42)
    cfg.setdefault("time_col", None)
    cfg.setdefault("drop", [])
    if not cfg.get("label"):
        raise ValueError(f"dataset {name!r} declares no label column")
    return cfg


def describe(name: str) -> str:
    cfg = get_config(name)
    return (f"{name}: {cfg['source_kind']} at {cfg['source_path']}, "
            f"label={cfg['label']}, "
            f"time_col={cfg['time_col'] or '(random split)'}")
