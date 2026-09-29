"""Configuration: every parameter lives in YAML.

``configs/default.yaml`` holds the method parameters (with comments naming the
paper section of each value), ``configs/datasets/<name>.yaml`` the dataset
schema. :func:`make_config` merges, in order: the defaults, the dataset file,
an optional user file and keyword overrides. Nested settings such as
``lightgbm: {n_estimators: 80}`` change only the keys they name.

Environment variables (optional): ``FORGE_THREADS``, ``FORGE_LLM_API_URL``,
``FORGE_LLM_MODEL`` and the API key variable named by ``llm_api_key_env``.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
"""Repository root (two levels above ``src/forge``)."""

ASSETS_DIR = PACKAGE_ROOT / "assets"
CONFIG_DIR = PACKAGE_ROOT / "configs"
LEARNERS = ("lightgbm", "xgboost", "catboost")


def load_yaml(path: Path | str) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def merge(base: dict, extra: dict) -> dict:
    """``base`` updated with ``extra``; nested dictionaries (learner settings) are merged key by key."""
    out = dict(base)
    for key, value in extra.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def make_config(dataset: str | None = None, config_file: Path | str | None = None, **overrides) -> dict:
    """Configuration for ``dataset`` (a name in ``configs/datasets/``) or for ``config_file``.

    Example: ``make_config("credit_default", seed=12, learner="xgboost")``.
    For your own data pass a YAML file with at least ``dataset``, ``label``
    and ``data_files`` (see ``configs/datasets/my_data_example.yaml``), or
    pass ``label=...`` and call ``run_forge(cfg, frame=df)``.
    """
    user = load_yaml(config_file) if config_file else {}
    name = dataset or user.get("dataset") or overrides.get("dataset") or "credit_default"
    cfg = load_yaml(CONFIG_DIR / "default.yaml")
    dataset_file = CONFIG_DIR / "datasets" / f"{name}.yaml"
    if dataset_file.exists():
        cfg = merge(cfg, load_yaml(dataset_file))
    cfg = merge(merge(cfg, user), overrides)
    cfg["dataset"] = name
    cfg.setdefault("loader", "frame")
    for key, default in (("drop", []), ("time_col", None), ("entity_cols", []), ("data_files", [])):
        cfg.setdefault(key, default)
    if not cfg.get("label"):
        raise ValueError(f"no label for dataset {name!r}: add configs/datasets/{name}.yaml, "
                         "pass a YAML file with 'label', or pass label=...")
    if cfg["learner"] not in LEARNERS and ":" not in cfg["learner"]:
        raise ValueError(f"learner must be one of {LEARNERS} or 'module:function', not {cfg['learner']!r}")
    cfg["threads"] = int(os.environ.get("FORGE_THREADS", cfg["threads"]))
    cfg["llm_api_url"] = os.environ.get("FORGE_LLM_API_URL", cfg["llm_api_url"])
    cfg["llm_model"] = os.environ.get("FORGE_LLM_MODEL", cfg["llm_model"])
    for key in ("data_dir", "cache_dir"):
        cfg[key] = (PACKAGE_ROOT / cfg[key]).resolve() if not Path(cfg[key]).is_absolute() else Path(cfg[key])
    cfg["drop"], cfg["entity_cols"] = list(cfg["drop"]), list(cfg["entity_cols"])
    return cfg


def learner_params(cfg: dict) -> dict:
    """Hyper-parameters of the configured learner (``cfg[cfg['learner']]``, empty if not set)."""
    return dict(cfg.get(cfg["learner"]) or {})
