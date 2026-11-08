from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.dont_write_bytecode = True

PACKAGE_DIR = Path(__file__).resolve().parent
FULL_ROOT = PACKAGE_DIR.parent
VENDOR_DIR = FULL_ROOT / "vendor"
RESULTS_DIR = FULL_ROOT / "results"
CONFIG_PATH = PACKAGE_DIR / "config.yaml"
THEORY_PATH = FULL_ROOT / "THEORM.md"
DESIGN_PATH = FULL_ROOT / "DESIGN.md"

for path in (VENDOR_DIR, FULL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from agg_executor import (AGG_WHITELIST, apply_agg_spec, entity_candidates,
                          validate_agg_spec, value_candidates)
from datasets import get_config
from forge_v7 import hetero_stack
from forge_v7.confirmation import confirmation_gate
from forge_v7.hetero_stack import fold_plan
from portfolio import FoldLog, build_member_model
from protocol import (FeatureSpec, _load_raw_partitions, apply_feature_spec,
                      design_matrix, load_outer_partitions, prepare_e1)
from residuals import build_residual_profile

from forge_v9.gpu_mlp import (GpuMlpConfig, fit_predict_gpu_mlp,
                              install_gpu_mlp_backend)
from forge_v9.manual_ag import (
    _attach_stack_columns,
    _ensemble_record,
    _members,
    _weighted_ensemble,
    manual_l1_library,
    manual_l2_library,
)
from forge_v13.dystack import ag_dystack_decision, holdout_prediction_table

from forge_v12.llm_v12 import VariableBundleLLM
from forge_v12.protocol import weighted_chronological_blocks

from forge_v13.factors_v13 import materialize_unified_factors
from forge_v13.stationarity_v32 import adversarial_auc, psi, screen_bundle
from forge_v13.templates_v21 import (expand_templates, propose_templates,
                                     seat_posneg_gap, template_scorecard)
from forge_v13.selection_v13 import (
    entry_sample_weights,
    Blacklist,
    entry_noise_floor,
    entry_scores,
    global_pair_bh,
    s4_per_seat_noise_floor,
    greedy_entry,
    label_free_representatives,
    lite_v2_entry,
    impute_columns,
    permutation_exit,
    probe_permutation,
)


METHOD = "FORGE_13_1_SPARSE"
EXPECTED_GRAPH = (35, 20)

L1_SEAT_SETS: dict[int, list[str]] = {
    "lgbm1": ["lgbm_default"],
    "cat1": ["cat_default"],
    "xgb1": ["xgb_default"],
    "torch1": ["torch_r86"],
    2: ["lgbm_default", "cat_default"],
    4: ["lgbm_default", "cat_default", "xgb_default", "torch_r86"],
    "4cpu": ["lgbm_default", "cat_default", "xgb_default", "xt_gini"],
    "10cpu": ["lgbm_xt", "lgbm_default", "lgbm_large",
              "cat_default", "cat_r177", "xgb_default", "xgb_r33",
              "rf_gini", "xt_gini", "xt_entropy"],
    "6cpu": ["lgbm_xt", "lgbm_large", "cat_default", "xgb_r33",
             "xt_gini", "xt_entropy"],
    "7nn": ["lgbm_xt", "lgbm_large", "cat_default", "xgb_r33",
            "xt_gini", "xt_entropy", "torch_r86"],
    "7nn_fast": ["lgbm_xt", "lgbm_large", "cat_default", "xgb_r33",
                 "xt_gini", "xt_entropy", "torch_r86"],
    "5nn_fast": ["lgbm_xt", "lgbm_large", "cat_default", "xgb_r33",
                 "torch_r86"],
    "20_fast": ["lgbm_xt", "lgbm_default", "lgbm_large", "lgbm_r131",
                "lgbm_r96", "cat_default", "cat_r177", "cat_r9", "cat_r137",
                "cat_r13", "xgb_default", "xgb_r33", "xgb_r89", "xgb_r194",
                "rf_gini", "rf_r195", "xt_gini", "xt_entropy", "xt_r42",
                "torch_r86"],
    "8nn": ["lgbm_xt", "lgbm_large", "cat_default", "xgb_r33",
            "xt_gini", "xt_entropy", "torch_r86", "torch_default"],
    10: ["lgbm_xt", "lgbm_default", "lgbm_large", "cat_default", "cat_r177",
         "xgb_default", "xgb_r33", "rf_gini", "xt_gini", "torch_r86"],
    20: ["lgbm_xt", "lgbm_default", "lgbm_large", "lgbm_r131", "lgbm_r96",
         "cat_default", "cat_r177", "cat_r9", "cat_r137", "cat_r13",
         "xgb_default", "xgb_r33", "xgb_r89", "xgb_r194",
         "rf_gini", "rf_r195", "xt_gini", "xt_entropy", "xt_r42",
         "torch_r86"],
    "20nn": ["lgbm_default", "lgbm_large", "lgbm_r131",
             "cat_default", "cat_r177", "cat_r9",
             "xgb_default", "xgb_r33",
             "rf_gini", "xt_gini",
             "fastai_default", "fastai_r102", "fastai_r145", "fastai_r191",
             "fastai_r11",
             "torch_default", "torch_r22", "torch_r30", "torch_r79",
             "torch_r86"],
    15: ["lgbm_default", "lgbm_large", "lgbm_r131",
         "cat_default", "cat_r177", "cat_r9",
         "xgb_default", "xgb_r33",
         "rf_gini", "xt_gini",
         "torch_default", "torch_r22", "torch_r30", "torch_r79",
         "torch_r86"],
    35: [name for name, _, _ in manual_l1_library()],
}
L1_SEAT_PARAM_OVERRIDES: dict[str, dict[str, dict]] = {
    "lgbm1": {"lgbm_default": {
        "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 20, "subsample": 0.8,
        "colsample_bytree": 0.8, "deterministic": True,
        "force_col_wise": True}},
    "torch1": {"torch_r86": {"batch_size": 4096, "epochs": 60}},
    "7nn_fast": {"torch_r86": {"batch_size": 4096, "epochs": 60}},
    "5nn_fast": {"torch_r86": {"batch_size": 4096, "epochs": 60}},
    "20_fast": {"torch_r86": {"batch_size": 4096, "epochs": 60}},
}

L2_SEAT_SETS: dict[int, list[str]] = {
    2: ["lgbm_default", "cat_default"],
    4: ["lgbm_default", "cat_default", "xgb_r33", "torch_r86"],
    10: ["lgbm_default", "cat_default", "xgb_r33", "lgbm_large", "rf_gini",
         "torch_r86", "torch_default", "torch_r22", "torch_r30", "torch_r79"],
    20: [name for name, _, _ in manual_l2_library()],
}

_GPU_RUNTIME: GpuMlpConfig | None = None
_MLP_DEVICES: list[str] = []


def _scoring_oof(y, oof):
    import os as _os
    import numpy as _np
    if _os.environ.get("FORGE_NO_RESIDUAL", "") == "1":
        y = _np.asarray(y, dtype=_np.float64)
        return _np.full(len(y), float(y.mean()))
    return oof

def _scoring_residual(y, oof):
    import os as _os
    import numpy as _np
    y = _np.asarray(y, dtype=_np.float64)
    if _os.environ.get("FORGE_NO_RESIDUAL", "") == "1":
        return y - y.mean()
    return y - _np.asarray(oof, dtype=_np.float64)

def log(message: str) -> None:
    print(f"[forge_v13 {time.strftime('%F %T')}] {message}", flush=True)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                               default=str), encoding="utf-8")


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def apply_device_override(cfg: dict, device: str | None) -> dict:
    if device:
        cfg["engine"]["mlp"]["device"] = str(device)
    return cfg


def apply_graph_override(cfg: dict, l1: int | None, l2: int | None) -> dict:
    graph = cfg["engine"].setdefault("graph", {})
    def _key(table: dict, value):
        if value in table:
            return value
        return int(value)

    if l1 is not None:
        key = _key(L1_SEAT_SETS, l1)
        graph["l1_seats"] = L1_SEAT_SETS[key]
        override = L1_SEAT_PARAM_OVERRIDES.get(key)
        if override:
            graph["l1_param_overrides"] = override
    if l2 is not None:
        graph["l2_seats"] = L2_SEAT_SETS[_key(L2_SEAT_SETS, l2)]
    return cfg


def apply_outer_split_override(dataset_cfg: dict, cfg: dict) -> dict:
    override = cfg.get("experiment", {}).get("outer_split_seed_override")
    if override is not None:
        dataset_cfg["split_seed"] = int(override)
    return dataset_cfg


def resolve_graph(cfg: dict | None = None) -> tuple[tuple, tuple]:
    l1_library = manual_l1_library()
    l2_library = manual_l2_library()
    graph = ((cfg or {}).get("engine", {}) or {}).get("graph", {}) or {}
    l1_names = graph.get("l1_seats")
    l2_names = graph.get("l2_seats")

    if not l1_names and not l2_names:
        if (len(l1_library), len(l2_library)) != EXPECTED_GRAPH:
            raise RuntimeError(
                f"frozen graph violated: "
                f"got {(len(l1_library), len(l2_library))},"
                f" expected {EXPECTED_GRAPH}")
        return l1_library, l2_library

    overrides = graph.get("l1_param_overrides") or {}

    def _select(library: tuple, names, layer: str) -> tuple:
        if not names:
            return library
        wanted = list(names)
        available = {entry[0]: entry for entry in library}
        missing = [name for name in wanted if name not in available]
        if missing:
            raise RuntimeError(
                f"{layer} seat set names not in the frozen library: {missing}")
        chosen = []
        for name in wanted:
            seat_name, learner, params = available[name]
            patch = overrides.get(name) if layer == "l1" else None
            if patch:
                params = {**(params or {}), **patch}
            chosen.append((seat_name, learner, params))
        return tuple(chosen)

    return (_select(l1_library, l1_names, "l1"),
            _select(l2_library, l2_names, "l2"))


def assert_frozen_graph(cfg: dict | None = None) -> tuple[tuple, tuple]:
    return resolve_graph(cfg)


def install_gpu(cfg: dict, out_dir: Path) -> dict:
    global _GPU_RUNTIME
    mlp = cfg["engine"]["mlp"]
    l1_library, l2_library = resolve_graph(cfg)
    if not any(learner == "mlp" for _, learner, _ in
               tuple(l1_library) + tuple(l2_library)):
        _GPU_RUNTIME = None
        runtime = {
            "backend": "disabled_no_mlp_seats",
            "device": "cpu",
            "device_name": "CPU-only tree roster",
            "cpu_fallback": False,
        }
        write_json(out_dir / "gpu_mlp.json", runtime)
        log("GPU MLP: disabled (resolved graph contains no MLP seats)")
        return runtime
    if mlp["backend"] != "pytorch_cuda":
        raise ValueError("FORGE_13_1_SPARSE requires the CUDA MLP backend")
    runtime = GpuMlpConfig(
        device=str(mlp["device"]), batch_size=int(mlp["batch_size"]),
        predict_batch_size=int(mlp["predict_batch_size"]),
        epochs=int(mlp["epochs"]), clip=float(mlp["clip"]))
    gpu = install_gpu_mlp_backend(runtime)
    _GPU_RUNTIME = runtime
    write_json(out_dir / "gpu_mlp.json", gpu)
    log(f"GPU MLP: {gpu['device_name']} ({gpu['device']})")
    return gpu


def write_baseline_records(out_dir: Path, records: list[dict],
                           we_l2: dict, we_l3: dict) -> None:
    (out_dir / "dystack_oof").mkdir(parents=True, exist_ok=True)
    (out_dir / "dystack_predictions").mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        member = str(record["member"])
        np.save(out_dir / "dystack_oof" / f"{member}.npy",
                np.asarray(record["oof"], dtype=np.float32))
        np.save(out_dir / "dystack_predictions" / f"{member}.npy",
                np.asarray(record["apply"], dtype=np.float32))
        if record["layer"] == "l1":
            weight = float(we_l2["weights"].get(member, 0.0))
            weight_stage = "weighted_ensemble_l2"
        else:
            weight = float(we_l3["weights"].get(member, 0.0))
            weight_stage = "weighted_ensemble_l3"
        rows.append({
            "member": member, "model_key": record["model_key"],
            "layer": record["layer"], "oof_auprc": record["oof_auprc"],
            "blend_weight": weight, "weight_stage": weight_stage,
            "fit_sec": record["fit_sec"],
            "n_columns": record["design"]["n_columns"],
            "learner": record["design"]["learner"],
        })
    pd.DataFrame(rows).to_csv(out_dir / "dystack_member_table.csv",
                              index=False)


def load_baseline_records(run_dir: Path) -> list[dict]:
    table = pd.read_csv(run_dir / "dystack_member_table.csv")
    records: list[dict] = []
    for row in table.to_dict(orient="records"):
        member = str(row["member"])
        records.append({
            "member": member,
            "model_key": str(row["model_key"]),
            "layer": str(row["layer"]),
            "oof": np.load(run_dir / "dystack_oof" / f"{member}.npy"),
            "apply": np.load(run_dir / "dystack_predictions" / f"{member}.npy"),
            "oof_auprc": float(row["oof_auprc"]),
            "fit_sec": float(row.get("fit_sec", 0.0)),
            "design": {"n_columns": row.get("n_columns"),
                       "learner": str(row.get("learner", "baseline_replay"))},
            "probe_oof": {},
            "probe_seeds": {},
            "probe_mode": None,
        })
    return records


def audit_baseline(baseline_dir: Path, *, dataset: str, n_train: int,
                   n_search: int, features: list[str]) -> dict:
    summary_path = baseline_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "dataset": summary.get("dataset") == dataset,
        "inner_rows": int(summary.get("split", {}).get("n_inner_train", -1))
                      == int(n_train),
        "search_rows": int(summary.get("split", {}).get("n_search_val", -1))
                       == int(n_search),
        "features": list(summary.get("features", [])) == list(features),
    }
    if not all(checks.values()):
        raise RuntimeError(f"FORGE_13_1_SPARSE baseline audit failed: {checks}")
    return {
        "checks": checks,
        "baseline_dir": str(baseline_dir),
        "baseline_summary_sha256": sha256(summary_path),
    }


def save_records(directory: Path, records: list[dict]) -> None:
    (directory / "oof").mkdir(parents=True, exist_ok=True)
    (directory / "predictions").mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        np.save(directory / "oof" / f"{record['member']}.npy",
                np.asarray(record["oof"], dtype=np.float32))
        np.save(directory / "predictions" / f"{record['member']}.npy",
                np.asarray(record["apply"], dtype=np.float32))
        rows.append({
            "member": record["member"], "model_key": record["model_key"],
            "layer": record["layer"], "oof_auprc": record["oof_auprc"],
            "fit_sec": record["fit_sec"],
            "n_columns": record["design"]["n_columns"],
            "learner": record["design"]["learner"],
        })
    pd.DataFrame(rows).to_csv(directory / "member_table.csv", index=False)


def _forest_unsampled_indices(model, n_samples: int) -> list[np.ndarray]:
    from sklearn.ensemble._forest import (_generate_unsampled_indices,
                                          _get_n_samples_bootstrap)

    n_bootstrap = _get_n_samples_bootstrap(n_samples, model.max_samples)
    return [
        _generate_unsampled_indices(estimator.random_state, n_samples,
                                    n_bootstrap)
        for estimator in model.estimators_
    ]


def _forest_oob_predict(model, unsampled: list[np.ndarray],
                        matrix: np.ndarray) -> np.ndarray:
    sums = np.zeros(len(matrix), dtype=np.float64)
    counts = np.zeros(len(matrix), dtype=np.float64)
    for estimator, rows in zip(model.estimators_, unsampled):
        if not len(rows):
            continue
        proba = estimator.predict_proba(matrix[rows])
        column = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        sums[rows] += column
        counts[rows] += 1.0
    with np.errstate(invalid="ignore"):
        out = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
    invalid = ~np.isfinite(out)
    if invalid.any():
        out[invalid] = float(np.nanmean(out[~invalid])) if (~invalid).any() else 0.5
    return out.astype(np.float32)


def _native_oob_probed(learner: str, params: dict, seed: int,
                       x_fit: np.ndarray, y_fit: np.ndarray,
                       x_apply: np.ndarray,
                       probes: list[tuple[str, np.ndarray, int]],
                       ) -> tuple[np.ndarray, np.ndarray, dict, str | None]:
    with np.errstate(all="ignore"):
        medians = np.nanmedian(x_fit, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    x_train = np.where(np.isfinite(x_fit), x_fit, medians).astype(np.float32)
    x_external = np.where(np.isfinite(x_apply), x_apply,
                          medians).astype(np.float32)
    model = build_member_model(learner, params, seed)
    model.set_params(bootstrap=True, oob_score=True)
    model.fit(x_train, y_fit)
    oob = np.asarray(model.oob_decision_function_, dtype=np.float32)
    if oob.ndim == 2:
        oob = oob[:, 1]
    invalid = ~np.isfinite(oob)
    if invalid.any():
        oob[invalid] = float(np.nanmean(oob[~invalid]))
    apply = model.predict_proba(x_external)[:, 1].astype(np.float32)
    probe_oof: dict[str, np.ndarray] = {}
    probe_mode = None
    if probes:
        try:
            unsampled = _forest_unsampled_indices(model, len(x_train))
            probe_mode = "native_oob_per_tree_unsampled"
        except Exception as exc:
            unsampled = None
            probe_mode = f"in_bag_full_model_predict_fallback ({exc!r})"
        for name, permuted_column, column_index in probes:
            xp = x_train.copy()
            xp[:, column_index] = np.where(
                np.isfinite(permuted_column), permuted_column,
                medians[column_index]).astype(np.float32)
            if unsampled is not None:
                probe_oof[name] = _forest_oob_predict(model, unsampled, xp)
            else:
                probe_oof[name] = model.predict_proba(xp)[:, 1].astype(
                    np.float32)
    return oob, apply, probe_oof, probe_mode


_POOL_LEARNERS = ("lgbm_base", "xgboost", "random_forest", "extra_trees",
                  "catboost")
_FIT_CONTEXT: dict = {}


def _fit_workers() -> int:
    return max(0, int(os.environ.get("FORGE_V13_FIT_WORKERS", "4")))


_LITE: dict = {}


def _lite(key: str, default=None):
    value = _LITE.get(key)
    return default if value in (None, False, "") else value


def _is_search_side(stage: str) -> bool:
    return stage.startswith("search_round_") or stage.startswith("dystack_")


def lite_bag_folds(cfg: dict, stage: str) -> int:
    base = int(cfg["engine"]["bag_folds"])
    reduced = _lite("search_bag_folds")
    if reduced and _is_search_side(stage):
        return max(2, int(reduced))
    return base


def lite_mlp_cap(x_fit: np.ndarray, y_fit: np.ndarray, cap_seed: int):
    cap = _lite("mlp_fit_cap")
    if not cap or len(y_fit) <= int(cap):
        return x_fit, y_fit
    rows = hetero_stack.stratified_cap(np.asarray(y_fit), int(cap), cap_seed)
    return x_fit[rows], np.asarray(y_fit)[rows]


def lite_prune_l1(cfg: dict, baseline_dir, top_m: int) -> dict:
    table = Path(baseline_dir) / "dystack_member_table.csv"
    if not table.is_file():
        raise SystemExit(
            f"--lite-prune-l1 needs {table}, which the frozen baseline should "
            "contain; re-run the baseline stage or drop the flag")
    scored: list[tuple[float, str]] = []
    with table.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("layer") != "l1":
                continue
            try:
                scored.append((float(row["oof_auprc"]), row["model_key"]))
            except (KeyError, ValueError):
                continue
    if not scored:
        raise SystemExit(f"no l1 rows with oof_auprc in {table}")
    scored.sort(key=lambda item: -item[0])
    keep = {key for _, key in scored[:int(top_m)]}

    graph = cfg["engine"].setdefault("graph", {})
    current = list(graph.get("l1_seats") or [])
    if not current:
        current = [key for key, _, _ in manual_l1_library()]
    pruned = [key for key in current if key in keep]
    if not pruned:
        raise SystemExit("pruning removed every L1 seat; check the baseline table")
    graph["l1_seats"] = pruned
    return {
        "requested_top_m": int(top_m),
        "kept": pruned,
        "dropped": [key for key in current if key not in keep],
        "ranking_source": str(table),
        "baseline_oof": {key: round(score, 7) for score, key in scored},
    }


def _mlp_worker_init(payload_path: str, lgbm_threads: str, omp_threads: str,
                     device: str, mlp_cfg: dict, lite: dict) -> None:
    import pickle

    global _GPU_RUNTIME
    os.environ["LGBM_THREADS"] = str(lgbm_threads)
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    _LITE.clear()
    _LITE.update(lite)
    runtime = GpuMlpConfig(
        device=str(device), batch_size=int(mlp_cfg["batch_size"]),
        predict_batch_size=int(mlp_cfg["predict_batch_size"]),
        epochs=int(mlp_cfg["epochs"]), clip=float(mlp_cfg["clip"]))
    install_gpu_mlp_backend(runtime)
    _GPU_RUNTIME = runtime
    with open(payload_path, "rb") as handle:
        _FIT_CONTEXT.clear()
        _FIT_CONTEXT.update(pickle.load(handle))


def _spawn_worker_init(payload_path: str, lgbm_threads: str,
                       omp_threads: str) -> None:
    import pickle

    os.environ["LGBM_THREADS"] = str(lgbm_threads)
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    with open(payload_path, "rb") as handle:
        _FIT_CONTEXT.clear()
        _FIT_CONTEXT.update(pickle.load(handle))


def _member_fit_payload(*, member_name: str, learner: str, params: dict,
                        seed: int, columns: list[str],
                        probe_columns: list[str], train: pd.DataFrame,
                        apply: pd.DataFrame, y_fit: np.ndarray,
                        folds, time_limit_sec: float | None = None) -> dict:
    limit_clock = time.time()

    def over_limit() -> bool:
        return (time_limit_sec is not None and float(time_limit_sec) > 0
                and time.time() - limit_clock > float(time_limit_sec))

    def dropped_sentinel(completed_folds) -> dict:
        return {
            "time_limit_dropped": True,
            "member": member_name,
            "learner": learner,
            "elapsed_sec": round(time.time() - limit_clock, 3),
            "limit_sec": float(time_limit_sec),
            "completed_folds": completed_folds,
        }

    x_fit = design_matrix(train, columns)
    x_apply = design_matrix(apply, columns)
    member_started = time.time()
    probes: list[tuple[str, np.ndarray, int]] = []
    probe_seeds: dict[str, int] = {}
    for column in probe_columns:
        column_index = columns.index(column)
        permutation, derived = probe_permutation(
            len(train), member_name, column, seed)
        probes.append((column, x_fit[permutation, column_index],
                       column_index))
        probe_seeds[column] = derived
    probe_oof: dict[str, np.ndarray] = {}
    probe_mode = None
    if learner in ("random_forest", "extra_trees"):
        oof, external, probe_oof, probe_mode = _native_oob_probed(
            learner, params, seed, x_fit, y_fit, x_apply, probes)
        oof_mode = "native_oob_single_child"
        if over_limit():
            return dropped_sentinel("native_oob_post_fit")
    else:
        oof = np.full(len(y_fit), np.nan, dtype=np.float32)
        probe_arrays = {name: np.full(len(y_fit), np.nan, dtype=np.float32)
                        for name, _, _ in probes}
        external_folds = []
        for fold, (fit_idx, oof_idx) in enumerate(folds):
            if over_limit():
                return dropped_sentinel(fold)
            probe_slices = []
            for name, permuted_column, column_index in probes:
                matrix = x_fit[oof_idx].copy()
                matrix[:, column_index] = permuted_column[oof_idx]
                probe_slices.append(matrix)
            if learner == "mlp":
                mlp_x, mlp_y = lite_mlp_cap(
                    x_fit[fit_idx], y_fit[fit_idx], 1000 * fold + seed)
                outputs = fit_predict_gpu_mlp(
                    seed, mlp_x, mlp_y,
                    [x_fit[oof_idx], x_apply, *probe_slices],
                    runtime=_GPU_RUNTIME, params=dict(params or {}),
                    validation=(x_fit[oof_idx], y_fit[oof_idx]))
            else:
                outputs = hetero_stack.fit_member(
                    learner, params, seed, x_fit[fit_idx], y_fit[fit_idx],
                    [x_fit[oof_idx], x_apply, *probe_slices],
                    cap_seed=1000 * fold + seed)
            oof[oof_idx] = outputs[0]
            external_folds.append(outputs[1])
            for probe_index, (name, _, _) in enumerate(probes):
                probe_arrays[name][oof_idx] = outputs[2 + probe_index]
        if over_limit():
            return dropped_sentinel(len(folds))
        if np.isnan(oof).any():
            raise RuntimeError(f"OOF coverage incomplete for {member_name}")
        for name, values in probe_arrays.items():
            if np.isnan(values).any():
                raise RuntimeError(
                    f"probe coverage incomplete for {member_name}:{name}")
        probe_oof = probe_arrays
        external = np.mean(external_folds, axis=0).astype(np.float32)
        oof_mode = ("shared_8fold_gpu_early_stopping" if learner == "mlp"
                    else "shared_8fold")
        if probes:
            probe_mode = "per_fold_model_repredict"
    return {
        "oof": np.asarray(oof, dtype=np.float32),
        "apply": np.asarray(external, dtype=np.float32),
        "fit_sec": round(time.time() - member_started, 3),
        "oof_mode": oof_mode,
        "probe_oof": probe_oof,
        "probe_seeds": probe_seeds,
        "probe_mode": probe_mode,
    }


def _pool_fit_member(member_name: str) -> dict:
    context = _FIT_CONTEXT
    task = context["tasks"][member_name]
    return _member_fit_payload(
        member_name=member_name, learner=task["learner"],
        params=task["params"], seed=context["seed"],
        columns=task["columns"], probe_columns=task["probe_columns"],
        train=context["train"], apply=context["apply"],
        y_fit=context["y_fit"], folds=context["folds"],
        time_limit_sec=context.get("time_limit_sec"))


def fit_layer(*, train: pd.DataFrame, apply: pd.DataFrame, label: str,
              library, surfaces: dict[str, list[str]], prefix: str,
              layer_index: int, cfg: dict, fold_log: FoldLog, stage: str,
              probe_cols: dict[str, list[str]] | None = None,
              time_limit_audit: list | None = None) -> list[dict]:
    y_fit = train[label].to_numpy(np.int8, copy=False)
    seed = int(cfg["engine"]["seed"])
    folds = fold_plan(y_fit, bag_folds=lite_bag_folds(cfg, stage), seed=seed)
    probe_cols = probe_cols or {}
    members = _members(library, prefix)
    workers = _fit_workers()
    raw_limit = cfg["engine"].get("member_time_limit_sec", 2700)
    time_limit = float(raw_limit) if raw_limit else None
    if time_limit is not None and time_limit <= 0:
        time_limit = None

    def record_drop(entry: dict, mode: str) -> None:
        entry = dict(entry)
        entry.pop("time_limit_dropped", None)
        entry.update({"stage": stage, "mode": mode})
        if time_limit_audit is not None:
            time_limit_audit.append(entry)
        if fold_log.path is not None:
            with open(fold_log.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    {"time_limit_dropped": True, **entry}) + "\n")
        log(f"time limit: dropped {entry['member']} after "
            f"{entry.get('elapsed_sec', '?')}s "
            f"(limit {entry.get('limit_sec')}s, mode {mode})")

    plans: dict[str, dict] = {}
    rep_cfg = cfg.get("representation") or {}
    native_cats = bool(rep_cfg.get("enabled"))
    for member_name, model_key, learner, params in members:
        columns = list(dict.fromkeys(surfaces[member_name]))
        if not columns or label in columns:
            raise ValueError(f"invalid feature surface for {member_name}")
        if native_cats and learner in ("lgbm_base", "catboost"):
            cat_idx = [i for i, c in enumerate(columns)
                       if str(c).endswith("_code")]
            if cat_idx and learner == "lgbm_base":
                params = {**params, "categorical_feature": cat_idx}
            elif cat_idx:
                params = {**params, "cat_features": cat_idx}
        probe_columns = []
        for column in probe_cols.get(member_name, []):
            if column not in columns:
                raise ValueError(
                    f"probe column {column!r} not on surface of {member_name}")
            probe_columns.append(column)
        plans[member_name] = {
            "model_key": model_key, "learner": learner,
            "params": params, "columns": columns,
            "probe_columns": probe_columns,
        }

    pooled = [name for name, plan in plans.items()
              if plan["learner"] in _POOL_LEARNERS] if workers > 0 else []
    mlp_pooled = ([name for name, plan in plans.items()
                   if plan["learner"] == "mlp"]
                  if len(_MLP_DEVICES) > 1 else [])
    if len(mlp_pooled) < 2:
        mlp_pooled = []
    payloads: dict[str, dict] = {}
    futures: dict[str, object] = {}
    pool = None
    mlp_pools: list = []
    payload_path: Path | None = None
    mlp_payload_path: Path | None = None
    try:
        if mlp_pooled:
            import multiprocessing as mp
            import pickle
            import tempfile
            from concurrent.futures import ProcessPoolExecutor

            mlp_columns = sorted({
                column for name in mlp_pooled
                for column in plans[name]["columns"]})
            handle = tempfile.NamedTemporaryFile(
                prefix="forge_v13_mlp_ctx_", suffix=".pkl", dir="/tmp",
                delete=False)
            mlp_payload_path = Path(handle.name)
            with handle:
                pickle.dump({
                    "train": train[mlp_columns], "apply": apply[mlp_columns],
                    "y_fit": y_fit, "folds": folds, "seed": seed,
                    "time_limit_sec": time_limit,
                    "tasks": {name: plans[name] for name in mlp_pooled},
                }, handle, protocol=pickle.HIGHEST_PROTOCOL)
            for device in _MLP_DEVICES[:len(mlp_pooled)]:
                mlp_pools.append(ProcessPoolExecutor(
                    max_workers=1, mp_context=mp.get_context("spawn"),
                    initializer=_mlp_worker_init,
                    initargs=(str(mlp_payload_path),
                              os.environ.get("LGBM_THREADS", "8"),
                              os.environ.get("OMP_NUM_THREADS", "8"),
                              str(device), dict(cfg["engine"]["mlp"]),
                              dict(_LITE))))
            log(f"MLP lane: {len(mlp_pooled)} nets across "
                f"{len(mlp_pools)} devices {_MLP_DEVICES[:len(mlp_pools)]}")
        if pooled:
            import multiprocessing as mp
            import pickle
            import tempfile
            from concurrent.futures import ProcessPoolExecutor

            pooled_columns = sorted({
                column for name in pooled for column in plans[name]["columns"]})
            handle = tempfile.NamedTemporaryFile(
                prefix="forge_v13_fit_ctx_", suffix=".pkl", dir="/tmp",
                delete=False)
            payload_path = Path(handle.name)
            with handle:
                pickle.dump({
                    "train": train[pooled_columns],
                    "apply": apply[pooled_columns],
                    "y_fit": y_fit, "folds": folds, "seed": seed,
                    "time_limit_sec": time_limit,
                    "tasks": {name: plans[name] for name in pooled},
                }, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pool = ProcessPoolExecutor(
                max_workers=min(int(workers), len(pooled)),
                mp_context=mp.get_context("spawn"),
                initializer=_spawn_worker_init,
                initargs=(str(payload_path),
                          os.environ.get("LGBM_THREADS", "8"),
                          os.environ.get("OMP_NUM_THREADS", "8")))
        pooled_set = set(pooled)
        mlp_pooled_set = set(mlp_pooled)
        for member_name, model_key, learner, params in members:
            if learner not in ("random_forest", "extra_trees"):
                for fold, (fit_idx, oof_idx) in enumerate(folds):
                    fold_log.record(
                        stage=stage, member=member_name, seed=seed, fold=fold,
                        n_rows=len(y_fit), train_idx=fit_idx, oof_idx=oof_idx)
            if pool is not None and member_name in pooled_set:
                futures[member_name] = pool.submit(_pool_fit_member,
                                                   member_name)
            elif member_name in mlp_pooled_set:
                target = mlp_pools[mlp_pooled.index(member_name)
                                   % len(mlp_pools)]
                futures[member_name] = target.submit(_pool_fit_member,
                                                     member_name)
        for member_name, model_key, learner, params in members:
            if member_name in futures:
                continue
            plan = plans[member_name]
            payload = _member_fit_payload(
                member_name=member_name, learner=learner, params=params,
                seed=seed, columns=plan["columns"],
                probe_columns=plan["probe_columns"], train=train,
                apply=apply, y_fit=y_fit, folds=folds,
                time_limit_sec=time_limit)
            if payload.get("time_limit_dropped"):
                record_drop(payload, "serial_lane_self_limit")
            else:
                payloads[member_name] = payload
        stuck_members: list[str] = []
        batch_deadline = None
        if futures and time_limit is not None:
            import math as _math

            waves = _math.ceil(len(futures) / max(1, int(workers)))
            batch_deadline = time.time() + time_limit * (waves + 1)
        for member_name in futures:
            try:
                if batch_deadline is None:
                    payload = futures[member_name].result()
                else:
                    remaining = max(60.0, batch_deadline - time.time())
                    payload = futures[member_name].result(timeout=remaining)
            except Exception as exc:
                stuck_members.append(member_name)
                record_drop({
                    "member": member_name,
                    "learner": plans[member_name]["learner"],
                    "elapsed_sec": None,
                    "limit_sec": time_limit,
                    "completed_folds": None,
                    "error": type(exc).__name__,
                }, "pool_batch_deadline_worker_killed")
                continue
            if payload.get("time_limit_dropped"):
                record_drop(payload, "pool_worker_self_limit")
            else:
                payloads[member_name] = payload
        if stuck_members and pool is not None:
            import signal as _signal

            for worker_pid in list(getattr(pool, "_processes", {}) or {}):
                try:
                    os.kill(worker_pid, _signal.SIGKILL)
                except ProcessLookupError:
                    pass
    finally:
        for mlp_pool in mlp_pools:
            mlp_pool.shutdown(wait=True)
        if mlp_payload_path is not None:
            mlp_payload_path.unlink(missing_ok=True)
        if pool is not None:
            pool.shutdown(wait=True)
        if payload_path is not None:
            payload_path.unlink(missing_ok=True)

    records: list[dict] = []
    for member_name, model_key, learner, params in members:
        if member_name not in payloads:
            continue
        plan = plans[member_name]
        payload = payloads[member_name]
        records.append({
            "member": member_name,
            "model_key": model_key,
            "layer": prefix,
            "oof": payload["oof"],
            "apply": payload["apply"],
            "oof_auprc": round(float(average_precision_score(
                y_fit, payload["oof"])), 7),
            "fit_sec": payload["fit_sec"],
            "design": {
                "model": model_key,
                "learner": learner,
                "n_columns": len(plan["columns"]),
                "columns": plan["columns"],
                "oof_mode": payload["oof_mode"],
            },
            "probe_oof": payload["probe_oof"],
            "probe_seeds": payload["probe_seeds"],
            "probe_mode": payload["probe_mode"],
        })
    return records


def build_ensembles(l1_records: list[dict], l2_records: list[dict],
                    y: np.ndarray, steps: int) -> tuple[dict, dict]:
    we_l2 = _weighted_ensemble(l1_records, y, steps)
    passthrough = _ensemble_record("weighted_ensemble_l2", "we_l2", we_l2)
    we_l3 = _weighted_ensemble(l2_records + [passthrough], y, steps)
    return we_l2, we_l3


def fit_layer_full_once(*, train: pd.DataFrame, apply: pd.DataFrame,
                        label: str, library, surfaces: dict[str, list[str]],
                        prefix: str, cfg: dict) -> list[dict]:
    y = train[label].to_numpy(np.int8, copy=False)
    seed = int(cfg["engine"]["seed"])
    records = []
    for member_name, model_key, learner, params in _members(library, prefix):
        columns = list(dict.fromkeys(surfaces[member_name]))
        x_fit = design_matrix(train, columns)
        x_apply = design_matrix(apply, columns)
        started = time.time()
        if learner == "mlp":
            outputs = fit_predict_gpu_mlp(
                seed, x_fit, y, [x_apply], runtime=_GPU_RUNTIME,
                params=dict(params or {}), validation=None)
        else:
            outputs = hetero_stack.fit_member(
                learner, params, seed, x_fit, y, [x_apply], cap_seed=seed)
        records.append({
            "member": member_name, "model_key": model_key, "layer": prefix,
            "oof": np.empty(0, dtype=np.float32),
            "apply": np.asarray(outputs[0], dtype=np.float32),
            "oof_auprc": None,
            "fit_sec": round(time.time() - started, 3),
            "design": {"model": model_key, "learner": learner,
                       "n_columns": len(columns), "columns": columns,
                       "oof_mode": "none_v8_full_train_single_fit"},
            "probe_oof": {}, "probe_seeds": {}, "probe_mode": None,
        })
    return records


def llm_wall_sec(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    total = 0.0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            total += float(json.loads(line).get("elapsed_sec") or 0.0)
        except (ValueError, TypeError):
            continue
    return round(total, 3)


def apply_frozen_weights(records: list[dict], weights: dict[str, float]) -> np.ndarray:
    available = {r["member"]: r for r in records}
    used = {name: float(weight) for name, weight in weights.items()
            if name in available and float(weight) > 0}
    total = sum(used.values())
    if total <= 0:
        raise RuntimeError("V8 frozen ensemble has no surviving positive weight")
    return np.sum([
        np.asarray(available[name]["apply"], dtype=np.float64) * (weight / total)
        for name, weight in used.items()
    ], axis=0).astype(np.float32)


def _cat_bases(features: list) -> list:
    return [c[: -len("_oof_te")] for c in features if c.endswith("_oof_te")]


def _candidate_surfaces(features: list, cat_bases: list,
                        code_bases: list | None = None) -> dict:
    te_freq = {f"{b}{s}" for b in cat_bases for s in ("_oof_te", "_freq")}
    common = [c for c in features if c not in te_freq]
    code_names = [f"{b}_code" for b in
                  (cat_bases if code_bases is None else code_bases)]
    return {"e1": {"columns": list(features), "codes": []},
            "identity": {"columns": common + code_names,
                         "codes": code_names},
            "raw": {"columns": list(common), "codes": []},
            "union": {"columns": list(features) + code_names,
                      "codes": code_names}}


def _materialize_codes(raw_fit, raw_apply_frame, train, apply,
                       cat_bases) -> None:
    lower_map = {str(c).lower(): c for c in raw_fit.columns}
    for base in cat_bases:
        name = f"{base}_code"
        if name in train.columns or base not in lower_map:
            continue
        col = lower_map[base]
        cat_fit = pd.Categorical(raw_fit[col])
        train[name] = cat_fit.codes.astype(np.float32)
        apply[name] = pd.Categorical(
            raw_apply_frame[col], categories=cat_fit.categories
        ).codes.astype(np.float32)


def _persistence_audit(raw_fit, cat_bases, dataset_cfg) -> dict:
    label = str(dataset_cfg["label"]).lower()
    time_col = (str(dataset_cfg["time_col"]).lower()
                if dataset_cfg.get("time_col") else None)
    if time_col and time_col in raw_fit.columns:
        order = np.argsort(raw_fit[time_col].to_numpy(), kind="stable")
    else:
        order = np.arange(len(raw_fit))
    cut = int(round(len(order) * 0.8))
    head_positions, tail_positions = order[:cut], order[cut:]
    lower_map = {str(c).lower(): c for c in raw_fit.columns}
    audit = {}
    for base in cat_bases:
        if base == label or base not in lower_map:
            continue
        series = raw_fit[lower_map[base]]
        head_values = set(series.iloc[head_positions].dropna().unique())
        tail = series.iloc[tail_positions]
        persistence = float(tail.isin(head_values).mean()) if len(tail) else 0.0
        audit[base] = {"nunique": int(series.nunique(dropna=True)),
                       "persistence": round(persistence, 4)}
    return audit


def select_and_apply_representation(raw_fit, raw_apply_frame, train, apply,
                                    features: list, dataset_cfg: dict,
                                    cfg: dict, *,
                                    frozen_choice: str | None = None) -> dict:
    rep = cfg.get("representation") or {}
    if not rep.get("enabled"):
        return {}
    label = str(dataset_cfg["label"]).lower()
    time_col = (str(dataset_cfg["time_col"]).lower()
                if dataset_cfg.get("time_col") else None)
    cat_bases = _cat_bases(features)
    if not cat_bases:
        return {"choice": "e1", "reason": "no_categorical_columns"}

    audit: dict = {"persistence_audit": _persistence_audit(
        raw_fit, cat_bases, dataset_cfg)}

    forced = str(_LITE.get("representation") or "")
    if frozen_choice is None and forced:
        choice = forced
        audit.update({"choice": choice, "forced": True, "frozen": False})
        log(f"representation forced: {choice} (probe skipped)")
    elif frozen_choice is None:
        row_cap = int(rep.get("probe_row_cap", 500000))
        head_frac = float(rep.get("probe_head_fraction", 0.8))
        tie_margin = float(rep.get("tie_margin_auprc", 0.0005))
        if time_col and time_col in raw_fit.columns:
            order = np.argsort(raw_fit[time_col].to_numpy(), kind="stable")
        else:
            order = np.arange(len(raw_fit))
        if len(order) > row_cap:
            order = order[-row_cap:]
        from sklearn.metrics import average_precision_score
        import lightgbm as lgb
        probe_folds = max(1, int(_LITE.get("probe_folds",
                                           rep.get("probe_folds", 1))))
        if probe_folds == 1:
            cuts = [(0, int(round(len(order) * head_frac)), len(order))]
        else:
            edges = [int(round(len(order) * i / (probe_folds + 1)))
                     for i in range(probe_folds + 2)]
            cuts = [(0, edges[i], edges[i + 1]) for i in range(1, probe_folds + 1)]
        per_fold: list[dict] = []
        for start, cut, stop in cuts:
            raw_head = raw_fit.iloc[order[start:cut]].reset_index(drop=True)
            raw_tail = raw_fit.iloc[order[cut:stop]].reset_index(drop=True)
            if len(raw_tail) == 0 or raw_tail[label].sum() == 0:
                continue
            p_train, p_apply, p_feats, _ = prepare_e1(raw_head, raw_tail,
                                                      dataset_cfg)
            p_bases = _cat_bases(p_feats)
            _materialize_codes(raw_head, raw_tail, p_train, p_apply, p_bases)
            surfaces = _candidate_surfaces(p_feats, p_bases)
            y_head = p_train[label].to_numpy(np.int8, copy=False)
            y_tail = p_apply[label].to_numpy(np.int8, copy=False)
            fold_scores: dict = {}
            for name in ("e1", "identity", "raw"):
                spec_c = surfaces[name]
                cols = [c for c in spec_c["columns"] if c in p_train.columns
                        or c in spec_c["codes"]]
                fit_matrix = design_matrix(p_train, cols)
                val_matrix = design_matrix(p_apply, cols)
                cat_idx = [i for i, c in enumerate(cols)
                           if c in set(spec_c["codes"])]
                kwargs = dict(n_estimators=400, learning_rate=.05,
                              num_leaves=31, min_child_samples=20,
                              n_jobs=4, random_state=20260730, verbosity=-1)
                if cat_idx:
                    kwargs["categorical_feature"] = cat_idx
                model = lgb.LGBMClassifier(**kwargs)
                model.fit(fit_matrix, y_head)
                pred = model.predict_proba(val_matrix)[:, 1]
                fold_scores[name] = float(average_precision_score(y_tail, pred))
            per_fold.append(fold_scores)
        if not per_fold:
            raise RuntimeError("representation probe produced no usable fold")
        scores = {name: float(np.mean([f[name] for f in per_fold]))
                  for name in ("e1", "identity", "raw")}
        spread = {name: float(np.std([f[name] for f in per_fold], ddof=1))
                  if len(per_fold) > 1 else None
                  for name in ("e1", "identity", "raw")}
        best = max(scores.values())
        choice = next(n for n in ("raw", "identity", "e1")
                      if scores[n] >= best - tie_margin)
        audit.update({"choice": choice, "probe_scores": scores,
                      "probe_rows": int(len(order)),
                      "probe_folds": len(per_fold),
                      "probe_scores_per_fold": per_fold,
                      "probe_score_sd": spread,
                      "tie_margin": tie_margin, "frozen": False})
        log(f"representation probe ({len(per_fold)} fold(s)): "
            f"{ {k: round(v, 5) for k, v in scores.items()} } -> {choice}"
            + ("" if len(per_fold) == 1 else
               f"  sd={ {k: (None if v is None else round(v, 5)) for k, v in spread.items()} }"))
    else:
        choice = frozen_choice
        audit.update({"choice": choice, "frozen": True})
        log(f"representation replay: {choice}")

    if choice != "raw":
        _materialize_codes(raw_fit, raw_apply_frame, train, apply, cat_bases)
    live_bases = [b for b in cat_bases if f"{b}_code" in train.columns]
    surfaces = _candidate_surfaces(features, cat_bases, code_bases=live_bases)
    audit["code_bases_requested"] = len(cat_bases)
    audit["code_bases_materialized"] = len(live_bases)
    if len(live_bases) != len(cat_bases) and choice != "raw":
        missing = sorted(set(cat_bases) - set(live_bases))
        audit["code_bases_missing"] = missing
        log(f"representation {choice}: {len(missing)} categorical base(s) have "
            f"no raw column and carry no _code: {missing[:6]}")
    features[:] = list(surfaces[choice]["columns"])
    audit["surfaces"] = {name: {"columns": list(spec_s["columns"]),
                                "codes": list(spec_s["codes"])}
                         for name, spec_s in surfaces.items()}
    return audit


def seat_representation_plan(l1_library, representation_audit,
                             features: list) -> dict:
    if str(_lite("rep_scope", "shared")) != "seat":
        return {}
    surfaces = (representation_audit or {}).get("surfaces") or {}
    if not surfaces:
        return {"seat_rep": {key: "e1" for key, _, _ in l1_library},
                "seat_base": {key: list(features) for key, _, _ in l1_library},
                "note": "no_categorical_columns_surfaces_coincide"}
    order = ("identity", "raw", "e1")
    seat_rep: dict[str, str] = {}
    seat_base: dict[str, list[str]] = {}
    tree_index = 0
    for key, learner, _ in l1_library:
        if str(learner).lower().startswith(("mlp", "torch", "fastai")):
            rep = "e1"
        else:
            rep = order[tree_index % len(order)]
            tree_index += 1
        seat_rep[key] = rep
        seat_base[key] = list(surfaces[rep]["columns"])
    return {"seat_rep": seat_rep, "seat_base": seat_base}


def negative_subsample(train: pd.DataFrame, label: str, ratio: float,
                       seed: int) -> np.ndarray:
    y = train[label].to_numpy(np.int8, copy=False)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep_neg = min(len(neg), int(round(float(ratio) * len(pos))))
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(neg, size=keep_neg, replace=False) if keep_neg else np.empty(0, int)
    mask = np.sort(np.concatenate([pos, chosen])).astype(np.int64)
    return mask


def materialize_bank(cfg: dict, raw_train, raw_apply, train, apply, *,
                     label: str, time_col: str | None,
                     forbidden: list[str],
                     frozen_spec: dict | None = None) -> tuple[list[str], dict]:
    schema = cfg["schema"]
    return materialize_unified_factors(
        raw_train, raw_apply, train, apply, label=label, time_col=time_col,
        forbidden=forbidden,
        min_repeat_fraction=float(schema["entity_min_repeat_fraction"]),
        min_unique=int(schema["entity_min_unique"]),
        top_entities=int(schema["entity_top_k"]),
        te_smooth=float(cfg["encoding"]["target_smooth"]),
        frozen_spec=frozen_spec,
        calendar_family=bool(_lite("calendar_family")),
        allow_te=not bool(_lite("bank_no_te")))


def reconcile_final_features(search_features: list[str],
                             final_train: pd.DataFrame,
                             final_test: pd.DataFrame,
                             raw_train: pd.DataFrame,
                             raw_test: pd.DataFrame, *, label: str,
                             time_col: str | None, split_seed: int,
                             final_features: list[str]) -> dict:
    from protocol import (_forward_encodings, _numeric_array,
                          _random_oof_encodings)

    missing = [name for name in search_features
               if name not in final_train.columns
               or name not in final_test.columns]
    failures: list[str] = []
    rebuilt: dict[str, str] = {}
    encoding_cache: dict[str, tuple] = {}
    for name in missing:
        if name.endswith("_oof_te") or name.endswith("_freq"):
            base = (name[:-len("_oof_te")] if name.endswith("_oof_te")
                    else name[:-len("_freq")])
            if base not in raw_train.columns or base not in raw_test.columns:
                failures.append(name)
                continue
            if base not in encoding_cache:
                if time_col:
                    encoding_cache[base] = _forward_encodings(
                        raw_train, raw_test, base, label, smooth=20.0)
                else:
                    encoding_cache[base] = _random_oof_encodings(
                        raw_train, raw_test, base, label, int(split_seed),
                        smooth=20.0)
            encoded = encoding_cache[base]
            if name.endswith("_oof_te"):
                final_train[name] = _numeric_array(encoded[0],
                                                   final_train.index)
                final_test[name] = _numeric_array(encoded[1],
                                                  final_test.index)
                rebuilt[name] = ("forward_target_encoding" if time_col
                                 else "random_oof_target_encoding")
            else:
                final_train[name] = _numeric_array(encoded[2],
                                                   final_train.index)
                final_test[name] = _numeric_array(encoded[3],
                                                  final_test.index)
                rebuilt[name] = "frequency_encoding"
        elif name.endswith("_log"):
            base = name[:-len("_log")]
            if base not in raw_train.columns or base not in raw_test.columns:
                failures.append(name)
                continue
            final_train[name] = _numeric_array(
                np.log1p(pd.to_numeric(raw_train[base],
                                       errors="coerce").clip(lower=0)),
                final_train.index)
            final_test[name] = _numeric_array(
                np.log1p(pd.to_numeric(raw_test[base],
                                       errors="coerce").clip(lower=0)),
                final_test.index)
            rebuilt[name] = "log1p_clipped"
        elif name.endswith("_code"):
            base = name[:-len("_code")]
            if base not in raw_train.columns or base not in raw_test.columns:
                failures.append(name)
                continue
            cat_fit = pd.Categorical(raw_train[base])
            final_train[name] = _numeric_array(
                cat_fit.codes.astype(np.float32), final_train.index)
            final_test[name] = _numeric_array(
                pd.Categorical(raw_test[base],
                               categories=cat_fit.categories)
                .codes.astype(np.float32), final_test.index)
            rebuilt[name] = "categorical_codes_fit_on_outer_train"
        else:
            failures.append(name)
    if failures:
        raise RuntimeError(
            "frozen representation replay failed; cannot rebuild search "
            f"features on the outer frames: {sorted(failures)}")
    return {
        "frozen_feature_count": len(search_features),
        "missing_before_replay": missing,
        "rebuilt": rebuilt,
        "outer_only_unused": [name for name in final_features
                              if name not in set(search_features)],
    }


def prefix_spec(spec: FeatureSpec, round_index: int, key: str) -> FeatureSpec:
    return FeatureSpec(
        name=f"v13_r{round_index}_{key}_{spec.name}",
        expression=spec.expression, family=spec.family,
        source=f"{spec.source}:model={key}:round={round_index}",
        description=spec.description, executor=spec.executor)


def _closed_llm_order(surface_names: list[str],
                      all_specs: dict) -> list[str]:
    ordered: list[str] = []
    placed: set[str] = set()

    def visit(name: str, stack: frozenset) -> None:
        if name in placed or name not in all_specs or name in stack:
            return
        expression = str(all_specs[name].expression)
        for other in all_specs:
            if other != name and other in expression:
                visit(other, stack | {name})
        if name not in placed:
            placed.add(name)
            ordered.append(name)

    for name in surface_names:
        visit(name, frozenset())
    return ordered


def composition_sweep_specs(primitives: list[str]) -> list[FeatureSpec]:
    def col(name: str) -> str:
        return f"pd.to_numeric(df['{name}'], errors='coerce').fillna(0.0)"

    specs: list[FeatureSpec] = []

    def add(name: str, expression: str) -> None:
        specs.append(FeatureSpec(
            name=name, expression=expression, family="sweep",
            source="composition_sweep", executor="pointwise",
            description="bounded composition of ranked bank primitives"))

    for a in primitives:
        add(f"v13_sw_log1p__{a}", f"np.log1p(np.abs({col(a)}))")
    for i, a in enumerate(primitives):
        for b in primitives[i + 1:]:
            add(f"v13_sw_mul__{a}__{b}", f"{col(a)} * {col(b)}")
        for b in primitives:
            if b == a:
                continue
            add(f"v13_sw_ratio__{a}__{b}",
                f"{col(a)} / (np.abs({col(b)}) + 1e-6)")
            add(f"v13_sw_mullog__{a}__{b}",
                f"{col(a)} * np.log1p(np.abs({col(b)}))")
    return specs


def bounded_idiom_specs(primitives: list[str], entities: list[str],
                        values: list[str], visible: list[str], label: str,
                        temporal: bool, order: int | None = None,
                        name_prefix: str = "vg") -> list[FeatureSpec]:
    def col(name: str) -> str:
        return f"pd.to_numeric(df['{name}'], errors='coerce').fillna(0.0)"

    def bound_pos(x: str) -> str:
        return f"np.log1p(np.maximum({x}, 0.0))"

    def bound_abs(x: str) -> str:
        return f"np.log1p(np.abs({x}))"

    specs: list[FeatureSpec] = []

    def add(name: str, expression: str, executor: str = "pointwise") -> None:
        specs.append(FeatureSpec(
            name=name, expression=expression, family="idiom",
            source="bounded_idiom_grid", executor=executor,
            description="bounded-idiom composition of bank primitives"))

    for i, a in enumerate(primitives):
        ca = col(a)
        add(f"{name_prefix}_bpos__{a}", bound_pos(ca))
        add(f"{name_prefix}_babs__{a}", bound_abs(ca))
        for b in primitives[i + 1:]:
            cb = col(b)
            add(f"{name_prefix}_mulBB__{a}__{b}", f"{bound_pos(ca)} * {bound_pos(cb)}")
            add(f"{name_prefix}_mulAA__{a}__{b}", f"{bound_abs(ca)} * {bound_abs(cb)}")
            add(f"{name_prefix}_ratB__{a}__{b}",
                f"{bound_pos(ca)} / ({bound_pos(cb)} + 1.0)")
            add(f"{name_prefix}_ratB__{b}__{a}",
                f"{bound_pos(cb)} / ({bound_pos(ca)} + 1.0)")
            add(f"{name_prefix}_difB__{a}__{b}", f"{bound_pos(ca)} - {bound_pos(cb)}")
            add(f"{name_prefix}_rawmulB__{a}__{b}", f"{ca} * {bound_abs(cb)}")
            add(f"{name_prefix}_rawmulB__{b}__{a}", f"{cb} * {bound_abs(ca)}")
    order = int(order if order is not None
                else (_lite("idiom_order", 2) or 2))
    if order >= 3:
        import itertools as _it
        top8 = primitives[:8]
        for a, b, c in _it.combinations(top8, 3):
            for x, y, z in ((a, b, c), (a, c, b), (b, c, a)):
                cx, cy, cz = col(x), col(y), col(z)
                add(f"{name_prefix}3_ratB__{x}__{y}__{z}",
                    f"{bound_pos(cx)} * {bound_pos(cy)} / "
                    f"({bound_pos(cz)} + 1.0)")
                add(f"{name_prefix}3_ratA__{x}__{y}__{z}",
                    f"{bound_abs(cx)} * {bound_abs(cy)} / "
                    f"({bound_abs(cz)} + 1.0)")
            add(f"{name_prefix}3_mulBBB__{a}__{b}__{c}",
                f"{bound_pos(col(a))} * {bound_pos(col(b))} * "
                f"{bound_pos(col(c))}")
    if order >= 4:
        import itertools as _it
        top6 = primitives[:6]
        for combo in _it.combinations(top6, 4):
            a, b, c, d = combo
            ca, cb, cc_, cd = (col(v) for v in combo)
            add(f"{name_prefix}4_rat2__{a}__{b}__{c}__{d}",
                f"{bound_pos(ca)} * {bound_pos(cb)} / "
                f"(({bound_pos(cc_)} + 1.0) * ({bound_pos(cd)} + 1.0))")
            add(f"{name_prefix}4_mul2rat2__{a}__{b}__{c}__{d}",
                f"({bound_abs(ca)} * {bound_abs(cb)}) - "
                f"({bound_abs(cc_)} * {bound_abs(cd)})")
    n_entities = 4 if order >= 3 else 2
    n_values = 2 if order >= 3 else 1
    windows = (5, 8, 10, 15, 20, 50) if order >= 3 else (8, 20)
    if temporal:
        for entity in entities[:n_entities]:
            for value in values[:n_values]:
                for agg in ("std", "mean", "max"):
                    for window in windows:
                        raw = {"op": "rolling_stat", "entity_cols": [entity],
                               "value_col": value, "agg": agg,
                               "window": window, "shift": 1}
                        try:
                            normalized = validate_agg_spec(
                                raw, columns=visible, label=label,
                                temporal=True)
                        except ValueError:
                            continue
                        add(f"{name_prefix}_roll_{entity}_{value}_{agg}_w{window}",
                            json.dumps(normalized, sort_keys=True), "agg")
    return specs


def s4_evaluation_folds(rows: np.ndarray, y: np.ndarray, *, seed: int,
                        n_folds: int = 3,
                        mode: str = "random") -> list[np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    if str(mode) == "chrono":
        ordered = np.sort(rows)
        head = int(len(ordered) * 0.4)
        tail = ordered[head:]
        blocks = [np.array(block, copy=True)
                  for block in np.array_split(tail, int(n_folds))]
        labels = np.asarray(y, dtype=np.int8)
        for index, block in enumerate(blocks):
            block_y = labels[block]
            if len(block) < 10 or int(block_y.sum()) == 0 \
                    or int((1 - block_y).sum()) == 0:
                raise ValueError(
                    "S4 chrono evaluation block needs at least 10 rows and "
                    f"both classes (block {index + 1}); increase discovery "
                    "rows or use --s4-eval random")
        return blocks
    labels = np.asarray(y, dtype=np.int8)[rows]
    rng = np.random.default_rng(int(seed))
    parts: list[list[np.ndarray]] = [[], []]
    for klass in (0, 1):
        class_rows = np.array(rows[labels == klass], copy=True)
        rng.shuffle(class_rows)
        parts[klass] = list(np.array_split(class_rows, int(n_folds)))
    folds: list[np.ndarray] = []
    for index in range(int(n_folds)):
        fold = np.sort(np.concatenate([parts[0][index], parts[1][index]]))
        fold_y = np.asarray(y, dtype=np.int8)[fold]
        if len(fold) < 10 or int(fold_y.sum()) == 0 \
                or int((1 - fold_y).sum()) == 0:
            raise ValueError(
                "S4 depth evaluation fold needs at least 10 rows and both "
                "classes; reduce depth count or increase discovery rows")
        folds.append(fold)
    return folds


def _s4_triplet(value, *, cast, name: str) -> tuple:
    raw = value.split(",") if isinstance(value, str) else list(value)
    if len(raw) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    parsed = tuple(cast(item) for item in raw)
    if any(item <= 0 for item in parsed):
        raise ValueError(f"{name} values must all be positive")
    return parsed


def _s4_spec_references(spec: FeatureSpec, known: set[str]) -> list[str]:
    if spec.executor == "agg":
        try:
            payload = json.loads(spec.expression)
        except (TypeError, json.JSONDecodeError):
            return []
        values = list(payload.get("entity_cols") or [])
        if payload.get("value_col"):
            values.append(str(payload["value_col"]))
        return [name for name in dict.fromkeys(values) if name in known]
    refs = re.findall(r"df\[['\"]([^'\"]+)['\"]\]", spec.expression)
    return [name for name in dict.fromkeys(refs) if name in known]


def s4_depth_grid_specs(
    *,
    depth: int,
    bank: list[str],
    previous_survivors: list[str],
    order_one_parents: list[str],
    entities: list[str],
    values: list[str],
    visible: list[str],
    label: str,
    temporal: bool,
) -> tuple[list[FeatureSpec], dict[str, dict]]:
    if int(depth) not in (1, 2, 3):
        raise ValueError("S4 depth must be 1, 2, or 3")

    def col(name: str) -> str:
        return f"pd.to_numeric(df['{name}'], errors='coerce').fillna(0.0)"

    def bpos(expr: str) -> str:
        return f"np.log1p(np.maximum({expr}, 0.0))"

    def babs(expr: str) -> str:
        return f"np.log1p(np.abs({expr}))"

    specs: list[FeatureSpec] = []
    lineage: dict[str, dict] = {}

    def add(name: str, expression: str, parents: list[str], *,
            executor: str = "pointwise", kind: str = "bounded") -> None:
        if name in lineage:
            return
        specs.append(FeatureSpec(
            name=name, expression=expression, family=f"s4_depth_{depth}",
            source="s4_depth_grid", executor=executor,
            description=(f"S4 depth-{depth} {kind}; parents="
                         + ",".join(parents))))
        lineage[name] = {
            "depth": int(depth), "parents": list(parents), "kind": kind,
            "source": "deterministic_safety_net",
        }

    if depth == 1:
        for parent in bank:
            cp = col(parent)
            add(f"s4d1_bpos__{parent}", bpos(cp), [parent])
            add(f"s4d1_babs__{parent}", babs(cp), [parent])
    else:
        right_parents = (list(dict.fromkeys(order_one_parents + bank))
                         if depth == 3 else
                         list(dict.fromkeys(previous_survivors + bank)))
        for left in previous_survivors:
            for right in right_parents:
                if right == left:
                    continue
                ca, cb = col(left), col(right)
                prefix = f"s4d{depth}"
                add(f"{prefix}_mulBB__{left}__{right}",
                    f"{bpos(ca)} * {bpos(cb)}", [left, right])
                add(f"{prefix}_ratB__{left}__{right}",
                    f"{bpos(ca)} / ({bpos(cb)} + 1.0)", [left, right])
                add(f"{prefix}_ratB__{right}__{left}",
                    f"{bpos(cb)} / ({bpos(ca)} + 1.0)", [right, left])
                add(f"{prefix}_difB__{left}__{right}",
                    f"{bpos(ca)} - {bpos(cb)}", [left, right])

    rolling_values = (list(values[:2]) if depth == 1
                      else list(previous_survivors[:4]))
    s4_aggs = tuple(
        part.strip() for part in str(_lite("s4_rolling_aggs", "max,std"))
        .split(",") if part.strip())
    s4_windows = tuple(
        int(part) for part in str(_lite("s4_rolling_windows", "5,10"))
        .split(",") if part.strip())
    if not set(s4_aggs) <= set(AGG_WHITELIST):
        raise ValueError(f"--s4-rolling-aggs must be within {AGG_WHITELIST}")
    if not s4_windows or any(w < 2 for w in s4_windows):
        raise ValueError("--s4-rolling-windows must be integers >= 2")
    if temporal and rolling_values:
        for entity in list(dict.fromkeys(entities))[:2]:
            for value in rolling_values:
                for agg in s4_aggs:
                    for window in s4_windows:
                        raw = {
                            "op": "rolling_stat", "entity_cols": [entity],
                            "value_col": value, "agg": agg,
                            "window": window, "shift": 1,
                        }
                        try:
                            normalized = validate_agg_spec(
                                raw, columns=visible, label=label,
                                temporal=True)
                        except ValueError:
                            continue
                        name = (f"s4d{depth}_roll__{entity}__{value}__"
                                f"{agg}__w{window}")
                        add(name, json.dumps(normalized, sort_keys=True),
                            [value], executor="agg", kind="strict_past_rolling")
    return specs, lineage


def _stump_gain(values: np.ndarray, residual: np.ndarray,
                bins: int = 64) -> float:
    mask = np.isfinite(values)
    if int(mask.sum()) < 1000:
        return 0.0
    v, r = values[mask], residual[mask]
    edges = np.unique(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]))
    if len(edges) == 0:
        return 0.0
    idx = np.searchsorted(edges, v, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(np.float64)
    sums = np.bincount(idx, weights=r, minlength=len(edges) + 1)
    csum_n = np.cumsum(counts)[:-1]
    csum_s = np.cumsum(sums)[:-1]
    n = float(len(r))
    total = float(r.sum())
    valid = (csum_n > 0) & (csum_n < n)
    if not valid.any():
        return 0.0
    mean_l = csum_s[valid] / csum_n[valid]
    mean_r = (total - csum_s[valid]) / (n - csum_n[valid])
    gain = csum_n[valid] * (n - csum_n[valid]) / n * (mean_l - mean_r) ** 2
    return float(gain.max() / max(np.var(r) * n, 1e-12))


def ruler_fusion_filter(pool: list[str], pool_matrix: np.ndarray,
                        conditioning: np.ndarray, residual: np.ndarray, *,
                        is_net: bool, top_m: int, entry_cfg: dict,
                        seed: int, weights) -> tuple[list[str], dict]:
    fwl = entry_scores(pool_matrix, pool, conditioning, residual,
                       folds=int(entry_cfg["residualize_folds"]),
                       ridge_alpha=float(entry_cfg["ridge_alpha"]),
                       seed=seed, weights=weights)
    order_fwl = sorted(pool, key=lambda n: -abs(float(fwl.get(n, 0.0))))
    stump = {name: _stump_gain(pool_matrix[:, j], residual)
             for j, name in enumerate(pool)}
    order_stump = sorted(pool, key=lambda n: -stump[n])
    keep = set(order_fwl[:top_m]) | set(order_stump[:top_m])
    rulers = {"fwl_top": order_fwl[:top_m], "stump_top": order_stump[:top_m]}
    if is_net:
        from sklearn.preprocessing import QuantileTransformer
        transformer = QuantileTransformer(
            output_distribution="normal",
            subsample=min(100_000, len(pool_matrix)), random_state=seed)
        qt_matrix = transformer.fit_transform(pool_matrix)
        qtfwl = entry_scores(qt_matrix, pool, conditioning, residual,
                             folds=int(entry_cfg["residualize_folds"]),
                             ridge_alpha=float(entry_cfg["ridge_alpha"]),
                             seed=seed + 1, weights=weights)
        order_qt = sorted(pool, key=lambda n: -abs(float(qtfwl.get(n, 0.0))))
        keep |= set(order_qt[:top_m])
        rulers["qtfwl_top"] = order_qt[:top_m]
    kept_idx = [j for j, name in enumerate(pool) if name in keep]
    filtered = [pool[j] for j in kept_idx]
    audit = {"pool_before": len(pool), "pool_after": len(filtered),
             "top_m": int(top_m), "is_net": bool(is_net), "rulers": rulers}
    return filtered, {"matrix_index": kept_idx, **audit}


def materialize_spec(*, spec: FeatureSpec, train: pd.DataFrame,
                     apply: pd.DataFrame, label: str,
                     time_values: np.ndarray | None) -> bool:
    if spec.name in train.columns or spec.name in apply.columns:
        return False
    if spec.executor == "agg":
        return bool(apply_agg_spec(
            train, apply, spec.name, json.loads(spec.expression), label=label,
            time_values=time_values))
    return bool(apply_feature_spec(train, apply, spec, label=label))


def top_s_evidence(scores: dict[str, float], families: dict[str, str],
                   floor: float, top: int = 6) -> str:
    lines = [
        "Entry-score evidence (S = cross-fitted orthogonalized correlation "
        f"between each candidate factor and this model's OOF residual; noise "
        f"floor tau = {floor:.4f}; candidates with |S| < tau will not enter):"]
    if not scores:
        lines.append("- candidate pool is empty for this model this round.")
        return "\n".join(lines)
    ordered = sorted(scores.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[:top]
    for name, value in ordered:
        lines.append(f"- {name} (family={families.get(name, 'llm')}): "
                     f"S={value:+.4f}")
    lines.append(
        "Mechanism families still carrying |S| above the floor are where "
        "unexplained signal lives; combine or vary those mechanisms "
        "(exploit) and keep at most one free hypothesis (explore).")
    return "\n".join(lines)


V4_PORTFOLIO_CONSTRAINT = (
    "\n\nV4 portfolio constraint: propose a diverse batch across "
    "distinct business mechanisms: engagement intensity, "
    "navigation friction, session composition, visitor/"
    "traffic context, and calendar context when supported. "
    "Use pagevalues in at most two proposals; do not invent "
    "event chronology when the dataset has no timestamp.")


def proposal_description_for(base: str) -> str:
    text = base
    if _LITE.get("v5"):
        text += (
            "\n\nV5 factor-search constraint: propose a diverse, "
            "domain-adaptive batch grounded only in the listed "
            "columns and residual evidence. Exact-value business "
            "rules may use pd.to_numeric(df['column'], "
            "errors='coerce').eq(value).astype('float64'). Include "
            "harmonic candidates only when the schema supports a "
            "real periodic mechanism: (a) sin/cos encoding of a "
            "cyclic scalar such as hour, weekday or month, or "
            "(b) row-wise first/second harmonic amplitude or phase "
            "across genuinely ordered lag columns such as monthly "
            "bill/payment histories. Harmonic expressions must be "
            "explicit combinations of available columns using "
            "np.sin, np.cos, np.sqrt and optionally np.arctan2; do "
            "not call np.fft, infer an ordering from unrelated "
            "columns, perform cross-row transforms, or invent "
            "timestamps. Prefer at most two complementary harmonic "
            "features and spend the remaining proposals on distinct "
            "business mechanisms."
        )
    elif _LITE.get("v4"):
        text += V4_PORTFOLIO_CONSTRAINT
    return text


def llm_fetch_for_model(*, llm: VariableBundleLLM, record: dict,
                        train: pd.DataFrame, label: str, meta: dict,
                        dataset_description: str, tried: set[str],
                        entities: list[str], values: list[str],
                        visible: list[str], disagreement: np.ndarray,
                        current_features: list[str], evidence: str,
                        point_cap: int, agg_cap: int,
                        temporal: bool,
                        blacklist_context: str = "") -> dict:
    key = str(record["model_key"])
    profile, profile_stats = build_residual_profile(
        train, record["oof"], disagreement, label=label,
        features=current_features, cats=list(meta.get("cats", [])))
    profile_text = f"{profile}\n\n{evidence}"
    if blacklist_context:
        profile_text += f"\n\n{blacklist_context}"
    raw_specs = []
    if int(point_cap) > 0:
        raw_specs = llm.propose_pointwise_variable(
            dataset_description=dataset_description, label=label,
            columns=visible, existing_features=current_features,
            tried=set(tried), residual_profile=profile_text,
            max_n=int(point_cap), source=f"v13_residual_pointwise:{key}")
    raw_aggs = []
    if int(agg_cap) > 0:
        raw_aggs = llm.propose_aggregates(
            dataset_description=dataset_description, label=label,
            entity_candidates=entities, value_candidates=values,
            existing_features=current_features, tried=set(tried),
            residual_profile=profile_text, n=int(agg_cap),
            source=f"v13_residual_aggregate:{key}")
    return {"model_key": key, "raw_specs": raw_specs, "raw_aggs": raw_aggs,
            "profile_stats": profile_stats, "evidence": evidence}


def llm_propose_for_model(*, llm: VariableBundleLLM, record: dict,
                          train: pd.DataFrame, apply: pd.DataFrame,
                          label: str, meta: dict, dataset_description: str,
                          round_index: int, tried: set[str],
                          entities: list[str], values: list[str],
                          time_values: np.ndarray | None,
                          visible: list[str], disagreement: np.ndarray,
                          current_features: list[str],
                          evidence: str, point_cap: int,
                          agg_cap: int,
                          blacklist_context: str = "",
                          prefetched: dict | None = None) -> tuple[list[FeatureSpec], dict]:
    key = str(record["model_key"])
    if prefetched is not None:
        profile_stats = prefetched["profile_stats"]
        raw_specs = list(prefetched["raw_specs"])
        raw_aggs_pre = list(prefetched["raw_aggs"])
    else:
        profile, profile_stats = build_residual_profile(
            train, record["oof"], disagreement, label=label,
            features=current_features, cats=list(meta.get("cats", [])))
        profile_text = f"{profile}\n\n{evidence}"
        if blacklist_context:
            profile_text += f"\n\n{blacklist_context}"
        raw_aggs_pre = None
    specs: list[FeatureSpec] = []
    audit_rows: list[dict] = []
    if prefetched is None:
        raw_specs = []
        if int(point_cap) > 0:
            raw_specs = llm.propose_pointwise_variable(
                dataset_description=dataset_description, label=label,
                columns=visible, existing_features=current_features,
                tried=tried, residual_profile=profile_text,
                max_n=int(point_cap), source=f"v13_residual_pointwise:{key}")
    for raw in raw_specs:
        tried.add(raw.name)
        spec = prefix_spec(raw, round_index, key)
        ok = materialize_spec(spec=spec, train=train, apply=apply, label=label,
                              time_values=time_values)
        audit_rows.append({"spec": spec.to_dict(),
                           "status": "materialized" if ok else "rejected"})
        if ok:
            specs.append(spec)
    if raw_aggs_pre is not None:
        raw_aggs = raw_aggs_pre
    else:
        raw_aggs = []
        if int(agg_cap) > 0:
            raw_aggs = llm.propose_aggregates(
                dataset_description=dataset_description, label=label,
                entity_candidates=entities, value_candidates=values,
                existing_features=current_features, tried=tried,
                residual_profile=profile_text, n=int(agg_cap),
                source=f"v13_residual_aggregate:{key}")
    for raw in raw_aggs:
        name = str(raw.get("name", ""))
        tried.add(name)
        try:
            normalized = validate_agg_spec(
                raw, columns=visible, label=label,
                temporal=time_values is not None)
            base = FeatureSpec(
                name=name, expression=json.dumps(normalized), family="agg",
                source=str(raw.get("source", "v13_residual_aggregate")),
                description=str(raw.get("description", "")), executor="agg")
            spec = prefix_spec(base, round_index, key)
            ok = materialize_spec(spec=spec, train=train, apply=apply,
                                  label=label, time_values=time_values)
            audit_rows.append({"spec": spec.to_dict(),
                               "status": "materialized" if ok else "rejected"})
            if ok:
                specs.append(spec)
        except ValueError as exc:
            audit_rows.append({"name": name, "status": "invalid_spec",
                               "error": str(exc)})
    return specs, {"residual_profile": profile_stats,
                   "evidence": evidence, "proposals": audit_rows}


def round_gate(*, y_apply: np.ndarray, block: np.ndarray,
               incumbent_pred: np.ndarray, candidate_pred: np.ndarray,
               cfg: dict, round_index: int, max_rounds: int) -> tuple:
    gate_cfg = cfg["experiment"]["gate"]
    alpha_round = float(gate_cfg["alpha_total"]) / int(max_rounds)
    result = confirmation_gate(
        y_apply[block], incumbent_pred[block], candidate_pred[block],
        bootstrap_repeats=int(gate_cfg["bootstrap_repeats"]),
        alpha=alpha_round,
        tau_auprc=float(gate_cfg["tau_auprc"]),
        se_multiplier=float(gate_cfg["se_multiplier"]),
        min_positive_fold_fraction=float(
            gate_cfg["min_positive_fold_fraction"]),
        seed=int(cfg["engine"]["seed"]) + int(round_index))
    audit = {
        "round": int(round_index),
        "block_rows": int(len(block)),
        "block_positives": int(np.asarray(y_apply)[block].sum()),
        "block_first_row": int(block[0]),
        "block_last_row": int(block[-1]),
        "alpha_round": float(alpha_round),
        "se_multiplier": float(gate_cfg["se_multiplier"]),
        "gate": result.to_dict(),
        "accepted": bool(result.accepted),
    }
    return result, audit


def run_baseline(dataset: str, out_dir: Path, cfg: dict) -> dict:
    l1_library, l2_library = assert_frozen_graph(cfg)
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=False)
    dataset_cfg = apply_outer_split_override(get_config(dataset), cfg)
    label = str(dataset_cfg["label"]).lower()
    gpu = install_gpu(cfg, out_dir)

    raw_inner, raw_search, split = _load_raw_partitions(
        dataset_cfg, purpose="search")
    train, apply, features, representation_meta = prepare_e1(
        raw_inner, raw_search, dataset_cfg)
    representation_audit = select_and_apply_representation(
        raw_inner, raw_search, train, apply, features, dataset_cfg, cfg)
    seat_plan = seat_representation_plan(l1_library, representation_audit,
                                         features)
    if seat_plan:
        log("rep-scope seat: "
            + json.dumps(seat_plan.get("seat_rep", {}), sort_keys=True))
    write_json(out_dir / "representation_audit.json", representation_audit)
    del raw_inner, raw_search
    gc.collect()
    neg_ratio = _lite("neg_subsample")
    if neg_ratio:
        mask = negative_subsample(train, label, float(neg_ratio),
                                  int(cfg["engine"]["seed"]))
        before = len(train)
        train = train.iloc[mask].reset_index(drop=True)
        log(f"neg-subsample {neg_ratio}x: inner {before} -> {len(train)}")
    y_train = train[label].to_numpy(np.int8, copy=False)
    log(f"baseline fit: inner={len(train)} search={len(apply)} "
        f"features={len(features)}")

    fold_log = FoldLog(out_dir / "baseline_folds.jsonl")
    baseline_drops: list = []
    log(f"member time limit: "
        f"{cfg['engine'].get('member_time_limit_sec', 2700)}s; "
        f"fit workers: {_fit_workers()}")
    seat_base_bl = (seat_plan or {}).get("seat_base") or {}
    l1_surfaces = {f"l1_{key}": list(seat_base_bl.get(key, features))
                   for key, _, _ in l1_library}
    l1_records = fit_layer(
        train=train, apply=apply, label=label, library=l1_library,
        surfaces=l1_surfaces, prefix="l1", layer_index=1, cfg=cfg,
        fold_log=fold_log, stage="baseline_raw_l1",
        time_limit_audit=baseline_drops)
    stack_columns = _attach_stack_columns(train, apply, l1_records)
    l2_surfaces = {f"l2_{key}": list(dict.fromkeys(features + stack_columns))
                   for key, _, _ in l2_library}
    l2_records = fit_layer(
        train=train, apply=apply, label=label, library=l2_library,
        surfaces=l2_surfaces, prefix="l2", layer_index=2, cfg=cfg,
        fold_log=fold_log, stage="baseline_raw_l2",
        time_limit_audit=baseline_drops)

    steps = int(cfg["engine"]["blend_steps"])
    we_l2, we_l3 = build_ensembles(l1_records, l2_records, y_train, steps)
    write_baseline_records(out_dir, l1_records + l2_records, we_l2, we_l3)

    summary = {
        "method": f"{METHOD}_BASELINE",
        "stage": "baseline",
        "dataset": dataset,
        "outer_split_seed": int(dataset_cfg["split_seed"]),
        "representation": str(representation_audit.get("choice", "e1")),
        "representation_probe": representation_audit.get("probe_scores"),
        "representation_scope": str(_lite("rep_scope", "shared")),
        "representation_seat_map": (seat_plan or {}).get("seat_rep"),
        "graph": {"l1_seats": [name for name, _, _ in l1_library],
                  "l2_seats": [name for name, _, _ in l2_library]},
        "split": split,
        "n_inner_train": len(train),
        "n_search_val": len(apply),
        "n_features": len(features),
        "features": features,
        "representation_meta": {
            key: representation_meta.get(key)
            for key in ("cats", "nums", "binaries", "time_col", "split_kind")
        },
        "n_l1": len(l1_records),
        "n_l2": len(l2_records),
        "time_limit_dropped": baseline_drops,
        "we_l2_oof_auprc": we_l2["oof_auprc"],
        "we_l3_oof_auprc": we_l3["oof_auprc"],
        "we_l2_blend_selection": we_l2.get("selection"),
        "we_l3_blend_selection": we_l3.get("selection"),
        "search_label_reads": 0,
        "outer_test_evaluations": 0,
        "engine": {
            "seed": int(cfg["engine"]["seed"]),
            "bag_folds": int(cfg["engine"]["bag_folds"]),
            "blend_steps": steps,
            "member_time_limit_sec": cfg["engine"].get(
                "member_time_limit_sec", 2700),
            "fit_workers": _fit_workers(),
            "lite": cfg["engine"].get("lite"),
            "lite_prune_audit": cfg["engine"].get("lite_prune_audit"),
        },
        "gpu_mlp": gpu,
        "config_sha256": sha256(CONFIG_PATH),
        "theory_sha256": sha256(THEORY_PATH),
        "design_sha256": sha256(DESIGN_PATH),
        "wall_total_sec": round(time.time() - started, 3),
    }
    write_json(out_dir / "summary.json", summary)
    log(f"baseline done: WE_L3 OOF={we_l3['oof_auprc']:.7f} "
        f"wall={summary['wall_total_sec']:.0f}s")
    return summary


def discovery_row_subsample(n_rows: int, cap: int, seed: int) -> np.ndarray:
    if n_rows <= int(cap):
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rows = rng.choice(n_rows, size=int(cap), replace=False)
    rows.sort()
    return rows.astype(np.int64)


def run_forge(dataset: str, out_dir: Path, baseline_dir: Path, cfg: dict, *,
              stub_llm: bool = False,
              max_rounds_override: int | None = None,
              domain_context: str | None = None,
              inject_factors: Path | None = None,
              inject_mode: str = "surface") -> dict:
    if not bool(cfg["llm"]["enabled"]):
        raise RuntimeError("THEORY section 4 requires the live LLM channel")
    l1_library, l2_library = assert_frozen_graph(cfg)
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=False)
    exp = cfg["experiment"]
    max_rounds = (int(max_rounds_override) if max_rounds_override is not None
                  else int(exp["max_rounds"]))
    if max_rounds < 1 or max_rounds > len(exp["gate_block_weights"]):
        raise ValueError("max_rounds exceeds precommitted gate-block budget")
    entry_cfg = cfg["selection"]["entry"]
    exit_cfg = cfg["selection"]["exit"]
    llm_cfg = cfg["llm"]
    steps = int(cfg["engine"]["blend_steps"])
    engine_seed = int(cfg["engine"]["seed"])
    dataset_cfg = apply_outer_split_override(get_config(dataset), cfg)
    label = str(dataset_cfg["label"]).lower()
    time_col = (str(dataset_cfg["time_col"]).lower()
                if dataset_cfg.get("time_col") else None)
    forbidden = [str(value).lower() for value in dataset_cfg.get("drop", [])]
    dataset_description = str(dataset_cfg.get("desc", dataset))
    domain_context = (domain_context or "").strip()
    if domain_context:
        dataset_description += (
            "\n\nDomain knowledge supplied by the researcher "
            "(mechanism/units/valid transformations only; never row labels "
            f"or outer-test statistics):\n{domain_context}")
    gpu = install_gpu(cfg, out_dir)

    raw_inner, raw_search, split = _load_raw_partitions(
        dataset_cfg, purpose="search")
    train, apply, features, meta = prepare_e1(raw_inner, raw_search,
                                              dataset_cfg)
    representation_audit = select_and_apply_representation(
        raw_inner, raw_search, train, apply, features, dataset_cfg, cfg)
    write_json(out_dir / "representation_audit.json", representation_audit)
    representation_choice = str(representation_audit.get("choice", "e1"))
    seat_plan = seat_representation_plan(l1_library, representation_audit,
                                         features)
    seat_base_map = (seat_plan or {}).get("seat_base") or {}
    if seat_plan:
        log("rep-scope seat: "
            + json.dumps(seat_plan.get("seat_rep", {}), sort_keys=True))

    def base_cols(key: str) -> list:
        return seat_base_map.get(str(key), features)

    learner_of = {str(key): str(learner) for key, learner, _ in l1_library}

    def seat_is_net(key: str) -> bool:
        return learner_of.get(str(key), "").lower().startswith(
            ("mlp", "torch", "fastai"))
    bank_names, bank_audit = materialize_bank(
        cfg, raw_inner, raw_search, train, apply,
        label=label, time_col=time_col, forbidden=forbidden)
    write_json(out_dir / "factor_bank.json", bank_audit)
    bank_spec = bank_audit["spec"]
    write_json(out_dir / "factor_spec.json", {
        "dataset": dataset,
        "outer_split_seed": int(dataset_cfg["split_seed"]),
        "stage": "search_frozen_discovery",
        "representation_features": list(features),
        "bank_spec": bank_spec,
    })
    families = {entry["name"]: entry["family"]
                for entry in bank_audit["factors"]}
    log(f"factor bank: {len(bank_names)} factors "
        f"entities={[e['name'] for e in bank_audit['entities']]}")
    time_values = None
    if time_col is not None:
        time_values = np.concatenate([
            pd.to_numeric(raw_inner[time_col], errors="coerce").to_numpy(float),
            pd.to_numeric(raw_search[time_col], errors="coerce").to_numpy(float),
        ])
    entities = entity_candidates(train, label=label, cats=meta["cats"],
                                 nums=meta["nums"])
    values = value_candidates(train, label=label, nums=meta["nums"])
    del raw_inner, raw_search
    gc.collect()

    y_train = train[label].to_numpy(np.int8, copy=False)
    y_apply = apply[label].to_numpy(np.int8, copy=False)
    blocks = weighted_chronological_blocks(
        y_apply, list(exp["gate_block_weights"][:max_rounds]))
    input_audit = audit_baseline(
        baseline_dir, dataset=dataset, n_train=len(train),
        n_search=len(apply), features=list(features))
    write_json(out_dir / "input_audit.json", input_audit)
    neg_ratio = _lite("neg_subsample")
    subsample_audit = None
    if neg_ratio:
        mask = negative_subsample(train, label, float(neg_ratio), engine_seed)
        before = len(train)
        train = train.iloc[mask].reset_index(drop=True)
        y_train = train[label].to_numpy(np.int8, copy=False)
        if time_values is not None:
            inner_times = time_values[:before][mask]
            time_values = np.concatenate([inner_times, time_values[before:]])
        subsample_audit = {
            "ratio": float(neg_ratio), "seed": engine_seed,
            "rows_before": int(before), "rows_after": int(len(train)),
            "positives": int(y_train.sum()),
            "applies_to": "inner-train fitting frame only; search-val intact",
        }
        log(f"neg-subsample {neg_ratio}x: inner {before} -> {len(train)} "
            f"(positives {int(y_train.sum())}); search-val untouched")
    baseline_records = load_baseline_records(baseline_dir)
    if _lite("prune_l1"):
        by_member = {record["member"]: record for record in baseline_records}
        ordered = ([f"l1_{key}" for key, _, _ in l1_library]
                   + [f"l2_{key}" for key, _, _ in l2_library])
        before = len(baseline_records)
        baseline_records = [by_member[m] for m in ordered if m in by_member]
        log(f"lite: baseline records narrowed {before} -> "
            f"{len(baseline_records)} to match the pruned roster")
    baseline_summary = json.loads(
        (baseline_dir / "summary.json").read_text(encoding="utf-8"))
    if _LITE.get("v9"):
        baseline_folds = int(
            (baseline_summary.get("engine") or {}).get(
                "bag_folds", cfg["engine"]["bag_folds"]))
        search_folds = lite_bag_folds(cfg, "search_round_01_l1")
        if baseline_folds != search_folds:
            raise SystemExit(
                "V9 fold-consistency check failed: the frozen baseline was "
                f"fitted with {baseline_folds} OOF folds but this search stage "
                f"would use {search_folds}. Selective refit blends baseline "
                "and refitted OOF vectors together, so the two must match. "
                f"Re-run the baseline with --oof-folds {search_folds}, or drop "
                "--lite-search-folds from the forge stage.")
        log(f"V9 fold-consistency check passed: baseline and search both "
            f"use {search_folds} OOF folds")
    baseline_representation = str(baseline_summary.get("representation") or "")
    if baseline_representation and baseline_representation != representation_choice:
        raise SystemExit(
            "representation-consistency check failed: the frozen baseline was "
            f"fitted on '{baseline_representation}' but this search stage's "
            f"probe chose '{representation_choice}'. Selective refit blends "
            "baseline and refitted OOF vectors, so both stages must sit on the "
            "same surface. Re-run the baseline with the same probe settings "
            "(--probe-folds) as the forge stage.")
    forge_seat_map = (seat_plan or {}).get("seat_rep") or {}
    baseline_seat_map = baseline_summary.get("representation_seat_map") or {}
    if forge_seat_map:
        narrowed = {key: baseline_seat_map.get(key) for key in forge_seat_map}
        if narrowed != forge_seat_map:
            raise SystemExit(
                "rep-scope consistency check failed: this forge stage assigns "
                f"{forge_seat_map} but the frozen baseline recorded "
                f"{baseline_seat_map or 'no seat map'}. Re-run the baseline "
                "with the same --rep-scope seat flag and roster "
                "(run_lite.sh: REP_SCOPE=seat reaches both stages).")
    elif baseline_seat_map:
        raise SystemExit(
            "rep-scope consistency check failed: the frozen baseline was "
            f"fitted per-seat ({baseline_seat_map}) but this forge stage "
            "runs a shared surface. Pass --rep-scope seat or re-run the "
            "baseline without it.")
    baseline_dropped = {
        str(entry.get("member"))
        for entry in baseline_summary.get("time_limit_dropped", [])}
    expected_present = [
        member for member in (
            [f"l1_{key}" for key, _, _ in l1_library]
            + [f"l2_{key}" for key, _, _ in l2_library])
        if member not in baseline_dropped]
    actual_members = [r["member"] for r in baseline_records]
    if sorted(actual_members) != sorted(expected_present):
        raise RuntimeError(
            "baseline member roster mismatch: "
            f"missing={sorted(set(expected_present) - set(actual_members))} "
            f"extra={sorted(set(actual_members) - set(expected_present))}")
    if baseline_dropped:
        log(f"baseline time-limit-dropped members tolerated: "
            f"{sorted(baseline_dropped)}")
    for record in baseline_records:
        if len(record["oof"]) != len(train) or len(record["apply"]) != len(apply):
            raise RuntimeError(f"baseline length mismatch: {record['member']}")
    inc_l1 = [r for r in baseline_records if r["layer"] == "l1"]
    inc_l2 = [r for r in baseline_records if r["layer"] == "l2"]
    expected_l1_present = [f"l1_{key}" for key, _, _ in l1_library
                           if f"l1_{key}" not in baseline_dropped]
    if [r["member"] for r in inc_l1] != expected_l1_present:
        raise RuntimeError("baseline L1 portfolio/order mismatch with code_9")


    llm = VariableBundleLLM(
        out_dir / "llm_calls.jsonl",
        model=(os.environ.get("FORGE_LLM_MODEL", "").strip()
               or str(llm_cfg["model"])),
        timeout=int(llm_cfg["timeout_sec"]), attempts=int(llm_cfg["attempts"]),
        max_calls=int(llm_cfg["max_calls"]), stub=bool(stub_llm))

    disc = discovery_row_subsample(
        len(train), int(entry_cfg["discovery_row_cap"]), engine_seed)
    y_disc = y_train[disc]
    n_pos = int(y_disc.sum())
    if n_pos <= 0 or int((1 - y_disc).sum()) <= 0:
        raise RuntimeError("discovery rows must contain both classes")
    alpha_entry = float(entry_cfg["alpha"])
    floor = entry_noise_floor(n_pos, alpha_entry)
    entry_weights = entry_sample_weights(y_disc, _lite("entry_weight", "none"))
    log(f"discovery rows={len(disc)} n_pos={n_pos} entry floor={floor:.5f} "
        f"entry_weight={_lite('entry_weight', 'none')}")

    s4_depth3 = bool(_LITE.get("s4_depth3"))
    s4_funnel = bool(_LITE.get("s4_funnel")) and s4_depth3
    s4_candidate_caps = _s4_triplet(
        _lite("s4_candidate_caps", "128,96,64"), cast=int,
        name="s4_candidate_caps")
    s4_survivor_caps = _s4_triplet(
        _lite("s4_survivor_caps", "8,6,4"), cast=int,
        name="s4_survivor_caps")
    s4_bh_q = _s4_triplet(
        _lite("s4_bh_q", "0.025,0.015,0.010"), cast=float,
        name="s4_bh_q")
    if any(value >= 1.0 for value in s4_bh_q):
        raise ValueError("s4_bh_q values must all be below 1")
    s4_eval_mode = str(_lite("s4_eval", "random"))
    s4_eval_rows = (s4_evaluation_folds(
        disc, y_train, seed=engine_seed + 404, n_folds=3,
        mode=s4_eval_mode)
        if s4_depth3 else [])
    if s4_depth3:
        log("S4 depth-3 fast: candidate caps="
            f"{s4_candidate_caps} survivor caps={s4_survivor_caps} "
            f"BH q={s4_bh_q}; funnel="
            + ("V21-4layer" if s4_funnel else "one-pass per-seat screen"))

    shadow_block = None
    if (_LITE.get("entry_shadow") and (not s4_depth3 or s4_funnel)
            and os.environ.get("FORGE_ABL_NO_TW", "") != "1"):
        shadow_frac = float(os.environ.get("FORGE_TW_FRAC", "")
                            or _lite("shadow_frac", 0.30))
        if not 0.0 < shadow_frac < 1.0:
            raise ValueError("--shadow-frac must be in (0, 1)")
        tail = np.arange(int(len(train) * (1.0 - shadow_frac)), len(train),
                         dtype=np.int64)
        cap = int(entry_cfg["discovery_row_cap"])
        if len(tail) > cap:
            rng = np.random.default_rng(engine_seed + 7)
            tail = np.sort(rng.choice(tail, size=cap, replace=False))
        y_shadow = y_train[tail]
        n_pos_shadow = int(y_shadow.sum())
        if n_pos_shadow > 0 and int((1 - y_shadow).sum()) > 0:
            shadow_block = {
                "rows": tail,
                "n_pos": n_pos_shadow,
                "floor": entry_noise_floor(n_pos_shadow, alpha_entry),
                "frac": shadow_frac,
            }
            log(f"entry shadow block: rows={len(tail)} n_pos={n_pos_shadow} "
                f"floor={shadow_block['floor']:.5f} frac={shadow_frac}")
        else:
            log("entry shadow block: DISABLED (tail slice is single-class); "
                "entry falls back to the discovery rule alone")
    effective_proposal_cycles = (3 if s4_depth3 else
                                 max(int(_lite("proposal_cycles", 1)), 1))
    if ((not s4_depth3 or s4_funnel) and effective_proposal_cycles > 1
            and shadow_block is None
            and str(_LITE.get("shadow_rule", "harm_only")) == "persistence"):
        log("proposal cycles: forced to 1 -- the persistence rule requires a "
            "usable shadow block and none is available")
        effective_proposal_cycles = 1
    if os.environ.get("FORGE_ABL_NO_RECURSION", "") == "1":
        log("ablation: FORGE_ABL_NO_RECURSION=1 -> proposal cycles = 1")
        effective_proposal_cycles = 1

    surfaces: dict[str, list[str]] = {key: [] for key, _, _ in l1_library}
    all_llm_specs: dict[str, FeatureSpec] = {}
    s4_lineage: dict[str, dict] = {
        name: {"depth": 1, "parents": [name], "kind": "bank",
               "source": "fixed_factor_bank"}
        for name in bank_names}
    inject_audit = None
    injected_candidates: list[str] = []
    gated_bundle: list[str] = []
    if inject_factors is not None:
        src_plan = json.loads(Path(inject_factors).read_text(encoding="utf-8"))
        src_specs = src_plan.get("accepted_llm_specs") or {}
        if not src_specs:
            raise SystemExit(
                f"--inject-factors: no accepted_llm_specs in {inject_factors}")
        spec_fields = {"name", "expression", "family", "source",
                       "description", "executor"}
        injected: list[str] = []
        for spec_name, payload in src_specs.items():
            spec = FeatureSpec(**{k: v for k, v in dict(payload).items()
                                  if k in spec_fields})
            if spec.name != spec_name:
                raise SystemExit(
                    f"--inject-factors: spec key {spec_name!r} disagrees with "
                    f"its name field {spec.name!r}")
            all_llm_specs[spec.name] = spec
            families.setdefault(spec.name, spec.family or "llm")
            injected.append(spec.name)
        known = (set(bank_names) | set(all_llm_specs)
                 | set(features) | set(map(str, train.columns)))
        referenced = sorted({
            column
            for name in injected
            for column in re.findall(r"df\[['\"]([^'\"]+)['\"]\]",
                                     all_llm_specs[name].expression)})
        missing = [column for column in referenced if column not in known]
        if missing:
            raise SystemExit(
                "--inject-factors: expressions reference columns this run's "
                f"bank/representation does not provide: {missing}")
        ordered = _closed_llm_order(injected, all_llm_specs)
        if inject_mode == "gated_surface":
            for spec_name in ordered:
                ok = materialize_spec(
                    spec=all_llm_specs[spec_name], train=train, apply=apply,
                    label=label, time_values=time_values)
                if not ok and spec_name not in train.columns:
                    raise SystemExit(
                        f"--inject-factors: materialization failed for "
                        f"{spec_name}")
            gated_bundle = list(ordered)
            mode_note = "gated_surface_bundle_pre_round"
        elif inject_mode == "bank":
            for spec_name in ordered:
                ok = materialize_spec(
                    spec=all_llm_specs[spec_name], train=train, apply=apply,
                    label=label, time_values=time_values)
                if not ok and spec_name not in train.columns:
                    raise SystemExit(
                        f"--inject-factors: materialization failed for "
                        f"{spec_name}")
            injected_candidates = list(ordered)
            mode_note = "bank_candidates_normal_search"
        else:
            for key, _, _ in l1_library:
                surfaces[key] = list(ordered)
            mode_note = "surface_union_all_seats_no_search"
        inject_audit = {
            "source": str(inject_factors),
            "source_sha256": hashlib.sha256(
                Path(inject_factors).read_bytes()).hexdigest(),
            "n_specs": len(injected),
            "specs": list(ordered),
            "referenced_columns": referenced,
            "mode": mode_note,
        }
        write_json(out_dir / "factor_injection.json", inject_audit)
        log(f"factor injection ({mode_note}): {len(injected)} specs from "
            f"{inject_factors}")
    tried: dict[str, set] = {key: set() for key, _, _ in l1_library}
    shared_tried: set[str] = set()
    cumulative_pool: dict[str, int] = {key: 0 for key, _, _ in l1_library}
    blacklist = Blacklist()
    rejected_hypotheses: list[dict] = []
    proposal_monitor: list[dict] = []
    proposal_signatures: set[str] = set()
    consecutive_empty_llm = 0
    selective_refit = bool(_LITE.get("v8") or s4_depth3)
    fast_deploy = bool(_LITE.get("v8")) and _lite("l2_policy") != "full"
    l2_enabled = (not fast_deploy) and bool(_LITE.get("search_l2", True))
    if s4_depth3 and l2_enabled:
        raise ValueError("--s4-depth3 requires --lite-search-l2 off")
    if not l2_enabled:
        inc_we_l2 = _weighted_ensemble(inc_l1, y_train, steps)
        inc_we_l3 = inc_we_l2
        inc_l2 = []
    else:
        inc_we_l2, inc_we_l3 = build_ensembles(inc_l1, inc_l2, y_train, steps)
    inc_we_l3_pred = np.asarray(inc_we_l3["prediction"], dtype=np.float64)
    search_fitted: set[str] = set()
    l1_fits_performed = 0
    l1_fits_saved = 0
    refit_l1_by_round: dict[str, list[str]] = {}
    reused_l1_by_round: dict[str, list[str]] = {}
    no_change_rounds = 0
    pruned_log: list[dict] = []
    llm_phase_wall_sec = 0.0

    fold_log = FoldLog(out_dir / "search_folds.jsonl")
    ledger: list[dict] = []
    accepted_rounds = 0
    gate_blocks_consumed = 0
    stop_reason = "round_cap"
    if (inject_audit is not None and not injected_candidates
            and not gated_bundle):
        max_rounds = 0
        stop_reason = "factor_injection_no_search"
    sweep_candidates: list[str] = []
    sweep_n = int(_lite("comp_sweep", 0) or 0)
    if not s4_depth3 and sweep_n > 0 and bank_names:
        cond_sweep = design_matrix(train, list(features))[disc]
        residual_ens = (y_train.astype(np.float64)
                        - np.asarray(inc_we_l3["oof"], dtype=np.float64))[disc]
        base_scores = entry_scores(
            impute_columns(design_matrix(train, list(bank_names))[disc]),
            list(bank_names), cond_sweep, residual_ens,
            folds=int(entry_cfg["residualize_folds"]),
            ridge_alpha=float(entry_cfg["ridge_alpha"]),
            seed=engine_seed + 777)
        primitives = [name for name, _ in sorted(
            base_scores.items(),
            key=lambda kv: (-abs(kv[1]), kv[0]))][:sweep_n]
        skipped_sweep: list[str] = []
        for spec in composition_sweep_specs(primitives):
            if spec.name in train.columns:
                skipped_sweep.append(spec.name)
                continue
            if not materialize_spec(spec=spec, train=train, apply=apply,
                                    label=label, time_values=time_values):
                skipped_sweep.append(spec.name)
                continue
            all_llm_specs[spec.name] = spec
            families.setdefault(spec.name, "sweep")
            sweep_candidates.append(spec.name)
        write_json(out_dir / "composition_sweep.json", {
            "n_primitives": sweep_n,
            "primitives": primitives,
            "primitive_scores": {name: round(float(value), 6)
                                 for name, value in base_scores.items()},
            "n_candidates": len(sweep_candidates),
            "candidates": sweep_candidates,
            "skipped": skipped_sweep,
        })
        log(f"composition sweep: {len(sweep_candidates)} candidates from "
            f"top-{sweep_n} primitives (lead: {primitives[:3]})")
    idiom_candidates: list[str] = []
    if not s4_depth3 and _lite("idiom_sweep") and bank_names:
        skipped_idiom: list[str] = []
        grid_entities = list(dict.fromkeys(
            [e["name"] for e in bank_audit.get("entities", [])] + entities))
        for spec in bounded_idiom_specs(
                list(bank_names), grid_entities, values, visible=
                [str(c) for c in train.columns], label=label,
                temporal=time_values is not None):
            if spec.name in train.columns:
                skipped_idiom.append(spec.name)
                continue
            if not materialize_spec(spec=spec, train=train, apply=apply,
                                    label=label, time_values=time_values):
                skipped_idiom.append(spec.name)
                continue
            all_llm_specs[spec.name] = spec
            families.setdefault(spec.name, "idiom")
            idiom_candidates.append(spec.name)
        write_json(out_dir / "idiom_sweep.json", {
            "n_primitives": len(bank_names),
            "n_candidates": len(idiom_candidates),
            "candidates": idiom_candidates, "skipped": skipped_idiom})
        log(f"idiom sweep: {len(idiom_candidates)} bounded-idiom candidates "
            f"from {len(bank_names)} primitives")
    pool_source_names = (list(bank_names) + injected_candidates
                         + sweep_candidates + idiom_candidates)
    template_channel = bool(_LITE.get("template_channel"))
    template_state = {"tried": set(), "scorecard_texts": [], "rounds": []}
    focus_state = {"enabled": True, "idiom_wins": {}, "last_admitted": [],
                   "cycle_new": [], "cycle_had_new": False}
    v34_router = bool(_LITE.get("v34_router"))
    v34_chain_on = True
    v34_selected_by_seat = None
    if v34_router:
        drift_rows = []
        import re as _re34
        _eng34 = _re34.compile(
            r"(_oof_te|_freq|_code|_log|_log1p|_minmax|_bin)$")
        card_columns = [
            str(c) for c in train.columns
            if str(c) != label
            and not str(c).startswith(("v9_pred_", "v13_", "vg_", "vf_",
                                       "sh34__", "s5_", "free"))
            and not _eng34.search(str(c))]
        for column in card_columns:
            series_i, series_v = train[column], apply[column]
            numeric_i = pd.to_numeric(series_i, errors="coerce")
            if float(numeric_i.notna().mean()) > 0.5:
                p_val = psi(numeric_i.to_numpy(np.float64),
                            pd.to_numeric(series_v, errors="coerce")
                            .to_numpy(np.float64))
                a_val = adversarial_auc(
                    numeric_i.to_numpy(np.float64),
                    pd.to_numeric(series_v, errors="coerce")
                    .to_numpy(np.float64))
                drifted = bool(p_val > 0.25 or abs(a_val - 0.5) > 0.15)
                drift_rows.append({"column": column, "psi": round(p_val, 4),
                                   "adv_auc": round(a_val, 4),
                                   "novelty": None, "drifted": drifted})
            else:
                seen = set(series_i.astype("string").dropna())
                vals = series_v.astype("string").dropna()
                novelty = (float((~vals.isin(seen)).mean())
                           if len(vals) else 0.0)
                drift_rows.append({"column": column, "psi": None,
                                   "adv_auc": None,
                                   "novelty": round(novelty, 4),
                                   "drifted": bool(novelty > 0.2)})
        drift_share = (float(np.mean([r["drifted"] for r in drift_rows]))
                       if drift_rows else 0.0)
        max_psi = max([r["psi"] for r in drift_rows
                       if r["psi"] is not None], default=0.0)
        d_hat = bool(drift_share > 0.2)
        h_ratio = 0.0
        if bank_names:
            residual_card = (y_train.astype(np.float64)
                             - np.asarray(inc_we_l3["oof"],
                                          dtype=np.float64))[disc]
            cond_card = design_matrix(train, list(features))[disc]
            card_scores = entry_scores(
                impute_columns(design_matrix(
                    train, list(bank_names))[disc]),
                list(bank_names), cond_card, residual_card,
                folds=int(entry_cfg["residualize_folds"]),
                ridge_alpha=float(entry_cfg["ridge_alpha"]),
                seed=engine_seed + 3434)
            floor_card = entry_noise_floor(
                int(y_train[disc].sum()), float(entry_cfg["alpha"]))
            if card_scores and floor_card > 0:
                h_ratio = max(abs(v) for v in card_scores.values()) \
                    / floor_card
        h_hat = bool(h_ratio >= 2.0)
        v34_chain_on = bool(d_hat or h_ratio >= 1.0
                            or time_values is not None)
        write_json(out_dir / "personality_card.json", {
            "drift": {"share": round(drift_share, 4),
                      "max_psi": round(float(max_psi), 4),
                      "columns": drift_rows},
            "headroom": {"ratio": round(float(h_ratio), 4)},
            "bits": {"D": d_hat, "H": h_hat},
            "routing": {"bundle_lane": ("off_drift" if d_hat else "on"),
                        "chain": ("on" if v34_chain_on
                                  else "off_saturated")},
        })
        log(f"V34 card: drift_share={drift_share:.2f} "
            f"max_psi={max_psi:.2f} D={int(d_hat)} | "
            f"headroom_ratio={h_ratio:.2f} H={int(h_hat)} -> "
            f"bundle={'OFF' if d_hat else 'on'} "
            f"chain={'on' if v34_chain_on else 'OFF'}")
        if d_hat and gated_bundle:
            gated_bundle = []
    if gated_bundle:
        bundle_started = time.time()
        bundle = [name for name in gated_bundle if name in train.columns]
        bundle, station_audit = screen_bundle(
            factors=bundle, train=train, apply=apply)
        write_json(out_dir / "stationarity_screen.json", station_audit)
        log(f"stationarity screen: kept {station_audit['n_kept']}"
            f"/{station_audit['n_in']} bundle factors "
            f"({station_audit['n_dropped']} dropped as drifting)")
        if (v34_router and bundle
                and os.environ.get("FORGE_ABL_NO_SHADOW", "") != "1"):
            rng34 = np.random.default_rng(engine_seed + 777)
            shadow_map = {}
            for c in bundle:
                sh_name = f"sh34__{c}"[:60]
                train[sh_name] = rng34.permutation(pd.to_numeric(
                    train[c], errors="coerce").fillna(0.0).to_numpy())
                apply[sh_name] = rng34.permutation(pd.to_numeric(
                    apply[c], errors="coerce").fillna(0.0).to_numpy())
                shadow_map[c] = sh_name
            probe_all = list(bundle) + list(shadow_map.values())
            phase_a_surfaces = {
                f"l1_{key}": list(dict.fromkeys(
                    base_cols(key) + list(surfaces[key]) + probe_all))
                for key, _, _ in l1_library}
            phase_a_probes = {f"l1_{key}": list(probe_all)
                              for key, _, _ in l1_library}
            phase_a_records = fit_layer(
                train=train, apply=apply, label=label, library=l1_library,
                surfaces=phase_a_surfaces, prefix="l1", layer_index=1,
                cfg=cfg, fold_log=fold_log,
                stage="search_bundle_shadowsel_l1",
                probe_cols=phase_a_probes, time_limit_audit=[])
            n_rows34 = len(train)
            halves34 = [np.arange(0, n_rows34 // 2),
                        np.arange(n_rows34 // 2, n_rows34)]
            y34 = y_train.astype(np.int8)

            def _pi34(record, column, rows):
                vec = (record.get("probe_oof") or {}).get(column)
                if vec is None or int(y34[rows].sum()) == 0:
                    return None
                oof34 = np.asarray(record["oof"], dtype=np.float64)
                return float(
                    average_precision_score(y34[rows], oof34[rows])
                    - average_precision_score(
                        y34[rows],
                        np.asarray(vec, dtype=np.float64)[rows]))

            v34_selected_by_seat = {}
            shadow_audit = {}
            for record in phase_a_records:
                seat_key = record["model_key"]
                theta = []
                for rows in halves34:
                    vals = [_pi34(record, sh, rows)
                            for sh in shadow_map.values()]
                    vals = [v for v in vals if v is not None]
                    theta.append(max(vals) if vals else 0.0)
                kept_c, table = [], []
                for c in bundle:
                    pis = [_pi34(record, c, rows) for rows in halves34]
                    ok = all(v is not None and v > theta[i]
                             for i, v in enumerate(pis))
                    table.append({
                        "factor": c,
                        "pi_halves": [None if v is None else round(v, 6)
                                      for v in pis],
                        "kept": bool(ok)})
                    if ok:
                        kept_c.append(c)
                v34_selected_by_seat[seat_key] = kept_c
                shadow_audit[seat_key] = {
                    "theta_halves": [round(t, 6) for t in theta],
                    "kept": kept_c, "table": table}
            train.drop(columns=list(shadow_map.values()), inplace=True,
                       errors="ignore")
            apply.drop(columns=list(shadow_map.values()), inplace=True,
                       errors="ignore")
            bundle = sorted({c for kept in v34_selected_by_seat.values()
                             for c in kept})
            write_json(out_dir / "shadow_selection.json", {
                "rule": ("keep for seat iff honest permutation importance "
                         "> max shadow importance on BOTH chrono halves"),
                "by_seat": shadow_audit, "union": bundle})
            log("V34 shadow selection: union "
                f"{len(bundle)} factors; per-seat "
                + json.dumps({k: len(v) for k, v
                              in v34_selected_by_seat.items()},
                             sort_keys=True))
    if gated_bundle and not bundle:
        bundle_row = {
            "round": 0,
            "kind": "free_bundle_all_dropped_by_stationarity",
            "bundle_size": 0,
            "bundle": [],
            "stationarity": station_audit,
            "time_limit_dropped": [],
            "per_model": {key: {"k_add": 0, "k_drop": 0, "adds": []}
                          for key, _, _ in l1_library},
            "gate": None,
            "accepted": False,
            "wall_sec": round(time.time() - bundle_started, 3),
        }
        ledger.append(bundle_row)
        write_json(out_dir / "rounds" / "round_00_bundle.json", bundle_row)
        log("bundle pre-round skipped: stationarity screen dropped every "
            "factor; gate block budget untouched")
    if gated_bundle and bundle:
        cand_surfaces = {
            key: list(dict.fromkeys(
                list(surfaces[key])
                + (v34_selected_by_seat.get(key, [])
                   if v34_selected_by_seat is not None else bundle)))
            for key, _, _ in l1_library}
        bundle_l1_surfaces = {
            f"l1_{key}": list(dict.fromkeys(base_cols(key)
                                            + cand_surfaces[key]))
            for key, _, _ in l1_library}
        bundle_probe_cols = {f"l1_{key}": list(cand_surfaces[key])
                             for key, _, _ in l1_library}
        bundle_drops: list = []
        bundle_records = fit_layer(
            train=train, apply=apply, label=label, library=l1_library,
            surfaces=bundle_l1_surfaces, prefix="l1", layer_index=1,
            cfg=cfg, fold_log=fold_log, stage="search_bundle_l1",
            probe_cols=bundle_probe_cols, time_limit_audit=bundle_drops)
        bundle_we = _weighted_ensemble(bundle_records, y_train, steps)
        bundle_pred = np.asarray(bundle_we["prediction"], dtype=np.float64)
        gate_block = blocks[0]
        gate_result, gate_audit = round_gate(
            y_apply=y_apply, block=gate_block,
            incumbent_pred=inc_we_l3_pred, candidate_pred=bundle_pred,
            cfg=cfg, round_index=1, max_rounds=max_rounds)
        gate_blocks_consumed += 1
        gate_audit["round"] = 0
        harm_k = float(cfg["experiment"]["gate"].get("harm_k", 2.0))
        seat_threshold = -harm_k * float(gate_result.bootstrap_se)
        half_point = len(gate_block) // 2
        gate_halves = [gate_block[:half_point], gate_block[half_point:]]
        half_results = []
        for h_index, h_block in enumerate(gate_halves):
            h_result, _ = round_gate(
                y_apply=y_apply, block=h_block,
                incumbent_pred=inc_we_l3_pred, candidate_pred=bundle_pred,
                cfg=cfg, round_index=2 + h_index, max_rounds=max_rounds)
            half_results.append(h_result)
        half_thresholds = [-harm_k * float(r.bootstrap_se)
                           for r in half_results]
        y_blk = np.asarray(y_apply)[gate_block]
        inc_ap = float(average_precision_score(
            y_blk, inc_we_l3_pred[gate_block]))
        inc_ap_halves = [
            float(average_precision_score(
                np.asarray(y_apply)[h], inc_we_l3_pred[h]))
            for h in gate_halves]
        inc_by_key = {r["model_key"]: r for r in inc_l1}
        cand_by_key = {r["model_key"]: r for r in bundle_records}
        order = [r["model_key"] for r in bundle_records]
        seat_deltas: dict[str, float] = {}
        seat_half_deltas: dict[str, list[float]] = {}
        for key in order:
            if key not in inc_by_key:
                continue
            mixed = [cand_by_key[k] if k == key
                     else inc_by_key.get(k, cand_by_key[k]) for k in order]
            mixed_we = _weighted_ensemble(mixed, y_train, steps)
            mixed_pred = np.asarray(mixed_we["prediction"],
                                    dtype=np.float64)
            seat_deltas[key] = float(average_precision_score(
                y_blk, mixed_pred[gate_block]) - inc_ap)
            seat_half_deltas[key] = [
                float(average_precision_score(
                    np.asarray(y_apply)[h], mixed_pred[h])
                    - inc_ap_halves[i])
                for i, h in enumerate(gate_halves)]
        kept_keys = {
            key for key, halves in seat_half_deltas.items()
            if all(delta >= threshold for delta, threshold
                   in zip(halves, half_thresholds))}
        dropped_keys = set(seat_half_deltas) - kept_keys
        h1 = np.array([seat_half_deltas[k][0] for k in sorted(seat_half_deltas)])
        h2 = np.array([seat_half_deltas[k][1] for k in sorted(seat_half_deltas)])
        sign_agreement = (float(np.mean(np.sign(h1) == np.sign(h2)))
                          if len(h1) else None)
        delta_corr = (float(np.corrcoef(h1, h2)[0, 1])
                      if len(h1) > 2 and h1.std() > 0 and h2.std() > 0
                      else None)
        if kept_keys:
            inc_l1 = [cand_by_key[k] if k in kept_keys
                      else inc_by_key.get(k, cand_by_key[k]) for k in order]
            inc_we_l2 = _weighted_ensemble(inc_l1, y_train, steps)
            inc_we_l3 = inc_we_l2
            inc_we_l3_pred = np.asarray(inc_we_l3["prediction"],
                                        dtype=np.float64)
            for key in kept_keys:
                surfaces[key] = list(cand_surfaces[key])
            search_fitted |= set(kept_keys)
            accepted_rounds += 1
        gate_audit["v13_1_accept_rule"] = \
            "bundle_per_seat_harm_only_two_half_persistence"
        gate_audit["harm_threshold"] = seat_threshold
        gate_audit["lcb_veto"] = "audit_only_v13_1"
        gate_audit["seat_scope"] = {
            "rule": "per_seat_harm_only_on_both_chrono_halves",
            "harm_k": harm_k,
            "full_block_threshold": seat_threshold,
            "half_thresholds": [round(t, 7) for t in half_thresholds],
            "half_rows": [int(len(h)) for h in gate_halves],
            "half_positives": [int(np.asarray(y_apply)[h].sum())
                               for h in gate_halves],
            "incumbent_block_auprc": inc_ap,
            "seat_deltas": {k: round(v, 7)
                            for k, v in sorted(seat_deltas.items())},
            "seat_half_deltas": {
                k: [round(v, 7) for v in halves]
                for k, halves in sorted(seat_half_deltas.items())},
            "consistency": {"sign_agreement": sign_agreement,
                            "delta_corr": delta_corr},
            "kept": sorted(kept_keys),
            "dropped": sorted(dropped_keys),
        }
        gate_audit["v13_1_accepted"] = bool(kept_keys)
        bundle_row = {
            "round": 0,
            "kind": "free_bundle_pre_round_v32",
            "bundle_size": len(bundle),
            "bundle": list(bundle),
            "stationarity": station_audit,
            "time_limit_dropped": bundle_drops,
            "per_model": {
                key: {"k_add": (len(bundle) if key in kept_keys else 0),
                      "k_drop": 0,
                      "adds": (list(bundle) if key in kept_keys else []),
                      "bundle_seat_delta": round(
                          seat_deltas.get(key, 0.0), 7)}
                for key, _, _ in l1_library},
            "gate": gate_audit,
            "accepted": bool(kept_keys),
            "wall_sec": round(time.time() - bundle_started, 3),
        }
        ledger.append(bundle_row)
        write_json(out_dir / "rounds" / "round_00_bundle.json", bundle_row)
        save_records(out_dir / "search" / "round_00_bundle" / "l1",
                     bundle_records)
        log(f"bundle pre-round (v32): {len(bundle)} factors, "
            f"full delta={gate_result.delta:+.6f} "
            f"halves={[round(float(r.delta), 5) for r in half_results]} "
            f"kept={sorted(kept_keys)} dropped={sorted(dropped_keys)}")
    for round_index in range(1, max_rounds + 1):
        if v34_router and not v34_chain_on:
            stop_reason = "v34_routed_no_chain"
            break
        if _LITE.get("v6") and gate_blocks_consumed >= len(blocks):
            stop_reason = "gate_budget_exhausted"
            break
        round_started = time.time()
        cycle_inc_l1 = list(inc_l1)
        cycle_inc_we_l2 = inc_we_l2
        cycle_inc_we_l3 = inc_we_l3
        by_key = {r["model_key"]: r for r in cycle_inc_l1}
        disagreement = np.std(
            np.column_stack([r["oof"] for r in cycle_inc_l1]), axis=1)
        s4_survivors: dict[int, list[str]] = {0: list(bank_names)}
        s4_round_audit: list[dict] = []
        visible = [str(column) for column in train.columns
                   if not str(column).startswith("v9_pred_")]
        new_surfaces = {key: list(cols) for key, cols in surfaces.items()}
        staged_blacklist: list[tuple[str, list[str], dict]] = []
        per_model: dict[str, dict] = {}
        total_added = 0
        total_dropped = 0
        shared_specs = None
        round_monitor = None
        proposal_cycles = effective_proposal_cycles
        cycle_ledger: list[dict] = []
        for cycle_index in range(1, proposal_cycles + 1):
            cycle_added = 0
            shared_specs = None
            s4_cycle_candidates: list[str] = (
                list(bank_names) if s4_depth3 and cycle_index == 1 else [])
            if s4_depth3:
                by_key = {r["model_key"]: r for r in cycle_inc_l1}
                disagreement = np.std(np.column_stack(
                    [r["oof"] for r in cycle_inc_l1]), axis=1)
            base_surfaces = (surfaces if cycle_index == 1
                             else {key: list(value)
                                   for key, value in new_surfaces.items()})
            if s4_depth3 and s4_eval_mode == "chrono":
                block_start = int(np.min(s4_eval_rows[cycle_index - 1]))
                evidence_rows = disc[disc < block_start]
            elif s4_depth3:
                evidence_rows = np.setdiff1d(
                    disc, s4_eval_rows[cycle_index - 1], assume_unique=False)
            else:
                evidence_rows = disc
            rich_blacklist = bool(_LITE.get("v9"))
            blacklist_lines = []
            for model_key, names in blacklist.entries.items():
                for name, info in names.items():
                    detail = (f"- model {model_key}: {name} permanently blocked "
                              f"after round {info.get('round')}")
                    if rich_blacklist:
                        pi = info.get("pi")
                        expr = (all_llm_specs[name].expression
                                if name in all_llm_specs else None)
                        if isinstance(pi, (int, float)):
                            detail += (f"; measured permutation importance "
                                       f"{float(pi):+.5f} (<= 0 means it carried "
                                       f"no usable signal for this model)")
                        if expr:
                            detail += f"; expression was: {expr}"
                    blacklist_lines.append(detail)
            for rejected in rejected_hypotheses[-40:]:
                detail = (f"- round {rejected['round']} {rejected['name']} "
                          f"(family={rejected.get('family', 'unknown')}): "
                          f"{rejected['reason']}")
                if rich_blacklist and rejected.get("expression"):
                    detail += f"; expression was: {rejected['expression']}"
                blacklist_lines.append(detail)
            blacklist_context = ""
            if blacklist_lines:
                blacklist_context = (
                    ("Permanent failed-hypothesis blacklist, with the measured "
                     "evidence that rejected each one."
                     if rich_blacklist else
                     "V6 permanent failed-hypothesis blacklist.")
                    + " Do not rename or algebraically restate these factors; "
                    "propose a materially different business mechanism "
                    "instead:\n" + "\n".join(blacklist_lines))
            templates_this_cycle: list[dict] = []
            template_expand_audit: dict = {}
            template_seat_rows: list[dict] = []
            template_cycle_audit: dict | None = None
            if template_channel:
                started_tmpl = time.time()
                cond_tmpl = design_matrix(train, list(features))[evidence_rows]
                residual_ens = _scoring_residual(
                    y_train, cycle_inc_we_l3["oof"])[evidence_rows]
                ens_scores: dict[str, float] = {}
                if bank_names:
                    ens_scores = entry_scores(
                        impute_columns(
                            design_matrix(train, list(bank_names))[evidence_rows]),
                        list(bank_names), cond_tmpl, residual_ens,
                        folds=int(entry_cfg["residualize_folds"]),
                        ridge_alpha=float(entry_cfg["ridge_alpha"]),
                        seed=(engine_seed + 777 + 1000 * round_index
                              + cycle_index))
                evidence_tmpl = top_s_evidence(ens_scores, families, floor,
                                               top=10)
                seat_sections: list[str] = []
                seat_section_keys: list[str] = []
                seat_eda_audit: list[dict] = []
                perceive_v2 = bool(_LITE.get("perceive_v2"))
                seat_weights = {
                    str(k).replace("l1_", "", 1): float(v)
                    for k, v in (cycle_inc_we_l2.get("weights") or {}).items()}
                weight_floor = (1.0 / max(1, len(cycle_inc_l1))
                                if perceive_v2 else 0.0)
                triage_skipped: list[dict] = []
                for record in cycle_inc_l1:
                    skey = str(record["model_key"])
                    oof_k = np.asarray(record["oof"], dtype=np.float64)
                    residual_k = _scoring_residual(
                        y_train, oof_k)[evidence_rows]
                    current_k = list(base_surfaces.get(skey, []))
                    conditioning_k = design_matrix(train, list(
                        dict.fromkeys(base_cols(skey)
                                      + current_k)))[evidence_rows]
                    pool_k = blacklist.filter(skey, [
                        name for name in bank_names
                        if name not in current_k])
                    scores_k: dict[str, float] = {}
                    if pool_k:
                        scores_k = entry_scores(
                            impute_columns(
                                design_matrix(train, pool_k)[evidence_rows]),
                            pool_k, conditioning_k, residual_k,
                            folds=int(entry_cfg["residualize_folds"]),
                            ridge_alpha=float(entry_cfg["ridge_alpha"]),
                            seed=(engine_seed + 777 + 1000 * round_index
                                  + cycle_index))
                    top_k = sorted(scores_k.items(),
                                   key=lambda kv: (-abs(kv[1]), kv[0]))[:5]
                    auprc_k = float(average_precision_score(y_train, oof_k))
                    weight_k = seat_weights.get(skey, 0.0)
                    headroom_k = max((abs(v) for v in scores_k.values()),
                                     default=0.0)
                    if (perceive_v2 and weight_k < weight_floor
                            and headroom_k < floor):
                        triage_skipped.append({
                            "seat": skey, "weight": round(weight_k, 4),
                            "max_abs_s": round(float(headroom_k), 6),
                            "reason": "zero_ensemble_weight_and_no_bank_headroom"})
                        continue
                    gaps_k, gap_text = seat_posneg_gap(
                        train, label=label, oof=_scoring_oof(y_train, oof_k),
                        rows=evidence_rows,
                        features=list(dict.fromkeys(
                            list(features) + list(bank_names))), top=4)
                    if perceive_v2:
                        base_k = base_cols(skey)
                        has_te = any(str(c).endswith(("_oof_te", "_freq"))
                                     for c in base_k)
                        has_code = any(str(c).endswith("_code") for c in base_k)
                        head = (f"- seat {skey} [{learner_of.get(skey, '?')}, "
                                f"rep={forge_seat_map.get(skey, 'shared')}, "
                                f"{len(base_k)} base cols, "
                                f"target-encodings {'available' if has_te else 'ABSENT'}, "
                                f"integer codes {'available' if has_code else 'ABSENT'}, "
                                f"ensemble weight {weight_k:.2f}]: "
                                f"OOF AUPRC {auprc_k:.4f}; ")
                    else:
                        head = f"- seat {skey}: OOF AUPRC {auprc_k:.4f}; "
                    section = (head + "top unexplained bank signal: "
                               + (", ".join(f"{name}={score:+.4f}"
                                            for name, score in top_k)
                                  if top_k else "none"))
                    if gap_text:
                        section += "; " + gap_text
                    seat_sections.append(section)
                    seat_section_keys.append(skey)
                    seat_eda_audit.append({
                        "seat": skey, "oof_auprc": round(auprc_k, 6),
                        "ensemble_weight": round(weight_k, 4),
                        "top_bank_s": [
                            {"name": name, "s": round(float(score), 6)}
                            for name, score in top_k],
                        "posneg_gap": gaps_k})
                if perceive_v2 and len(seat_sections) < 2 and triage_skipped:
                    for entry in sorted(triage_skipped,
                                        key=lambda e: -e["weight"]):
                        rec = next(
                            (r for r in cycle_inc_l1
                             if str(r["model_key"]) == entry["seat"]),
                            None)
                        if rec is None:
                            continue
                        seat_sections.append(
                            f"- seat {entry['seat']}: OOF AUPRC "
                            f"{float(average_precision_score(y_train, np.asarray(rec['oof'], dtype=np.float64))):.4f}; "
                            "top unexplained bank signal: none")
                        seat_section_keys.append(entry["seat"])
                        entry["reason"] = "restored_min_two_seats"
                        if len(seat_sections) >= 2:
                            break
                vocabulary_summary = None
                if perceive_v2:
                    generated_prefixes = ("vg_", "vg3_", "vg4_", "vf_", "v21_")
                    raw_cols = [c for c in visible
                                if not str(c).startswith(generated_prefixes)
                                and not str(c).startswith("v13_")]
                    on_surface = {
                        k: list(base_surfaces.get(k, []))
                        for k, _, _ in l1_library}
                    n_grid = sum(1 for c in pool_source_names
                                 if str(c).startswith(("vg_", "vg3_", "vg4_")))
                    n_focus = sum(1 for c in pool_source_names
                                  if str(c).startswith("vf_"))
                    n_tmpl = sum(1 for c in pool_source_names
                                 if str(c).startswith("v21_"))
                    lines = [
                        "Slot vocabulary, by layer (use these names in slots):",
                        f"- base columns ({len(raw_cols)}): "
                        + ", ".join(map(str, raw_cols)),
                        f"- factor bank ({len(bank_names)}), strictly-past "
                        "entity statistics: " + ", ".join(map(str, bank_names)),
                    ]
                    if s4_depth3:
                        previous_names = s4_survivors.get(
                            max(cycle_index - 1, 0), [])
                        lines += [
                            f"- previous-depth survivors ({len(previous_names)}): "
                            + ", ".join(previous_names),
                            "- S4 does NOT pre-enumerate a global binary/ternary/"
                            "quaternary grid. Obey the depth contract and use "
                            "survivors as the beam; the machine will expand only "
                            "your small slot vocabularies.",
                        ]
                    else:
                        lines += [
                            "- ALREADY ENUMERATED by the deterministic engine, do "
                            "NOT restate these as templates: bounded-idiom "
                            "compositions vg_{bpos|babs|mulBB|mulAA|ratB|difB|"
                            "rawmulB} over every factor-bank pair (plus vg3_/vg4_ "
                            f"ternary and quaternary forms) -- {n_grid} live; "
                            f"focus cross-compositions vf_* -- {n_focus} live; "
                            f"earlier templates' expansions v21_* -- {n_tmpl} live.",
                            "  Propose mechanisms these compositions cannot reach "
                            "(new entity keys, new windows, new aggregations), not "
                            "algebra over columns already in the pool.",
                        ]
                    surf_bits = [f"{k}: {len(v)}" for k, v in on_surface.items()
                                 if v]
                    if surf_bits:
                        lines.append("- factors already admitted per seat: "
                                     + ", ".join(surf_bits))
                    vocabulary_summary = "\n".join(lines)
                scorecard_recent = "\n\n".join(
                    template_state["scorecard_texts"][-3:])
                depth_contract = None
                if s4_depth3:
                    previous = s4_survivors.get(cycle_index - 1, [])
                    order_one = list(dict.fromkeys(
                        s4_survivors.get(1, []) + list(bank_names)))
                    if cycle_index == 1:
                        depth_contract = (
                            "Depth 1 only. Pointwise templates must have exactly "
                            "ONE slot and use factor-bank columns. Aggregate "
                            "templates may propose strictly-past rolling/history "
                            "features. Do not propose any interaction, product, "
                            "ratio, or difference involving two columns.")
                    elif cycle_index == 2:
                        depth_contract = (
                            "Depth 2 only. Pointwise templates must have exactly "
                            "TWO slots and every binding must use at least one "
                            "depth-1 survivor. Depth-1 survivors: "
                            + ", ".join(previous[:24])
                            + ". Aggregate templates may roll only these "
                              "survivors. Do not propose ternary forms.")
                    else:
                        depth_contract = (
                            "Depth 3 only. Compose exactly one depth-2 survivor "
                            "with order-1 parents; never compose depth-2 with "
                            "depth-2. Depth-2 survivors: "
                            + ", ".join(previous[:24])
                            + ". Order-1 parents: " + ", ".join(order_one[:24])
                            + ". Aggregate templates may roll depth-2 survivors. "
                              "No quaternary form is allowed.")
                template_count = int(_lite("template_count", 4))
                scope = str(_lite("template_scope", "shared"))
                if scope == "seat" and seat_sections:
                    per_seat_audits: list[dict] = []
                    fetched: list[tuple[str, list[dict], dict]] = []
                    workers = min(int(_lite("parallel_llm", 8) or 8),
                                  max(len(seat_sections), 1))
                    with ThreadPoolExecutor(max_workers=workers) as pool_t:
                        futures = {}
                        for skey, section in zip(seat_section_keys,
                                                 seat_sections):
                            futures[pool_t.submit(
                                propose_templates, llm,
                                dataset_description=dataset_description,
                                label=label, visible=visible,
                                entities=entities, values=values,
                                temporal=time_values is not None,
                                evidence=evidence_tmpl,
                                blacklist_context=blacklist_context,
                                scorecard_text=scorecard_recent,
                                max_templates=template_count,
                                tried=set(template_state["tried"]),
                                seat_sections=[section],
                                vocabulary_summary=vocabulary_summary,
                                depth_contract=depth_contract,
                                call_kind=f"v21_templates_{skey}",
                                source=(f"v21_templates:{skey}:"
                                        f"r{round_index}c{cycle_index}"))]\
                                = skey
                        for future in as_completed(futures):
                            skey = futures[future]
                            try:
                                seat_templates, seat_audit = future.result()
                            except Exception as exc:
                                log(f"round {round_index} cycle "
                                    f"{cycle_index} {skey}: template call "
                                    f"failed ({type(exc).__name__}: {exc})")
                                continue
                            fetched.append((skey, seat_templates, seat_audit))
                    templates_this_cycle = []
                    seen_names: set[str] = set()
                    for skey, seat_templates, seat_audit in sorted(
                            fetched, key=lambda item: item[0]):
                        per_seat_audits.append(
                            {"seat": skey, **seat_audit})
                        for template in seat_templates:
                            if template["name"] in seen_names:
                                continue
                            seen_names.add(template["name"])
                            if not template.get("target_seats"):
                                template["target_seats"] = [skey]
                            templates_this_cycle.append(template)
                    template_state["tried"].update(seen_names)
                    propose_audit = {"scope": "seat",
                                     "n_templates": len(templates_this_cycle),
                                     "per_seat": per_seat_audits}
                else:
                    templates_this_cycle, propose_audit = propose_templates(
                        llm, dataset_description=dataset_description,
                        label=label, visible=visible, entities=entities,
                        values=values, temporal=time_values is not None,
                        evidence=evidence_tmpl,
                        blacklist_context=blacklist_context,
                        scorecard_text=scorecard_recent,
                        max_templates=template_count,
                        tried=template_state["tried"],
                        seat_sections=seat_sections,
                        vocabulary_summary=vocabulary_summary,
                        depth_contract=depth_contract,
                        source=f"v21_templates:r{round_index}c{cycle_index}")
                    propose_audit["scope"] = "shared"
                if s4_depth3:
                    previous_set = set(s4_survivors.get(
                        cycle_index - 1, []))
                    order_one_set = set(s4_survivors.get(1, [])) \
                        | set(bank_names)
                    allowed = (set(bank_names) if cycle_index == 1 else
                               previous_set | (set(bank_names)
                                               if cycle_index == 2
                                               else order_one_set))
                    expected_slots = 1 if cycle_index == 1 else 2
                    depth_kept: list[dict] = []
                    depth_rejected: list[dict] = []
                    for template in templates_this_cycle:
                        candidate = dict(template)
                        if candidate.get("kind") == "pointwise":
                            slots = {
                                token: [value for value in vocab
                                        if value in allowed]
                                for token, vocab in
                                (candidate.get("slots") or {}).items()}
                            if (len(slots) != expected_slots
                                    or any(not vocab
                                           for vocab in slots.values())):
                                depth_rejected.append({
                                    "template": candidate.get("name"),
                                    "reason": "pointwise_slot_arity_or_vocabulary",
                                })
                                continue
                            union_vocab = set().union(*map(set, slots.values()))
                            if cycle_index > 1 and not (
                                    union_vocab & previous_set):
                                depth_rejected.append({
                                    "template": candidate.get("name"),
                                    "reason": "missing_previous_depth_parent",
                                })
                                continue
                            candidate["slots"] = slots
                        elif cycle_index > 1:
                            if candidate.get("op") not in {
                                    "rolling_stat", "lag", "expanding_stat"}:
                                depth_rejected.append({
                                    "template": candidate.get("name"),
                                    "reason": "later_depth_aggregate_not_rolling",
                                })
                                continue
                            candidate["value_slots"] = [
                                value for value in
                                candidate.get("value_slots", [])
                                if value in previous_set]
                            if not candidate["value_slots"]:
                                depth_rejected.append({
                                    "template": candidate.get("name"),
                                    "reason": "rolling_value_not_previous_survivor",
                                })
                                continue
                        depth_kept.append(candidate)
                    templates_this_cycle = depth_kept
                    propose_audit["s4_depth_filter"] = {
                        "depth": cycle_index,
                        "expected_direct_slots": expected_slots,
                        "kept": [t.get("name") for t in depth_kept],
                        "rejected": depth_rejected,
                    }
                specs_tmpl, template_expand_audit = expand_templates(
                    templates_this_cycle, visible=visible, label=label,
                    temporal=time_values is not None,
                    existing=(set(map(str, train.columns))
                              | set(all_llm_specs)),
                    cap_per_template=int(_lite("template_per_cap", 60)),
                    cap_total=(min(
                        int(_lite("template_cap", 240)),
                        max(1, s4_candidate_caps[cycle_index - 1] // 2))
                        if s4_depth3 else
                        int(_lite("template_cap", 240))))
                if s4_depth3:
                    previous_set = set(s4_survivors.get(
                        cycle_index - 1, []))
                    order_one_set = (set(s4_survivors.get(1, []))
                                     | set(bank_names))
                    known_factor_names = set(s4_lineage) | set(all_llm_specs)
                    filtered_specs: list[FeatureSpec] = []
                    for spec in specs_tmpl:
                        refs = set(_s4_spec_references(
                            spec, known_factor_names))
                        valid = True
                        if spec.executor == "pointwise":
                            if cycle_index == 1:
                                valid = len(refs & set(bank_names)) == 1
                            elif cycle_index == 2:
                                valid = (len(refs) == 2
                                         and bool(refs & previous_set))
                            else:
                                valid = (len(refs) == 2
                                         and bool(refs & previous_set)
                                         and bool(refs & order_one_set))
                        if valid:
                            filtered_specs.append(spec)
                    specs_tmpl = filtered_specs
                materialized_tmpl: list[str] = []
                for spec in specs_tmpl:
                    if not materialize_spec(spec=spec, train=train,
                                            apply=apply, label=label,
                                            time_values=time_values):
                        continue
                    all_llm_specs[spec.name] = spec
                    families.setdefault(spec.name, spec.family)
                    pool_source_names.append(spec.name)
                    materialized_tmpl.append(spec.name)
                    if s4_depth3:
                        s4_cycle_candidates.append(spec.name)
                        parents = _s4_spec_references(
                            spec, set(s4_lineage) | set(all_llm_specs))
                        s4_lineage[spec.name] = {
                            "depth": int(cycle_index),
                            "parents": parents,
                            "kind": ("strict_past_aggregate"
                                     if spec.executor == "agg"
                                     else "llm_exact_form"),
                            "source": "llm_template",
                        }
                llm_phase_wall_sec += time.time() - started_tmpl
                template_cycle_audit = {
                    "round": int(round_index), "cycle": int(cycle_index),
                    "propose": propose_audit,
                    "seat_evidence": seat_eda_audit,
                    "expand": template_expand_audit,
                    "n_materialized": len(materialized_tmpl),
                    "materialized": materialized_tmpl}
                if perceive_v2:
                    template_cycle_audit["perceive_v2"] = {
                        "seats_carried": list(seat_section_keys),
                        "seats_skipped": triage_skipped,
                        "weight_floor": round(weight_floor, 6),
                        "entry_floor": round(float(floor), 6),
                        "vocabulary_summary_chars": len(vocabulary_summary or ""),
                    }
                log(f"round {round_index} cycle {cycle_index}: template "
                    f"sweep {len(templates_this_cycle)} template(s) -> "
                    f"{len(materialized_tmpl)} candidates materialized")
                if s4_depth3:
                    grid_entities = list(dict.fromkeys(
                        [e["name"] for e in bank_audit.get("entities", [])]
                        + entities))
                    depth_specs, depth_lineage = s4_depth_grid_specs(
                        depth=cycle_index, bank=list(bank_names),
                        previous_survivors=s4_survivors.get(
                            cycle_index - 1, []),
                        order_one_parents=list(dict.fromkeys(
                            s4_survivors.get(1, []) + list(bank_names))),
                        entities=grid_entities, values=values,
                        visible=[str(c) for c in train.columns], label=label,
                        temporal=time_values is not None)
                    cap = int(s4_candidate_caps[cycle_index - 1])
                    skipped_by_cap = 0
                    materialized_grid: list[str] = []
                    for spec in depth_specs:
                        if len(dict.fromkeys(s4_cycle_candidates)) >= cap:
                            skipped_by_cap += 1
                            continue
                        if spec.name in train.columns:
                            continue
                        if not materialize_spec(
                                spec=spec, train=train, apply=apply,
                                label=label, time_values=time_values):
                            continue
                        all_llm_specs[spec.name] = spec
                        families.setdefault(spec.name, spec.family)
                        pool_source_names.append(spec.name)
                        s4_cycle_candidates.append(spec.name)
                        s4_lineage[spec.name] = depth_lineage[spec.name]
                        materialized_grid.append(spec.name)
                    s4_cycle_candidates = list(dict.fromkeys(
                        s4_cycle_candidates))[:cap]
                    template_cycle_audit["s4_depth_generation"] = {
                        "depth": int(cycle_index),
                        "candidate_cap": cap,
                        "raw_bank_candidates": (len(bank_names)
                                                if cycle_index == 1 else 0),
                        "template_candidates": len(materialized_tmpl),
                        "grid_candidates": len(materialized_grid),
                        "n_candidates": len(s4_cycle_candidates),
                        "skipped_by_cap": int(skipped_by_cap),
                        "lineage": {
                            name: s4_lineage.get(name, {})
                            for name in s4_cycle_candidates},
                    }
                n_focus_total = sum(1 for n in pool_source_names
                                    if n.startswith("vf_"))
                if (not s4_depth3 and _LITE.get("focus_grid")
                        and focus_state["enabled"]
                        and (materialized_tmpl or cycle_index > 1)
                        and n_focus_total < int(_lite("focus_cap", 600))):
                    if _LITE.get("focus_beam"):
                        focus_prims = list(dict.fromkeys(
                            list(focus_state["last_admitted"])[:4]
                            + list(materialized_tmpl)[:4]
                            + list(bank_names)[:4]))
                    else:
                        survivors = [n for n in dict.fromkeys(
                            name for cols in base_surfaces.values()
                            for name in cols) if n in all_llm_specs]
                        focus_prims = list(dict.fromkeys(
                            survivors[-6:] + list(materialized_tmpl)[:6]
                            + list(bank_names)[:6]))
                    n_focus = 0
                    focus_specs = bounded_idiom_specs(
                        focus_prims, [], [], visible=[], label=label,
                        temporal=False, order=2, name_prefix="vf")
                    if _LITE.get("focus_beam") and focus_state["idiom_wins"]:
                        winners = set(focus_state["idiom_wins"])
                        focus_specs = [
                            s for s in focus_specs
                            if s.name.split("__")[0][3:] in winners
                            or s.name.startswith(("vf_bpos", "vf_babs"))]
                    focus_state["cycle_new"] = []
                    for spec in focus_specs:
                        if spec.name in train.columns:
                            continue
                        if not materialize_spec(
                                spec=spec, train=train, apply=apply,
                                label=label, time_values=time_values):
                            continue
                        all_llm_specs[spec.name] = spec
                        families.setdefault(spec.name, "focus_idiom")
                        pool_source_names.append(spec.name)
                        focus_state["cycle_new"].append(spec.name)
                        n_focus += 1
                    focus_state["cycle_had_new"] = n_focus > 0
                    if template_cycle_audit is not None:
                        template_cycle_audit["n_focus_grid"] = n_focus
                    log(f"round {round_index} cycle {cycle_index}: focus "
                        f"grid +{n_focus} cross-composition candidates")
            s4_selected_by_seat: dict[str, list[str]] = {}
            s4_scores_by_seat: dict[str, dict[str, float]] = {}
            s4_bh_audit: dict | None = None
            s4_dedup_audit: dict | None = None
            s4_representatives: list[str] = []
            if s4_depth3:
                eval_rows = s4_eval_rows[cycle_index - 1]
                candidate_names = [
                    name for name in dict.fromkeys(s4_cycle_candidates)
                    if name in train.columns]
                if candidate_names:
                    eval_matrix = impute_columns(
                        design_matrix(train, candidate_names)[eval_rows])
                    s4_representatives, s4_dedup_audit = (
                        label_free_representatives(
                            eval_matrix, candidate_names,
                            correlation_threshold=0.95))
                pair_scores: dict[tuple[str, str], float] = {}
                n_pos_by_seat: dict[str, int] = {}
                for key, _, _ in l1_library:
                    record = by_key.get(key)
                    if record is None:
                        continue
                    current = list(base_surfaces[key])
                    eligible = blacklist.filter(key, [
                        name for name in s4_representatives
                        if name not in current])
                    if not eligible:
                        s4_scores_by_seat[key] = {}
                        continue
                    residual_eval = (
                        y_train.astype(np.float64)
                        - np.asarray(record["oof"], dtype=np.float64)
                    )[eval_rows]
                    conditioning_eval = design_matrix(
                        train, list(dict.fromkeys(
                            base_cols(key) + current)))[eval_rows]
                    scores = entry_scores(
                        impute_columns(
                            design_matrix(train, eligible)[eval_rows]),
                        eligible, conditioning_eval, residual_eval,
                        folds=int(entry_cfg["residualize_folds"]),
                        ridge_alpha=float(entry_cfg["ridge_alpha"]),
                        seed=(engine_seed + 6100 + 1000 * round_index
                              + cycle_index))
                    s4_scores_by_seat[key] = scores
                    n_pos_by_seat[key] = int(y_train[eval_rows].sum())
                    pair_scores.update({
                        (key, name): float(score)
                        for name, score in scores.items()})
                if str(_lite("s4_entry", "bh")) == "noise_floor":
                    s4_selected_by_seat, s4_bh_audit = s4_per_seat_noise_floor(
                        pair_scores, n_pos_by_seat,
                        alpha=float(entry_cfg["alpha"]),
                        unique_factor_cap=int(
                            s4_survivor_caps[cycle_index - 1]))
                else:
                    s4_selected_by_seat, s4_bh_audit = global_pair_bh(
                        pair_scores, n_pos_by_seat,
                        q=float(s4_bh_q[cycle_index - 1]),
                        unique_factor_cap=int(
                            s4_survivor_caps[cycle_index - 1]))
                if template_cycle_audit is not None:
                    template_cycle_audit["s4_selection"] = {
                        "depth": int(cycle_index),
                        "evidence_rows": int(len(evidence_rows)),
                        "evaluation_rows": int(len(eval_rows)),
                        "evaluation_n_pos": int(y_train[eval_rows].sum()),
                        "candidate_cap": int(
                            s4_candidate_caps[cycle_index - 1]),
                        "survivor_cap": int(
                            s4_survivor_caps[cycle_index - 1]),
                        "dedup": s4_dedup_audit,
                        "bh": s4_bh_audit,
                    }
            seat_evidence: dict[str, dict] = {}
            prefetched_by_key: dict[str, dict] = {}
            if (_lite("parallel_llm") and not _LITE.get("shared_llm")
                    and not template_channel):
                for key, _, _ in l1_library:
                    record = by_key.get(key)
                    if record is None:
                        continue
                    oof_k = np.asarray(record["oof"], dtype=np.float64)
                    residual_k = _scoring_residual(y_train, oof_k)[disc]
                    current_k = list(base_surfaces[key])
                    conditioning_k = design_matrix(
                        train,
                        list(dict.fromkeys(base_cols(key) + current_k)))[disc]
                    pool_bank_k = blacklist.filter(key, [
                        name for name in pool_source_names
                        if name not in current_k])
                    bank_scores_k: dict[str, float] = {}
                    if pool_bank_k:
                        bank_scores_k = entry_scores(
                            impute_columns(design_matrix(train, pool_bank_k)[disc]),
                            pool_bank_k, conditioning_k, residual_k,
                            folds=int(entry_cfg["residualize_folds"]),
                            ridge_alpha=float(entry_cfg["ridge_alpha"]),
                            seed=engine_seed + 1000 * round_index,
                            weights=entry_weights)
                    seat_evidence[key] = {
                        "residual": residual_k, "conditioning": conditioning_k,
                        "pool_bank": pool_bank_k, "bank_scores": bank_scores_k,
                        "evidence": top_s_evidence(bank_scores_k, families, floor),
                        "current_factors": current_k,
                    }
                llm_draws = max(int(_lite("llm_draws", 1)), 1)
                workers = min(int(_lite("parallel_llm")),
                              max(len(seat_evidence) * llm_draws, 1))
                started_llm = time.time()
                fetched_parts: dict[str, list[dict]] = {
                    key: [] for key in seat_evidence}
                with ThreadPoolExecutor(max_workers=workers) as pool_llm:
                    futures = {
                        pool_llm.submit(
                            llm_fetch_for_model,
                            llm=llm, record=by_key[key], train=train, label=label,
                            meta=meta, dataset_description=proposal_description_for(
                                dataset_description),
                            tried=tried[key], entities=entities, values=values,
                            visible=visible, disagreement=disagreement,
                            current_features=list(dict.fromkeys(
                                base_cols(key)
                                + seat_evidence[key]["current_factors"])),
                            evidence=seat_evidence[key]["evidence"],
                            point_cap=int(llm_cfg["max_pointwise_per_model_per_round"]),
                            agg_cap=int(llm_cfg["max_aggregate_per_model_per_round"]),
                            temporal=time_values is not None,
                            blacklist_context=blacklist_context): key
                        for key in seat_evidence
                        for _draw in range(llm_draws)}
                    for future in as_completed(futures):
                        seat = futures[future]
                        try:
                            fetched_parts[seat].append(future.result())
                        except Exception as exc:
                            log(f"round {round_index} {seat}: LLM fetch failed "
                                f"({type(exc).__name__}: {exc})")
                for seat, parts in fetched_parts.items():
                    if not parts:
                        continue
                    if len(parts) == 1:
                        prefetched_by_key[seat] = parts[0]
                        continue
                    seen_sig: set = set()
                    raw_specs, raw_aggs = [], []
                    for part in parts:
                        for raw in part["raw_specs"]:
                            sig = ("spec", raw.name,
                                   " ".join(raw.expression.split()).lower())
                            if sig[1] in {r.name for r in raw_specs} \
                                    or sig in seen_sig:
                                continue
                            seen_sig.add(sig)
                            raw_specs.append(raw)
                        for raw in part["raw_aggs"]:
                            name = str(raw.get("name", ""))
                            if any(str(r.get("name", "")) == name
                                   for r in raw_aggs):
                                continue
                            raw_aggs.append(raw)
                    prefetched_by_key[seat] = {
                        **parts[0], "raw_specs": raw_specs,
                        "raw_aggs": raw_aggs, "n_draws": len(parts)}
                phase_wall = time.time() - started_llm
                llm_phase_wall_sec += phase_wall
                pooled = {seat: len(part.get("raw_specs", []))
                          for seat, part in prefetched_by_key.items()}
                log(f"round {round_index}: {len(prefetched_by_key)} seats x "
                    f"{llm_draws} draw(s) over {workers} workers in "
                    f"{phase_wall:.1f}s; pooled specs {pooled}")

            for key, _, _ in l1_library:
                record = by_key.get(key)
                if record is None:
                    per_model[key] = {
                        "skipped": "no_incumbent_record_time_limit_dropped",
                        "k_add": 0, "adds": [], "k_drop": 0, "drops": [],
                    }
                    continue
                oof = np.asarray(record["oof"], dtype=np.float64)
                current_factors = list(base_surfaces[key])
                surface_columns = list(dict.fromkeys(base_cols(key)
                                                     + current_factors))
                cached = seat_evidence.get(key)
                if cached is not None:
                    residual = cached["residual"]
                    conditioning = cached["conditioning"]
                    pool_bank = cached["pool_bank"]
                    evidence = cached["evidence"]
                else:
                    score_rows = (s4_eval_rows[cycle_index - 1]
                                  if s4_depth3 and not s4_funnel else disc)
                    residual = _scoring_residual(y_train, oof)[score_rows]
                    conditioning = design_matrix(
                        train, surface_columns)[score_rows]

                    pool_bank = blacklist.filter(key, [
                        name for name in (s4_representatives
                                          if s4_depth3
                                          else pool_source_names)
                        if name not in current_factors])
                    bank_scores: dict[str, float] = {}
                    if pool_bank and not s4_depth3:
                        bank_matrix = impute_columns(
                            design_matrix(train, pool_bank)[score_rows])
                        bank_scores = entry_scores(
                            bank_matrix, pool_bank, conditioning, residual,
                            folds=int(entry_cfg["residualize_folds"]),
                            ridge_alpha=float(entry_cfg["ridge_alpha"]),
                            seed=engine_seed + 1000 * round_index,
                            weights=entry_weights)
                    evidence = top_s_evidence(bank_scores, families, floor)

                if template_channel:
                    specs = []
                    llm_audit = {"template_channel": True, "proposals": []}
                elif _LITE.get("shared_llm") and shared_specs is not None:
                    specs = list(shared_specs)
                    llm_audit = {"shared": True, "source": "shared_ensemble",
                                 "proposals": []}
                else:
                    proposal_record = record
                    proposal_tried = tried[key]
                    if _LITE.get("shared_llm"):
                        proposal_record = dict(record)
                        proposal_record["model_key"] = "shared_ensemble"
                        proposal_record["oof"] = np.asarray(
                            inc_we_l3["oof"], dtype=np.float64)
                        proposal_tried = (shared_tried if _LITE.get("v6") else
                                          set().union(*tried.values()))
                    proposal_description = proposal_description_for(
                        dataset_description)
                    specs, llm_audit = llm_propose_for_model(
                        llm=llm, record=proposal_record, train=train, apply=apply,
                        label=label, meta=meta,
                        dataset_description=proposal_description,
                        round_index=round_index, tried=proposal_tried,
                        entities=entities, values=values,
                        time_values=time_values, visible=visible,
                        disagreement=disagreement,
                        current_features=surface_columns, evidence=evidence,
                        prefetched=prefetched_by_key.get(key),
                        point_cap=int(llm_cfg["max_pointwise_per_model_per_round"]),
                        agg_cap=int(llm_cfg["max_aggregate_per_model_per_round"]),
                        blacklist_context=blacklist_context)
                    if _LITE.get("shared_llm"):
                        shared_specs = list(specs)
                        if _LITE.get("v6"):
                            entries = []
                            for spec in specs:
                                signature = " ".join(spec.expression.split()).lower()
                                entries.append({
                                    "name": spec.name, "family": spec.family,
                                    "expression": spec.expression,
                                    "repeated_expression": signature in proposal_signatures,
                                    "blacklist_name_hit": any(
                                        str(item["name"]).split("_", 4)[-1] in spec.name
                                        for item in rejected_hypotheses),
                                })
                                proposal_signatures.add(signature)
                            round_monitor = {
                                "round": round_index,
                                "blacklist_prompt_entries": len(blacklist_lines),
                                "proposals": entries,
                                "families": sorted({str(s.family) for s in specs}),
                                "outcome": None,
                            }
                            proposal_monitor.append(round_monitor)
                for spec in specs:
                    all_llm_specs[spec.name] = spec
                    families.setdefault(spec.name, spec.family or "llm")
                pool = pool_bank + [spec.name for spec in specs]
                if s4_depth3 and not s4_funnel:
                    pool = []
                elif s4_depth3 and s4_funnel:
                    pool = [
                        name for name in dict.fromkeys(
                            list(pool_bank) + list(s4_representatives))
                        if name in train.columns
                        and name not in set(current_factors)]
                    pool = blacklist.filter(key, pool)

                cumulative_pool[key] += len(pool)
                fusion_audit = None
                if pool:
                    pool_matrix = impute_columns(design_matrix(train, pool)[disc])
                    ruler_top = int(_lite("ruler_top", 20) or 20)
                    if (_LITE.get("ruler_fusion")
                            and len(pool) > 2 * ruler_top):
                        pool, fusion_audit = ruler_fusion_filter(
                            pool, pool_matrix, conditioning, residual,
                            is_net=seat_is_net(key), top_m=ruler_top,
                            entry_cfg=entry_cfg,
                            seed=engine_seed + 2000 * round_index,
                            weights=entry_weights)
                        pool_matrix = pool_matrix[
                            :, fusion_audit.pop("matrix_index")]
                    use_bh = (bool(_LITE.get("v2_selection"))
                              and _lite("entry_rule", "bh") != "noise_floor")
                    selector = lite_v2_entry if use_bh else greedy_entry
                    selector_kwargs = dict(
                        weights=entry_weights,
                        alpha=alpha_entry,
                        folds=int(entry_cfg["residualize_folds"]),
                        ridge_alpha=float(entry_cfg["ridge_alpha"]),
                        seed=engine_seed + 1000 * round_index)
                    if use_bh:
                        family = (len(pool) if _lite("bh_family") == "per_round"
                                  else cumulative_pool[key])
                        selector_kwargs.update(
                            incumbent=set(current_factors),
                            correlation_threshold=0.95,
                            total_hypotheses=max(family, 1))
                    else:
                        selector_kwargs.update(
                            k_cap=int(entry_cfg["k_cap"]),
                            bh_total_candidates=cumulative_pool[key])
                    selected, entry_audit = selector(
                        pool_matrix, pool, conditioning, residual, n_pos,
                        **selector_kwargs)
                    if fusion_audit is not None and isinstance(entry_audit,
                                                               dict):
                        entry_audit["ruler_fusion"] = fusion_audit
                else:
                    selected, entry_audit = greedy_entry(
                        np.zeros((len(disc), 0)), [], conditioning, residual,
                        n_pos, alpha=alpha_entry, k_cap=int(entry_cfg["k_cap"]),
                        bh_total_candidates=max(cumulative_pool[key], 1))

                if s4_depth3 and not s4_funnel:
                    pool = list(pool_bank)
                    selected = list(s4_selected_by_seat.get(key, []))
                    entry_audit = {
                        "rule": "s4_approximate_global_pair_bh",
                        "depth": int(cycle_index),
                        "evaluation_rows": int(len(
                            s4_eval_rows[cycle_index - 1])),
                        "q": float(s4_bh_q[cycle_index - 1]),
                        "pool_size": len(pool),
                        "first_pass_scores": dict(
                            s4_scores_by_seat.get(key, {})),
                        "selected": list(selected),
                        "k_add": len(selected),
                        "stop_reason": "global_pair_bh",
                        "dedup": s4_dedup_audit,
                        "global_bh": s4_bh_audit,
                    }

                if isinstance(entry_audit, dict) and "first_pass_scores" in entry_audit:
                    first_pass = entry_audit["first_pass_scores"] or {}
                    would_pass = sorted(
                        name for name, score in first_pass.items()
                        if abs(float(score)) >= floor)
                    entry_audit["counterfactual_noise_floor_first_pass"] = {
                        "rule": "abs_orthogonalized_score >= z_{1-alpha}/sqrt(n_pos)",
                        "noise_floor": float(floor),
                        "n_would_pass": len(would_pass),
                        "would_pass": would_pass,
                        "n_bh_selected": len(selected),
                        "caveat": ("first-pass scores only; greedy entry "
                                   "re-orthogonalizes after each acceptance, so "
                                   "this is an upper bound, not a replay"),
                    }

                if shadow_block is not None and selected:
                    rows_s = shadow_block["rows"]
                    residual_shadow = _scoring_residual(y_train, oof)[rows_s]
                    shadow_scores = entry_scores(
                        impute_columns(design_matrix(train, selected)[rows_s]),
                        list(selected),
                        design_matrix(train, surface_columns)[rows_s],
                        residual_shadow,
                        folds=int(entry_cfg["residualize_folds"]),
                        ridge_alpha=float(entry_cfg["ridge_alpha"]),
                        seed=engine_seed + 5000 + 1000 * round_index)
                    first_pass = (entry_audit.get("first_pass_scores") or {}
                                  if isinstance(entry_audit, dict) else {})
                    persistence = (str(_LITE.get("shadow_rule", "harm_only"))
                                   == "persistence" and cycle_index > 1)
                    shadow_table: dict[str, dict] = {}
                    shadow_drops: list[str] = []
                    for name in list(selected):
                        s_disc = float(first_pass.get(name, 0.0))
                        s_shadow = float(shadow_scores.get(name, 0.0))
                        flipped = bool(s_disc * s_shadow < 0.0
                                       and abs(s_shadow) >= shadow_block["floor"])
                        if persistence:
                            dropped = not bool(
                                s_disc * s_shadow > 0.0
                                and abs(s_shadow) >= shadow_block["floor"])
                        else:
                            dropped = flipped
                        shadow_table[name] = {
                            "s_discovery": round(s_disc, 7),
                            "s_shadow": round(s_shadow, 7),
                            "rule": ("persistence" if persistence
                                     else "harm_only"),
                            "dropped": dropped,
                        }
                        if dropped:
                            shadow_drops.append(name)
                            if flipped:
                                blacklist.add(key, [name],
                                              round_index=round_index)
                            rejected_hypotheses.append({
                                "round": round_index, "name": name,
                                "family": families.get(name, "unknown"),
                                "expression": (all_llm_specs[name].expression
                                               if name in all_llm_specs
                                               else None),
                                "reason": (
                                    "sign-flipped on the chronological shadow "
                                    "block: informative in-period, confidently "
                                    "reversed out-of-period" if flipped else
                                    "marginal-cycle factor with no persistent "
                                    "shadow-block evidence (same-sign "
                                    "|S_shadow| below the shadow floor)"),
                            })
                    if shadow_drops:
                        selected = [name for name in selected
                                    if name not in shadow_drops]
                        log(f"round {round_index} {key} shadow gate dropped "
                            f"{shadow_drops}")
                    if isinstance(entry_audit, dict):
                        entry_audit["shadow_gate"] = {
                            "rule": ("cycle1_harm_only__marginal_cycles_"
                                     "persistence" if str(_LITE.get(
                                         "shadow_rule", "harm_only"))
                                     == "persistence" else
                                     "drop_iff_sign_flip_and_abs_shadow_ge_"
                                     "shadow_noise_floor"),
                            "cycle": int(cycle_index),
                            "block_rows": int(len(rows_s)),
                            "block_n_pos": int(shadow_block["n_pos"]),
                            "block_frac": float(shadow_block["frac"]),
                            "floor": float(shadow_block["floor"]),
                            "scores": shadow_table,
                            "dropped": shadow_drops,
                        }

                if template_channel:
                    first_pass_tmpl: dict[str, float] = {}
                    shadow_dropped_tmpl: list[str] = []
                    if isinstance(entry_audit, dict):
                        first_pass_tmpl = {
                            name: float(score) for name, score in
                            (entry_audit.get("first_pass_scores")
                             or {}).items()
                            if str(families.get(name, "")).startswith("tmpl_")}
                        shadow_dropped_tmpl = [
                            name for name in
                            ((entry_audit.get("shadow_gate") or {})
                             .get("dropped") or [])
                            if str(families.get(name, "")).startswith("tmpl_")]
                    template_seat_rows.append({
                        "seat": key,
                        "admitted": [
                            name for name in selected
                            if str(families.get(name, "")).startswith("tmpl_")],
                        "first_pass": first_pass_tmpl,
                        "shadow_dropped": shadow_dropped_tmpl,
                    })

                exit_audit = None
                drops: list[str] = []
                probe_available = {name: values_
                                   for name, values_ in record["probe_oof"].items()
                                   if name in current_factors}
                if ((not s4_depth3 or s4_funnel) and current_factors
                        and probe_available and cycle_index == 1):
                    exit_audit = permutation_exit(
                        y_disc, oof[disc],
                        {name: np.asarray(values_, dtype=np.float64)[disc]
                         for name, values_ in probe_available.items()},
                        d_max=int(exit_cfg["d_max_per_model_per_round"]),
                        tau_drop=float(exit_cfg["tau_drop"]))
                    drops = list(exit_audit["drops"])
                    if drops:
                        staged_blacklist.append((key, drops, exit_audit["pi"]))

                new_surfaces[key] = ([name for name in current_factors
                                      if name not in drops] + selected)
                total_added += len(selected)
                total_dropped += len(drops)
                cycle_added += len(selected)
                seat_row = {
                    "k_add": len(selected),
                    "adds": selected,
                    "k_drop": len(drops),
                    "drops": drops,
                    "surface_before": current_factors,
                    "surface_after": new_surfaces[key],
                    "pool_size": len(pool),
                    "cumulative_pool_p": cumulative_pool[key],
                    "noise_floor": float(floor),
                    "n_pos": n_pos,
                    "entry": entry_audit,
                    "exit": exit_audit,
                    "exit_skipped_reason": (None if exit_audit is not None else
                                            ("no_factors_on_surface"
                                             if not current_factors
                                             else "no_probes_available")),
                    "llm": llm_audit,
                    "probe_mode": record.get("probe_mode"),
                }
                previous = per_model.get(key)
                if cycle_index > 1 and isinstance(previous, dict) \
                        and "skipped" not in previous:
                    seat_row["adds"] = list(previous.get("adds", [])) + selected
                    seat_row["k_add"] = len(seat_row["adds"])
                    seat_row["drops"] = list(previous.get("drops", []))
                    seat_row["k_drop"] = len(seat_row["drops"])
                    seat_row["surface_before"] = previous.get(
                        "surface_before", current_factors)
                    seat_row["pool_size"] = (int(previous.get("pool_size", 0))
                                             + len(pool))
                    seat_row["exit"] = previous.get("exit")
                    seat_row["exit_skipped_reason"] = previous.get(
                        "exit_skipped_reason")
                    seat_row["entry"] = previous.get("entry")
                    seat_row["entry_cycles"] = (
                        previous.get("entry_cycles")
                        or [previous.get("entry")]) + [entry_audit]
                    seat_row["llm"] = previous.get("llm")
                    seat_row["llm_cycles"] = (
                        previous.get("llm_cycles")
                        or [previous.get("llm")]) + [llm_audit]
                per_model[key] = seat_row
                log(f"round {round_index} {key}: +{len(selected)} "
                    f"{selected} / -{len(drops)} {drops}")

            if template_channel:
                score_json, score_text = template_scorecard(
                    round_index=round_index, cycle_index=cycle_index,
                    templates=templates_this_cycle, families=families,
                    seat_rows=template_seat_rows, floor=floor,
                    expand_audit=template_expand_audit)
                template_state["scorecard_texts"].append(score_text)
                if template_cycle_audit is not None:
                    template_cycle_audit["scorecard"] = score_json
                    template_state["rounds"].append(template_cycle_audit)
                write_json(out_dir / "template_sweep.json",
                           {"rounds": template_state["rounds"]})
            if not s4_depth3 and _LITE.get("focus_grid"):
                on_surface = set()
                for cols in new_surfaces.values():
                    on_surface.update(cols)
                dead = [n for n in list(pool_source_names)
                        if n.startswith("vf_") and n not in on_surface]
                if dead:
                    train.drop(columns=dead, inplace=True, errors="ignore")
                    apply.drop(columns=dead, inplace=True, errors="ignore")
                    for n in dead:
                        pool_source_names.remove(n)
                        all_llm_specs.pop(n, None)
                    log(f"round {round_index} cycle {cycle_index}: evicted "
                        f"{len(dead)} rejected focus-grid columns")
                kept_new = [n for n in focus_state.get("cycle_new", [])
                            if n in on_surface]
                if _LITE.get("focus_beam"):
                    admitted_any = [n for key2 in new_surfaces
                                    for n in new_surfaces[key2]
                                    if n not in set(base_surfaces.get(key2,
                                                                      []))]
                    focus_state["last_admitted"] = list(
                        dict.fromkeys(admitted_any))
                    for n in kept_new:
                        idiom = n.split("__")[0][3:]
                        focus_state["idiom_wins"][idiom] = (
                            focus_state["idiom_wins"].get(idiom, 0) + 1)
                    if (focus_state["cycle_had_new"] and not kept_new
                            and cycle_index > 2):
                        focus_state["enabled"] = False
                        log(f"round {round_index} cycle {cycle_index}: "
                            "focus-beam early stop (zero admissions)")
                focus_state["cycle_new"] = []
                focus_state["cycle_had_new"] = False
            if s4_depth3:
                on_surface = {
                    name for columns in new_surfaces.values()
                    for name in columns
                }
                admitted = list(dict.fromkeys(
                    name for key, _, _ in l1_library
                    for name in new_surfaces[key]
                    if (name in set(s4_cycle_candidates)
                        and name not in set(base_surfaces[key]))
                ))
                s4_survivors[cycle_index] = admitted

                dead = [
                    name for name in s4_cycle_candidates
                    if name not in set(bank_names) and name not in on_surface
                ]
                if dead:
                    train.drop(columns=dead, inplace=True, errors="ignore")
                    apply.drop(columns=dead, inplace=True, errors="ignore")
                    dead_set = set(dead)
                    pool_source_names[:] = [
                        name for name in pool_source_names
                        if name not in dead_set
                    ]
                    for name in dead:
                        all_llm_specs.pop(name, None)
                    log(f"round {round_index} depth {cycle_index}: evicted "
                        f"{len(dead)} globally rejected S4 candidates")

                depth_audit = {
                    "depth": cycle_index,
                    "candidate_cap": s4_candidate_caps[cycle_index - 1],
                    "candidate_count": len(s4_cycle_candidates),
                    "pair_admissions": int(cycle_added),
                    "unique_survivors": admitted,
                    "unique_survivor_count": len(admitted),
                    "survivor_cap": s4_survivor_caps[cycle_index - 1],
                    "bh_q": s4_bh_q[cycle_index - 1],
                    "evicted_count": len(dead),
                    "oof_refreshed_seats": [],
                }

                if admitted and cycle_index < 3:
                    changed = {
                        key for key, _, _ in l1_library
                        if new_surfaces[key] != base_surfaces[key]
                    }
                    refresh_library = tuple(
                        entry for entry in l1_library if entry[0] in changed)
                    refresh_surfaces = {
                        f"l1_{key}": list(dict.fromkeys(
                            base_cols(key) + new_surfaces[key]))
                        for key in changed
                    }
                    refreshed = fit_layer(
                        train=train, apply=apply, label=label,
                        library=refresh_library, surfaces=refresh_surfaces,
                        prefix="l1", layer_index=1, cfg=cfg,
                        fold_log=fold_log,
                        stage=(f"search_round_{round_index:02d}_s4_"
                               f"depth_{cycle_index}_refresh"),
                        probe_cols=None)
                    replacement = {
                        record["model_key"]: record for record in refreshed
                    }
                    cycle_inc_l1 = [
                        replacement.get(record["model_key"], record)
                        for record in cycle_inc_l1
                    ]
                    cycle_inc_we_l2 = _weighted_ensemble(
                        cycle_inc_l1, y_train, steps)
                    cycle_inc_we_l3 = cycle_inc_we_l2
                    l1_fits_performed += len(refreshed)
                    depth_audit["oof_refreshed_seats"] = sorted(changed)
                    depth_audit["oof_refresh_fit_count"] = len(refreshed)
                    log(f"round {round_index} depth {cycle_index}: "
                        f"selective OOF refresh for {sorted(changed)}")

                s4_round_audit.append(depth_audit)
                if template_cycle_audit is not None:
                    template_cycle_audit["s4_depth_outcome"] = depth_audit
                    write_json(out_dir / "template_sweep.json",
                               {"rounds": template_state["rounds"]})
            cycle_ledger.append({"cycle": cycle_index,
                                 "added": int(cycle_added)})
            if proposal_cycles > 1:
                log(f"round {round_index} cycle {cycle_index}/"
                    f"{proposal_cycles}: +{cycle_added} factors admitted")
            if cycle_added == 0:
                break

        row = {
            "round": round_index,
            "wall_sec": None,
            "total_added": total_added,
            "total_dropped": total_dropped,
            "per_model": per_model,
            "gate": None,
            "accepted": None,
        }
        if proposal_cycles > 1:
            row["proposal_cycles"] = cycle_ledger
        if s4_depth3:
            row["s4_depth3"] = s4_round_audit

        if total_added == 0 and total_dropped == 0:
            no_change_rounds += 1
            if round_monitor is not None:
                round_monitor["outcome"] = "no_bh_changes"
            if _LITE.get("v6"):
                if not shared_specs:
                    consecutive_empty_llm += 1
                else:
                    consecutive_empty_llm = 0
                    for spec in shared_specs:
                        rejected_hypotheses.append({
                            "round": round_index, "name": spec.name,
                            "family": spec.family, "expression": spec.expression,
                            "reason": "failed BH entry for every consumer",
                        })
                    for key, _, _ in l1_library:
                        blacklist.add(key, [s.name for s in shared_specs],
                                      round_index=round_index)
                if round_index < max_rounds and consecutive_empty_llm < 2:
                    row["continued_after_no_changes"] = True
                    row["wall_sec"] = round(time.time() - round_started, 3)
                    ledger.append(row)
                    write_json(out_dir / "rounds" / f"round_{round_index:02d}.json", row)
                    continue
            stop_reason = f"round_{round_index}_converged_no_changes"
            row["wall_sec"] = round(time.time() - round_started, 3)
            ledger.append(row)
            write_json(out_dir / "rounds" / f"round_{round_index:02d}.json",
                       row)
            break

        compute_probes = round_index < max_rounds
        round_drops: list = []
        l1_surfaces = {
            f"l1_{key}": list(dict.fromkeys(base_cols(key) + new_surfaces[key]))
            for key, _, _ in l1_library}
        probe_cols = ({f"l1_{key}": list(new_surfaces[key])
                       for key, _, _ in l1_library} if compute_probes else {})
        changed_keys = {key for key, _, _ in l1_library
                        if new_surfaces[key] != surfaces[key]}
        by_key_now = {r["model_key"]: r for r in inc_l1}
        reuse_blocked: dict[str, str] = {}
        if selective_refit:
            reusable = set()
            for key, _, _ in l1_library:
                if key in changed_keys:
                    continue
                record = by_key_now.get(key)
                if record is None:
                    reuse_blocked[key] = "no_incumbent_record"
                elif _LITE.get("v9") and key not in search_fitted:
                    reuse_blocked[key] = "incumbent_is_from_baseline_stage"
                elif (_LITE.get("v9") and compute_probes
                        and not set(new_surfaces[key]).issubset(
                            set(record.get("probe_oof") or {}))):
                    reuse_blocked[key] = "probes_missing_for_current_surface"
                else:
                    reusable.add(key)
            fit_library = tuple(entry for entry in l1_library
                                if entry[0] not in reusable)
        else:
            reusable = set()
            fit_library = l1_library
        changed_records = (fit_layer(
            train=train, apply=apply, label=label, library=fit_library,
            surfaces=l1_surfaces, prefix="l1", layer_index=1, cfg=cfg,
            fold_log=fold_log, stage=f"search_round_{round_index:02d}_l1",
            probe_cols=probe_cols, time_limit_audit=round_drops)
            if fit_library else [])
        if selective_refit:
            replacement = {r["model_key"]: r for r in changed_records}
            cand_l1 = [replacement.get(key, by_key_now[key])
                       for key, _, _ in l1_library
                       if key in replacement or key in by_key_now]
            refit_names = [r["model_key"] for r in changed_records]
            reused_names = [r["model_key"] for r in cand_l1
                            if r["model_key"] not in replacement]
            l1_fits_performed += len(refit_names)
            l1_fits_saved += len(reused_names)
            refit_l1_by_round[str(round_index)] = refit_names
            reused_l1_by_round[str(round_index)] = reused_names
            row["selective_oof"] = {
                "changed_models": sorted(changed_keys),
                "refitted_models": refit_names,
                "reused_models": reused_names,
                "reuse_blocked": reuse_blocked,
                "l2_enabled": l2_enabled,
            }
        else:
            cand_l1 = changed_records
            l1_fits_performed += len(cand_l1)
            refit_l1_by_round[str(round_index)] = [
                r["model_key"] for r in cand_l1]
            reused_l1_by_round[str(round_index)] = []
        if l2_enabled:
            stack_columns = _attach_stack_columns(train, apply, cand_l1)
            l2_surfaces = {
                f"l2_{key}": list(dict.fromkeys(features + stack_columns))
                for key, _, _ in l2_library}
            cand_l2 = fit_layer(
                train=train, apply=apply, label=label, library=l2_library,
                surfaces=l2_surfaces, prefix="l2", layer_index=2, cfg=cfg,
                fold_log=fold_log, stage=f"search_round_{round_index:02d}_l2",
                time_limit_audit=round_drops)
            cand_we_l2, cand_we_l3 = build_ensembles(cand_l1, cand_l2,
                                                     y_train, steps)
        else:
            cand_l2 = []
            cand_we_l2 = _weighted_ensemble(cand_l1, y_train, steps)
            cand_we_l3 = cand_we_l2
        cand_we_l3_pred = np.asarray(cand_we_l3["prediction"],
                                     dtype=np.float64)
        row["probe_columns"] = {member: list(columns)
                                for member, columns in probe_cols.items()
                                if columns}
        row["time_limit_dropped"] = round_drops

        gate_block = blocks[(gate_blocks_consumed if _LITE.get("v6")
                             else round_index - 1)]
        gate_result, gate_audit = round_gate(
            y_apply=y_apply, block=gate_block,
            incumbent_pred=inc_we_l3_pred, candidate_pred=cand_we_l3_pred,
            cfg=cfg, round_index=round_index, max_rounds=max_rounds)
        gate_blocks_consumed += 1
        row["gate"] = gate_audit
        gate_rule = str(cfg["experiment"]["gate"].get("rule", "delta_ge_tau"))
        if gate_rule == "harm_only_2se":
            harm_k = float(cfg["experiment"]["gate"].get("harm_k", 2.0))
            threshold = -harm_k * float(gate_result.bootstrap_se)
            v13_1_accept = bool(gate_result.delta >= threshold)
            gate_audit["v13_1_accept_rule"] = "harm_only_delta_ge_minus_2se"
            gate_audit["harm_threshold"] = threshold
        else:
            v13_1_accept = bool(gate_result.delta
                                >= float(cfg["experiment"]["gate"]["tau_auprc"]))
            gate_audit["v13_1_accept_rule"] = "delta_ge_tau"
        gate_audit["lcb_veto"] = "audit_only_v13_1"
        seat_gate = (str(_LITE.get("gate_scope", "round")) == "seat"
                     and bool(changed_keys))
        if seat_gate and l2_enabled:
            raise RuntimeError("--gate-scope seat requires --lite-search-l2 off")
        if seat_gate:
            y_blk = np.asarray(y_apply)[gate_block]
            inc_ap = float(average_precision_score(
                y_blk, inc_we_l3_pred[gate_block]))
            harm_k_seat = float(cfg["experiment"]["gate"].get("harm_k", 2.0))
            seat_threshold = -harm_k_seat * float(gate_result.bootstrap_se)
            inc_by_key = {r["model_key"]: r for r in inc_l1}
            cand_by_key = {r["model_key"]: r for r in cand_l1}
            order = [r["model_key"] for r in cand_l1]
            seat_deltas: dict[str, float] = {}
            for key in sorted(changed_keys):
                if key not in cand_by_key or key not in inc_by_key:
                    continue
                mixed = [cand_by_key[k] if k == key
                         else inc_by_key.get(k, cand_by_key[k]) for k in order]
                mixed_we = _weighted_ensemble(mixed, y_train, steps)
                mixed_pred = np.asarray(mixed_we["prediction"], dtype=np.float64)
                seat_deltas[key] = float(average_precision_score(
                    y_blk, mixed_pred[gate_block]) - inc_ap)
            kept_keys = {k for k, d in seat_deltas.items()
                         if d >= seat_threshold}
            dropped_keys = set(seat_deltas) - kept_keys
            gate_audit["seat_scope"] = {
                "rule": "per_seat_harm_only_delta_ge_minus_k_se",
                "harm_k": harm_k_seat,
                "threshold": seat_threshold,
                "incumbent_block_auprc": inc_ap,
                "seat_deltas": {k: round(v, 7)
                                for k, v in sorted(seat_deltas.items())},
                "kept": sorted(kept_keys),
                "dropped": sorted(dropped_keys),
            }
            salvage_cap = int(_LITE.get("salvage_max_refits", 0))
            salvage_log: list[dict] = []
            if dropped_keys and salvage_cap > 0:
                salvage_order = sorted(dropped_keys,
                                       key=lambda k: -seat_deltas[k])
                refits_spent = 0
                for key in salvage_order:
                    entry = per_model.get(key)
                    adds = list(entry.get("adds", [])) if isinstance(entry, dict) else []
                    if len(adds) < 2 or refits_spent >= salvage_cap:
                        continue
                    seat_lib = tuple(e for e in l1_library if e[0] == key)
                    best = None
                    for drop_factor in adds:
                        if refits_spent >= salvage_cap:
                            break
                        subset = [c for c in new_surfaces[key] if c != drop_factor]
                        refits_spent += 1
                        try:
                            recs = fit_layer(
                                train=train, apply=apply, label=label,
                                library=seat_lib,
                                surfaces={f"l1_{key}": list(dict.fromkeys(
                                    base_cols(key) + subset))},
                                prefix="l1", layer_index=1, cfg=cfg,
                                fold_log=fold_log,
                                stage=f"search_round_{round_index:02d}_salvage",
                                time_limit_audit=round_drops)
                        except Exception as exc:
                            log(f"round {round_index} salvage {key} "
                                f"-{drop_factor}: fit failed "
                                f"({type(exc).__name__}: {exc})")
                            continue
                        if not recs:
                            continue
                        salv = recs[0]
                        mixed = [salv if k == key
                                 else inc_by_key.get(k, cand_by_key[k])
                                 for k in order]
                        mixed_we = _weighted_ensemble(mixed, y_train, steps)
                        d = float(average_precision_score(
                            y_blk, np.asarray(mixed_we["prediction"],
                                              dtype=np.float64)[gate_block])
                            - inc_ap)
                        salvage_log.append({
                            "round": round_index, "seat": key,
                            "removed": drop_factor, "kept_subset": subset,
                            "delta": round(d, 7),
                            "seat_delta_before": round(seat_deltas[key], 7),
                            "rescued": bool(d >= seat_threshold),
                        })
                        if d >= seat_threshold and (best is None
                                                    or d > best["delta"]):
                            best = {"delta": d, "record": salv,
                                    "surface": subset, "removed": drop_factor}
                    l1_fits_performed += refits_spent
                    if best is not None:
                        new_surfaces[key] = list(best["surface"])
                        cand_by_key[key] = best["record"]
                        changed_records = [r for r in changed_records
                                           if r["model_key"] != key]
                        changed_records.append(best["record"])
                        seat_deltas[key] = best["delta"]
                        kept_keys.add(key)
                        dropped_keys.discard(key)
                        if isinstance(entry, dict):
                            entry["adds"] = [a for a in adds
                                             if a != best["removed"]]
                            entry["k_add"] = len(entry["adds"])
                            entry["surface_after"] = list(best["surface"])
                            entry["salvaged_by_removing"] = best["removed"]
                            blacklist.add(key, [best["removed"]],
                                          round_index=round_index)
                gate_audit["seat_scope"]["salvage"] = {
                    "cap": salvage_cap, "refits": refits_spent,
                    "attempts": salvage_log,
                    "rescued": sorted(x["seat"] for x in salvage_log
                                      if x["rescued"]),
                }
                gate_audit["seat_scope"]["kept"] = sorted(kept_keys)
                gate_audit["seat_scope"]["dropped"] = sorted(dropped_keys)
                if salvage_log:
                    log(f"round {round_index} salvage: {refits_spent} refits, "
                        f"rescued {sorted(set(x['seat'] for x in salvage_log if x['rescued']))}")
            if dropped_keys or any(x["rescued"] for x in salvage_log):
                for key in dropped_keys:
                    new_surfaces[key] = list(surfaces[key])
                    entry = per_model.get(key)
                    if isinstance(entry, dict):
                        entry["seat_gate_dropped"] = True
                        entry["seat_gate_delta"] = round(seat_deltas[key], 7)
                        blacklist.add(key, list(entry.get("adds", [])),
                                      round_index=round_index)
                        entry["adds"], entry["k_add"] = [], 0
                        entry["surface_after"] = list(surfaces[key])
                cand_l1 = [inc_by_key.get(k, cand_by_key[k])
                           if k in dropped_keys else cand_by_key[k]
                           for k in order]
                changed_records = [r for r in changed_records
                                   if r["model_key"] not in dropped_keys]
                cand_we_l2 = _weighted_ensemble(cand_l1, y_train, steps)
                cand_we_l3 = cand_we_l2
                cand_we_l3_pred = np.asarray(cand_we_l3["prediction"],
                                             dtype=np.float64)
                gate_audit["seat_scope"]["combined_delta_after_drop"] = round(
                    float(average_precision_score(
                        y_blk, cand_we_l3_pred[gate_block]) - inc_ap), 7)
            v13_1_accept = bool(kept_keys)
            gate_audit["v13_1_accept_rule"] = "per_seat_harm_only"
            log(f"round {round_index} seat gate: kept {sorted(kept_keys)} "
                f"dropped {sorted(dropped_keys)} "
                f"threshold={seat_threshold:+.6f} "
                f"deltas={ {k: round(v, 5) for k, v in seat_deltas.items()} }")
        gate_audit["v13_1_accepted"] = v13_1_accept
        row["accepted"] = v13_1_accept
        row["wall_sec"] = round(time.time() - round_started, 3)
        ledger.append(row)
        write_json(out_dir / "rounds" / f"round_{round_index:02d}.json", row)
        save_records(out_dir / "search" / f"round_{round_index:02d}_candidate"
                     / "l1", cand_l1)
        save_records(out_dir / "search" / f"round_{round_index:02d}_candidate"
                     / "l2", cand_l2)
        log(f"round {round_index} gate: delta={gate_result.delta:+.6f} "
            f"lcb={gate_result.bootstrap_lcb:+.6f} "
            f"v13_1_accepted={v13_1_accept} "
            f"lcb_audit_accepted={gate_result.accepted}")
        if not v13_1_accept:
            if round_monitor is not None:
                round_monitor["outcome"] = "gate_rejected"
            if _LITE.get("v6"):
                for key, entry in per_model.items():
                    failed = list(entry.get("adds", []))
                    if failed:
                        blacklist.add(key, failed, round_index=round_index)
                selected_names = {name for entry in per_model.values()
                                  for name in entry.get("adds", [])}
                for spec in (shared_specs or []):
                    if spec.name in selected_names:
                        rejected_hypotheses.append({
                            "round": round_index, "name": spec.name,
                            "family": spec.family, "expression": spec.expression,
                            "reason": "candidate graph failed chronological gate",
                        })
                if round_index < max_rounds:
                    row["continued_after_gate_rejection"] = True
                    continue
            stop_reason = f"round_{round_index}_gate_rejected"
            break
        surfaces = new_surfaces
        for key, drops, pi in staged_blacklist:
            blacklist.add(key, drops, round_index=round_index, pi=pi)
        search_fitted.update(r["model_key"] for r in changed_records)
        inc_l1, inc_l2 = cand_l1, cand_l2
        inc_we_l2, inc_we_l3 = cand_we_l2, cand_we_l3
        inc_we_l3_pred = cand_we_l3_pred
        accepted_rounds = round_index
        if round_monitor is not None:
            round_monitor["outcome"] = "accepted"

        prune_frac = _lite("prune_weight")
        if prune_frac:
            min_seats = int(_lite("prune_min_seats", 4))
            weights = dict(inc_we_l2.get("weights") or {})
            active = [entry for entry in l1_library]
            threshold = float(prune_frac) / max(len(active), 1)
            nets = {key for key, learner, _ in active if learner == "mlp"}
            scored = [(key, float(weights.get(f"l1_{key}", 0.0)))
                      for key, _, _ in active]
            doomed = {key for key, weight in scored if weight < threshold}
            surviving_nets = nets - doomed
            if nets and not surviving_nets:
                keep_net = max((k for k in nets),
                               key=lambda k: float(weights.get(f"l1_{k}", 0.0)))
                doomed.discard(keep_net)
            if len(active) - len(doomed) < min_seats:
                for key, _ in sorted(scored, key=lambda kv: -kv[1]):
                    if len(active) - len(doomed) >= min_seats:
                        break
                    doomed.discard(key)
            if doomed:
                l1_library = tuple(entry for entry in l1_library
                                   if entry[0] not in doomed)
                inc_l1 = [r for r in inc_l1 if r["model_key"] not in doomed]
                inc_we_l2 = _weighted_ensemble(inc_l1, y_train, steps)
                if l2_enabled:
                    inc_we_l2, inc_we_l3 = build_ensembles(
                        inc_l1, inc_l2, y_train, steps)
                else:
                    inc_we_l3 = inc_we_l2
                inc_we_l3_pred = np.asarray(inc_we_l3["prediction"],
                                            dtype=np.float64)
                pruned_log.append({
                    "round": round_index, "threshold": threshold,
                    "dropped": sorted(doomed),
                    "dropped_weights": {k: float(weights.get(f"l1_{k}", 0.0))
                                        for k in sorted(doomed)},
                    "remaining": [key for key, _, _ in l1_library],
                    "l2_carried_over_stale": bool(l2_enabled and inc_l2),
                })
                row["seat_pruning"] = pruned_log[-1]
                log(f"V10 prune round {round_index}: dropped "
                    f"{sorted(doomed)} (weight < {threshold:.4f}); "
                    f"{len(l1_library)} seats remain")

    final_started = time.time()
    search_wall_sec = round(
        sum(float(row.get("wall_sec") or 0.0) for row in ledger), 3)
    write_json(out_dir / "round_ledger.json", ledger)
    write_json(out_dir / "blacklist.json", blacklist.to_dict())
    write_json(out_dir / "llm_proposal_monitor.json", {
        "policy": "v6_blacklist_in_prompt_and_expression_drift",
        "rounds": proposal_monitor,
        "rejected_hypotheses": rejected_hypotheses,
    })
    save_records(out_dir / "search" / "accepted" / "l1", inc_l1)
    save_records(out_dir / "search" / "accepted" / "l2", inc_l2)

    plan = {key: list(surfaces[key]) for key, _, _ in l1_library}
    for key, columns in plan.items():
        unknown = [column for column in columns
                   if column not in set(bank_names)
                   and column not in all_llm_specs]
        if unknown:
            raise RuntimeError(f"unaccounted factors for {key}: {unknown}")
    surface_llm_names = sorted({
        column for columns in plan.values() for column in columns
        if column in all_llm_specs})
    accepted_llm_names = _closed_llm_order(surface_llm_names, all_llm_specs)
    intermediate_llm_names = [name for name in accepted_llm_names
                              if name not in set(surface_llm_names)]
    if intermediate_llm_names:
        log(f"frozen spec closure added {len(intermediate_llm_names)} "
            f"intermediate LLM factor(s): {intermediate_llm_names}")
    s4_plan_audit = None
    if s4_depth3:
        accepted_factor_names = sorted({
            name for columns in plan.values() for name in columns})
        s4_plan_audit = {
            "enabled": True,
            "protocol": "S4_DEPTH3_FAST",
            "candidate_caps": list(s4_candidate_caps),
            "survivor_caps": list(s4_survivor_caps),
            "global_pair_bh_q": list(s4_bh_q),
            "candidate_caps_are": "hard_materialization_ceilings_not_targets",
            "depths": [
                "bank_plus_unary_plus_strictly_past_rolling",
                "depth1_survivor_binary_plus_strictly_past_rolling",
                "depth2_survivor_plus_order1_parent_plus_strictly_past_rolling",
            ],
            "selection_family": "global_seat_factor_pairs",
            "dedup": "label_free_absolute_spearman_before_bh",
            "residual_refresh": "changed_seats_only_after_depths_1_and_2",
            "final_verdict": "existing_chronological_per_seat_round_gate",
            "accepted_factor_lineage": {
                name: s4_lineage.get(name, {})
                for name in accepted_factor_names
                if name in s4_lineage},
            "rounds": [
                {"round": row.get("round"),
                 "depths": row.get("s4_depth3", [])}
                for row in ledger],
        }
        write_json(out_dir / "s4_depth3_audit.json", s4_plan_audit)
    write_json(out_dir / "model_factor_plan.json", {
        "l1_factor_surfaces": plan,
        "l2_rule": "raw_features_plus_stack_columns_code9_graph_rule",
        "accepted_llm_specs": {
            name: all_llm_specs[name].to_dict()
            for name in accepted_llm_names},
        "stack_graph": (f"{len(l1_library)}_L1_to_{len(l2_library)}"
                        "_L2_to_WE_L3_full_caruana"),
        "accepted_rounds": accepted_rounds,
        "stop_reason": stop_reason,
        "s4_depth3": s4_plan_audit,
    })
    del baseline_records, train, apply
    gc.collect()

    raw_train, raw_test, outer_split = load_outer_partitions(dataset_cfg)
    final_train, final_test, final_features, _ = prepare_e1(
        raw_train, raw_test, dataset_cfg)
    final_representation_audit = select_and_apply_representation(
        raw_train, raw_test, final_train, final_test, final_features,
        dataset_cfg, cfg, frozen_choice=representation_choice)
    write_json(out_dir / "representation_audit_final.json",
               final_representation_audit)
    representation_repair = reconcile_final_features(
        list(features), final_train, final_test, raw_train, raw_test,
        label=label, time_col=time_col,
        split_seed=int(dataset_cfg["split_seed"]),
        final_features=list(final_features))
    write_json(out_dir / "final_representation_repair.json",
               representation_repair)
    if representation_repair["rebuilt"]:
        log(f"final representation replay rebuilt "
            f"{sorted(representation_repair['rebuilt'])}")
    final_bank_names, final_bank_audit = materialize_bank(
        cfg, raw_train, raw_test, final_train, final_test,
        label=label, time_col=time_col, forbidden=forbidden,
        frozen_spec=bank_spec)
    if final_bank_names != bank_names:
        raise RuntimeError("factor bank drift between search and final")
    write_json(out_dir / "factor_bank_final.json", final_bank_audit)
    final_time = None
    if time_col is not None:
        final_time = np.concatenate([
            pd.to_numeric(raw_train[time_col], errors="coerce").to_numpy(float),
            pd.to_numeric(raw_test[time_col], errors="coerce").to_numpy(float),
        ])
    del raw_train, raw_test
    gc.collect()
    for name in accepted_llm_names:
        if not materialize_spec(spec=all_llm_specs[name], train=final_train,
                                apply=final_test, label=label,
                                time_values=final_time):
            raise RuntimeError(f"final LLM materialization failed: {name}")
    for key, columns in plan.items():
        missing = [column for column in columns
                   if column not in final_train.columns
                   or column not in final_test.columns]
        if missing:
            raise RuntimeError(f"final factor columns missing for {key}: "
                               f"{missing}")
    final_y = final_train[label].to_numpy(np.int8, copy=False)

    if fast_deploy:
        final_l1_surfaces = {
            f"l1_{key}": list(dict.fromkeys(base_cols(key) + plan[key]))
            for key, _, _ in l1_library}
        final_l1 = fit_layer_full_once(
            train=final_train, apply=final_test, label=label,
            library=l1_library, surfaces=final_l1_surfaces, prefix="l1",
            cfg=cfg)
        frozen_weights = dict(inc_we_l2["weights"])
        prediction = apply_frozen_weights(final_l1, frozen_weights)
        final_dir = out_dir / "final"
        save_records(final_dir / "l1", final_l1)
        np.save(final_dir / "test_pred.npy", prediction)
        write_json(final_dir / "ensemble.json", {
            "stack_enabled": False,
            "final_ensemble": "v8_frozen_search_oof_weighted_l1",
            "weight_source": "accepted_search_2fold_oof_only",
            "weights": frozen_weights,
            "final_fit": "one_full_train_fit_per_l1_member_no_oof",
        })
        y_test = final_test[label].to_numpy(np.int8, copy=False)
        auprc = float(average_precision_score(y_test, prediction))
        auc = float(roc_auc_score(y_test, prediction))
        baseline_wall = float(baseline_summary.get("wall_total_sec") or 0.0)
        llm_wall = llm_wall_sec(out_dir / "llm_calls.jsonl")
        references = dict(cfg.get("references", {}).get(dataset, {}) or {})
        reference_deltas = {
            f"delta_vs_{name}": round(auprc - float(value), 7)
            for name, value in references.items()}
        summary = {
            "method": METHOD,
            "variant": ("FORGE_LITE_V9" if _LITE.get("v9")
                        else "FORGE_LITE_V8"),
            "stage": "forge", "dataset": dataset,
            "outer_split_seed": int(dataset_cfg["split_seed"]),
            "representation": str(representation_choice),
            "representation_probe": representation_audit.get("probe_scores"),
            "representation_scope": str(_lite("rep_scope", "shared")),
            "representation_seat_map": (seat_plan or {}).get("seat_rep"),
            "graph": {"l1_seats": [key for key, _, _ in l1_library],
                      "l2_seats": []},
            "final_auprc": round(auprc, 7), "final_auc": round(auc, 7),
            "references": references, "reference_deltas": reference_deltas,
            "factor_injection": inject_audit,
            "s4_depth3": s4_plan_audit,
            "outer_test_evaluations": 1,
            "outer_train_rows": len(final_train),
            "outer_test_rows": len(final_test),
            "outer_split": outer_split, "search_split": split,
            "n_l1": len(final_l1), "n_l2": 0, "stack_enabled": False,
            "neg_subsample": subsample_audit,
            "theory": {
                "entry": ("s4_approximate_global_pair_bh"
                          if s4_depth3 else
                          "forge_lite_v2_collinearity_then_bh"
                          if _LITE.get("v2_selection") else
                          "greedy_orthogonalized_residual_corr_noise_floor"),
                "entry_alpha": alpha_entry,
                "entry_noise_floor": float(floor),
                "bh_family": _lite("bh_family", "cumulative"),
                "entry_rule_selected": _lite("entry_rule", "bh"),
                "exit": ("disabled_inside_s4_depths"
                         if s4_depth3 else str(exit_cfg["rule"])),
                "exit_tau_drop": float(exit_cfg["tau_drop"]),
                "exit_d_max": int(exit_cfg["d_max_per_model_per_round"]),
                "discovery_rows": len(disc), "discovery_n_pos": n_pos,
                "bh_role": ("global_seat_factor_decision_rule"
                            if s4_depth3 else
                            "decision_rule_v2" if _LITE.get("v2_selection")
                            else "audit_only"),
            },
            "rounds": {"max_rounds": max_rounds,
                       "accepted_rounds": accepted_rounds,
                       "attempted_rounds": len(ledger),
                       "stop_reason": stop_reason,
                       "gate_block_weights": list(
                           exp["gate_block_weights"][:max_rounds]),
                       "gate_block_weights_source": exp.get(
                           "gate_block_weights_source", "config"),
                       "gate_blocks_consumed": gate_blocks_consumed,
                       "gate_readings": [row.get("gate") for row in ledger],
                       "no_change_rounds": no_change_rounds,
                       "refit_l1_by_round": refit_l1_by_round,
                       "reused_l1_by_round": reused_l1_by_round,
                       "l1_fits_performed": l1_fits_performed,
                       "l1_fits_saved": l1_fits_saved,
                       "seat_pruning": pruned_log},
            "n_factor_bank": len(bank_names), "factor_bank": list(bank_names),
            "model_factor_plan": plan,
            "accepted_llm_factors": surface_llm_names,
            "intermediate_llm_factors": intermediate_llm_names,
            "blacklist": blacklist.to_dict(),
            "neg_subsample": subsample_audit,
            "oof_policy": {
                "baseline": f"{int(baseline_summary.get('engine', {}).get('bag_folds', cfg['engine']['bag_folds']))}fold_oof",
                "search": (f"{lite_bag_folds(cfg, 'search_round_01_l1')}fold"
                           "_only_for_changed_l1_models"),
                "unchanged_models": "reuse_incumbent_oof_and_predictions",
                "reuse_guard": ("v7_strict_search_round_and_probes_required"
                                if _LITE.get("v9") else "v8_surface_diff_only"),
                "l2_and_dystack": "disabled",
                "final": "single_full_train_fit_no_oof",
                "weight_source": "accepted_search_oof",
            },
            "final_weights": frozen_weights,
            "llm": {**llm.audit(),
                    "domain_context_supplied": bool(domain_context),
                    "domain_context_sha256": (
                        hashlib.sha256(
                            domain_context.encode("utf-8")).hexdigest()
                        if domain_context else None)},
            "llm_caps": {
                "pointwise_per_model_per_round": int(
                    llm_cfg["max_pointwise_per_model_per_round"]),
                "aggregate_per_model_per_round": int(
                    llm_cfg["max_aggregate_per_model_per_round"])},
            "input_audit": input_audit,
            "gpu_mlp": gpu,
            "time_limit": {
                "member_time_limit_sec": int(
                    cfg["engine"]["member_time_limit_sec"]),
                "semantics": "ag_soft_member_cap_plus_pool_batch_deadline",
                "round_drops": [row.get("time_limit_dropped")
                                for row in ledger],
            },
            "stage_wall_sec": {
                "baseline": round(baseline_wall, 3),
                "search_rounds_total": search_wall_sec,
                "per_round": {str(row["round"]): row.get("wall_sec")
                              for row in ledger},
                "dystack": 0.0,
                "final_rebuild": round(time.time() - final_started, 3),
                "forge_stage_total": round(time.time() - started, 3),
                "baseline_plus_forge": round(
                    baseline_wall + (time.time() - started), 3),
                "llm": llm_wall,
            "llm_work_sec": llm_wall_sec(out_dir / "llm_calls.jsonl"),
            "llm_concurrent_phase_wall_sec": round(llm_phase_wall_sec, 3),
                "llm_share_of_forge_stage": (
                    round(llm_wall / (time.time() - started), 5)
                    if time.time() > started else None),
            },
            "engine": {"seed": engine_seed,
                       "bag_folds": int(cfg["engine"]["bag_folds"]),
                       "blend_steps": steps,
                       "member_time_limit_sec": int(
                           cfg["engine"]["member_time_limit_sec"]),
                       "fit_workers": _fit_workers(),
                       "lgbm_threads": os.environ.get("LGBM_THREADS"),
                       "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                       "contended_lane": os.environ.get("FORGE_LANE"),
                       "lite": cfg["engine"].get("lite"),
                       "lite_prune_audit": cfg["engine"].get(
                           "lite_prune_audit")},
            "config_sha256": sha256(CONFIG_PATH),
            "theory_sha256": sha256(THEORY_PATH),
            "design_sha256": sha256(DESIGN_PATH),
            "wall_total_sec": round(time.time() - started, 3),
        }
        write_json(out_dir / "summary.json", summary)
        log(f"{summary['variant']} outer AUPRC={auprc:.7f} AUC={auc:.7f}; "
            f"final fit used no OOF; L1 fits performed={l1_fits_performed} "
            f"saved={l1_fits_saved}")
        return summary

    if not _LITE.get("dystack", True):
        stack_enabled = False
        detection = {
            "rule": "skipped",
            "skipped_reason": "lite_dystack_off",
            "stack_enabled": False,
            "stacked_overfitting": None,
            "delta_holdout": None,
            "se_bootstrap": None,
        }
        dystack_drops = []
        write_json(out_dir / "dystack.json", {
            "protocol": "skipped",
            "rule": "skipped",
            "rule_source": "lite_dystack_off",
            "ag_stacked_overfitting_detection": detection,
            "stack_enabled": False,
            "time_limit_dropped": [],
            "outer_test_read": False,
        })
        log("DyStack: skipped (--lite-dystack off); shipping WE_L2")
    else:
        holdout_frac = float(cfg["dystack"]["holdout_frac"])
        random_state = int(cfg["dystack"]["random_state"])
        ds_protocol = str(cfg["dystack"].get("protocol", "autogluon_1_4_holdout"))
        if ds_protocol == "chronological_tail_holdout":
            frame = (final_train.sort_values(time_col, kind="stable")
                     if time_col and time_col in final_train.columns
                     else final_train)
            cut = int(round(len(frame) * (1.0 - holdout_frac)))
            ds_train = frame.iloc[:cut].copy().reset_index(drop=True)
            ds_holdout = frame.iloc[cut:].copy().reset_index(drop=True)
        else:
            ds_train, ds_holdout = train_test_split(
                final_train, test_size=holdout_frac, shuffle=True,
                random_state=random_state, stratify=final_train[label])
            ds_train = ds_train.copy().reset_index(drop=True)
            ds_holdout = ds_holdout.copy().reset_index(drop=True)
        log(f"DyStack sub-fit: inner={len(ds_train)} holdout={len(ds_holdout)}")
        ds_fold_log = FoldLog(out_dir / "dystack_folds.jsonl")
        ds_l1_surfaces = {
            f"l1_{key}": list(dict.fromkeys(base_cols(key) + plan[key]))
            for key, _, _ in l1_library}
        dystack_drops: list = []
        ds_l1 = fit_layer(
            train=ds_train, apply=ds_holdout, label=label, library=l1_library,
            surfaces=ds_l1_surfaces, prefix="l1", layer_index=1, cfg=cfg,
            fold_log=ds_fold_log, stage="dystack_factorized_l1",
            time_limit_audit=dystack_drops)
        ds_stack = _attach_stack_columns(ds_train, ds_holdout, ds_l1)
        ds_l2_surfaces = {
            f"l2_{key}": list(dict.fromkeys(features + ds_stack))
            for key, _, _ in l2_library}
        ds_l2 = fit_layer(
            train=ds_train, apply=ds_holdout, label=label, library=l2_library,
            surfaces=ds_l2_surfaces, prefix="l2", layer_index=2, cfg=cfg,
            fold_log=ds_fold_log, stage="dystack_factorized_l2",
            time_limit_audit=dystack_drops)
        y_ds_train = ds_train[label].to_numpy(np.int8, copy=False)
        y_ds_holdout = ds_holdout[label].to_numpy(np.int8, copy=False)
        ds_we_l2, ds_we_l3 = build_ensembles(ds_l1, ds_l2, y_ds_train, steps)
        ds_search_fit = {
            "records": ds_l1 + ds_l2,
            "we_l2_oof": ds_we_l2["oof"],
            "we_l2_prediction": ds_we_l2["prediction"],
            "we_l3_oof": ds_we_l3["oof"],
            "we_l3_prediction": ds_we_l3["prediction"],
            "train_labels": y_ds_train,
        }
        detection = ag_dystack_decision(
            ds_search_fit, y_ds_holdout,
            rule=str(cfg["dystack"].get("rule", "legacy_ag")),
            k=float(cfg["dystack"].get("k", 1.0)),
            bootstrap_repeats=int(cfg["dystack"].get("bootstrap_repeats", 1000)),
            bootstrap_seed=int(cfg["dystack"].get("bootstrap_seed", 20260802)))
        stack_enabled = bool(cfg["dystack"]["enabled"]) and bool(
            detection["stack_enabled"])
        holdout_dir = out_dir / "dystack_holdout_predictions"
        holdout_dir.mkdir(parents=True, exist_ok=True)
        for member, vector in holdout_prediction_table(ds_search_fit).items():
            np.save(holdout_dir / f"{member}.npy", vector)
        np.save(out_dir / "dystack_holdout_labels.npy", y_ds_holdout)
        pd.DataFrame([{
            "member": r["member"], "model_key": r["model_key"],
            "layer": r["layer"], "oof_auprc": r["oof_auprc"],
            "fit_sec": r["fit_sec"], "n_columns": r["design"]["n_columns"],
            "learner": r["design"]["learner"],
        } for r in ds_l1 + ds_l2]).to_csv(
            out_dir / "dystack_member_table.csv", index=False)
        write_json(out_dir / "dystack.json", {
            "protocol": str(cfg["dystack"]["protocol"]),
            "rule": str(cfg["dystack"].get("rule", "legacy_ag")),
            "rule_source": ("forge_v13.dystack.ag_dystack_decision "
                            "(V9 r6 rule + V19 complexity gate)"),
            "holdout_frac": holdout_frac,
            "random_state": random_state,
            "n_subfit_train": len(ds_train),
            "n_holdout": len(ds_holdout),
            "ag_stacked_overfitting_detection": detection,
            "stack_enabled": stack_enabled,
            "time_limit_dropped": dystack_drops,
            "outer_test_read": False,
        })
        log(f"DyStack rule={detection['rule']} "
            f"delta={detection['delta_holdout']:+.5f} "
            f"se={detection.get('se_bootstrap')} "
            f"stacked_overfitting={detection['stacked_overfitting']} "
            f"stack_enabled={stack_enabled} "
            f"flipped_vs_legacy={detection.get('flipped_vs_legacy')}")
        del ds_train, ds_holdout, ds_l1, ds_l2, ds_we_l2, ds_we_l3
        gc.collect()

    final_fold_log = FoldLog(out_dir / "final_folds.jsonl")
    final_drops: list = []
    final_l1_surfaces = {
        f"l1_{key}": list(dict.fromkeys(base_cols(key) + plan[key]))
        for key, _, _ in l1_library}
    final_l1 = fit_layer(
        train=final_train, apply=final_test, label=label, library=l1_library,
        surfaces=final_l1_surfaces, prefix="l1", layer_index=1,
        cfg=cfg, fold_log=final_fold_log, stage="final_factorized_l1",
        time_limit_audit=final_drops)
    final_we_l2 = _weighted_ensemble(final_l1, final_y, steps)
    final_l2: list[dict] = []
    final_we_l3 = None
    if stack_enabled:
        final_stack = _attach_stack_columns(final_train, final_test, final_l1)
        final_l2_surfaces = {
            f"l2_{key}": list(dict.fromkeys(features + final_stack))
            for key, _, _ in l2_library}
        final_l2 = fit_layer(
            train=final_train, apply=final_test, label=label,
            library=l2_library, surfaces=final_l2_surfaces, prefix="l2",
            layer_index=2, cfg=cfg, fold_log=final_fold_log,
            stage="final_factorized_l2", time_limit_audit=final_drops)
        passthrough = _ensemble_record(
            "weighted_ensemble_l2", "we_l2", final_we_l2)
        final_we_l3 = _weighted_ensemble(
            final_l2 + [passthrough], final_y, steps)
    chosen = final_we_l3 if final_we_l3 is not None else final_we_l2
    fallback_audit = None
    if _LITE.get("fallback_gate"):
        nf_surfaces = {f"l1_{key}": list(base_cols(key))
                       for key, _, _ in l1_library}
        nf_l1 = fit_layer(
            train=final_train, apply=final_test, label=label,
            library=l1_library, surfaces=nf_surfaces, prefix="l1",
            layer_index=1, cfg=cfg, fold_log=final_fold_log,
            stage="final_nofactor_l1", time_limit_audit=final_drops)
        nf_we = _weighted_ensemble(nf_l1, final_y, steps)
        y_test_fb = final_test[label].to_numpy(np.int8, copy=False)
        fallback_audit = {
            "oof_with_factors": float(chosen["oof_auprc"]),
            "oof_no_factors": float(nf_we["oof_auprc"]),
            "shipped": ("no_factors" if nf_we["oof_auprc"]
                        > chosen["oof_auprc"] else "with_factors"),
            "outer_with_factors_DIAGNOSTIC": round(float(
                average_precision_score(
                    y_test_fb, np.asarray(chosen["prediction"],
                                          np.float64))), 7),
            "outer_no_factors_DIAGNOSTIC": round(float(
                average_precision_score(
                    y_test_fb, np.asarray(nf_we["prediction"],
                                          np.float64))), 7),
        }
        if fallback_audit["shipped"] == "no_factors":
            chosen = nf_we
            log("fallback gate: shipped NO-FACTOR ensemble "
                f"(oof {fallback_audit['oof_no_factors']:.5f} > "
                f"{fallback_audit['oof_with_factors']:.5f})")
        write_json(out_dir / "fallback_gate.json", fallback_audit)
    final_dir = out_dir / "final"
    save_records(final_dir / "l1", final_l1)
    if final_l2:
        save_records(final_dir / "l2", final_l2)
    np.save(final_dir / "oof.npy",
            np.asarray(chosen["oof"], dtype=np.float32))
    np.save(final_dir / "test_pred.npy",
            np.asarray(chosen["prediction"], dtype=np.float32))
    write_json(final_dir / "ensemble.json", {
        "stack_enabled": stack_enabled,
        "final_ensemble": ("weighted_ensemble_l3" if stack_enabled
                           else "weighted_ensemble_l2"),
        "weight_source": "outer_train_oof_only",
        "we_l2": {"oof_auprc": final_we_l2["oof_auprc"],
                  "weights": final_we_l2["weights"],
                  "selection": final_we_l2.get("selection")},
        "we_l3": (None if final_we_l3 is None else
                  {"oof_auprc": final_we_l3["oof_auprc"],
                   "weights": final_we_l3["weights"],
                   "selection": final_we_l3.get("selection")}),
    })

    y_test = final_test[label].to_numpy(np.int8, copy=False)
    auprc = float(average_precision_score(y_test, chosen["prediction"]))
    auc = float(roc_auc_score(y_test, chosen["prediction"]))
    member_outer_auprc = {
        str(r["member"]): round(float(average_precision_score(
            y_test, np.asarray(r["apply"], np.float64))), 7)
        for r in final_l1 if len(r.get("apply", [])) == len(y_test)}
    accepted_names = sorted({name for cols in plan.values()
                             for name in cols}) if isinstance(plan, dict) else []
    factor_provenance = {
        "grid": sum(1 for n in accepted_names
                    if n.startswith(("vg_", "vg3_", "vg4_"))),
        "focus": sum(1 for n in accepted_names if n.startswith("vf_")),
        "template": sum(1 for n in accepted_names if n.startswith("v21_")),
        "other": sum(1 for n in accepted_names
                     if not n.startswith(("vg_", "vg3_", "vg4_", "vf_",
                                          "v21_"))),
    }
    references = dict(cfg.get("references", {}).get(dataset, {}) or {})
    reference_deltas = {
        f"delta_vs_{name}": round(auprc - float(value), 7)
        for name, value in references.items()
    }
    k_add_values = [entry["k_add"] for row in ledger
                    for entry in row["per_model"].values()]
    k_drop_values = [entry["k_drop"] for row in ledger
                     for entry in row["per_model"].values()]
    summary = {
        "method": METHOD,
        "stage": "forge",
        "dataset": dataset,
        "outer_split_seed": int(dataset_cfg["split_seed"]),
        "representation": str(representation_choice),
        "representation_probe": representation_audit.get("probe_scores"),
        "representation_scope": str(_lite("rep_scope", "shared")),
        "representation_seat_map": (seat_plan or {}).get("seat_rep"),
        "member_outer_auprc": member_outer_auprc,
        "factor_provenance": factor_provenance,
        "fallback_gate": fallback_audit,
        "graph": {"l1_seats": [name for name, _, _ in l1_library],
                  "l2_seats": [name for name, _, _ in l2_library]},
        "final_auprc": round(auprc, 7),
        "final_auc": round(auc, 7),
        "references": references,
        "reference_deltas": reference_deltas,
        "factor_injection": inject_audit,
        "s4_depth3": s4_plan_audit,
        "outer_test_evaluations": 1,
        "outer_train_rows": len(final_train),
        "outer_test_rows": len(final_test),
        "outer_split": outer_split,
        "search_split": split,
        "n_l1": len(final_l1),
        "n_l2": len(final_l2),
        "stack_enabled": stack_enabled,
        "dystack_detection": detection,
        "theory": {
            "entry": ("s4_approximate_global_pair_bh"
                      if s4_depth3 else
                      "forge_lite_v2_collinearity_then_bh"
                      if _LITE.get("v2_selection") else
                      "greedy_orthogonalized_residual_corr_noise_floor"),
            "entry_alpha": alpha_entry,
            "entry_noise_floor": float(floor),
            "entry_k_cap": int(entry_cfg["k_cap"]),
            "bh_family": _lite("bh_family", "cumulative"),
            "exit": ("disabled_inside_s4_depths"
                     if s4_depth3 else
                     "oof_permutation_importance_discovery"),
            "exit_tau_drop": float(exit_cfg["tau_drop"]),
            "exit_d_max": int(exit_cfg["d_max_per_model_per_round"]),
            "discovery_rows": int(len(disc)),
            "discovery_n_pos": n_pos,
            "bh_role": ("global_seat_factor_decision_rule"
                        if s4_depth3 else
                        "decision_rule_v2" if _LITE.get("v2_selection")
                        else "audit_only"),
        },
        "rounds": {
            "max_rounds": max_rounds,
            "accepted_rounds": accepted_rounds,
            "attempted_rounds": len(ledger),
            "stop_reason": stop_reason,
            "gate_block_weights": list(exp["gate_block_weights"][:max_rounds]),
            "gate_block_weights_source": exp.get("gate_block_weights_source",
                                                 "config"),
            "gate_blocks_consumed": gate_blocks_consumed,
            "gate_readings": [row["gate"] for row in ledger],
            "no_change_rounds": no_change_rounds,
            "refit_l1_by_round": refit_l1_by_round,
            "reused_l1_by_round": reused_l1_by_round,
            "l1_fits_performed": l1_fits_performed,
            "l1_fits_saved": l1_fits_saved,
            "seat_pruning": pruned_log,
            "k_add_by_round_model": {
                str(row["round"]): {key: entry["k_add"]
                                    for key, entry in row["per_model"].items()}
                for row in ledger},
            "k_add_total": int(np.sum(k_add_values)) if k_add_values else 0,
            "k_drop_total": int(np.sum(k_drop_values)) if k_drop_values else 0,
            "k_add_histogram": {
                str(value): int(count) for value, count in zip(
                    *np.unique(k_add_values, return_counts=True))
            } if k_add_values else {},
        },
        "n_factor_bank": len(bank_names),
        "factor_bank": bank_names,
        "model_factor_plan": plan,
        "accepted_llm_factors": surface_llm_names,
        "intermediate_llm_factors": intermediate_llm_names,
        "blacklist": blacklist.to_dict(),
        "llm": {
            **llm.audit(),
            "domain_context_supplied": bool(domain_context),
            "domain_context_sha256": (
                hashlib.sha256(domain_context.encode("utf-8")).hexdigest()
                if domain_context else None),
        },
        "llm_caps": {
            "pointwise_per_model_per_round": int(
                llm_cfg["max_pointwise_per_model_per_round"]),
            "aggregate_per_model_per_round": int(
                llm_cfg["max_aggregate_per_model_per_round"]),
        },
        "input_audit": input_audit,
        "final_we_l2_oof_auprc": final_we_l2["oof_auprc"],
        "final_we_l3_oof_auprc": (None if final_we_l3 is None
                                  else final_we_l3["oof_auprc"]),
        "final_we_l2_weights": final_we_l2["weights"],
        "final_we_l3_weights": (None if final_we_l3 is None
                                else final_we_l3["weights"]),
        "final_we_l2_blend_selection": final_we_l2.get("selection"),
        "final_we_l3_blend_selection": (
            None if final_we_l3 is None else final_we_l3.get("selection")),
        "gpu_mlp": gpu,
        "time_limit": {
            "member_time_limit_sec": cfg["engine"].get(
                "member_time_limit_sec", 2700),
            "semantics": "ag_soft_member_cap_plus_pool_batch_deadline",
            "baseline_dropped_tolerated": sorted(baseline_dropped),
            "round_drops": [row.get("time_limit_dropped", [])
                            for row in ledger],
            "dystack_dropped": dystack_drops,
            "final_dropped": final_drops,
        },
        "stage_wall_sec": {
            "baseline": round(
                float(baseline_summary.get("wall_total_sec") or 0.0), 3),
            "search_rounds_total": search_wall_sec,
            "per_round": {str(row["round"]): row.get("wall_sec")
                          for row in ledger},
            "final_rebuild": round(time.time() - final_started, 3),
            "final_rebuild_includes_dystack": bool(_LITE.get("dystack", True)),
            "forge_stage_total": round(time.time() - started, 3),
            "baseline_plus_forge": round(
                float(baseline_summary.get("wall_total_sec") or 0.0)
                + (time.time() - started), 3),
            "llm": llm_wall_sec(out_dir / "llm_calls.jsonl"),
            "llm_work_sec": llm_wall_sec(out_dir / "llm_calls.jsonl"),
            "llm_concurrent_phase_wall_sec": round(llm_phase_wall_sec, 3),
        },
        "engine": {
            "seed": engine_seed,
            "bag_folds": int(cfg["engine"]["bag_folds"]),
            "blend_steps": steps,
            "member_time_limit_sec": cfg["engine"].get(
                "member_time_limit_sec", 2700),
            "fit_workers": _fit_workers(),
            "lgbm_threads": os.environ.get("LGBM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "contended_lane": os.environ.get("FORGE_LANE"),
            "lite": cfg["engine"].get("lite"),
            "lite_prune_audit": cfg["engine"].get("lite_prune_audit"),
        },
        "config_sha256": sha256(CONFIG_PATH),
        "theory_sha256": sha256(THEORY_PATH),
        "design_sha256": sha256(DESIGN_PATH),
        "wall_total_sec": round(time.time() - started, 3),
    }
    write_json(out_dir / "summary.json", summary)
    reference_note = " ".join(
        f"{name}={delta:+.7f}" for name, delta in reference_deltas.items())
    log(f"outer AUPRC={auprc:.7f} AUC={auc:.7f} "
        f"stop={stop_reason} {reference_note}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--stage", choices=("baseline", "forge"),
                        required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--stub-llm", action="store_true")
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument(
        "--mlp-devices", type=str, metavar="cuda:0,cuda:1")
    parser.add_argument(
        "--gate-block-weights", type=str, metavar="W1,W2,...")
    parser.add_argument(
        "--seed", type=int)
    parser.add_argument(
        "--outer-split-seed", type=int)
    parser.add_argument(
        "--l1", type=str, choices=[str(k) for k in L1_SEAT_SETS])
    parser.add_argument(
        "--l2", type=str, choices=[str(k) for k in L2_SEAT_SETS])
    parser.add_argument(
        "--domain-context", type=Path)
    parser.add_argument(
        "--inject-factors", type=Path, metavar="PLAN_JSON")
    parser.add_argument(
        "--inject-mode", choices=("surface", "bank", "gated_surface"),
        default="surface")
    parser.add_argument(
        "--oof-folds", type=int, choices=range(2, 33), metavar="K")

    lite = parser.add_argument_group(
        "FORGE-lite (CHANGES THE NUMBERS -- see LITE_DESIGN.md)")
    lite.add_argument(
        "--lite-v21", action="store_true")
    lite.add_argument(
        "--template-count", type=int, metavar="T", default=None)
    lite.add_argument(
        "--template-cap", type=int, metavar="N", default=None)
    lite.add_argument(
        "--template-per-cap", type=int, metavar="N", default=None)
    lite.add_argument(
        "--idiom-sweep", action="store_true")
    lite.add_argument(
        "--ruler-fusion", action="store_true")
    lite.add_argument(
        "--ruler-top", type=int, metavar="M", default=None)
    lite.add_argument(
        "--focus-beam", action="store_true")
    lite.add_argument(
        "--perceive-v2", action="store_true")
    lite.add_argument(
        "--s4-depth3", action="store_true")
    lite.add_argument(
        "--s4-funnel", action="store_true")
    lite.add_argument(
        "--s4-rolling-windows", metavar="W1,W2,...", default=None)
    lite.add_argument(
        "--s4-rolling-aggs", metavar="A1,A2,...", default=None)
    lite.add_argument(
        "--v34-router", action="store_true")
    lite.add_argument(
        "--s4-candidate-caps", metavar="N1,N2,N3", default=None)
    lite.add_argument(
        "--s4-survivor-caps", metavar="K1,K2,K3", default=None)
    lite.add_argument(
        "--s4-bh-q", metavar="Q1,Q2,Q3", default=None)
    lite.add_argument(
        "--bank-no-te", action="store_true")
    lite.add_argument(
        "--s4-eval", choices=("random", "chrono"), default=None)
    lite.add_argument(
        "--s4-entry", choices=("bh", "noise_floor"), default=None)
    lite.add_argument(
        "--fallback-gate", action="store_true")
    lite.add_argument(
        "--focus-grid", action="store_true")
    lite.add_argument(
        "--idiom-order", type=int, choices=(2, 3, 4), default=None)
    lite.add_argument(
        "--rep-scope", choices=("shared", "seat"), default=None)
    lite.add_argument(
        "--template-scope", choices=("shared", "seat"), default=None)
    lite.add_argument(
        "--lite-v19", action="store_true")
    lite.add_argument(
        "--proposal-cycles", type=int, metavar="K", default=None)
    lite.add_argument(
        "--entry-shadow", choices=("on", "off"), default=None)
    lite.add_argument(
        "--shadow-frac", type=float, metavar="F", default=None)
    lite.add_argument(
        "--llm-draws", type=int, metavar="D", default=None)
    lite.add_argument(
        "--composition-sweep", type=int, metavar="N", default=None)
    lite.add_argument(
        "--shadow-rule", choices=("harm_only", "persistence"), default=None)
    lite.add_argument(
        "--lite-v12", action="store_true")
    lite.add_argument(
        "--lite-v11", action="store_true")
    lite.add_argument(
        "--neg-subsample", type=float, metavar="K", default=None)
    lite.add_argument(
        "--calendar-family", action="store_true")
    lite.add_argument(
        "--entry-weight", choices=("none", "balanced", "positive"),
        default=None)
    lite.add_argument(
        "--parallel-llm", type=int, metavar="N", default=None)
    lite.add_argument(
        "--lite-v10", action="store_true")
    lite.add_argument(
        "--lite-prune-weight", type=float, metavar="FRAC")
    lite.add_argument(
        "--lite-prune-min-seats", type=int, metavar="N", default=None)
    lite.add_argument(
        "--entry-rule", choices=("bh", "noise_floor"), default=None)
    lite.add_argument(
        "--lite-v9", action="store_true")
    lite.add_argument(
        "--lite-l2", choices=("off", "full"), default=None)
    lite.add_argument(
        "--bh-family", choices=("cumulative", "per_round"), default=None)
    lite.add_argument(
        "--lite-v8", action="store_true")
    lite.add_argument(
        "--lite-v6", action="store_true")
    lite.add_argument(
        "--lite-v5", action="store_true")
    lite.add_argument(
        "--lite-v4", action="store_true")
    lite.add_argument(
        "--lite-v3", action="store_true")
    lite.add_argument(
        "--lite-v2", action="store_true")
    lite.add_argument(
        "--lite", action="store_true")
    lite.add_argument(
        "--representation",
        choices=("probe", "union", "e1", "identity", "raw"), default=None)
    lite.add_argument(
        "--probe-folds", type=int, metavar="K", default=None)
    lite.add_argument(
        "--salvage-max-refits", type=int, metavar="N", default=None)
    lite.add_argument(
        "--gate-scope", choices=("round", "seat"), default=None)
    lite.add_argument(
        "--lite-search-l2", choices=("on", "off"), default=None)
    lite.add_argument(
        "--lite-dystack", choices=("on", "off"), default=None)
    lite.add_argument(
        "--lite-mlp-cap", type=int, metavar="ROWS")
    lite.add_argument(
        "--lite-prune-l1", type=int, metavar="M")
    lite.add_argument(
        "--lite-search-folds", type=int, metavar="K")
    lite.add_argument(
        "--lite-llm-pointwise", type=int, metavar="N")
    lite.add_argument(
        "--lite-llm-aggregate", type=int, metavar="N")
    return parser


def resolve_lite(args) -> dict:
    lite: dict = {}

    def v2():
        lite["v2_selection"] = True

    def v3():
        v2()
        lite.update(v3=True, llm_pointwise=2, llm_aggregate=1)

    def shared_batch():
        v2()
        lite.update(shared_llm=True, llm_aggregate=0)

    def v4():
        shared_batch()
        lite.update(v4=True, llm_pointwise=6)

    def v5():
        shared_batch()
        lite.update(v5=True, llm_pointwise=8)

    def v6():
        v5()
        lite["v6"] = True

    def v8():
        v6()
        lite.update(v8=True, search_bag_folds=2)

    def v9():
        v8()
        lite["v9"] = True

    def v10():
        v9()
        lite.update(v10=True, prune_weight=1.0, prune_min_seats=4,
                    entry_rule="noise_floor")

    def v12():
        v11()
        lite.update(v12=True, calendar_family=True)

    def v11():
        v10()
        lite.update(v11=True, parallel_llm=8)
        lite.pop("shared_llm", None)

    def v19():
        v11()
        lite.update(v19=True, dystack=False, search_l2=False,
                    gate_scope="seat", proposal_cycles=3, entry_shadow=True,
                    shadow_rule="persistence")

    def v21():
        v19()
        lite.update(v21=True, template_channel=True,
                    llm_pointwise=0, llm_aggregate=0,
                    template_count=4, template_cap=240, template_per_cap=60)

    for flag, builder in (("lite_v21", v21), ("lite_v19", v19),
                          ("lite_v12", v12),
                          ("lite_v11", v11),
                          ("lite_v10", v10),
                          ("lite_v9", v9),
                          ("lite_v8", v8), ("lite_v6", v6),
                          ("lite_v5", v5), ("lite_v4", v4), ("lite_v3", v3),
                          ("lite_v2", v2)):
        if getattr(args, flag, False):
            builder()
            break
    if getattr(args, "lite", False):
        lite.update(mlp_fit_cap=200_000, prune_l1=8, search_bag_folds=4)
    if lite.get("v9") and not lite.get("v10"):
        lite.setdefault("l2_policy", "full")
        lite.setdefault("bh_family", "cumulative")
    if lite.get("v10") or lite.get("v11") or lite.get("v12"):
        lite.setdefault("l2_policy", "full")
        lite.setdefault("bh_family", "cumulative")
    l2_policy = getattr(args, "lite_l2", None)
    if l2_policy is not None:
        lite["l2_policy"] = str(l2_policy)
    bh_family = getattr(args, "bh_family", None)
    if bh_family is not None:
        lite["bh_family"] = str(bh_family)
    dystack_flag = getattr(args, "lite_dystack", None)
    if dystack_flag is not None:
        lite["dystack"] = (dystack_flag == "on")
    search_l2_flag = getattr(args, "lite_search_l2", None)
    if search_l2_flag is not None:
        lite["search_l2"] = (search_l2_flag == "on")
    if getattr(args, "v34_router", False):
        lite["v34_router"] = True
    gate_scope = getattr(args, "gate_scope", None)
    if gate_scope is not None:
        lite["gate_scope"] = str(gate_scope)
    proposal_cycles = getattr(args, "proposal_cycles", None)
    if proposal_cycles is not None:
        lite["proposal_cycles"] = int(proposal_cycles)
    entry_shadow = getattr(args, "entry_shadow", None)
    if entry_shadow is not None:
        lite["entry_shadow"] = (entry_shadow == "on")
    shadow_frac = getattr(args, "shadow_frac", None)
    if shadow_frac is not None:
        lite["shadow_frac"] = float(shadow_frac)
    shadow_rule = getattr(args, "shadow_rule", None)
    if shadow_rule is not None:
        lite["shadow_rule"] = str(shadow_rule)
    comp_sweep = getattr(args, "composition_sweep", None)
    if comp_sweep is not None:
        lite["comp_sweep"] = int(comp_sweep)
    for flag in ("template_count", "template_cap", "template_per_cap"):
        value = getattr(args, flag, None)
        if value is not None:
            lite[flag] = int(value)
            lite.setdefault("template_channel", True)
    template_scope = getattr(args, "template_scope", None)
    if template_scope is not None:
        lite["template_scope"] = str(template_scope)
        lite.setdefault("template_channel", True)
    if getattr(args, "idiom_sweep", False):
        lite["idiom_sweep"] = True
    idiom_order = getattr(args, "idiom_order", None)
    if idiom_order is not None:
        lite["idiom_order"] = int(idiom_order)
        lite.setdefault("idiom_sweep", True)
    if getattr(args, "focus_grid", False):
        lite["focus_grid"] = True
    if getattr(args, "fallback_gate", False):
        lite["fallback_gate"] = True
    if getattr(args, "bank_no_te", False):
        lite["bank_no_te"] = True
    if getattr(args, "focus_beam", False):
        lite["focus_beam"] = True
        lite.setdefault("focus_grid", True)
    if getattr(args, "perceive_v2", False):
        lite["perceive_v2"] = True
        lite.setdefault("template_channel", True)
    if getattr(args, "s4_funnel", False):
        args.s4_depth3 = True
    if getattr(args, "s4_depth3", False):
        lite.update({
            "s4_depth3": True,
            "template_channel": True,
            "perceive_v2": True,
            "search_l2": False,
            "entry_shadow": bool(getattr(args, "s4_funnel", False)),
        })
        if getattr(args, "s4_funnel", False):
            lite["s4_funnel"] = True
            lite.setdefault("shadow_rule", "persistence")
    for flag in ("s4_candidate_caps", "s4_survivor_caps", "s4_bh_q",
                 "s4_entry", "s4_eval", "s4_rolling_windows",
                 "s4_rolling_aggs"):
        value = getattr(args, flag, None)
        if value is not None:
            lite[flag] = str(value)
    if getattr(args, "ruler_fusion", False):
        lite["ruler_fusion"] = True
    ruler_top = getattr(args, "ruler_top", None)
    if ruler_top is not None:
        lite["ruler_top"] = int(ruler_top)
        lite.setdefault("ruler_fusion", True)
    rep_scope = getattr(args, "rep_scope", None)
    if rep_scope is not None:
        lite["rep_scope"] = str(rep_scope)
        if lite["rep_scope"] == "seat":
            lite["representation"] = "union"
    llm_draws = getattr(args, "llm_draws", None)
    if llm_draws is not None:
        lite["llm_draws"] = int(llm_draws)
    salvage_cap = getattr(args, "salvage_max_refits", None)
    if salvage_cap is not None:
        lite["salvage_max_refits"] = int(salvage_cap)
    probe_folds = getattr(args, "probe_folds", None)
    if probe_folds is not None:
        lite["probe_folds"] = int(probe_folds)
    representation = getattr(args, "representation", None)
    if representation is not None and representation != "probe":
        lite["representation"] = str(representation)
    neg_sub = getattr(args, "neg_subsample", None)
    if neg_sub is not None:
        lite["neg_subsample"] = float(neg_sub)
    if getattr(args, "calendar_family", False):
        lite["calendar_family"] = True
    entry_weight = getattr(args, "entry_weight", None)
    if entry_weight is not None:
        lite["entry_weight"] = str(entry_weight)
    parallel_llm = getattr(args, "parallel_llm", None)
    if parallel_llm is not None:
        lite["parallel_llm"] = int(parallel_llm)
    entry_rule = getattr(args, "entry_rule", None)
    if entry_rule is not None:
        lite["entry_rule"] = str(entry_rule)
    prune_weight = getattr(args, "lite_prune_weight", None)
    if prune_weight is not None:
        lite["prune_weight"] = float(prune_weight)
    prune_min = getattr(args, "lite_prune_min_seats", None)
    if prune_min is not None:
        lite["prune_min_seats"] = int(prune_min)
    for flag, key in (("lite_mlp_cap", "mlp_fit_cap"),
                      ("lite_prune_l1", "prune_l1"),
                      ("lite_search_folds", "search_bag_folds"),
                      ("lite_llm_pointwise", "llm_pointwise"),
                      ("lite_llm_aggregate", "llm_aggregate")):
        value = getattr(args, flag, None)
        if value is not None:
            lite[key] = int(value)
    if lite.get("rep_scope") == "seat":
        lite["representation"] = "union"
    return {k: v for k, v in lite.items() if v is not None}


def main() -> None:
    args = build_parser().parse_args()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if os.environ.get("FORGE_HARM_K", "").strip():
        cfg["experiment"]["gate"]["harm_k"] = float(os.environ["FORGE_HARM_K"])
    if os.environ.get("FORGE_ENTRY_ALPHA", "").strip():
        cfg["selection"]["entry"]["alpha"] = float(
            os.environ["FORGE_ENTRY_ALPHA"])
    apply_device_override(cfg, args.device)
    apply_graph_override(cfg, args.l1, args.l2)
    if args.seed is not None:
        cfg["engine"]["seed"] = int(args.seed)
    if args.outer_split_seed is not None:
        cfg["experiment"]["outer_split_seed_override"] = int(
            args.outer_split_seed)
    if args.oof_folds is not None:
        cfg["engine"]["bag_folds"] = int(args.oof_folds)
    if args.mlp_devices:
        _MLP_DEVICES.clear()
        _MLP_DEVICES.extend(
            d.strip() for d in str(args.mlp_devices).split(",") if d.strip())
    if args.gate_block_weights is not None:
        weights = [int(part) for part in
                   str(args.gate_block_weights).replace(" ", "").split(",")
                   if part]
        if not weights or any(w < 1 for w in weights):
            raise SystemExit("--gate-block-weights must be positive integers")
        cfg["experiment"]["gate_block_weights"] = weights
        cfg["experiment"]["gate_block_weights_source"] = "cli_override"

    _LITE.clear()
    _LITE.update(resolve_lite(args))
    if _LITE:
        if "llm_pointwise" in _LITE:
            cfg["llm"]["max_pointwise_per_model_per_round"] = int(
                _LITE["llm_pointwise"])
        if "llm_aggregate" in _LITE:
            cfg["llm"]["max_aggregate_per_model_per_round"] = int(
                _LITE["llm_aggregate"])
        prune_m = _LITE.get("prune_l1")
        if prune_m:
            if args.baseline_dir is None:
                raise SystemExit(
                    "--lite-prune-l1 ranks seats using the frozen baseline, so "
                    "it needs --baseline-dir (and therefore --stage forge)")
            cfg["engine"]["lite_prune_audit"] = lite_prune_l1(
                cfg, args.baseline_dir, prune_m)
        cfg["engine"]["lite"] = dict(_LITE)
        log("FORGE-LITE ACTIVE -- this run is NOT comparable with a "
            "full-protocol run: " + json.dumps(_LITE, sort_keys=True))
        if prune_m:
            log("lite: L1 seats kept = "
                + ",".join(cfg["engine"]["lite_prune_audit"]["kept"]))

    if args.stage == "baseline":
        out = args.out or (RESULTS_DIR / args.dataset
                           / "baseline" / "r1")
        run_baseline(args.dataset, out, cfg)
    else:
        if args.baseline_dir is None:
            raise SystemExit("--baseline-dir is required for --stage forge")
        domain_context = (
            args.domain_context.read_text(encoding="utf-8")
            if args.domain_context is not None else None)
        out = args.out or (RESULTS_DIR / args.dataset
                           / "forge_v13" / "r1")
        run_forge(args.dataset, out, args.baseline_dir, cfg,
                  stub_llm=bool(args.stub_llm),
                  max_rounds_override=args.max_rounds,
                  domain_context=domain_context,
                  inject_factors=args.inject_factors,
                  inject_mode=str(args.inject_mode))


if __name__ == "__main__":
    os.environ.setdefault("LGBM_THREADS", "8")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    main()
