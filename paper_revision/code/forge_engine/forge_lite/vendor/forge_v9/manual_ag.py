from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from forge_v7 import hetero_stack
from forge_v7.hetero_stack import fold_plan
from portfolio import FoldLog, build_member_model, caruana_blend
from protocol import design_matrix


ENGINE_NAME = "forge-v9-manual-ag"


@dataclass(frozen=True)
class ManualAgConfig:
    seed: int = 20260717
    bag_folds: int = 5
    blend_steps: int = 100
    include_l2: bool = True


def _lgbm(*, learning_rate=.05, num_leaves=31, min_leaf=20,
          feature_fraction=1.0, extra_trees=False) -> dict:
    return dict(n_estimators=400, learning_rate=learning_rate,
                num_leaves=num_leaves, min_child_samples=min_leaf,
                colsample_bytree=feature_fraction, extra_trees=extra_trees)


def _cat(*, depth=6, learning_rate=.05, l2=3.0,
         grow_policy="SymmetricTree") -> dict:
    return dict(n_estimators=400, depth=depth, learning_rate=learning_rate,
                l2_leaf_reg=l2, grow_policy=grow_policy)


def _xgb(*, learning_rate=.05, max_depth=6, min_child_weight=1.0,
         colsample=1.0, enable_categorical=False) -> dict:
    return dict(n_estimators=400, learning_rate=learning_rate,
                max_depth=max_depth, min_child_weight=min_child_weight,
                colsample_bytree=colsample,
                enable_categorical=enable_categorical)


def _mlp(hidden: tuple[int, ...], *, learning_rate: float, alpha: float,
         activation="relu", dropout=.1, batchnorm=False, batch_size=128,
         epochs=200) -> dict:
    return dict(hidden_layer_sizes=hidden, learning_rate_init=learning_rate,
                alpha=alpha, activation=activation, dropout_prob=dropout,
                use_batchnorm=batchnorm, batch_size=batch_size, epochs=epochs)


def manual_l1_library() -> tuple[tuple[str, str, dict], ...]:
    rf = dict(n_estimators=300, min_samples_leaf=1, max_leaf_nodes=15000)
    return (
        ("lgbm_xt", "lgbm_base", _lgbm(extra_trees=True)),
        ("lgbm_default", "lgbm_base", _lgbm()),
        ("rf_gini", "random_forest", {**rf, "criterion": "gini"}),
        ("rf_entropy", "random_forest", {**rf, "criterion": "entropy"}),
        ("cat_default", "catboost", _cat()),
        ("xt_gini", "extra_trees", {**rf, "criterion": "gini"}),
        ("xt_entropy", "extra_trees", {**rf, "criterion": "entropy"}),
        ("fastai_default", "mlp", _mlp((200, 100), learning_rate=.001,
                                         alpha=1e-4, batch_size=256,
                                         epochs=30)),
        ("xgb_default", "xgboost", _xgb()),
        ("torch_default", "mlp", _mlp((128,) * 4, learning_rate=3e-4,
                                        alpha=1e-6)),
        ("lgbm_large", "lgbm_base", _lgbm(learning_rate=.03,
                                            num_leaves=128, min_leaf=3,
                                            feature_fraction=.9)),
        ("cat_r177", "catboost", _cat(depth=6,
                                       learning_rate=.06864209415792857,
                                       l2=2.1542798306067823)),
        ("torch_r79", "mlp", _mlp((108,) * 4,
                                    learning_rate=.002735937344002146,
                                    alpha=1.356433327634438e-12,
                                    activation="elu", dropout=.10077639529843717,
                                    batchnorm=True)),
        ("lgbm_r131", "lgbm_base", _lgbm(
            learning_rate=.012144796373999013, num_leaves=53, min_leaf=14,
            feature_fraction=.7023601671276614)),
        ("fastai_r191", "mlp", _mlp((800, 400), learning_rate=.008,
                                      alpha=1e-4, dropout=.23782946566604385,
                                      batch_size=256, epochs=43)),
        ("cat_r9", "catboost", _cat(depth=8,
                                     learning_rate=.031375015734637225,
                                     l2=2.7997999596449104,
                                     grow_policy="Depthwise")),
        ("lgbm_r96", "lgbm_base", _lgbm(
            learning_rate=.01518660230385841, num_leaves=16, min_leaf=48,
            feature_fraction=.5636931414546802, extra_trees=True)),
        ("torch_r22", "mlp", _mlp((213,) * 4,
                                    learning_rate=.0010474382260641949,
                                    alpha=5.594471067786272e-10,
                                    activation="elu", dropout=.11897478034205347)),
        ("xgb_r33", "xgboost", _xgb(
            learning_rate=.018063876087523967, max_depth=10,
            min_child_weight=.6028633586934382,
            colsample=.6917311125174739)),
        ("xt_r42", "extra_trees", dict(n_estimators=300,
                                         max_features=.75,
                                         max_leaf_nodes=18392,
                                         min_samples_leaf=1)),
        ("cat_r137", "catboost", _cat(depth=4,
                                       learning_rate=.04939557741379516,
                                       l2=4.559174625782161)),
        ("fastai_r102", "mlp", _mlp((200, 100), learning_rate=.008,
                                      alpha=1e-4, dropout=.10393466140748028,
                                      batch_size=2048, epochs=29)),
        ("cat_r13", "catboost", _cat(depth=8,
                                      learning_rate=.017301189655111057,
                                      l2=3.3274013177541373)),
        ("rf_r195", "random_forest", dict(n_estimators=300,
                                            max_features=.75,
                                            max_leaf_nodes=37308,
                                            min_samples_leaf=1)),
        ("lgbm_r188", "lgbm_base", _lgbm(
            learning_rate=.033929021353492905, num_leaves=127, min_leaf=6,
            feature_fraction=.8282601210460099, extra_trees=True)),
        ("fastai_r145", "mlp", _mlp((400, 200, 100),
                                      learning_rate=.008615195908919904,
                                      alpha=1e-4, dropout=.19220253419114286,
                                      batch_size=128, epochs=31)),
        ("xgb_r89", "xgboost", _xgb(
            learning_rate=.08775715546881824, max_depth=5,
            min_child_weight=.6294123374222513,
            colsample=.6628423832084077)),
        ("torch_r30", "mlp", _mlp((159,) * 5,
                                    learning_rate=.008507536855608535,
                                    alpha=1.8201539594953562e-6,
                                    activation="elu", dropout=.24622382571353768,
                                    batchnorm=True)),
        ("lgbm_r130", "lgbm_base", _lgbm(
            learning_rate=.04711573688184715, num_leaves=89, min_leaf=56,
            feature_fraction=.6245777099925497)),
        ("torch_r86", "mlp", _mlp((135,) * 5,
                                    learning_rate=.001631450730978947,
                                    alpha=3.867683394425807e-5,
                                    dropout=.09976801642258049)),
        ("cat_r50", "catboost", _cat(depth=4,
                                      learning_rate=.07092851311746352,
                                      l2=2.7018061518087038,
                                      grow_policy="Depthwise")),
        ("fastai_r11", "mlp", _mlp((800, 400), learning_rate=.008,
                                     alpha=1e-4, dropout=.4569532219038436,
                                     batch_size=128, epochs=31)),
        ("xgb_r194", "xgboost", _xgb(
            learning_rate=.09290221350439203, max_depth=7,
            min_child_weight=.8041986915994078,
            colsample=.9090166528779192, enable_categorical=True)),
        ("xt_r172", "extra_trees", dict(n_estimators=300,
                                          max_features=1.0,
                                          max_leaf_nodes=12845,
                                          min_samples_leaf=4)),
        ("cat_r69", "catboost", _cat(depth=5,
                                      learning_rate=.050294288910022224,
                                      l2=1.0457098345001241)),
    )


def manual_l2_library() -> tuple[tuple[str, str, dict], ...]:
    rf = dict(n_estimators=300, min_samples_leaf=1, max_leaf_nodes=15000)
    return (
        ("lgbm_default", "lgbm_base", _lgbm()),
        ("cat_default", "catboost", _cat()),
        ("xgb_r33", "xgboost", _xgb(
            learning_rate=.018063876087523967, max_depth=10,
            min_child_weight=.6028633586934382,
            colsample=.6917311125174739)),
        ("torch_r86", "mlp", _mlp(
            (135,) * 5, learning_rate=.001631450730978947,
            alpha=3.867683394425807e-5,
            dropout=.09976801642258049, batch_size=4096, epochs=60)),
        ("lgbm_large", "lgbm_base", _lgbm(learning_rate=.03, num_leaves=128,
                                          min_leaf=3, feature_fraction=.9)),
        ("lgbm_xt", "lgbm_base", _lgbm(extra_trees=True)),
        ("lgbm_r131", "lgbm_base", _lgbm(
            learning_rate=.012144796373999013, num_leaves=53, min_leaf=14,
            feature_fraction=.7023601671276614)),
        ("lgbm_r96", "lgbm_base", _lgbm(
            learning_rate=.01518660230385841, num_leaves=16, min_leaf=48,
            feature_fraction=.5636931414546802, extra_trees=True)),
        ("cat_r177", "catboost", _cat(depth=6,
                                      learning_rate=.06864209415792857,
                                      l2=2.1542798306067823)),
        ("cat_r9", "catboost", _cat(depth=8,
                                    learning_rate=.031375015734637225,
                                    l2=2.7997999596449104,
                                    grow_policy="Depthwise")),
        ("cat_r137", "catboost", _cat(depth=4,
                                      learning_rate=.04939557741379516,
                                      l2=4.559174625782161)),
        ("xgb_default", "xgboost", _xgb()),
        ("xgb_r89", "xgboost", _xgb(
            learning_rate=.08775715546881824, max_depth=5,
            min_child_weight=.6294123374222513,
            colsample=.6628423832084077)),
        ("rf_gini", "random_forest", {**rf, "criterion": "gini"}),
        ("xt_gini", "extra_trees", {**rf, "criterion": "gini"}),
        ("xt_entropy", "extra_trees", {**rf, "criterion": "entropy"}),
        ("torch_r30", "mlp", _mlp(
            (159,) * 5, learning_rate=.008507536855608535,
            alpha=1.8201539594953562e-6, activation="elu",
            dropout=.24622382571353768)),
        ("torch_r79", "mlp", _mlp(
            (108,) * 4, learning_rate=.002735937344002146,
            alpha=1.356433327634438e-12, activation="elu",
            dropout=.10077639529843717)),
        ("torch_default", "mlp", _mlp((128,) * 4, learning_rate=3e-4,
                                      alpha=1e-6)),
        ("torch_r22", "mlp", _mlp(
            (213,) * 4, learning_rate=.0010474382260641949,
            alpha=5.594471067786272e-10, activation="elu",
            dropout=.1189747803420534)),
    )

def _members(library: tuple[tuple[str, str, dict], ...], prefix: str):
    return [(f"{prefix}_{key}", key, learner, params)
            for key, learner, params in library]


def _native_oob_member(learner: str, params: dict, seed: int,
                       x_fit: np.ndarray, y_fit: np.ndarray,
                       x_apply: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        medians = np.nanmedian(x_fit, axis=0).astype(np.float32)
    medians[~np.isfinite(medians)] = 0.0
    x_train = np.where(np.isfinite(x_fit), x_fit, medians).astype(np.float32)
    x_external = np.where(np.isfinite(x_apply), x_apply, medians).astype(np.float32)
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
    return oob, apply


def _fit_manual_layer(train_frame: pd.DataFrame, apply_frame: pd.DataFrame,
                      y_fit: np.ndarray,
                      members: list[tuple[str, str, str, dict]],
                      feature_sets: dict[str, list[str]], *, layer_index: int,
                      forbidden: str, folds, seed: int, fold_log: FoldLog,
                      stage: str) -> dict:
    started = time.time()
    names: list[str] = []
    member_oof: dict[str, np.ndarray] = {}
    member_apply: dict[str, np.ndarray] = {}
    seconds: dict[str, float] = {}
    designs: dict[str, dict] = {}
    for member_name, model_key, learner, params in members:
        columns = list(dict.fromkeys(feature_sets[member_name]))
        if not columns or forbidden in columns:
            raise ValueError(f"invalid feature surface for {member_name}")
        x_fit = design_matrix(train_frame, columns)
        x_apply = design_matrix(apply_frame, columns)
        member_started = time.time()
        if learner in ("random_forest", "extra_trees"):
            oof, external = _native_oob_member(
                learner, params, seed, x_fit, y_fit, x_apply)
            oof_mode = "native_oob_single_child"
        elif learner == "mlp":
            from forge_v9.gpu_mlp import fit_member_gpu_with_validation

            oof = np.full(len(y_fit), np.nan, dtype=np.float32)
            external_folds = []
            for fold, (fit_idx, oof_idx) in enumerate(folds):
                fold_log.record(
                    stage=stage, member=member_name, seed=seed, fold=fold,
                    n_rows=len(y_fit), train_idx=fit_idx, oof_idx=oof_idx)
                fold_oof, fold_external = fit_member_gpu_with_validation(
                    params, seed, x_fit[fit_idx], y_fit[fit_idx],
                    x_fit[oof_idx], y_fit[oof_idx], x_apply)
                oof[oof_idx] = fold_oof
                external_folds.append(fold_external)
            external = np.mean(external_folds, axis=0).astype(np.float32)
            oof_mode = "shared_8fold_gpu_early_stopping"
        else:
            oof = np.full(len(y_fit), np.nan, dtype=np.float32)
            external_folds = []
            for fold, (fit_idx, oof_idx) in enumerate(folds):
                fold_log.record(
                    stage=stage, member=member_name, seed=seed, fold=fold,
                    n_rows=len(y_fit), train_idx=fit_idx, oof_idx=oof_idx)
                fold_oof, fold_external = hetero_stack.fit_member(
                    learner, params, seed, x_fit[fit_idx], y_fit[fit_idx],
                    [x_fit[oof_idx], x_apply], cap_seed=1000 * fold + seed)
                oof[oof_idx] = fold_oof
                external_folds.append(fold_external)
            if np.isnan(oof).any():
                raise RuntimeError(f"OOF coverage incomplete for {member_name}")
            external = np.mean(external_folds, axis=0).astype(np.float32)
            oof_mode = "shared_8fold"
        names.append(member_name)
        member_oof[member_name] = np.asarray(oof, dtype=np.float32)
        member_apply[member_name] = np.asarray(external, dtype=np.float32)
        seconds[member_name] = round(time.time() - member_started, 3)
        designs[member_name] = {
            "model": model_key,
            "learner": learner,
            "n_columns": len(columns),
            "columns": columns,
            "oof_mode": oof_mode,
        }
    return {
        "member_names": names,
        "member_oof": member_oof,
        "member_holdout": member_apply,
        "member_oof_auprc": {
            name: round(float(average_precision_score(y_fit, member_oof[name])), 7)
            for name in names
        },
        "fit_seconds": seconds,
        "truncated": [],
        "design_meta": designs,
        "layer_index": int(layer_index),
        "engine": ENGINE_NAME,
        "engine_seed": int(seed),
        "bag_folds": len(folds),
        "oof_policy": "ag_mixed_cv_and_native_oob",
        "elapsed_sec": round(time.time() - started, 3),
    }


def _records(layer: dict, prefix: str) -> list[dict]:
    records: list[dict] = []
    for member in layer["member_names"]:
        records.append({
            "member": member,
            "model_key": member.removeprefix(prefix + "_"),
            "layer": prefix,
            "oof": layer["member_oof"][member],
            "apply": layer["member_holdout"][member],
            "oof_auprc": float(layer["member_oof_auprc"][member]),
            "fit_sec": float(layer["fit_seconds"][member]),
            "design": layer["design_meta"][member],
        })
    return records


def _attach_stack_columns(train: pd.DataFrame, apply: pd.DataFrame,
                          records: list[dict]) -> list[str]:
    columns: list[str] = []
    for record in records:
        column = f"v9_pred_{record['member']}"
        train[column] = np.asarray(record["oof"], dtype=np.float32)
        apply[column] = np.asarray(record["apply"], dtype=np.float32)
        columns.append(column)
    return columns


def _weighted_ensemble(records: list[dict], y: np.ndarray,
                       steps: int) -> dict:
    matrix = np.column_stack([record["oof"] for record in records])
    weights = caruana_blend(matrix, y, max_steps=int(steps))
    oof = matrix @ weights
    apply_matrix = np.column_stack([record["apply"] for record in records])
    prediction = apply_matrix @ weights
    weight_map = {
        record["member"]: float(weight)
        for record, weight in zip(records, weights)
    }
    return {
        "weights": weight_map,
        "oof": np.asarray(oof, dtype=np.float32),
        "prediction": np.asarray(prediction, dtype=np.float32),
        "oof_auprc": float(average_precision_score(y, oof)),
        "selection": {"method": "caruana_greedy_with_replacement",
                      "steps": int(steps)},
    }


def _ensemble_record(name: str, layer: str, ensemble: dict) -> dict:
    return {
        "member": name,
        "model_key": name,
        "layer": layer,
        "oof": ensemble["oof"],
        "apply": ensemble["prediction"],
        "oof_auprc": ensemble["oof_auprc"],
        "fit_sec": 0.0,
        "design": {"n_columns": 0, "learner": "weighted_ensemble"},
    }


def fit_manual_ag(
    train: pd.DataFrame,
    apply: pd.DataFrame,
    *,
    features: list[str],
    label: str,
    config: ManualAgConfig,
    fold_log: FoldLog,
    prefit_l1_records: list[dict] | None = None,
) -> dict:
    started = time.time()
    columns = list(dict.fromkeys(features))
    if label in columns:
        raise ValueError("label must not be a model feature")
    y = train[label].to_numpy(np.int8, copy=False)
    l1_library = manual_l1_library()
    l2_library = manual_l2_library()
    folds = fold_plan(y, bag_folds=int(config.bag_folds), seed=int(config.seed))

    if prefit_l1_records is None:
        l1_surfaces = {f"l1_{key}": columns for key, _, _ in l1_library}
        l1 = _fit_manual_layer(
            train, apply, y, _members(l1_library, "l1"), l1_surfaces,
            layer_index=1, forbidden=label, folds=folds, seed=int(config.seed),
            fold_log=fold_log, stage="parity_l1",
        )
        l1_records = _records(l1, "l1")
        l1_elapsed_sec = l1["elapsed_sec"]
    else:
        expected = [f"l1_{key}" for key, _, _ in l1_library]
        actual = [str(record["member"]) for record in prefit_l1_records]
        if actual != expected:
            raise ValueError("prefit L1 portfolio/order does not match code_9 library")
        for record in prefit_l1_records:
            if len(record["oof"]) != len(train):
                raise ValueError(f"prefit OOF length mismatch: {record['member']}")
            if len(record["apply"]) != len(apply):
                raise ValueError(f"prefit apply length mismatch: {record['member']}")
        l1_records = list(prefit_l1_records)
        l1_elapsed_sec = 0.0
    we_l2 = _weighted_ensemble(l1_records, y, int(config.blend_steps))
    we_l2_record = _ensemble_record(
        "weighted_ensemble_l2", "we_l2", we_l2)
    stack_columns = _attach_stack_columns(train, apply, l1_records)

    l2_records: list[dict] = []
    l2 = None
    if config.include_l2:
        l2_columns = columns + stack_columns
        l2_surfaces = {f"l2_{key}": l2_columns for key, _, _ in l2_library}
        l2 = _fit_manual_layer(
            train, apply, y, _members(l2_library, "l2"), l2_surfaces,
            layer_index=2, forbidden=label, folds=folds, seed=int(config.seed),
            fold_log=fold_log, stage="parity_l2",
        )
        l2_records = _records(l2, "l2")

    we_l3 = None
    if l2_records:
        we_l3 = _weighted_ensemble(
            l2_records + [we_l2_record], y, int(config.blend_steps))
    final_ensemble = we_l3 if we_l3 is not None else we_l2
    records = l1_records + l2_records
    rows = []
    for record in records:
        if record["layer"] == "l1":
            weight = we_l2["weights"].get(record["member"], 0.0)
            weight_stage = "weighted_ensemble_l2"
        else:
            weight = (we_l3["weights"].get(record["member"], 0.0)
                      if we_l3 is not None else 0.0)
            weight_stage = "weighted_ensemble_l3"
        rows.append({
            "member": record["member"],
            "model_key": record["model_key"],
            "layer": record["layer"],
            "oof_auprc": record["oof_auprc"],
            "blend_weight": weight,
            "weight_stage": weight_stage,
            "fit_sec": record["fit_sec"],
            "n_columns": record["design"]["n_columns"],
            "learner": record["design"]["learner"],
        })
    active_l2 = [name for name, value in we_l2["weights"].items()
                 if value > 0]
    active_l3 = ([name for name, value in we_l3["weights"].items()
                  if value > 0] if we_l3 is not None else [])
    meta = {
        "engine": ENGINE_NAME,
        "seed": int(config.seed),
        "bag_folds": int(config.bag_folds),
        "blend_steps": int(config.blend_steps),
        "include_l2": bool(config.include_l2),
        "l1_config_list": [key for key, _, _ in l1_library],
        "l2_config_list": [key for key, _, _ in l2_library],
        "n_l1": len(l1_records),
        "n_l2": len(l2_records),
        "n_candidates": len(records),
        "weighted_ensemble_l2": {
            "active_members": active_l2,
            "weights": we_l2["weights"],
            "oof_auprc": we_l2["oof_auprc"],
        },
        "weighted_ensemble_l3": ({
            "active_members": active_l3,
            "weights": we_l3["weights"],
            "oof_auprc": we_l3["oof_auprc"],
        } if we_l3 is not None else None),
        "final_ensemble": ("weighted_ensemble_l3"
                           if we_l3 is not None else "weighted_ensemble_l2"),
        "l1_elapsed_sec": l1_elapsed_sec,
        "l1_reused": prefit_l1_records is not None,
        "l2_elapsed_sec": l2["elapsed_sec"] if l2 is not None else 0.0,
        "fit_sec": round(time.time() - started, 3),
        "n_features_l1": len(columns),
        "n_features_l2": len(columns) + len(stack_columns),
    }
    return {
        "train_labels": y,
        "prediction": final_ensemble["prediction"],
        "oof": final_ensemble["oof"],
        "we_l2_prediction": we_l2["prediction"],
        "we_l2_oof": we_l2["oof"],
        "we_l3_prediction": (we_l3["prediction"]
                             if we_l3 is not None else None),
        "we_l3_oof": we_l3["oof"] if we_l3 is not None else None,
        "member_rows": rows,
        "records": records,
        "meta": meta,
    }
