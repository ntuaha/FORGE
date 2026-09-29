"""FORGE end to end: S1 feature search -> S2 optional experts -> S3 freeze and test.

:func:`run_forge` chains the stage functions below (Algorithm 1 of the paper);
each stage function can also be called on its own (the walkthrough notebook
does this).

========================  ============================================  ==========================================
Stage                     Paper                                         Function
========================  ============================================  ==========================================
Partition                 Sec. IV-B, Eq. (1), Fig. 2                    :func:`forge.data.split_dataset`
Baseline (Raw)            Sec. IV-B, four-fold OOF, Eq. (2)             :func:`initialize_state`
Routing rule              Sec. IV-C, S1.2 "Routing"                     :func:`routing_rule`
R0                        Sec. IV-C, S1.1.1-S1.1.3, S1.2                :func:`run_initial_round`
CoF rounds R1-R3          Sec. IV-C, S1.1.1-S1.1.3, S1.2, Algorithm 1   :func:`run_cof_round`, :func:`run_cof_cycle`
CoF core selection        Sec. IV-C, S1.2, AP on the complete V         :func:`forge.validation.choose_configuration`
Core fit on I+V           Sec. IV-D (p_core^OOF) and IV-E               :func:`fit_core`
Experts                   Sec. IV-D, S2.1, S2.2, Eqs. (15)-(28)         :func:`run_experts`
Freeze and test           Sec. IV-E                                     :func:`freeze_and_evaluate`
========================  ============================================  ==========================================

Implementation notes (details the paper does not specify)
---------------------------------------------------------
* Search representation: target and frequency encodings of categorical-like
  columns are computed once on ``I`` with five shuffled out-of-fold folds and
  are available to candidate expressions (e.g. ``pay_0_oof_te``); they are
  not base learner columns.
* R0 shadow permutation importance permutes each probe column with a fixed
  permutation drawn over all rows of ``I`` and restricted to the held-out fold.
* Without an event-time column, the partition order serves as time for the
  trailing window and for entity histories.

Paper differences
-----------------
* The CoF rounds here were written from the paper's description. The paper's
  CoF numbers (all datasets except Credit Default and Twitterbot, where the
  routing rule skips CoF) come from the authors' experiment engine and its
  recorded LLM answers, which are not part of this release, so they are not
  reproduced exactly.
* Only the R0 proposals of Credit Default / LightGBM / seed 11 are shipped;
  other learners and seeds reuse them (see ``llm.frozen_replay_applies``).
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import llm, templates
from .admission import permutation_exit, shadow_admission
from .config import learner_params
from .data import Partition, canonical_hash, frame_hash, split_dataset, validate_dataset
from .datasets import load_dataset
from .experts import blend_prediction, recency_candidates, slice_candidates, target_encoding_candidates, \
    validate_experts
from .features import Surface, dependency_registry, make_representation_plan
from .history import detect_entity_columns, entity_history_library
from .learners import fit_oof
from .scoring import admission_threshold, discovery_rows, greedy_entry, score_pool
from .screening import drift_table, screen_candidates, trailing_window_check
from .validation import ValidationBudget, choose_configuration, harm_only_gate

log = logging.getLogger("forge")


# --------------------------------------------------------------------------- state
def initialize_state(I: pd.DataFrame, Vx: pd.DataFrame, cfg: dict) -> dict:
    """Baseline (Raw) configuration: four-fold OOF predictions on ``I`` and the mean of the fold models on ``V``.

    A *state* is a dict with the materialised ``surface``, model ``columns``,
    added ``features``, ``oof`` predictions on ``I`` and ``V_prediction``.
    """
    plan = make_representation_plan(I, cfg)
    surface = Surface(I, Vx, plan, {}, cfg)
    fitted = fit_oof(surface, plan["base"])
    return {"surface": surface, "columns": list(plan["base"]), "features": [],
            "oof": fitted["oof"], "V_prediction": fitted["apply"], "oof_ap": fitted["oof_ap"]}


def _state_from_fit(surface, columns, features, fitted) -> dict:
    return {"surface": surface, "columns": list(columns), "features": list(features), "oof": fitted["oof"],
            "V_prediction": fitted["apply"], "oof_ap": fitted["oof_ap"]}


def routing_rule(state: dict, V: pd.DataFrame, cfg: dict) -> dict:
    """Routing rule applied once before search (paper S1.2, "Routing").

    * R0 is skipped when more than 20% of the raw columns shift between ``I``
      and ``V`` (Eq. (8) for numeric columns, unseen-category rate > 20%
      for categorical columns).
    * CoF is skipped only when no such shift is detected, the dataset has no
      event-time column and no feature of the fixed entity-history library
      reaches the admission threshold ``tau``.

    Returns ``shift`` (the share of shifted columns > 20%), ``ratio`` (max
    library ``|S|`` / tau), ``tau``, ``run_r0``, ``run_cof``, the library and
    the diagnostics.
    """
    surface = state["surface"]
    drift = drift_table(surface.raw, V, cfg)
    shift = bool(drift["drifted"].mean() > cfg["route_shift_share"])
    library = entity_history_library(surface.raw, cfg, surface.plan)
    scores = score_pool(state, library, cfg)
    y = surface.raw[cfg["label"]].to_numpy()
    tau = admission_threshold(int(y[discovery_rows(len(y), cfg)].sum()), cfg["entry_alpha"])
    ratio = max([abs(r["score"]) for r in scores], default=0.) / tau
    has_time = cfg["time_col"] is not None
    return {"shift": shift, "ratio": float(ratio), "tau": tau, "run_r0": not shift,
            "run_cof": bool(shift or has_time or ratio >= 1), "has_event_time": has_time,
            "entity_columns": detect_entity_columns(surface.raw, cfg),
            "library": library, "library_scores": scores, "drift": drift}


@dataclass
class SearchState:
    """Mutable bookkeeping shared by the S1 rounds."""

    I: pd.DataFrame
    V: pd.DataFrame
    cfg: dict
    router: dict
    current: dict
    baseline: dict
    budget: ValidationBudget
    pool: dict = field(default_factory=dict)
    scorecard: list = field(default_factory=list)
    tried: set = field(default_factory=set)
    dropped: list = field(default_factory=list)
    accepted: list = field(default_factory=list)
    rounds: list = field(default_factory=list)

    @property
    def Vx(self) -> pd.DataFrame:
        return self.V.drop(columns=self.cfg["label"])

    @property
    def y_V(self) -> np.ndarray:
        return self.V[self.cfg["label"]].to_numpy(np.int8)


def _validate_round(search: SearchState, surface, pre: dict, columns, features, initial: bool) -> dict:
    """S1.2: refit the round update on ``I`` and apply the harm-only gate on the next block.

    Paper difference (code detail): the V prediction of a configuration is the
    mean of its four OOF fold models fitted on ``I``, not one refit on all of ``I``.
    """
    block, regions = search.budget.next_regions(initial_round=initial)
    fitted = fit_oof(surface, columns)
    passed, table = harm_only_gate(search.y_V, pre["V_prediction"], fitted["apply"], regions, search.cfg)
    if passed:
        search.current = _state_from_fit(surface, columns, features, fitted)
        search.accepted.append(search.current)
    else:
        search.current = pre
    return {"validation_block": block, "accepted": passed, "gate": table.to_dict("records")}


# --------------------------------------------------------------------------- R0
def run_initial_round(search: SearchState) -> dict:
    """R0: free-form proposal, PSI/AdvAUC screen, shadow admission, then S1.2 on both halves of a block."""
    cfg, current = search.cfg, search.current
    proposal = llm.propose_r0(current, cfg)
    specs = {k: v for k, v in proposal["specs"].items() if k not in search.tried and k not in current["columns"]}
    valid, validity = llm.validate_proposals(specs, current["surface"], cfg, limit=cfg["max_initial_candidates"])
    search.tried.update(valid)
    search.pool.update(valid)
    surface = Surface(search.I, search.Vx, current["surface"].plan, search.pool, cfg)
    screened, screen = screen_candidates(surface, list(valid), cfg)
    search.dropped.extend(n for n in valid if n not in screened)
    admitted, admission = shadow_admission(surface, screened, current["columns"], cfg)
    record = {"round": 0, "channel": "free_form", "proposal_audit": proposal["audit"],
              "proposal_validity": validity, "proposed": len(valid), "screened": len(screened),
              "admitted": admitted, "screen": screen.to_dict("records"),
              "admission": admission.to_dict("records")}
    if admitted:
        columns = list(dict.fromkeys(current["columns"] + admitted))
        record.update(_validate_round(search, surface, current, columns, current["features"] + admitted, True))
    else:
        record.update(accepted=False, validation_block=None, reason="no change; no block consumed")
    search.rounds.append(record)
    log.info("R0: %d proposed, %d screened, %d admitted, accepted=%s", len(valid), len(screened),
             len(admitted), record["accepted"])
    return record


# --------------------------------------------------------------------------- CoF
def run_cof_cycle(search: SearchState, prov: dict, pre: dict, round_index: int, cycle: int,
                  admitted_by_cycle: dict) -> tuple[dict, dict]:
    """One CoF cycle: propose -> expand -> cap and deduplicate -> (i) greedy entry,
    (ii) trailing-window check, (iii) permutation exit (cycle 1 only).

    Returns the provisional state with the cycle's changes and the audit record.
    """
    cfg = search.cfg
    raw_columns = [c for c in pre["surface"].train if c != cfg["label"]]
    contract = templates.depth_contract(cycle, raw_columns, admitted_by_cycle)

    # S1.1.1 proposal: evidence = best scores so far, depth contract, template scorecard.
    outside = {k: v for k, v in search.pool.items() if k not in prov["columns"]}
    scored = score_pool(prov, dict(list(outside.items())[-cfg["candidate_caps"][0]:]), cfg)
    evidence = {"entry_scores": {"tau": search.router["tau"],
                                 "top": sorted(scored, key=lambda r: -abs(r["score"]))[:cfg["evidence_top"]]},
                "depth_contract": contract, "template_scorecard": search.scorecard[-40:],
                "templates_used": sorted({row["template"] for row in search.scorecard})}
    proposal = llm.propose_cof_templates(prov, evidence, round_index, cycle, cfg)
    specs, expansion = templates.expand_templates(proposal["templates"], contract, raw_columns,
                                                  search.router["entity_columns"], cfg)
    if cycle == 1:                                  # the fixed library first, so the cap never drops it
        specs = {**search.router["library"], **specs}
    specs = {k: v for k, v in specs.items() if k not in search.tried and k not in prov["columns"]}
    valid, validity = llm.validate_proposals(specs, prov["surface"], cfg, limit=cfg["candidate_caps"][cycle - 1])
    search.tried.update(valid)
    search.pool.update(valid)
    surface = Surface(search.I, search.Vx, pre["surface"].plan, search.pool, cfg)
    names, dedup = llm.deduplicate(surface.train, list(valid), cfg["dedup_abs_spearman"])

    # S1.1.3 admission.
    residual = search.I[cfg["label"]].to_numpy(float) - prov["oof"]
    entered, entry = greedy_entry(surface.train, names, prov["columns"], residual, cfg)
    kept, trailing = trailing_window_check(surface.train, entered, prov["columns"], residual, cycle, cfg)
    removed, exit_table = [], pd.DataFrame()
    if cycle == 1:
        removed, exit_table = permutation_exit(surface, prov["columns"], pre["features"], cfg)

    templates.update_scorecard(search.scorecard, round_index, cycle, search.pool, names,
                               [n for n, _ in entered], kept)
    prov = {**prov, "surface": surface,
            "columns": [c for c in prov["columns"] if c not in removed] + kept,
            "features": [f for f in prov["features"] if f not in removed] + kept}
    record = {"cycle": cycle, "depth_contract": contract["rule"], "proposal_audit": proposal["audit"],
              "templates": expansion, "proposal_validity": validity, "candidates": len(names),
              "dedup": dedup, "entry": entry, "trailing_window": trailing.to_dict("records"),
              "admitted": kept, "exit": exit_table.to_dict("records"), "removed": removed}
    return prov, record


def run_cof_round(search: SearchState, round_index: int) -> dict:
    """One CoF round (R1-R3): up to ``cof_cycles`` cycles, then S1.2 on the whole next block.

    A deeper cycle runs only if the previous one admitted a feature; OOF
    residuals are refreshed before it. A round that changes nothing consumes
    no validation block.
    """
    cfg, pre = search.cfg, search.current
    prov, admitted_by_cycle, cycles = dict(pre), {}, []
    for cycle in range(1, cfg["cof_cycles"] + 1):
        prov, record = run_cof_cycle(search, prov, pre, round_index, cycle, admitted_by_cycle)
        cycles.append(record)
        if not record["admitted"]:
            break
        admitted_by_cycle[cycle] = record["admitted"]
        if cycle < cfg["cof_cycles"]:
            fitted = fit_oof(prov["surface"], prov["columns"])      # refresh residuals for the next cycle
            prov = _state_from_fit(prov["surface"], prov["columns"], prov["features"], fitted)

    record = {"round": round_index, "channel": "template", "cycles": cycles,
              "changed": set(prov["features"]) != set(pre["features"]),
              "admitted_any": any(c["admitted"] for c in cycles)}
    if record["changed"]:
        record.update(_validate_round(search, prov["surface"], pre, prov["columns"], prov["features"], False))
        record["added"] = [f for f in prov["features"] if f not in pre["features"]]
        record["removed"] = [f for f in pre["features"] if f not in prov["features"]]
    else:
        record.update(accepted=False, validation_block=None, reason="no change; no block consumed")
    search.rounds.append(record)
    log.info("R%d: %d cycle(s), changed=%s, accepted=%s, block=%s", round_index, len(cycles), record["changed"],
             record["accepted"], record["validation_block"])
    return record


# --------------------------------------------------------------------------- S2
def run_experts(state: dict, V: pd.DataFrame, pool: dict, dropped: list, cfg: dict) -> dict:
    """S2: build TE, slice and recency candidates (S2.1) and validate them (S2.2)."""
    te, te_audit = target_encoding_candidates(state, V, cfg)
    slices, slice_audit = slice_candidates(state, V, cfg)
    recency, recency_audit = recency_candidates(state, V, dropped, pool, cfg)
    result = validate_experts(state, V, pool, te + slices + recency, cfg)
    result["candidates"] = {"te": te_audit, "slice": slice_audit, "recency": recency_audit}
    return result


# --------------------------------------------------------------------------- S3
def fit_core(state: dict, part: Partition, pool: dict, cfg: dict) -> dict:
    """The CoF core fitted by stratified four-fold on ``I + V`` (S2 and S3).

    Returns the OOF predictions of the rows of ``I`` and ``V`` (the core
    prediction ``p_core^OOF`` used in S2), the mean of the four fold models on
    ``T`` (the rebuilt core of S3) and the feature registry.
    """
    registry = dependency_registry(state["features"], pool)
    fitted = fit_oof(Surface(part.outer, part.X_T, state["surface"].plan, registry, cfg), state["columns"])
    position = {row: i for i, row in enumerate(part.outer_ids)}
    rows_I = [position[row] for row in part.ids["I"]]
    rows_V = [position[row] for row in part.ids["V"]]
    return {"oof_I": fitted["oof"][rows_I], "oof_V": fitted["oof"][rows_V], "T": fitted["apply"],
            "registry": registry}


def freeze_and_evaluate(state: dict, baseline: dict, core: dict, part: Partition, experts: dict, cfg: dict):
    """S3: freeze the specification, predict ``T`` with the rebuilt core and the frozen experts,
    and open ``y_T`` once.

    The rebuilt core (``core['T']``, from :func:`fit_core`) averages the four
    fold models fitted on ``I + V``; the baseline is rebuilt the same way.
    Experts are not refitted. Returns ``(scores, predictions, frozen_spec)``.
    """
    frozen = {"representation": copy.deepcopy(state["surface"].plan), "features": core["registry"],
              "columns": list(state["columns"]), "learner": cfg["learner"],
              "model_parameters": {**learner_params(cfg), "seed": cfg["seed"], "oof_folds": cfg["oof_folds"]},
              "final_model": f"mean of the {cfg['oof_folds']} fold models fitted on I+V",
              "experts": [{"name": e.spec["name"], "weight": e.weight} for e in experts["experts"]]}
    if state["features"]:
        base_pred = fit_oof(Surface(part.outer, part.X_T, baseline["surface"].plan, {}, cfg),
                            baseline["columns"])["apply"]
    else:
        base_pred = core["T"]                       # the core is the baseline
    blended = core["T"].copy()
    for expert in experts["experts"]:
        pred, mask = expert.predict(part.X_T)
        blended = blend_prediction(blended, pred, mask, expert.weight)
    predictions = {"baseline": base_pred, "cof": core["T"].copy(), "cof_experts": blended}
    scores = part.vault.evaluate(predictions)
    return scores, predictions, {**frozen, "sha256": canonical_hash(frozen)}


# --------------------------------------------------------------------------- driver
@dataclass
class ForgeResult:
    """Everything produced by :func:`run_forge`."""

    cfg: dict
    scores: dict
    predictions: dict
    router: dict
    rounds: list
    scorecard: list
    selection: pd.DataFrame
    accepted_features: list
    experts: dict
    frozen_spec: dict
    partition: dict
    elapsed_seconds: float

    def report(self) -> dict:
        router = {k: v for k, v in self.router.items() if k not in {"library", "drift", "library_scores"}}
        router["drift"] = self.router["drift"].to_dict("records")
        return {"dataset": self.cfg["dataset"], "seed": self.cfg["seed"], "learner": self.cfg["learner"],
                "llm_mode": self.cfg["llm_mode"], "scores": self.scores, "router": router,
                "accepted_features": self.accepted_features,
                "accepted_experts": [e.spec["name"] for e in self.experts["experts"]],
                "rounds": self.rounds, "template_scorecard": self.scorecard,
                "core_selection": self.selection.to_dict("records"),
                "expert_validation": self.experts["audit"], "partition": self.partition,
                "frozen_spec_sha256": self.frozen_spec["sha256"], "elapsed_seconds": self.elapsed_seconds}

    def save(self, output: Path) -> None:
        """Write ``config.yaml`` (all settings of the run), ``report.json``, ``metrics.csv``,
        ``features.json`` and ``predictions.npz``."""
        import json

        import yaml

        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        settings = {k: str(v) if isinstance(v, Path) else v for k, v in self.cfg.items()}
        (output / "config.yaml").write_text(yaml.safe_dump(_jsonable(settings), sort_keys=False))
        (output / "report.json").write_text(json.dumps(_jsonable(self.report()), indent=2) + "\n")
        (output / "features.json").write_text(json.dumps(_jsonable(self.frozen_spec), indent=2) + "\n")
        pd.DataFrame(self.scores).T.rename_axis("configuration").to_csv(output / "metrics.csv")
        np.savez_compressed(output / "predictions.npz", **self.predictions)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.name
    return value


def run_forge(cfg: dict, frame: pd.DataFrame | None = None) -> ForgeResult:
    """Run FORGE on ``cfg['dataset']`` (or on ``frame``) and evaluate the sealed test set once."""
    started = time.perf_counter()
    llm.check_llm_settings(cfg)
    raw = load_dataset(cfg) if frame is None else validate_dataset(frame, cfg)
    part = split_dataset(raw, cfg)
    I, V = part.I, part.V
    log.info("Partition: I=%d, V=%d, T=%d", len(I), len(V), len(part.X_T))
    baseline = initialize_state(I, V.drop(columns=cfg["label"]), cfg)
    router = routing_rule(baseline, V, cfg)
    log.info("Routing: shift=%s ratio=%.3f event_time=%s -> R0 %s, CoF %s", router["shift"], router["ratio"],
             router["has_event_time"], "runs" if router["run_r0"] else "skipped",
             "runs" if router["run_cof"] else "skipped")
    search = SearchState(I=I, V=V, cfg=cfg, router=router, current=baseline, baseline=baseline,
                         budget=ValidationBudget(len(V), cfg), accepted=[baseline])
    if router["run_r0"]:
        run_initial_round(search)
    empty_rounds = 0
    for t in range(1, cfg["max_cof_rounds"] + 1):
        if not router["run_cof"] or search.budget.exhausted or empty_rounds >= cfg["max_empty_cof_rounds"]:
            break
        record = run_cof_round(search, t)
        rejected = record["changed"] and not record["accepted"]
        empty_rounds += not record["admitted_any"] and not rejected     # gate-rejected rounds do not count
    final, selection = choose_configuration(search.accepted, search.y_V, cfg)
    log.info("CoF core: %d feature(s) %s", len(final["features"]), final["features"])
    core = fit_core(final, part, search.pool, cfg)
    s2_state = {**final, "oof": core["oof_I"], "V_prediction": core["oof_V"]}
    experts = run_experts(s2_state, V, search.pool, search.dropped, cfg)
    log.info("S2: %d expert(s) accepted", len(experts["experts"]))
    scores, predictions, frozen = freeze_and_evaluate(final, baseline, core, part, experts, cfg)
    return ForgeResult(cfg=cfg, scores=scores, predictions=predictions, router=router, rounds=search.rounds,
                       scorecard=search.scorecard, selection=selection, accepted_features=list(final["features"]), experts=experts,
                       frozen_spec=frozen,
                       partition={"rows": {"I": len(I), "V": len(V), "T": len(part.X_T)},
                                  "class_balance": part.class_balance, "data_sha256": frame_hash(raw),
                                  "split_sha256": canonical_hash(part.ids)},
                       elapsed_seconds=time.perf_counter() - started)
