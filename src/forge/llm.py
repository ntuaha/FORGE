"""LLM proposer (paper S1.1.1): prompts, three ways to get answers, and a response cache.

R0 makes five free-form calls (unary, binary, ternary, related-column and
complement features); each returns pointwise expressions. Only the
complement call receives the probe-model summary (:func:`probe_summary`). In
every CoF cycle the LLM proposes templates with slot vocabularies
(:mod:`forge.templates`), guided by residual-score evidence, the depth
contract of the cycle and the template scorecard.

``llm_mode`` (configs/default.yaml)
-----------------------------------
``replay`` (default)
    No LLM. Credit Default uses the R0 proposals recorded in the paper's
    LightGBM / seed-11 run (``assets/credit_default_r0_proposals.json``), for
    any learner and seed. ``temporal_demo`` uses a hand-written fixture.
    Anything else needs a response recorded earlier in ``.cache/llm/``.
``api``
    An OpenAI-compatible chat-completions endpoint with a JSON-schema
    response format (``llm_api_url``, key in the variable ``llm_api_key_env``).
``command``
    Any command-line tool (``llm_command``), e.g. Codex, Claude Code or
    Ollama. The prompt goes to its stdin; the answer is read from the file
    ``{output}`` if the tool writes it, otherwise from stdout; ``{schema}`` is
    the path of the answer's JSON schema.

``api`` and ``command`` answers are cached in ``cache_dir`` (keyed by the data,
settings and prompt, not by the tool), so a later ``replay`` run with the same
inputs gives the same result. Use a separate ``cache_dir`` per LLM to compare
LLMs.
"""

from __future__ import annotations

import copy
import os
import hashlib
import itertools
import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ASSETS_DIR, learner_params
from .data import canonical_hash, frame_hash
from .features import matrix, materialize_expression
from .history import entity_history
from .learners import make_learner
from .templates import template_schema

R0_FAMILIES = ("unary", "binary", "ternary", "related_column", "complement")

CREDIT_PROPOSALS_FILE = ASSETS_DIR / "credit_default_r0_proposals.json"
CREDIT_DISCOVERY_HASH = "0783e61092ce8d10463e8e9c97dabf42ca099ee8cf3eff17f70b0b5072e5d6cc"
"""``frame_hash`` of the Credit Default training partition ``I`` (split seed 42)."""


# --------------------------------------------------------------------------- frozen proposals
def load_frozen_proposals(path: Path = CREDIT_PROPOSALS_FILE) -> dict:
    """Load frozen R0 proposals and verify their checksum."""
    record = json.loads(Path(path).read_text())
    if canonical_hash(record["proposals"]) != record["proposals_sha256"]:
        raise ValueError(f"checksum mismatch in {path}")
    return record


def frozen_replay_applies(cfg: dict, I: pd.DataFrame) -> bool:
    """True when this run has the published Credit Default training partition ``I``.

    The proposals were recorded in the LightGBM / seed-11 run. Paper
    difference: the paper's other learners and seeds used their own LLM
    proposals, which are not shipped; reusing the seed-11 proposals for them
    gives a comparable but not identical run.
    """
    return cfg["dataset"] == "credit_default" and frame_hash(I) == CREDIT_DISCOVERY_HASH


# --------------------------------------------------------------------------- prompts
def proposal_schema() -> dict:
    """JSON schema of an R0 free-form response."""
    item = {"type": "object", "additionalProperties": False,
            "properties": {k: {"type": "string"} for k in ("name", "family", "description", "python_expr")},
            "required": ["name", "family", "description", "python_expr"]}
    return {"type": "object", "additionalProperties": False,
            "properties": {"features": {"type": "array", "items": item}}, "required": ["features"]}


def probe_summary(state: dict, cfg: dict) -> dict:
    """What the learner already exploits (context of the R0 complement call).

    A probe model of the same learner with ``probe_trees`` trees is fitted on
    up to the first ``probe_rows`` rows of ``I``. Returns its most important
    columns and the column pairs that most often occur in the same tree
    (CatBoost: its pairwise interaction scores).
    """
    surface = state["surface"]
    columns = list(state["columns"])
    rows = min(len(surface.train), cfg["probe_rows"])
    y = surface.train[cfg["label"]].to_numpy(np.int8)[:rows]
    size = {"lightgbm": "n_estimators", "xgboost": "n_estimators", "catboost": "iterations"}.get(cfg["learner"])
    model = make_learner(cfg, **({size: cfg["probe_trees"]} if size else {}))
    model.fit(matrix(surface.train.iloc[:rows], columns), y)
    importance = np.asarray(getattr(model, "feature_importances_", np.zeros(len(columns))), float)
    top = [columns[i] for i in np.argsort(-importance, kind="stable")[:cfg["probe_top_columns"]]
           if importance[i] > 0]
    pairs: dict = {}
    if cfg["learner"] == "catboost":
        for i, j, score in model.get_feature_importance(type="Interaction"):
            pairs[tuple(sorted((int(i), int(j))))] = score
    elif cfg["learner"] in ("lightgbm", "xgboost"):          # other learners: no pair summary
        for found in _features_per_tree(model, cfg["learner"]):
            for pair in itertools.combinations(sorted(found), 2):
                pairs[pair] = pairs.get(pair, 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:cfg["probe_top_pairs"]]
    return {"top_columns": top, "co_used_pairs": [[columns[a], columns[b]] for (a, b), _ in ranked]}


def _features_per_tree(model, learner: str) -> list[set]:
    """Indices of the split features of every tree (LightGBM, XGBoost)."""
    if learner == "xgboost":
        trees = model.get_booster().trees_to_dataframe()
        trees = trees[trees["Feature"] != "Leaf"]
        return [{int(f[1:]) for f in tree["Feature"]} for _, tree in trees.groupby("Tree")]

    def walk(node, found):
        if "split_feature" in node:
            found.add(node["split_feature"])
            walk(node["left_child"], found)
            walk(node["right_child"], found)
        return found

    return [walk(tree["tree_structure"], set()) for tree in model.booster_.dump_model()["tree_info"]]


def _column_summary(state: dict, cfg: dict) -> list[dict]:
    train = state["surface"].train
    return [{"name": c, "missing_rate": float(train[c].isna().mean()), "n_unique": int(train[c].nunique())}
            for c in train if c != cfg["label"]]


R0_PROMPT = ("You propose features; FORGE alone decides admission. Propose up to {budget} {family} numeric "
             "pointwise features. Return only schema JSON. No labels, predictions, external state, I/O, "
             "aggregates, groupby, rolling or apply. Expressions may use df['column'], basic arithmetic, "
             "np.abs/maximum/minimum/log1p/sqrt/clip/where and pd.to_numeric. Guard divisions/logs/square "
             "roots.")

R0_FAMILY_TASKS = {
    "unary": "Each feature transforms one column.",
    "binary": "Each feature combines two columns.",
    "ternary": "Each feature combines three columns.",
    "related_column": "Each feature summarizes the shape of a group of sibling columns (such as x1..x6).",
    "complement": "Target columns and mechanisms outside those the learner already exploits "
                  "(listed under already_exploited).",
}

# Paper difference: the paper prints abridged prompts. These are the full
# prompts of this code; the template prompt also carries one example form.
TEMPLATE_PROMPT = (
    "You do not propose individual factors. You propose a searchable set: templates with an exact functional "
    "form plus slot vocabularies. A deterministic engine expands every slot binding into a candidate, and a "
    "statistical verifier decides what survives. Write the form as you would ship it (clipping, log1p "
    "bounding, guarded division, explicit fillna policy); leave only the column and parameter choices as "
    "slots. Target column {label} is forbidden in every expression. Pointwise templates use {{A}}-style slots "
    "that stand for df['column']; groupby, rank, rolling, expanding, merge, join, and apply are forbidden. "
    "Aggregate templates are strictly causal specifications over entity slots, value slots, aggregations, "
    "windows, and shifts; shift >= 1 is mandatory. Span at least four distinct mechanisms. Return only schema "
    "JSON. Example pointwise form: \"np.log1p(np.abs({{A}}))/(np.abs({{B}})+1.0)\".")


def build_prompt(channel: str, state: dict, cfg: dict, family: str | None = None,
                 evidence: dict | None = None) -> str:
    """Prompt of the R0 free-form channel or the CoF template channel.

    R0: dataset name, learner type and a summary of the available columns
    (name, missing rate, number of unique values); the complement call also
    receives the probe summary (``evidence``). CoF: the template instructions
    and the evidence (entry scores with ``tau``, depth contract, template
    scorecard, template names already used). The LLM never sees row-level
    labels, residual vectors or test data.
    """
    payload = {"dataset": cfg["dataset"], "learner": cfg["learner"], "columns": _column_summary(state, cfg)}
    if channel == "template":
        return TEMPLATE_PROMPT.format(label=cfg["label"]) + " Evidence:\n" + json.dumps(
            {**payload, **(evidence or {})}, default=str, ensure_ascii=False)
    if family == "complement":
        payload["already_exploited"] = evidence
    return (R0_PROMPT.format(budget=cfg["llm_proposals_per_order"], family=family) + " "
            + R0_FAMILY_TASKS[family] + " Evidence:\n" + json.dumps(payload, default=str, ensure_ascii=False))


# --------------------------------------------------------------------------- transport + cache
def call_api(prompt: str, schema: dict, cfg: dict) -> dict:
    """``api`` mode: one chat-completions request with a strict JSON-schema response format."""
    key = os.environ.get(cfg["llm_api_key_env"], "")
    body = {"model": cfg["llm_model"], "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "forge_proposals", "strict": True, "schema": schema}}}
    request = urllib.request.Request(cfg["llm_api_url"], data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=cfg["llm_timeout_seconds"]) as response:
        payload = json.loads(response.read().decode())
    return parse_json(payload["choices"][0]["message"]["content"])


def call_command(prompt: str, schema: dict, cfg: dict) -> dict:
    """``command`` mode: run ``llm_command`` with the prompt on stdin and parse its JSON answer.

    The prompt also states the schema, for tools that cannot enforce one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        schema_path, output_path = Path(tmp) / "schema.json", Path(tmp) / "output.json"
        schema_path.write_text(json.dumps(schema))
        command = cfg["llm_command"].format(schema=schema_path, output=output_path)
        text = prompt + "\nAnswer with one JSON object that follows this JSON schema:\n" + json.dumps(schema)
        done = subprocess.run(command, shell=True, input=text, capture_output=True, text=True,
                              timeout=cfg["llm_timeout_seconds"])
        if done.returncode != 0:
            raise RuntimeError(f"llm_command failed ({done.returncode}): {done.stderr.strip()[-500:]}")
        answer = output_path.read_text() if output_path.exists() and output_path.stat().st_size else done.stdout
    return parse_json(answer)


def parse_json(text: str) -> dict:
    """The JSON object in ``text`` (plain JSON, a ```json block, or the outermost {...})."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    elif not text.startswith("{"):
        text = text[text.find("{"):text.rfind("}") + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the LLM answer is not JSON: {text[:200]!r}") from exc


def check_llm_settings(cfg: dict) -> None:
    """Fail early (before any model is trained) if the LLM mode cannot work."""
    mode = cfg["llm_mode"]
    if mode == "api" and not (cfg["llm_api_url"] and os.environ.get(cfg["llm_api_key_env"])):
        raise ValueError(f"llm_mode 'api' needs llm_api_url (or FORGE_LLM_API_URL) and the key in "
                         f"${cfg['llm_api_key_env']}")
    if mode == "command" and not cfg["llm_command"]:
        raise ValueError("llm_mode 'command' needs llm_command, e.g. "
                         "\"codex exec --skip-git-repo-check --output-schema {schema} -o {output} -\"")
    if mode not in ("replay", "api", "command"):
        raise ValueError("llm_mode must be 'replay', 'api' or 'command'")


def _cache_path(prompt: str, identity: dict, cfg: dict) -> tuple[str, Path]:
    key = canonical_hash({"identity": identity, "prompt": prompt})
    return key, Path(cfg["cache_dir"]) / "llm" / f"{key}.json"


def _demo_fixture_applies(prompt: str, identity: dict, cfg: dict) -> bool:
    """Replay of ``temporal_demo`` without a recorded response uses the hand-written fixture."""
    return (cfg["llm_mode"] == "replay" and cfg["dataset"] == "temporal_demo"
            and not _cache_path(prompt, identity, cfg)[1].exists())


def cached_call(prompt: str, schema: dict, identity: dict, cfg: dict) -> tuple[dict, dict]:
    """Ask the LLM (``api``/``command``) and cache the answer, or return the cached answer (``replay``)."""
    key, path = _cache_path(prompt, identity, cfg)
    if cfg["llm_mode"] == "replay":
        if not path.exists():
            raise FileNotFoundError("no recorded LLM response for this prompt (recorded proposals are shipped only "
                                    "for Credit Default); set llm_mode to 'api' or 'command'")
        record = json.loads(path.read_text())
        if record["key"] != key or record["response_hash"] != canonical_hash(record["response"]):
            raise ValueError("recorded LLM response failed verification")
        return record["response"], {"source": "recorded_llm", "key": key, "llm": record.get("llm")}
    response = call_api(prompt, schema, cfg) if cfg["llm_mode"] == "api" else call_command(prompt, schema, cfg)
    record = {"key": key, "identity": identity, "prompt": prompt, "response": response,
              "response_hash": canonical_hash(response),
              "llm": cfg["llm_model"] if cfg["llm_mode"] == "api" else cfg["llm_command"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return response, {"source": cfg["llm_mode"], "key": key}


def _identity(state: dict, cfg: dict, round_index: int, cycle: int = 0) -> dict:
    surface = state["surface"]
    return {"dataset": cfg["dataset"], "data_hash": frame_hash(surface.raw),
            "validation_features_hash": frame_hash(surface.apply_raw), "seed": cfg["seed"],
            "learner": cfg["learner"], "round": round_index, "cycle": cycle,
            "model_config_hash": canonical_hash({"learner": learner_params(cfg), "folds": cfg["oof_folds"],
                                                 "split_seed": cfg["split_seed"]})}


# --------------------------------------------------------------------------- validation of proposals
def validate_proposals(specs: dict, surface, cfg: dict, limit: int | None = None) -> tuple[dict, list[dict]]:
    """Materialise each proposal on ``I`` and ``V``; reject invalid, non-finite or constant ones.

    With ``limit``, at most ``limit`` valid proposals are kept in order; the
    remaining ones are reported as over the proposal budget.
    """
    accepted, audit = {}, []
    fit, apply = surface.train.copy(), surface.apply.copy()
    for name, spec in specs.items():
        reason = "valid"
        if limit is not None and len(accepted) >= limit:
            audit.append({"feature": name, "valid": False, "reason": "over the proposal budget"})
            continue
        try:
            if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", name) or name in fit:
                raise ValueError("invalid name or name of an existing column")
            if spec.get("executor", "pointwise") == "entity_history":
                fi = entity_history(surface.raw, None, spec, cfg)
                ap = entity_history(surface.raw, surface.apply_raw, spec, cfg)
            else:
                fi = materialize_expression(fit, spec["expression"], cfg["label"])
                ap = materialize_expression(apply, spec["expression"], cfg["label"])
            if not np.isfinite(fi).all() or not np.isfinite(ap).all():
                raise ValueError("non-finite values")
            if np.unique(fi).size <= 1:
                raise ValueError("constant on I")
            fit[name], apply[name] = fi, ap
            accepted[name] = spec
        except (ValueError, KeyError, TypeError, SyntaxError) as exc:
            reason = str(exc)
        audit.append({"feature": name, "valid": name in accepted, "reason": reason})
    return accepted, audit


# --------------------------------------------------------------------------- R0
def propose_r0(state: dict, cfg: dict) -> dict:
    """R0 free-form proposal stage (five calls; the pipeline keeps at most ``max_initial_candidates`` valid ones).

    Returns ``{"specs", "audit", "prompts"}`` with raw (not yet validated) specs.
    """
    surface = state["surface"]
    if cfg["llm_mode"] == "replay" and frozen_replay_applies(cfg, surface.raw):
        record = load_frozen_proposals()
        return {"specs": copy.deepcopy(record["proposals"]), "prompts": [],
                "audit": [{"source": "frozen_llm_proposals", "model": record["model"],
                           "calls": len(R0_FAMILIES), "sha256": record["proposals_sha256"]}]}
    specs, audit, prompts = {}, [], []
    probe = probe_summary(state, cfg)
    first = build_prompt("free_form", state, cfg, family=R0_FAMILIES[0], evidence=probe)
    if _demo_fixture_applies(first, {**_identity(state, cfg, 0), "family": R0_FAMILIES[0]}, cfg):
        specs = {"demo_burden": {"expression": "df['debt']/(np.abs(df['income'])+1.0)", "family": "ratio",
                                 "description": "teaching fixture: debt burden"}}
        return {"specs": specs, "prompts": [], "audit": [{"source": "handwritten_teaching_fixture"}]}
    for family in R0_FAMILIES:
        prompt = build_prompt("free_form", state, cfg, family=family, evidence=probe)
        response, record = cached_call(prompt, proposal_schema(), {**_identity(state, cfg, 0), "family": family}, cfg)
        for item in response["features"][:cfg["llm_proposals_per_order"]]:     # up to eight per call
            name = item["name"] if item["name"] not in specs else f"{family}__{item['name']}"   # same name, two calls
            specs[name] = {"expression": item["python_expr"], "family": item["family"],
                                   "description": item["description"], "executor": "pointwise"}
        audit.append({**record, "family": family})
        prompts.append({"family": family, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "prompt": prompt})
    return {"specs": specs, "audit": audit, "prompts": prompts}


# --------------------------------------------------------------------------- CoF
DEMO_TEMPLATE_FIXTURE = {
    1: [{"name": "burden", "kind": "pointwise", "mechanism": "debt burden", "form": "{A}/(np.abs({B})+1.0)",
         "slots": [{"slot": "A", "columns": ["debt"]}, {"slot": "B", "columns": ["income"]}],
         "entities": [], "values": [], "aggregations": [], "windows": [], "shift": 1},
        {"name": "interaction", "kind": "pointwise", "mechanism": "two-factor interaction", "form": "{A}*{B}",
         "slots": [{"slot": "A", "columns": ["x1"]}, {"slot": "B", "columns": ["x2", "income"]}],
         "entities": [], "values": [], "aggregations": [], "windows": [], "shift": 1},
        {"name": "custhist", "kind": "aggregate", "mechanism": "customer history", "form": "", "slots": [],
         "entities": ["customer"], "values": ["debt"], "aggregations": ["mean", "deviation"],
         "windows": [0, 5], "shift": 1}],
    2: [{"name": "prod", "kind": "pointwise", "mechanism": "product of admitted features", "form": "{A}*{B}",
         "slots": [{"slot": "A", "columns": ["*"]}, {"slot": "B", "columns": ["*"]}],
         "entities": [], "values": [], "aggregations": [], "windows": [], "shift": 1},
        {"name": "sq", "kind": "pointwise", "mechanism": "signed square", "form": "{A}*np.abs({A})",
         "slots": [{"slot": "A", "columns": ["*"]}],
         "entities": [], "values": [], "aggregations": [], "windows": [], "shift": 1}],
    3: [{"name": "ratio", "kind": "pointwise", "mechanism": "guarded ratio", "form": "{A}/(np.abs({B})+1.0)",
         "slots": [{"slot": "A", "columns": ["*"]}, {"slot": "B", "columns": ["*"]}],
         "entities": [], "values": [], "aggregations": [], "windows": [], "shift": 1}],
}
"""Hand-written templates used in replay mode for ``temporal_demo`` (no LLM), keyed by cycle."""


def propose_cof_templates(state: dict, evidence: dict, round_index: int, cycle: int, cfg: dict) -> dict:
    """CoF proposal of cycle ``cycle``: the LLM returns templates (see :mod:`forge.templates`).

    Returns ``{"templates", "audit", "prompts"}``; expansion is done by the caller.
    """
    prompt = build_prompt("template", state, cfg, evidence=evidence)
    identity = _identity(state, cfg, round_index, cycle)
    if _demo_fixture_applies(prompt, identity, cfg):
        return {"templates": copy.deepcopy(DEMO_TEMPLATE_FIXTURE.get(cycle, [])), "prompts": [],
                "audit": [{"source": "handwritten_teaching_fixture", "round": round_index, "cycle": cycle}]}
    response, record = cached_call(prompt, template_schema(), identity, cfg)
    return {"templates": response["templates"], "audit": [record],
            "prompts": [{"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "prompt": prompt}]}


def deduplicate(train: pd.DataFrame, names: list[str], threshold: float) -> tuple[list[str], list[dict]]:
    """Label-free near-duplicate removal: drop a candidate with ``|Spearman| >= threshold`` to a kept one."""
    kept, rows = [], []
    for name in names:
        twin = None
        for other in kept:
            corr = train[name].corr(train[other], method="spearman")
            if np.isfinite(corr) and abs(corr) >= threshold:
                twin = other
                break
        if twin is None:
            kept.append(name)
        rows.append({"feature": name, "duplicate_of": twin})
    return kept, rows
