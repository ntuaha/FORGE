"""CoF templates (paper S1.1.1, "Template proposal and composition depth").

In each CoF cycle the LLM proposes *templates*: an exact functional form with
column slots and a small vocabulary of admissible columns per slot. FORGE
expands every slot binding deterministically into a candidate feature.

Template fields (all required by the JSON schema of :func:`template_schema`)::

    name          short identifier
    kind          "pointwise" or "aggregate"
    mechanism     one line: the mechanism the template targets
    form          pointwise: expression over slots, e.g. "{A}/(np.abs({B})+1.0)"
    slots         pointwise: [{"slot": "A", "columns": [...]}, ...]
    entities      aggregate: entity columns; "a+b" denotes an entity pair
    values        aggregate: value columns
    aggregations  aggregate: subset of features.HISTORY_AGGREGATIONS
    windows       aggregate: numbers of most recent past events (0 = all)
    shift         aggregate: >= 1 (mandatory causal shift)

Composition depth (the depth contract of cycle ``d``):

* cycle 1 builds first-order features from the raw representation; aggregate
  templates (strictly causal entity histories) are allowed only here;
* cycle 2 combines features admitted in cycle 1 (every slot);
* cycle 3 composes features admitted in cycle 2 (at least one slot) with
  first-order features (features admitted in cycle 1 and raw columns).

A slot entry ``"*"`` stands for the whole vocabulary permitted at this depth.
Entries outside the permitted vocabulary are dropped and reported.
"""

from __future__ import annotations

import hashlib
import itertools
import re

from .history import HISTORY_AGGREGATIONS, history_feature_name

SLOT = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")
MAX_WINDOWS = 4


def template_schema() -> dict:
    """Strict JSON schema of a CoF template response."""
    strings = {"type": "array", "items": {"type": "string"}}
    slot = {"type": "object", "additionalProperties": False,
            "properties": {"slot": {"type": "string"}, "columns": strings}, "required": ["slot", "columns"]}
    fields = {"name": {"type": "string"}, "kind": {"type": "string", "enum": ["pointwise", "aggregate"]},
              "mechanism": {"type": "string"}, "form": {"type": "string"},
              "slots": {"type": "array", "items": slot}, "entities": strings, "values": strings,
              "aggregations": {"type": "array", "items": {"type": "string", "enum": list(HISTORY_AGGREGATIONS)}},
              "windows": {"type": "array", "items": {"type": "integer"}}, "shift": {"type": "integer"}}
    item = {"type": "object", "additionalProperties": False, "properties": fields, "required": list(fields)}
    return {"type": "object", "additionalProperties": False,
            "properties": {"templates": {"type": "array", "items": item}}, "required": ["templates"]}


def depth_contract(cycle: int, raw_columns: list[str], admitted: dict[int, list[str]]) -> dict:
    """Vocabularies permitted in cycle ``cycle``.

    ``admitted[d]`` lists the features admitted in cycle ``d`` of the current
    round. Returns ``{"cycle", "rule", "vocabulary", "required"}``: every slot
    binds to ``vocabulary``; when ``required`` is non-empty at least one slot
    must bind to it.
    """
    if cycle == 1:
        return {"cycle": 1, "rule": "first-order features built directly from raw columns; "
                                    "aggregate templates allowed",
                "vocabulary": list(raw_columns), "required": []}
    if cycle == 2:
        return {"cycle": 2, "rule": "every slot binds to a feature admitted in cycle 1; pointwise only",
                "vocabulary": list(admitted.get(1, [])), "required": []}
    first_order = list(dict.fromkeys(admitted.get(1, []) + list(raw_columns)))
    higher = list(admitted.get(2, []))
    return {"cycle": 3, "rule": "at least one slot binds to a feature admitted in cycle 2, the others to "
                                "first-order features (cycle-1 features or raw columns); pointwise only",
            "vocabulary": list(dict.fromkeys(higher + first_order)), "required": higher}


def _candidate_name(template: str, parts: list[str]) -> str:
    base = "tpl_" + "__".join([template] + parts)
    base = re.sub(r"[^A-Za-z0-9_]", "_", base)
    if len(base) <= 80:
        return base
    return base[:69] + "_" + hashlib.sha256(base.encode()).hexdigest()[:10]


def _vocabulary(entries: list[str], allowed: list[str], dropped: list[str]) -> list[str]:
    out = []
    for entry in entries:
        if entry == "*":
            out += allowed
        elif entry in allowed:
            out.append(entry)
        else:
            dropped.append(entry)
    return list(dict.fromkeys(out))


def expand_template(template: dict, contract: dict, raw_columns: list[str], entity_columns: list[str],
                    cfg: dict) -> tuple[dict, dict]:
    """Expand one template into candidate specs under ``contract``.

    Returns ``(specs, audit)`` with specs keyed by candidate name, in binding order.
    """
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(template.get("name", "")))[:32] or "t"
    audit = {"template": name, "kind": template.get("kind"), "mechanism": template.get("mechanism", ""),
             "dropped_columns": [], "candidates": 0, "reason": "expanded"}
    specs = {}
    if template.get("kind") == "aggregate":
        if contract["cycle"] != 1:
            audit["reason"] = "aggregate templates are only allowed in cycle 1"
            return {}, audit
        if int(template.get("shift", 0)) < 1:
            audit["reason"] = "shift >= 1 is mandatory"
            return {}, audit
        allowed_entities = [c for c in raw_columns if c not in {cfg["label"], cfg["time_col"]}]
        allowed_entities = list(dict.fromkeys(entity_columns + allowed_entities))
        entities = []
        for entry in template.get("entities", []):
            key = [part for part in str(entry).split("+") if part]
            if key and len(key) <= 2 and all(part in allowed_entities for part in key):
                entities.append(key)
            else:
                audit["dropped_columns"].append(entry)
        values = _vocabulary(template.get("values", []), [c for c in raw_columns if c not in entity_columns],
                             audit["dropped_columns"])
        windows = sorted({max(int(w), 0) for w in template.get("windows", [0])})[:MAX_WINDOWS] or [0]
        for key, agg, window in itertools.product(entities, template.get("aggregations", []), windows):
            if agg not in HISTORY_AGGREGATIONS:
                continue
            for value in (values if agg not in ("count", "repeat", "time_since") else [""]):
                if agg == "time_since" and not cfg["time_col"]:
                    continue
                spec = {"executor": "entity_history", "entity": key, "value": value, "agg": agg,
                        "window": window, "shift": int(template["shift"]), "family": name,
                        "template": name, "description": template.get("mechanism", "")}
                specs[history_feature_name(spec)] = spec
    elif template.get("kind") == "pointwise":
        form = str(template.get("form", ""))
        if not SLOT.search(form):          # tolerate "A - B" written without braces
            for slot in (s["slot"] for s in template.get("slots", [])):
                form = re.sub(rf"(?<![\w'\"]){re.escape(slot)}(?![\w'\"])", "{" + slot + "}", form)
        slots = list(dict.fromkeys(SLOT.findall(form)))
        vocab = {s["slot"]: _vocabulary(s.get("columns", []), contract["vocabulary"], audit["dropped_columns"])
                 for s in template.get("slots", [])}
        if not slots or any(not vocab.get(s) for s in slots):
            audit["reason"] = "every slot of the form needs a non-empty permitted vocabulary"
            return {}, audit
        required = set(contract["required"])
        for binding in itertools.product(*(vocab[s] for s in slots)):
            if len(set(binding)) < len(binding):
                continue
            if required and not required & set(binding):
                continue
            expression = form
            for slot, column in zip(slots, binding):
                expression = expression.replace("{" + slot + "}", f"df[{column!r}]")
            specs[_candidate_name(name, list(binding))] = {
                "executor": "pointwise", "expression": expression, "family": name, "template": name,
                "description": template.get("mechanism", "")}
    else:
        audit["reason"] = "kind must be 'pointwise' or 'aggregate'"
    audit["candidates"] = len(specs)
    return specs, audit


def expand_templates(templates: list[dict], contract: dict, raw_columns: list[str], entity_columns: list[str],
                     cfg: dict) -> tuple[dict, list[dict]]:
    """Expand all templates and interleave their candidates (one per template in turn).

    Interleaving keeps one large template from filling the candidate cap.
    """
    expanded, audit = [], []
    for template in templates:
        specs, row = expand_template(template, contract, raw_columns, entity_columns, cfg)
        expanded.append(list(specs.items()))
        audit.append(row)
    merged = {}
    for group in itertools.zip_longest(*expanded):
        for item in group:
            if item is not None and item[0] not in merged:
                merged[item[0]] = item[1]
    return merged, audit


def update_scorecard(scorecard: list[dict], round_index: int, cycle: int, pool: dict, candidates: list[str],
                     entered: list[str], admitted: list[str]) -> None:
    """Append one row per template: candidates offered, entered by greedy entry, admitted."""
    by_template = {}
    for name in candidates:
        by_template.setdefault(pool[name].get("template", pool[name].get("family", "")), []).append(name)
    for template, names in by_template.items():
        scorecard.append({"round": round_index, "cycle": cycle, "template": template, "candidates": len(names),
                          "entered": sum(n in entered for n in names),
                          "admitted": sum(n in admitted for n in names)})
