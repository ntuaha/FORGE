from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
FULL_ROOT = PACKAGE_DIR.parent
VENDOR_DIR = FULL_ROOT / "vendor"
for _path in (VENDOR_DIR, FULL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd

from agg_executor import (AGG_WHITELIST, STATIC_OPS,
                          TEMPORAL_OPS, validate_agg_spec)
from protocol import FeatureSpec, compile_pointwise_expression


TEMPLATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "pointwise_templates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "target_seats": {"type": "array",
                                     "items": {"type": "string"}},
                    "python_expr": {"type": "string"},
                    "slots": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "token": {"type": "string"},
                                "values": {"type": "array",
                                           "items": {"type": "string"}},
                            },
                            "required": ["token", "values"],
                        },
                    },
                },
                "required": ["name", "description", "mechanism",
                             "target_seats", "python_expr", "slots"],
            },
        },
        "agg_templates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "target_seats": {"type": "array",
                                     "items": {"type": "string"}},
                    "op": {"type": "string"},
                    "entity_slots": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                    "value_slots": {"type": "array", "items": {"type": "string"}},
                    "aggs": {"type": "array", "items": {"type": "string"}},
                    "windows": {"type": "array", "items": {"type": "integer"}},
                    "shifts": {"type": "array", "items": {"type": "integer"}},
                    "units": {"type": "array", "items": {"type": "string"}},
                    "encodings": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "description", "mechanism",
                             "target_seats", "op", "entity_slots",
                             "value_slots", "aggs", "windows", "shifts",
                             "units", "encodings"],
            },
        },
    },
    "required": ["pointwise_templates", "agg_templates"],
}

_SLOT_TOKEN = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")
_NAME_SAFE = re.compile(r"[^a-z0-9_]+")


def _safe_name(text: str, limit: int = 40) -> str:
    return _NAME_SAFE.sub("_", str(text).strip().lower()).strip("_")[:limit]


def _seat_fields(raw: dict) -> dict:
    seats = raw.get("target_seats")
    if isinstance(seats, str):
        seats = [seats]
    if not isinstance(seats, (list, tuple)):
        seats = []
    return {"target_seats": list(dict.fromkeys(
                str(s).strip().lower() for s in seats if str(s).strip()))[:8],
            "mechanism": _safe_name(raw.get("mechanism", ""), 32) or "unspecified"}


def seat_posneg_gap(train: pd.DataFrame, *, label: str, oof: np.ndarray,
                    rows: np.ndarray, features: list[str],
                    top: int = 5) -> tuple[list[dict], str]:
    y = train[label].to_numpy(np.int8, copy=False)[rows]
    p = np.asarray(oof, dtype=np.float64)[rows]
    pos = y == 1
    neg = ~pos
    if int(pos.sum()) < 20 or int(neg.sum()) < 50:
        return [], ""
    hard_pos = pos & (p <= np.median(p[pos]))
    hard_neg = neg & (p >= np.quantile(p[neg], 0.90))
    if int(hard_pos.sum()) < 10 or int(hard_neg.sum()) < 10:
        return [], ""
    gaps: list[dict] = []
    for column in features:
        if column not in train.columns or column == label:
            continue
        if not pd.api.types.is_numeric_dtype(train[column]):
            continue
        values = pd.to_numeric(train[column].iloc[rows],
                               errors="coerce").to_numpy(np.float64)
        with np.errstate(invalid="ignore"):
            scale = float(np.nanquantile(values, 0.75)
                          - np.nanquantile(values, 0.25))
            if not np.isfinite(scale) or scale <= 0:
                scale = float(np.nanstd(values))
            if not np.isfinite(scale) or scale <= 0:
                continue
            med_pos = float(np.nanmedian(values[hard_pos]))
            med_neg = float(np.nanmedian(values[hard_neg]))
        if not (np.isfinite(med_pos) and np.isfinite(med_neg)):
            continue
        gap = (med_pos - med_neg) / scale
        if abs(gap) < 1e-9:
            continue
        gaps.append({"column": column, "gap": round(float(gap), 4),
                     "median_hard_pos": round(med_pos, 5),
                     "median_false_alarm": round(med_neg, 5)})
    gaps.sort(key=lambda g: -abs(g["gap"]))
    gaps = gaps[:top]
    if not gaps:
        return [], ""
    text = ("hard-miss positives vs false-alarm negatives differ most on: "
            + ", ".join(f"{g['column']} gap={g['gap']:+.2f} "
                        f"(med {g['median_hard_pos']:g} vs "
                        f"{g['median_false_alarm']:g})" for g in gaps))
    return gaps, text


def _template_prompt(*, dataset_description: str, label: str,
                     visible: list[str], entities: list[str],
                     values: list[str], temporal: bool, evidence: str,
                     blacklist_context: str, scorecard_text: str,
                     max_templates: int, tried: set[str],
                     seat_sections: list[str] | None = None,
                     vocabulary_summary: str | None = None,
                     depth_contract: str | None = None) -> str:
    allowed_ops = TEMPORAL_OPS if temporal else STATIC_OPS
    parts = [
        "You are the PERCEIVER of a two-stage factor-discovery system for "
        "tabular risk models.  You do NOT propose individual factors.  You "
        "propose a SEARCHABLE SET: templates with an exact functional form "
        "plus slot vocabularies; a deterministic engine expands every slot "
        "binding into a candidate and a statistical verifier (orthogonalized "
        "residual correlation with a noise floor, a chronological shadow "
        "screen, and an out-of-period gate) decides what survives.  Your job "
        "is WHERE TO LOOK; the machine does the per-candidate bookkeeping.",
        "",
        "Measured division of labour you must respect: a naive operator grid "
        "(raw a*b, a/(|b|+eps)) over the RIGHT primitives was measured to "
        "fail out-of-period -- the value of a winning factor lives in its "
        "exact form: clipping (np.maximum/np.minimum), log1p bounding, "
        "guarded division, explicit fillna policy, one-sided windows.  Write "
        "the form as you would ship it; leave only the COLUMN/PARAMETER "
        "choices as slots.",
        "",
        f"Dataset: {dataset_description}",
        f"Target column (FORBIDDEN in every expression): {label}",
        (vocabulary_summary if vocabulary_summary else
         f"Available columns (slot vocabulary): {', '.join(visible[:150])}"),
        f"Entity-key candidates (for aggregation templates): "
        f"{', '.join(entities[:24])}",
        f"Value-column candidates: {', '.join(values[:40])}",
        "",
        "Pointwise templates: `python_expr` uses df['{A}']-style slot tokens "
        "({A}, {B}, ...); every token must have a slot vocabulary entry in "
        "`slots`, an array of {\"token\": \"A\", \"values\": [columns...]} "
        "objects.  Allowed: df['col'], np.*, pd.to_numeric and vectorized "
        "Series methods; FORBIDDEN: groupby, rank, rolling, expanding, "
        "merge, join, apply/lambda, the target column.  Guard every division "
        "and logarithm and state the fillna policy explicitly.",
        "",
        "Aggregate templates: strictly-causal structured specs; the engine "
        f"expands entity_slots x value_slots x aggs x windows x shifts.  "
        f"Allowed ops for THIS dataset: {', '.join(allowed_ops)}"
        + (", periodic" if temporal else "")
        + f".  Allowed aggs: {', '.join(AGG_WHITELIST)}.  shift >= 1 is "
        "mandatory (strict past).  Use [] for fields an op does not consume "
        "(e.g. value_slots/aggs for cum_count, time_since_prev, "
        "group_count; windows for lag).  For periodic use units "
        "(hour/dayofweek) and encodings (sin/cos/raw) with entity_slots [].",
        "",
        "Slot vocabularies should be SMALL and reasoned (2-6 columns each): "
        "name the columns whose business mechanism matches the template, "
        "not everything available.",
        "",
        "DIVERSITY CONTRACT: your templates must span at least four DISTINCT "
        "mechanisms (at most two templates per mechanism) -- e.g. entity "
        "rolling history, recency/velocity, pair novelty, counterparty "
        "churn, peer-group dispersion, arithmetic composition of ranked "
        "primitives, calendar periodicity.  Tag each template with its "
        "`mechanism`.  Each ensemble seat listed below must be targeted by "
        "at least one template (`target_seats`): different seats fail on "
        "different rows, and the entry rule selects PER SEAT against that "
        "seat's own residual -- a template aimed at one seat's deficit will "
        "be picked up by exactly the seats it helps.",
    ]
    if evidence:
        parts += ["", evidence]
    if depth_contract:
        parts += ["", "S4 DEPTH CONTRACT (mandatory):", depth_contract]
    if seat_sections:
        parts += ["", "Per-seat differential evidence (each seat's own OOF "
                      "residual view; repair factors must live where these "
                      "deficits are):"]
        parts += seat_sections
    if scorecard_text:
        parts += ["", "Template scorecard from earlier cycles (family-level "
                      "evidence; revise the search set with it -- widen what "
                      "worked, replace what died):", scorecard_text]
    if blacklist_context:
        parts += ["", blacklist_context]
    if tried:
        parts += ["", "Template names already used (pick new ones): "
                      + ", ".join(sorted(tried)[:60])]
    parts += [
        "",
        f"Propose at most {max_templates} pointwise_templates and at most "
        f"{max_templates} agg_templates.  Fewer is fine; an empty array is a "
        "valid answer when no defensible mechanism remains.  Output only the "
        "requested JSON object.",
    ]
    return "\n".join(parts)


def _stub_templates(*, entities: list[str], values: list[str],
                    temporal: bool, max_templates: int) -> dict:
    numeric = [v for v in values if v] or ["amount"]
    a = numeric[0]
    b = numeric[1 % len(numeric)]
    pointwise = [{
        "name": "stub_guarded_ratio",
        "description": "log1p-bounded one-sided ratio (stub)",
        "python_expr": ("np.log1p(np.maximum(pd.to_numeric(df['{A}'], "
                        "errors='coerce').fillna(0.0), 0.0)) / "
                        "(np.abs(pd.to_numeric(df['{B}'], "
                        "errors='coerce').fillna(0.0)) + 1.0)"),
        "slots": {"A": [a, b], "B": [b, a]},
    }]
    ents = [e for e in entities[:2] if e] or entities[:1]
    if temporal:
        aggs = [
            {"name": "stub_recency", "description": "per-entity recency (stub)",
             "op": "time_since_prev",
             "entity_slots": [[e] for e in ents], "value_slots": [],
             "aggs": [], "windows": [], "shifts": [1],
             "units": [], "encodings": []},
            {"name": "stub_rolling", "description": "entity rolling mean (stub)",
             "op": "rolling_stat",
             "entity_slots": [[e] for e in ents],
             "value_slots": numeric[:1], "aggs": ["mean"],
             "windows": [5, 20], "shifts": [1],
             "units": [], "encodings": []},
        ]
    else:
        aggs = [
            {"name": "stub_groupstat", "description": "entity mean (stub)",
             "op": "group_stat",
             "entity_slots": [[e] for e in ents],
             "value_slots": numeric[:1], "aggs": ["mean"],
             "windows": [], "shifts": [], "units": [], "encodings": []},
        ]
    return {"pointwise_templates": pointwise[:max_templates],
            "agg_templates": aggs[:max_templates]}


def _normalize_pointwise(raw: dict, tried: set[str]) -> dict | None:
    name = _safe_name(raw.get("name", ""))
    expr = str(raw.get("python_expr", "")).strip()
    slots_raw = raw.get("slots")
    if isinstance(slots_raw, list):
        converted: dict[str, list] = {}
        for entry in slots_raw:
            if not isinstance(entry, dict):
                return None
            token = str(entry.get("token", "")).strip().strip("{}")
            values = entry.get("values")
            if not token or not isinstance(values, (list, tuple)):
                return None
            converted[token] = list(values)
        slots_raw = converted
    if not name or not expr or name in tried or not isinstance(slots_raw, dict):
        return None
    tokens = sorted(set(_SLOT_TOKEN.findall(expr)))
    slots: dict[str, list[str]] = {}
    for token in tokens:
        vocab = slots_raw.get(token)
        if isinstance(vocab, str):
            vocab = [vocab]
        if not isinstance(vocab, (list, tuple)) or not vocab:
            return None
        slots[token] = list(dict.fromkeys(
            str(c).strip().lower() for c in vocab if str(c).strip()))
        if not slots[token]:
            return None
    if not tokens:
        return None
    return {"kind": "pointwise", "name": name, "python_expr": expr,
            "slots": slots, "description": str(raw.get("description", "")),
            **_seat_fields(raw)}


def _normalize_agg(raw: dict, tried: set[str], temporal: bool) -> dict | None:
    name = _safe_name(raw.get("name", ""))
    op = str(raw.get("op", "")).strip().lower()
    allowed = set(TEMPORAL_OPS if temporal else STATIC_OPS) | (
        {"periodic"} if temporal else set())
    if not name or name in tried or op not in allowed:
        return None

    def _strs(key: str) -> list[str]:
        out = raw.get(key)
        if isinstance(out, str):
            out = [out]
        if not isinstance(out, (list, tuple)):
            return []
        return list(dict.fromkeys(
            str(x).strip().lower() for x in out if str(x).strip()))

    def _ints(key: str) -> list[int]:
        out = raw.get(key)
        if not isinstance(out, (list, tuple)):
            return []
        result = []
        for x in out:
            try:
                result.append(int(x))
            except (TypeError, ValueError):
                continue
        return list(dict.fromkeys(result))

    entity_slots: list[list[str]] = []
    for group in (raw.get("entity_slots") or []):
        if isinstance(group, str):
            group = [group]
        if not isinstance(group, (list, tuple)):
            continue
        cols = [str(c).strip().lower() for c in group if str(c).strip()]
        if cols and cols not in entity_slots:
            entity_slots.append(cols)
    return {"kind": "agg", "name": name, "op": op,
            "entity_slots": entity_slots, "value_slots": _strs("value_slots"),
            "aggs": _strs("aggs"), "windows": _ints("windows"),
            "shifts": _ints("shifts"), "units": _strs("units"),
            "encodings": _strs("encodings"),
            "description": str(raw.get("description", "")),
            **_seat_fields(raw)}


def propose_templates(llm, *, dataset_description: str, label: str,
                      visible: list[str], entities: list[str],
                      values: list[str], temporal: bool, evidence: str,
                      blacklist_context: str, scorecard_text: str,
                      max_templates: int, tried: set[str],
                      source: str,
                      seat_sections: list[str] | None = None,
                      vocabulary_summary: str | None = None,
                      depth_contract: str | None = None,
                      call_kind: str = "v21_templates") -> tuple[list[dict], dict]:
    if llm.stub:
        parsed = _stub_templates(entities=entities, values=values,
                                 temporal=temporal,
                                 max_templates=max_templates)
    else:
        prompt = _template_prompt(
            dataset_description=dataset_description, label=label,
            visible=[c for c in visible if c != label],
            entities=entities, values=values, temporal=temporal,
            evidence=evidence, blacklist_context=blacklist_context,
            scorecard_text=scorecard_text, max_templates=max_templates,
            tried=tried, seat_sections=seat_sections,
            vocabulary_summary=vocabulary_summary,
            depth_contract=depth_contract)
        parsed = llm.call_json(call_kind, prompt, TEMPLATE_SCHEMA)
    templates: list[dict] = []
    rejected: list[dict] = []
    for raw in (parsed.get("pointwise_templates") or [])[:max_templates]:
        normalized = _normalize_pointwise(dict(raw), tried)
        if normalized is None:
            rejected.append({"raw": raw, "kind": "pointwise"})
            continue
        tried.add(normalized["name"])
        templates.append(normalized)
    for raw in (parsed.get("agg_templates") or [])[:max_templates]:
        normalized = _normalize_agg(dict(raw), tried, temporal)
        if normalized is None:
            rejected.append({"raw": raw, "kind": "agg"})
            continue
        tried.add(normalized["name"])
        templates.append(normalized)
    audit = {"source": source, "stub": bool(llm.stub),
             "n_templates": len(templates), "n_rejected": len(rejected),
             "templates": [dict(t) for t in templates],
             "rejected": rejected,
             "depth_contract": depth_contract}
    return templates, audit


def _expand_pointwise(template: dict, *, columns: set[str], label: str,
                      cap: int) -> tuple[list[FeatureSpec], int]:
    tokens = sorted(template["slots"])
    vocabularies = []
    for token in tokens:
        vocab = [c for c in template["slots"][token]
                 if c in columns and c != label]
        if not vocab:
            return [], 0
        vocabularies.append(vocab)
    specs: list[FeatureSpec] = []
    invalid = 0
    for binding in itertools.product(*vocabularies):
        if len(set(binding)) != len(binding):
            continue
        expr = template["python_expr"]
        for token, column in zip(tokens, binding):
            expr = expr.replace("{" + token + "}", column)
        try:
            compile_pointwise_expression(expr, forbidden_columns=(label,))
        except Exception:
            invalid += 1
            continue
        name = "v21_" + template["name"] + "__" + "__".join(
            _safe_name(c, 24) for c in binding)
        specs.append(FeatureSpec(
            name=name, expression=expr, family=f"tmpl_{template['name']}",
            source="v21_template_sweep", executor="pointwise",
            description=template.get("description", "")))
        if len(specs) >= cap:
            break
    return specs, invalid


def _expand_agg(template: dict, *, columns: list[str], label: str,
                temporal: bool, cap: int) -> tuple[list[FeatureSpec], int]:
    entity_slots = template["entity_slots"] or [[]]
    value_slots = template["value_slots"] or [""]
    aggs = template["aggs"] or [""]
    windows = template["windows"] or [1]
    shifts = template["shifts"] or [1]
    units = template["units"] or [""]
    encodings = template["encodings"] or [""]
    specs: list[FeatureSpec] = []
    invalid = 0
    seen: set[str] = set()
    combos = (itertools.product([[]], [""], [""], [1], [1], units, encodings)
              if template["op"] == "periodic" else
              itertools.product(entity_slots, value_slots, aggs, windows,
                                shifts, [""], [""]))
    for entity_cols, value_col, agg, window, shift, unit, encoding in combos:
        raw = {"op": template["op"], "entity_cols": list(entity_cols),
               "value_col": value_col, "agg": agg, "window": window,
               "shift": shift, "unit": unit, "encoding": encoding}
        try:
            normalized = validate_agg_spec(
                raw, columns=columns, label=label, temporal=temporal)
        except ValueError:
            invalid += 1
            continue
        expression = json.dumps(normalized, sort_keys=True)
        if expression in seen:
            continue
        seen.add(expression)
        pieces = [template["op"]]
        if normalized.get("entity_cols"):
            pieces.append("_".join(_safe_name(c, 16)
                                   for c in normalized["entity_cols"]))
        if normalized.get("value_col"):
            pieces.append(_safe_name(normalized["value_col"], 16))
        if normalized.get("agg"):
            pieces.append(str(normalized["agg"]))
        if normalized.get("window"):
            pieces.append(f"w{normalized['window']}")
        if normalized.get("shift"):
            pieces.append(f"s{normalized['shift']}")
        if normalized.get("unit"):
            pieces.append(f"{normalized['unit']}_{normalized['encoding']}")
        name = "v21_" + template["name"] + "__" + "_".join(pieces)
        specs.append(FeatureSpec(
            name=name, expression=expression,
            family=f"tmpl_{template['name']}", source="v21_template_sweep",
            executor="agg", description=template.get("description", "")))
        if len(specs) >= cap:
            break
    return specs, invalid


def expand_templates(templates: list[dict], *, visible: list[str], label: str,
                     temporal: bool, existing: set[str], cap_per_template: int,
                     cap_total: int) -> tuple[list[FeatureSpec], dict]:
    column_set = {str(c).lower() for c in visible if str(c) != label}
    specs: list[FeatureSpec] = []
    per_template: list[dict] = []
    for template in templates:
        if len(specs) >= cap_total:
            per_template.append({"template": template["name"],
                                 "skipped": "cap_total_reached"})
            continue
        cap = min(cap_per_template, cap_total - len(specs))
        if template["kind"] == "pointwise":
            expanded, invalid = _expand_pointwise(
                template, columns=column_set, label=label, cap=cap)
        else:
            expanded, invalid = _expand_agg(
                template, columns=sorted(column_set), label=label,
                temporal=temporal, cap=cap)
        fresh = [spec for spec in expanded if spec.name not in existing]
        specs.extend(fresh)
        per_template.append({
            "template": template["name"], "kind": template["kind"],
            "n_expanded": len(expanded), "n_invalid": int(invalid),
            "n_deduped": len(expanded) - len(fresh),
            "candidates": [spec.name for spec in fresh]})
    audit = {"n_templates": len(templates), "n_candidates": len(specs),
             "cap_per_template": int(cap_per_template),
             "cap_total": int(cap_total), "per_template": per_template}
    return specs, audit


def template_scorecard(*, round_index: int, cycle_index: int,
                       templates: list[dict], families: dict[str, str],
                       seat_rows: list[dict], floor: float,
                       expand_audit: dict) -> tuple[dict, str]:
    def template_of(name: str) -> str | None:
        family = str(families.get(name, ""))
        return family[5:] if family.startswith("tmpl_") else None

    by_name = {t["name"]: t for t in templates}
    rows: dict[str, dict] = {}
    for entry in (expand_audit or {}).get("per_template", []):
        tname = entry.get("template")
        if tname:
            rows[tname] = {
                "template": tname, "kind": entry.get("kind"),
                "n_candidates": len(entry.get("candidates", [])),
                "n_invalid": int(entry.get("n_invalid", 0) or 0),
                "n_scored": 0, "best_abs_s": 0.0, "best_candidate": None,
                "n_admitted": 0, "admitted": [], "seats": {},
                "n_shadow_dropped": 0}
    for template in templates:
        rows.setdefault(template["name"], {
            "template": template["name"], "kind": template["kind"],
            "n_candidates": 0, "n_invalid": 0, "n_scored": 0,
            "best_abs_s": 0.0, "best_candidate": None, "n_admitted": 0,
            "admitted": [], "seats": {}, "n_shadow_dropped": 0})
    for tname, row in rows.items():
        template = by_name.get(tname, {})
        row["mechanism"] = template.get("mechanism", "unspecified")
        row["target_seats"] = list(template.get("target_seats", []))
    for seat_row in seat_rows:
        seat = str(seat_row.get("seat"))
        for name, score in (seat_row.get("first_pass") or {}).items():
            tname = template_of(name)
            if tname is None or tname not in rows:
                continue
            row = rows[tname]
            row["n_scored"] += 1
            if abs(float(score)) > row["best_abs_s"]:
                row["best_abs_s"] = abs(float(score))
                row["best_candidate"] = name
        for name in seat_row.get("admitted") or []:
            tname = template_of(name)
            if tname is None or tname not in rows:
                continue
            row = rows[tname]
            row["n_admitted"] += 1
            if name not in row["admitted"]:
                row["admitted"].append(name)
            row["seats"][seat] = int(row["seats"].get(seat, 0)) + 1
        for name in seat_row.get("shadow_dropped") or []:
            tname = template_of(name)
            if tname is not None and tname in rows:
                rows[tname]["n_shadow_dropped"] += 1
    for row in rows.values():
        row["best_abs_s"] = round(float(row["best_abs_s"]), 6)

    lines = [f"[round {round_index} cycle {cycle_index}] entry noise floor "
             f"= {floor:.5f}; per-template outcome:"]
    for row in sorted(rows.values(), key=lambda r: (-r["best_abs_s"],
                                                    r["template"])):
        verdict = ("ADMITTED" if row["n_admitted"] else
                   ("signal below floor" if row["best_abs_s"] < floor
                    else "scored above floor but not kept"))
        targeted = set(row.get("target_seats") or [])
        landed = set(row["seats"])
        aim_note = ""
        if targeted:
            aim_note = (", aim held" if landed & targeted else
                        (f", aimed at {sorted(targeted)} but landed on "
                         f"{sorted(landed)}" if landed else
                         f", aimed at {sorted(targeted)}, landed nowhere"))
        lines.append(
            f"- {row['template']} ({row['kind']}, "
            f"mechanism={row.get('mechanism', 'unspecified')}): "
            f"{row['n_candidates']} bindings, best |S|={row['best_abs_s']:.5f}"
            + (f" ({row['best_candidate']})" if row["best_candidate"] else "")
            + f", admitted {row['n_admitted']} on "
            f"{len(row['seats'])} seat(s)"
            + (f", {row['n_shadow_dropped']} shadow-dropped"
               if row["n_shadow_dropped"] else "")
            + aim_note + f" -> {verdict}")
    payload = {"round": int(round_index), "cycle": int(cycle_index),
               "floor": float(floor),
               "templates": sorted(rows.values(),
                                   key=lambda r: r["template"])}
    return payload, "\n".join(lines)
