from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CandidateOrigin(str, Enum):
    MODEL_DERIVED = "model_derived"
    LLM_INFERRED = "llm_inferred"


@dataclass(frozen=True)
class GateDecision:
    origin: CandidateOrigin
    requires_gate: bool
    accepted: bool
    reason: str


def gate_decision(
    origin: CandidateOrigin | str,
    *,
    gate_accepted: bool | None = None,
) -> GateDecision:
    origin = CandidateOrigin(origin)
    if origin is CandidateOrigin.MODEL_DERIVED:
        return GateDecision(
            origin=origin,
            requires_gate=False,
            accepted=True,
            reason="leak_safe_model_pipeline_bypasses_effect_gate",
        )
    if gate_accepted is None:
        return GateDecision(
            origin=origin,
            requires_gate=True,
            accepted=False,
            reason="llm_hypothesis_requires_consumer_gate",
        )
    return GateDecision(
        origin=origin,
        requires_gate=True,
        accepted=bool(gate_accepted),
        reason=("llm_consumer_gate_passed" if gate_accepted
                else "llm_consumer_gate_rejected"),
    )
