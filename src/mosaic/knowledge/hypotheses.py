"""Hypotheses: explicit interpretations that must remain testable."""

from __future__ import annotations

from dataclasses import dataclass

from mosaic.kernel.events import Event
from mosaic.kernel.identity import AgentIdentity

from ._common import new_entity_id, require_text, validate_confidence


EVENT_TYPE = "hypothesis.proposed"


@dataclass(frozen=True, slots=True)
class Hypothesis:
    hypothesis_id: str
    claim: str
    initial_confidence: float | None
    proposed_by: str
    event_id: str

    @classmethod
    def create_event(
        cls,
        *,
        stream_id: str,
        claim: str,
        actor: AgentIdentity,
        confidence: float | None = None,
        hypothesis_id: str | None = None,
    ) -> Event:
        claim = require_text(claim, "claim")
        confidence = validate_confidence(confidence)

        return Event(
            event_type=EVENT_TYPE,
            stream_id=stream_id,
            actor=actor,
            payload={
                "hypothesis_id": hypothesis_id or new_entity_id("H"),
                "claim": claim,
                "initial_confidence": confidence,
            },
        )

    @classmethod
    def from_event(cls, event: Event) -> "Hypothesis":
        if event.event_type != EVENT_TYPE:
            raise ValueError(f"expected {EVENT_TYPE}, got {event.event_type}")
        p = event.payload
        return cls(
            hypothesis_id=str(p["hypothesis_id"]),
            claim=str(p["claim"]),
            initial_confidence=(
                None
                if p["initial_confidence"] is None
                else float(p["initial_confidence"])
            ),
            proposed_by=event.actor.ref,
            event_id=event.event_id,
        )
