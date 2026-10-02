"""Predictions committed before their outcomes are known."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mosaic.kernel.events import Event
from mosaic.kernel.identity import AgentIdentity

from ._common import new_entity_id, require_text, validate_confidence


EVENT_TYPE = "prediction.created"


@dataclass(frozen=True, slots=True)
class Prediction:
    prediction_id: str
    hypothesis_id: str
    statement: str
    confidence: float
    conditions: Mapping[str, Any]
    created_by: str
    event_id: str

    @classmethod
    def create_event(
        cls,
        *,
        stream_id: str,
        hypothesis_id: str,
        statement: str,
        confidence: float,
        actor: AgentIdentity,
        conditions: Mapping[str, Any] | None = None,
        prediction_id: str | None = None,
    ) -> Event:
        hypothesis_id = require_text(hypothesis_id, "hypothesis_id")
        statement = require_text(statement, "statement")
        normalized_confidence = validate_confidence(confidence)
        assert normalized_confidence is not None

        return Event(
            event_type=EVENT_TYPE,
            stream_id=stream_id,
            actor=actor,
            payload={
                "prediction_id": prediction_id or new_entity_id("P"),
                "hypothesis_id": hypothesis_id,
                "statement": statement,
                "confidence": normalized_confidence,
                "conditions": dict(conditions or {}),
            },
        )

    @classmethod
    def from_event(cls, event: Event) -> "Prediction":
        if event.event_type != EVENT_TYPE:
            raise ValueError(f"expected {EVENT_TYPE}, got {event.event_type}")
        p = event.payload
        return cls(
            prediction_id=str(p["prediction_id"]),
            hypothesis_id=str(p["hypothesis_id"]),
            statement=str(p["statement"]),
            confidence=float(p["confidence"]),
            conditions=p["conditions"],
            created_by=event.actor.ref,
            event_id=event.event_id,
        )
