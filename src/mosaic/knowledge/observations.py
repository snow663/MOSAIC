"""Observation records: measured or supplied facts, not interpretations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mosaic.kernel.events import Event
from mosaic.kernel.identity import AgentIdentity, KERNEL_IDENTITY

from ._common import new_entity_id, require_text


EVENT_TYPE = "observation.recorded"


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    name: str
    value: Any
    unit: str | None
    source: str
    uncertainty: float | None
    context_id: str | None
    context_label: str | None
    event_id: str

    @classmethod
    def create_event(
        cls,
        *,
        stream_id: str,
        name: str,
        value: Any,
        source: str,
        unit: str | None = None,
        uncertainty: float | None = None,
        context_id: str | None = None,
        context_label: str | None = None,
        actor: AgentIdentity = KERNEL_IDENTITY,
        observation_id: str | None = None,
    ) -> Event:
        name = require_text(name, "name")
        source = require_text(source, "source")
        if unit is not None:
            unit = require_text(unit, "unit")
        if uncertainty is not None:
            uncertainty = float(uncertainty)
            if uncertainty < 0:
                raise ValueError("uncertainty must be non-negative")
        if context_id is not None:
            context_id = require_text(context_id, "context_id")
        if context_label is not None:
            context_label = require_text(context_label, "context_label")
        if (context_id is None) != (context_label is None):
            raise ValueError(
                "context_id and context_label must be supplied together"
            )

        return Event(
            event_type=EVENT_TYPE,
            stream_id=stream_id,
            actor=actor,
            payload={
                "observation_id": observation_id or new_entity_id("O"),
                "name": name,
                "value": value,
                "unit": unit,
                "source": source,
                "uncertainty": uncertainty,
                "context_id": context_id,
                "context_label": context_label,
            },
        )

    @classmethod
    def from_event(cls, event: Event) -> "Observation":
        if event.event_type != EVENT_TYPE:
            raise ValueError(f"expected {EVENT_TYPE}, got {event.event_type}")
        p = event.payload
        return cls(
            observation_id=str(p["observation_id"]),
            name=str(p["name"]),
            value=p["value"],
            unit=None if p["unit"] is None else str(p["unit"]),
            source=str(p["source"]),
            uncertainty=(
                None if p["uncertainty"] is None else float(p["uncertainty"])
            ),
            context_id=(
                None
                if p.get("context_id") is None
                else str(p["context_id"])
            ),
            context_label=(
                None
                if p.get("context_label") is None
                else str(p["context_label"])
            ),
            event_id=event.event_id,
        )
