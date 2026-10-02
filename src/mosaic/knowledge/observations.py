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
            event_id=event.event_id,
        )
