"""Explicit edges in the MOSAIC epistemic graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mosaic.kernel.events import Event
from mosaic.kernel.identity import AgentIdentity

from ._common import new_entity_id, require_text


EVENT_TYPE = "relation.added"


class RelationType(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ALTERNATIVE_TO = "alternative_to"
    DEPENDS_ON = "depends_on"
    SUPERSEDES = "supersedes"
    TESTED_BY = "tested_by"
    DERIVED_FROM = "derived_from"


@dataclass(frozen=True, slots=True)
class Relation:
    relation_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    rationale: str | None
    created_by: str
    event_id: str

    @classmethod
    def create_event(
        cls,
        *,
        stream_id: str,
        source_id: str,
        target_id: str,
        relation_type: RelationType | str,
        actor: AgentIdentity,
        rationale: str | None = None,
        relation_id: str | None = None,
    ) -> Event:
        source_id = require_text(source_id, "source_id")
        target_id = require_text(target_id, "target_id")
        if source_id == target_id:
            raise ValueError("a relation cannot point an entity at itself")

        relation_type = RelationType(relation_type)
        if rationale is not None:
            rationale = require_text(rationale, "rationale")

        return Event(
            event_type=EVENT_TYPE,
            stream_id=stream_id,
            actor=actor,
            payload={
                "relation_id": relation_id or new_entity_id("R"),
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type.value,
                "rationale": rationale,
            },
        )

    @classmethod
    def from_event(cls, event: Event) -> "Relation":
        if event.event_type != EVENT_TYPE:
            raise ValueError(f"expected {EVENT_TYPE}, got {event.event_type}")
        p = event.payload
        return cls(
            relation_id=str(p["relation_id"]),
            source_id=str(p["source_id"]),
            target_id=str(p["target_id"]),
            relation_type=RelationType(str(p["relation_type"])),
            rationale=None if p["rationale"] is None else str(p["rationale"]),
            created_by=event.actor.ref,
            event_id=event.event_id,
        )
