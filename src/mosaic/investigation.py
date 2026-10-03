"""Replayable investigation state built entirely from kernel events."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from mosaic.kernel.events import Event
from mosaic.kernel.identity import AgentIdentity, KERNEL_IDENTITY
from mosaic.kernel.store import SQLiteEventStore, StoredEvent
from mosaic.knowledge.hypotheses import (
    EVENT_TYPE as HYPOTHESIS_PROPOSED,
    Hypothesis,
)
from mosaic.knowledge.observations import (
    EVENT_TYPE as OBSERVATION_RECORDED,
    Observation,
)
from mosaic.knowledge.predictions import (
    EVENT_TYPE as PREDICTION_CREATED,
    Prediction,
)
from mosaic.knowledge.relations import (
    EVENT_TYPE as RELATION_ADDED,
    Relation,
    RelationType,
)
from mosaic.snapshot import InvestigationSnapshot


class Investigation:
    """Current epistemic state reconstructed from an append-only event stream."""

    def __init__(self, store: SQLiteEventStore, investigation_id: str) -> None:
        investigation_id = investigation_id.strip()
        if not investigation_id:
            raise ValueError("investigation_id must be non-empty")

        self.store = store
        self.investigation_id = investigation_id
        self.observations: dict[str, Observation] = {}
        self.hypotheses: dict[str, Hypothesis] = {}
        self.predictions: dict[str, Prediction] = {}
        self.relations: list[Relation] = []
        self.last_sequence = 0
        self.replay()

    def replay(self) -> None:
        """Discard derived state and rebuild it from the ledger."""

        self.observations.clear()
        self.hypotheses.clear()
        self.predictions.clear()
        self.relations.clear()
        self.last_sequence = 0

        for stored in self.store.events_for_stream(self.investigation_id):
            self._apply(stored)

    def _apply(self, stored: StoredEvent) -> None:
        event = stored.event

        if event.event_type == OBSERVATION_RECORDED:
            observation = Observation.from_event(event)
            self._ensure_new_entity(observation.observation_id)
            self.observations[observation.observation_id] = observation

        elif event.event_type == HYPOTHESIS_PROPOSED:
            hypothesis = Hypothesis.from_event(event)
            self._ensure_new_entity(hypothesis.hypothesis_id)
            self.hypotheses[hypothesis.hypothesis_id] = hypothesis

        elif event.event_type == PREDICTION_CREATED:
            prediction = Prediction.from_event(event)
            self._ensure_new_entity(prediction.prediction_id)
            if prediction.hypothesis_id not in self.hypotheses:
                raise ValueError(
                    "prediction references unknown hypothesis "
                    f"{prediction.hypothesis_id}"
                )
            self.predictions[prediction.prediction_id] = prediction

        elif event.event_type == RELATION_ADDED:
            relation = Relation.from_event(event)
            self._ensure_new_entity(relation.relation_id)
            if not self.has_entity(relation.source_id):
                raise ValueError(
                    f"relation source does not exist: {relation.source_id}"
                )
            if not self.has_entity(relation.target_id):
                raise ValueError(
                    f"relation target does not exist: {relation.target_id}"
                )
            self.relations.append(relation)

        self.last_sequence = stored.sequence

    def _ensure_new_entity(self, entity_id: str) -> None:
        if self.has_entity(entity_id):
            raise ValueError(f"duplicate entity id: {entity_id}")

    def has_entity(self, entity_id: str) -> bool:
        return (
            entity_id in self.observations
            or entity_id in self.hypotheses
            or entity_id in self.predictions
            or any(r.relation_id == entity_id for r in self.relations)
        )

    def _append_and_apply(self, event: Event) -> StoredEvent:
        stored = self.store.append(event)
        self._apply(stored)
        return stored

    def record_observation(
        self,
        *,
        name: str,
        value: Any,
        source: str,
        unit: str | None = None,
        uncertainty: float | None = None,
        context_id: str | None = None,
        context_label: str | None = None,
        actor: AgentIdentity = KERNEL_IDENTITY,
    ) -> Observation:
        event = Observation.create_event(
            stream_id=self.investigation_id,
            name=name,
            value=value,
            source=source,
            unit=unit,
            uncertainty=uncertainty,
            context_id=context_id,
            context_label=context_label,
            actor=actor,
        )
        self._append_and_apply(event)
        return self.observations[str(event.payload["observation_id"])]

    def propose_hypothesis(
        self,
        *,
        claim: str,
        actor: AgentIdentity,
        confidence: float | None = None,
    ) -> Hypothesis:
        event = Hypothesis.create_event(
            stream_id=self.investigation_id,
            claim=claim,
            actor=actor,
            confidence=confidence,
        )
        self._append_and_apply(event)
        return self.hypotheses[str(event.payload["hypothesis_id"])]

    def add_prediction(
        self,
        *,
        hypothesis_id: str,
        statement: str,
        confidence: float,
        actor: AgentIdentity,
        conditions: Mapping[str, Any] | None = None,
    ) -> Prediction:
        if hypothesis_id not in self.hypotheses:
            raise KeyError(f"unknown hypothesis: {hypothesis_id}")

        event = Prediction.create_event(
            stream_id=self.investigation_id,
            hypothesis_id=hypothesis_id,
            statement=statement,
            confidence=confidence,
            actor=actor,
            conditions=conditions,
        )
        self._append_and_apply(event)
        return self.predictions[str(event.payload["prediction_id"])]

    def link(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType | str,
        *,
        actor: AgentIdentity,
        rationale: str | None = None,
    ) -> Relation:
        if not self.has_entity(source_id):
            raise KeyError(f"unknown source entity: {source_id}")
        if not self.has_entity(target_id):
            raise KeyError(f"unknown target entity: {target_id}")

        event = Relation.create_event(
            stream_id=self.investigation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            actor=actor,
            rationale=rationale,
        )
        self._append_and_apply(event)
        return self.relations[-1]

    def snapshot(self) -> InvestigationSnapshot:
        """Return a frozen point-in-time view for independent analysis."""

        return InvestigationSnapshot(
            investigation_id=self.investigation_id,
            ledger_sequence=self.last_sequence,
            observations=tuple(self.observations.values()),
            hypotheses=tuple(self.hypotheses.values()),
            predictions=tuple(self.predictions.values()),
            relations=tuple(self.relations),
        )

    def predictions_for(self, hypothesis_id: str) -> tuple[Prediction, ...]:
        return tuple(
            prediction
            for prediction in self.predictions.values()
            if prediction.hypothesis_id == hypothesis_id
        )

    def relations_for(self, entity_id: str) -> tuple[Relation, ...]:
        return tuple(
            relation
            for relation in self.relations
            if relation.source_id == entity_id or relation.target_id == entity_id
        )

    def iter_entities(
        self,
    ) -> Iterable[Observation | Hypothesis | Prediction | Relation]:
        yield from self.observations.values()
        yield from self.hypotheses.values()
        yield from self.predictions.values()
        yield from self.relations
