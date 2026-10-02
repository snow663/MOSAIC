"""Cross-domain synthesis and controlled promotion into the hypothesis graph."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable
from uuid import uuid4

from mosaic.investigation import Investigation
from mosaic.kernel.events import Event
from mosaic.kernel.identity import AgentIdentity
from mosaic.kernel.store import SQLiteEventStore
from mosaic.knowledge.hypotheses import EVENT_TYPE as HYPOTHESIS_PROPOSED
from mosaic.knowledge.observations import EVENT_TYPE as OBSERVATION_RECORDED
from mosaic.knowledge.predictions import EVENT_TYPE as PREDICTION_CREATED
from mosaic.knowledge.relations import EVENT_TYPE as RELATION_ADDED
from mosaic.snapshot import InvestigationSnapshot

from .independent import AgentProtocolError
from .roles import ExaminationDisposition, ExaminationResult


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid4()}"


def _text(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, MappingABC):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("mapping keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


class FindingRelationType(StrEnum):
    """Relationship between two independently examined findings."""

    AGREES_WITH = "agrees_with"
    CONTRADICTS = "contradicts"
    OVERLAPS = "overlaps"
    DEPENDS_ON = "depends_on"
    DISTINCT = "distinct"


@dataclass(frozen=True, slots=True)
class FindingRelation:
    source_examination_id: str
    target_examination_id: str
    relation_type: FindingRelationType
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_examination_id",
            _text(self.source_examination_id, "source_examination_id"),
        )
        object.__setattr__(
            self,
            "target_examination_id",
            _text(self.target_examination_id, "target_examination_id"),
        )
        if self.source_examination_id == self.target_examination_id:
            raise ValueError("a finding relation cannot point to itself")
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale"))


@dataclass(frozen=True, slots=True)
class MinorityReport:
    """A finding that must remain visible even if it is not the majority view."""

    examination_id: str
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "examination_id",
            _text(self.examination_id, "examination_id"),
        )
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale"))


@dataclass(frozen=True, slots=True)
class PromotionCandidate:
    """One examined hypothesis selected for institutional graph promotion."""

    examination_id: str
    hypothesis_index: int
    rationale: str
    minority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "examination_id",
            _text(self.examination_id, "examination_id"),
        )
        if self.hypothesis_index < 0:
            raise ValueError("hypothesis_index must be non-negative")
        object.__setattr__(self, "rationale", _text(self.rationale, "rationale"))


@dataclass(frozen=True, slots=True)
class CrossDomainSynthesis:
    """Structured comparison of completed private examinations."""

    reviewer_ref: str
    investigation_id: str
    source_ledger_sequence: int
    finding_ids: tuple[str, ...]
    summary: str
    relations: tuple[FindingRelation, ...] = ()
    minority_reports: tuple[MinorityReport, ...] = ()
    promotions: tuple[PromotionCandidate, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    statistics: Mapping[str, Any] = field(default_factory=dict)
    synthesis_id: str = field(default_factory=lambda: _id("SYN"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "reviewer_ref", _text(self.reviewer_ref, "reviewer_ref"))
        object.__setattr__(
            self,
            "investigation_id",
            _text(self.investigation_id, "investigation_id"),
        )
        if self.source_ledger_sequence < 0:
            raise ValueError("source_ledger_sequence must be non-negative")
        if not self.finding_ids:
            raise ValueError("synthesis requires at least one finding")
        normalized_ids = tuple(_text(item, "finding_id") for item in self.finding_ids)
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("finding_ids must be unique")
        object.__setattr__(self, "finding_ids", normalized_ids)
        object.__setattr__(self, "summary", _text(self.summary, "summary"))
        object.__setattr__(
            self,
            "unresolved_questions",
            tuple(_text(item, "unresolved_question") for item in self.unresolved_questions),
        )
        object.__setattr__(self, "statistics", _freeze_json(self.statistics))


@dataclass(frozen=True, slots=True)
class PromotionRecord:
    """Trace from an examined proposal into durable graph entities."""

    examination_id: str
    source_proposal_id: str
    source_hypothesis_index: int
    hypothesis_id: str
    prediction_ids: tuple[str, ...]
    minority: bool


@runtime_checkable
class CrossDomainReviewer(Protocol):
    """Compares examined findings without directly mutating research state."""

    identity: AgentIdentity

    async def synthesize(
        self,
        snapshot: InvestigationSnapshot,
        findings: Sequence[ExaminationResult],
    ) -> CrossDomainSynthesis:
        ...


class SynthesisProtocolError(AgentProtocolError):
    """Raised when cross-domain synthesis violates institutional constraints."""


class SynthesisSession:
    """Validate and audit one cross-domain review."""

    def __init__(self, *, audit_store: SQLiteEventStore | None = None) -> None:
        self.audit_store = audit_store

    @staticmethod
    def _validate(
        synthesis: CrossDomainSynthesis,
        *,
        snapshot: InvestigationSnapshot,
        findings: Sequence[ExaminationResult],
        reviewer: CrossDomainReviewer,
    ) -> None:
        if synthesis.reviewer_ref != reviewer.identity.ref:
            raise SynthesisProtocolError("cross-domain reviewer attribution mismatch")
        if synthesis.investigation_id != snapshot.investigation_id:
            raise SynthesisProtocolError("synthesis targets a different investigation")
        if synthesis.source_ledger_sequence != snapshot.ledger_sequence:
            raise SynthesisProtocolError("synthesis targets a different snapshot")

        finding_map = {finding.examination_id: finding for finding in findings}
        if len(finding_map) != len(findings):
            raise SynthesisProtocolError("examination ids must be unique")

        expected_ids = set(finding_map)
        actual_ids = set(synthesis.finding_ids)
        missing = expected_ids - actual_ids
        unknown = actual_ids - expected_ids
        if missing:
            raise SynthesisProtocolError(
                "synthesis omitted examined findings: " + ", ".join(sorted(missing))
            )
        if unknown:
            raise SynthesisProtocolError(
                "synthesis referenced unknown findings: " + ", ".join(sorted(unknown))
            )

        for relation in synthesis.relations:
            if relation.source_examination_id not in expected_ids:
                raise SynthesisProtocolError("finding relation has unknown source")
            if relation.target_examination_id not in expected_ids:
                raise SynthesisProtocolError("finding relation has unknown target")

        for report in synthesis.minority_reports:
            if report.examination_id not in expected_ids:
                raise SynthesisProtocolError("minority report references unknown finding")

        seen_promotions: set[tuple[str, int]] = set()
        for candidate in synthesis.promotions:
            if candidate.examination_id not in expected_ids:
                raise SynthesisProtocolError("promotion references unknown finding")
            key = (candidate.examination_id, candidate.hypothesis_index)
            if key in seen_promotions:
                raise SynthesisProtocolError("duplicate hypothesis promotion")
            seen_promotions.add(key)

            finding = finding_map[candidate.examination_id]
            if finding.disposition is ExaminationDisposition.REJECTED:
                raise SynthesisProtocolError("rejected findings cannot be promoted")
            if candidate.hypothesis_index >= len(finding.proposal.hypotheses):
                raise SynthesisProtocolError("promotion hypothesis index is out of range")

    async def run(
        self,
        snapshot: InvestigationSnapshot,
        findings: Sequence[ExaminationResult],
        reviewer: CrossDomainReviewer,
    ) -> CrossDomainSynthesis:
        if not findings:
            raise ValueError("at least one examined finding is required")

        synthesis = await reviewer.synthesize(snapshot, findings)
        self._validate(
            synthesis,
            snapshot=snapshot,
            findings=findings,
            reviewer=reviewer,
        )

        if self.audit_store is not None:
            self.audit_store.append(
                Event(
                    event_type="synthesis.completed",
                    stream_id=snapshot.investigation_id,
                    actor=reviewer.identity,
                    correlation_id=synthesis.synthesis_id,
                    payload={
                        "synthesis_id": synthesis.synthesis_id,
                        "source_ledger_sequence": synthesis.source_ledger_sequence,
                        "finding_ids": list(synthesis.finding_ids),
                        "summary": synthesis.summary,
                        "relations": [
                            {
                                "source_examination_id": relation.source_examination_id,
                                "target_examination_id": relation.target_examination_id,
                                "relation_type": relation.relation_type.value,
                                "rationale": relation.rationale,
                            }
                            for relation in synthesis.relations
                        ],
                        "minority_reports": [
                            {
                                "examination_id": report.examination_id,
                                "rationale": report.rationale,
                            }
                            for report in synthesis.minority_reports
                        ],
                        "promotions": [
                            {
                                "examination_id": candidate.examination_id,
                                "hypothesis_index": candidate.hypothesis_index,
                                "rationale": candidate.rationale,
                                "minority": candidate.minority,
                            }
                            for candidate in synthesis.promotions
                        ],
                        "unresolved_questions": list(synthesis.unresolved_questions),
                        "statistics": dict(synthesis.statistics),
                    },
                )
            )

        return synthesis


class PromotionExecutor:
    """Apply an already-validated synthesis plan to the hypothesis graph."""

    _EPISTEMIC_EVENT_TYPES = {
        OBSERVATION_RECORDED,
        HYPOTHESIS_PROPOSED,
        PREDICTION_CREATED,
        RELATION_ADDED,
    }

    @classmethod
    def _assert_no_epistemic_drift(
        cls,
        investigation: Investigation,
        source_sequence: int,
    ) -> None:
        changed = [
            stored
            for stored in investigation.store.events_for_stream(
                investigation.investigation_id
            )
            if stored.sequence > source_sequence
            and stored.event.event_type in cls._EPISTEMIC_EVENT_TYPES
        ]
        if changed:
            raise SynthesisProtocolError(
                "investigation changed after the synthesis snapshot; "
                "rerun cross-domain review before promotion"
            )

    @staticmethod
    def _prevalidate(
        *,
        investigation: Investigation,
        synthesis: CrossDomainSynthesis,
        findings: Sequence[ExaminationResult],
        thinker_identities: Mapping[str, AgentIdentity],
        promoter: AgentIdentity,
    ) -> list[tuple[PromotionCandidate, ExaminationResult, AgentIdentity]]:
        if synthesis.investigation_id != investigation.investigation_id:
            raise SynthesisProtocolError("promotion targets a different investigation")
        if promoter.ref != synthesis.reviewer_ref:
            raise SynthesisProtocolError("promotion actor is not the synthesis reviewer")

        finding_map = {finding.examination_id: finding for finding in findings}
        if set(synthesis.finding_ids) != set(finding_map):
            raise SynthesisProtocolError("promotion findings do not match synthesis")

        prepared: list[
            tuple[PromotionCandidate, ExaminationResult, AgentIdentity]
        ] = []
        for candidate in synthesis.promotions:
            finding = finding_map[candidate.examination_id]
            if finding.disposition is ExaminationDisposition.REJECTED:
                raise SynthesisProtocolError("rejected findings cannot be promoted")
            if candidate.hypothesis_index >= len(finding.proposal.hypotheses):
                raise SynthesisProtocolError("promotion hypothesis index is out of range")

            identity = thinker_identities.get(finding.thinker_ref)
            if identity is None:
                raise SynthesisProtocolError(
                    f"missing identity for source Thinker {finding.thinker_ref}"
                )
            if identity.ref != finding.thinker_ref:
                raise SynthesisProtocolError("Thinker identity registry mismatch")
            prepared.append((candidate, finding, identity))

        return prepared

    @classmethod
    def apply(
        cls,
        *,
        investigation: Investigation,
        synthesis: CrossDomainSynthesis,
        findings: Sequence[ExaminationResult],
        thinker_identities: Mapping[str, AgentIdentity],
        promoter: AgentIdentity,
    ) -> tuple[PromotionRecord, ...]:
        cls._assert_no_epistemic_drift(
            investigation,
            synthesis.source_ledger_sequence,
        )
        prepared = cls._prevalidate(
            investigation=investigation,
            synthesis=synthesis,
            findings=findings,
            thinker_identities=thinker_identities,
            promoter=promoter,
        )

        records: list[PromotionRecord] = []
        for candidate, finding, thinker_identity in prepared:
            source = finding.proposal.hypotheses[candidate.hypothesis_index]
            hypothesis = investigation.propose_hypothesis(
                claim=source.claim,
                actor=thinker_identity,
                confidence=source.confidence,
            )

            prediction_ids: list[str] = []
            for proposed_prediction in source.predictions:
                prediction = investigation.add_prediction(
                    hypothesis_id=hypothesis.hypothesis_id,
                    statement=proposed_prediction.statement,
                    confidence=proposed_prediction.confidence,
                    actor=thinker_identity,
                    conditions=proposed_prediction.conditions,
                )
                prediction_ids.append(prediction.prediction_id)

            record = PromotionRecord(
                examination_id=finding.examination_id,
                source_proposal_id=finding.proposal.proposal_id,
                source_hypothesis_index=candidate.hypothesis_index,
                hypothesis_id=hypothesis.hypothesis_id,
                prediction_ids=tuple(prediction_ids),
                minority=candidate.minority,
            )
            records.append(record)

            investigation.store.append(
                Event(
                    event_type="synthesis.hypothesis_promoted",
                    stream_id=investigation.investigation_id,
                    actor=promoter,
                    correlation_id=synthesis.synthesis_id,
                    payload={
                        "synthesis_id": synthesis.synthesis_id,
                        "examination_id": record.examination_id,
                        "source_proposal_id": record.source_proposal_id,
                        "source_hypothesis_index": record.source_hypothesis_index,
                        "hypothesis_id": record.hypothesis_id,
                        "prediction_ids": list(record.prediction_ids),
                        "source_thinker_ref": finding.thinker_ref,
                        "examiner_ref": finding.examiner_ref,
                        "disposition": finding.disposition.value,
                        "minority": record.minority,
                        "promotion_rationale": candidate.rationale,
                    },
                )
            )

        investigation.replay()
        return tuple(records)
