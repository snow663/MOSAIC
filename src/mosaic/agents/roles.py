"""Institutional role contracts for MOSAIC research orchestration."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, TypeAlias, runtime_checkable
from uuid import uuid4

from mosaic.kernel.identity import AgentIdentity
from mosaic.snapshot import InvestigationSnapshot

from .contracts import HypothesisProposal


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


class ExaminationDisposition(StrEnum):
    """Permitted terminal states for a private examination."""

    ACCEPTED = "accepted"
    ACCEPTED_WITH_RESERVATIONS = "accepted_with_reservations"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ThinkerTask:
    """A neutral assignment prepared by the Coordinator for one Thinker."""

    assigned_to: str
    question: str
    scope: str
    observation_ids: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    task_id: str = field(default_factory=lambda: _id("T"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "assigned_to", _text(self.assigned_to, "assigned_to"))
        object.__setattr__(self, "question", _text(self.question, "question"))
        object.__setattr__(self, "scope", _text(self.scope, "scope"))
        object.__setattr__(
            self,
            "observation_ids",
            tuple(_text(item, "observation_id") for item in self.observation_ids),
        )
        object.__setattr__(
            self,
            "constraints",
            tuple(_text(item, "constraint") for item in self.constraints),
        )



@dataclass(frozen=True, slots=True)
class ObservationDraft:
    """A direct user-reported observation extracted during intake."""

    name: str
    value: Any
    unit: str | None = None
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name"))
        if self.unit is not None:
            object.__setattr__(self, "unit", _text(self.unit, "unit"))
        if self.uncertainty is not None:
            uncertainty = float(self.uncertainty)
            if uncertainty < 0:
                raise ValueError("uncertainty must be non-negative")
            object.__setattr__(self, "uncertainty", uncertainty)


@dataclass(frozen=True, slots=True)
class InvestigationPlan:
    """Coordinator output that converts user input into neutral specialist work."""

    coordinator_ref: str
    question: str
    normalized_input: str
    tasks: tuple[ThinkerTask, ...]
    observations: tuple[ObservationDraft, ...] = ()
    ambiguities: tuple[str, ...] = ()
    plan_id: str = field(default_factory=lambda: _id("PLAN"))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coordinator_ref",
            _text(self.coordinator_ref, "coordinator_ref"),
        )
        object.__setattr__(self, "question", _text(self.question, "question"))
        object.__setattr__(
            self,
            "normalized_input",
            _text(self.normalized_input, "normalized_input"),
        )
        if not self.tasks:
            raise ValueError("an investigation plan requires at least one Thinker task")
        object.__setattr__(
            self,
            "ambiguities",
            tuple(_text(item, "ambiguity") for item in self.ambiguities),
        )


@dataclass(frozen=True, slots=True)
class ThinkerProposal:
    """A Thinker's independently generated scientific proposal."""

    thinker_ref: str
    task_id: str
    summary: str
    hypotheses: tuple[HypothesisProposal, ...] = ()
    open_questions: tuple[str, ...] = ()
    proposal_id: str = field(default_factory=lambda: _id("TP"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "thinker_ref", _text(self.thinker_ref, "thinker_ref"))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "summary", _text(self.summary, "summary"))
        object.__setattr__(
            self,
            "open_questions",
            tuple(_text(item, "open_question") for item in self.open_questions),
        )


@dataclass(frozen=True, slots=True)
class ExaminationChallenge:
    """A focused objection or question from an Examiner to one Thinker."""

    examiner_ref: str
    thinker_ref: str
    proposal_id: str
    question: str
    targeted_claim: str | None = None
    evidence_refs: tuple[str, ...] = ()
    challenge_id: str = field(default_factory=lambda: _id("Q"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "examiner_ref", _text(self.examiner_ref, "examiner_ref"))
        object.__setattr__(self, "thinker_ref", _text(self.thinker_ref, "thinker_ref"))
        object.__setattr__(self, "proposal_id", _text(self.proposal_id, "proposal_id"))
        object.__setattr__(self, "question", _text(self.question, "question"))
        if self.targeted_claim is not None:
            object.__setattr__(
                self,
                "targeted_claim",
                _text(self.targeted_claim, "targeted_claim"),
            )
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(_text(item, "evidence_ref") for item in self.evidence_refs),
        )


@dataclass(frozen=True, slots=True)
class ThinkerResponse:
    """A Thinker's direct response to an Examiner challenge."""

    thinker_ref: str
    challenge_id: str
    answer: str
    revised_proposal: ThinkerProposal | None = None
    response_id: str = field(default_factory=lambda: _id("A"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "thinker_ref", _text(self.thinker_ref, "thinker_ref"))
        object.__setattr__(self, "challenge_id", _text(self.challenge_id, "challenge_id"))
        object.__setattr__(self, "answer", _text(self.answer, "answer"))


@dataclass(frozen=True, slots=True)
class ExaminationExchange:
    """One Examiner challenge paired with the Thinker's answer."""

    challenge: ExaminationChallenge
    response: ThinkerResponse

    def __post_init__(self) -> None:
        if self.response.challenge_id != self.challenge.challenge_id:
            raise ValueError("response does not answer this challenge")
        if self.response.thinker_ref != self.challenge.thinker_ref:
            raise ValueError("response thinker does not match challenged thinker")


@dataclass(frozen=True, slots=True)
class ExaminationResult:
    """Terminal examined finding eligible for later cross-domain review."""

    examiner_ref: str
    thinker_ref: str
    task_id: str
    proposal: ThinkerProposal
    disposition: ExaminationDisposition
    findings_summary: str
    reservations: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    statistics: Mapping[str, Any] = field(default_factory=dict)
    transcript: tuple[ExaminationExchange, ...] = ()
    initial_proposal_id: str | None = None
    examination_id: str = field(default_factory=lambda: _id("EX"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "examiner_ref", _text(self.examiner_ref, "examiner_ref"))
        object.__setattr__(self, "thinker_ref", _text(self.thinker_ref, "thinker_ref"))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(
            self,
            "findings_summary",
            _text(self.findings_summary, "findings_summary"),
        )
        if self.proposal.thinker_ref != self.thinker_ref:
            raise ValueError("result proposal belongs to a different Thinker")
        if self.proposal.task_id != self.task_id:
            raise ValueError("result proposal belongs to a different task")
        object.__setattr__(
            self,
            "reservations",
            tuple(_text(item, "reservation") for item in self.reservations),
        )
        object.__setattr__(
            self,
            "unresolved_questions",
            tuple(
                _text(item, "unresolved_question")
                for item in self.unresolved_questions
            ),
        )
        object.__setattr__(self, "statistics", _freeze_json(self.statistics))
        if self.initial_proposal_id is not None:
            object.__setattr__(
                self,
                "initial_proposal_id",
                _text(self.initial_proposal_id, "initial_proposal_id"),
            )


@dataclass(frozen=True, slots=True)
class CoordinatorReport:
    """Structured professional report produced from examined findings."""

    coordinator_ref: str
    answer: str
    finding_ids: tuple[str, ...]
    observations: tuple[str, ...] = ()
    established: tuple[str, ...] = ()
    surviving_hypotheses: tuple[str, ...] = ()
    key_test: str | None = None
    examiner_status: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    report_id: str = field(default_factory=lambda: _id("REPORT"))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coordinator_ref",
            _text(self.coordinator_ref, "coordinator_ref"),
        )
        object.__setattr__(self, "answer", _text(self.answer, "answer"))
        object.__setattr__(
            self,
            "finding_ids",
            tuple(_text(item, "finding_id") for item in self.finding_ids),
        )
        for field_name, item_name in (
            ("observations", "observation"),
            ("established", "established statement"),
            ("surviving_hypotheses", "surviving hypothesis"),
            ("examiner_status", "examiner status"),
            ("unresolved_questions", "unresolved question"),
            ("caveats", "caveat"),
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(
                    _text(item, item_name)
                    for item in getattr(self, field_name)
                ),
            )
        if self.key_test is not None:
            object.__setattr__(
                self,
                "key_test",
                _text(self.key_test, "key_test"),
            )


ExaminerReview: TypeAlias = ExaminationChallenge | ExaminationResult


@runtime_checkable
class Coordinator(Protocol):
    """Constrained human interface; it organizes but does not conduct science."""

    identity: AgentIdentity

    async def intake(
        self,
        user_input: str,
        snapshot: InvestigationSnapshot,
    ) -> InvestigationPlan:
        ...

    async def report(
        self,
        snapshot: InvestigationSnapshot,
        findings: Sequence[ExaminationResult],
        synthesis_context: Mapping[str, Any] | None = None,
    ) -> CoordinatorReport:
        ...


@runtime_checkable
class Thinker(Protocol):
    """Independent domain reasoner."""

    identity: AgentIdentity

    async def investigate(
        self,
        snapshot: InvestigationSnapshot,
        task: ThinkerTask,
    ) -> ThinkerProposal:
        ...

    async def answer_examination(
        self,
        snapshot: InvestigationSnapshot,
        task: ThinkerTask,
        proposal: ThinkerProposal,
        challenge: ExaminationChallenge,
    ) -> ThinkerResponse:
        ...


@runtime_checkable
class Examiner(Protocol):
    """Analytical gatekeeper for one Thinker's proposal."""

    identity: AgentIdentity

    async def examine(
        self,
        snapshot: InvestigationSnapshot,
        task: ThinkerTask,
        proposal: ThinkerProposal,
        transcript: tuple[ExaminationExchange, ...],
    ) -> ExaminerReview:
        ...
