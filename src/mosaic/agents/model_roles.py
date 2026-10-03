"""Model-backed implementations of MOSAIC institutional roles."""

from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC, Sequence
from typing import Any, Mapping

from mosaic.kernel.identity import AgentIdentity
from mosaic.snapshot import InvestigationSnapshot

from .contracts import (
    HypothesisProposal,
    ModelBackend,
    ModelRequest,
    PredictionProposal,
)
from .roles import (
    CoordinatorReport,
    ExaminationChallenge,
    ExaminationDisposition,
    ExaminationExchange,
    ExaminationResult,
    ExaminerReview,
    InvestigationPlan,
    ObservationDraft,
    ThinkerProposal,
    ThinkerResponse,
    ThinkerTask,
)
from .synthesis import (
    CrossDomainSynthesis,
    FindingRelation,
    FindingRelationType,
    MinorityReport,
    PromotionCandidate,
)


class StructuredOutputError(RuntimeError):
    """Model output could not be converted into the requested MOSAIC object."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


_SCALAR_SCHEMA = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "integer"},
        {"type": "boolean"},
        {"type": "null"},
    ]
}


def _prediction_schema() -> dict[str, Any]:
    condition = _object_schema(
        {
            "name": {"type": "string"},
            "value": _SCALAR_SCHEMA,
        },
        ["name", "value"],
    )
    return _object_schema(
        {
            "statement": {"type": "string"},
            "confidence": {"type": "number"},
            "conditions": {"type": "array", "items": condition},
        },
        ["statement", "confidence", "conditions"],
    )


def _hypothesis_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "claim": {"type": "string"},
            "confidence": {
                "anyOf": [{"type": "number"}, {"type": "null"}]
            },
            "rationale": {
                "anyOf": [{"type": "string"}, {"type": "null"}]
            },
            "predictions": {
                "type": "array",
                "items": _prediction_schema(),
            },
        },
        ["claim", "confidence", "rationale", "predictions"],
    )


def _proposal_schema() -> dict[str, Any]:
    return _object_schema(
        {
            "summary": {"type": "string"},
            "hypotheses": {
                "type": "array",
                "items": _hypothesis_schema(),
            },
            "open_questions": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        ["summary", "hypotheses", "open_questions"],
    )


def _proposal_payload(proposal: ThinkerProposal) -> dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "thinker_ref": proposal.thinker_ref,
        "task_id": proposal.task_id,
        "summary": proposal.summary,
        "hypotheses": [
            {
                "claim": hypothesis.claim,
                "confidence": hypothesis.confidence,
                "rationale": hypothesis.rationale,
                "predictions": [
                    {
                        "statement": prediction.statement,
                        "confidence": prediction.confidence,
                        "conditions": _json_ready(prediction.conditions),
                    }
                    for prediction in hypothesis.predictions
                ],
            }
            for hypothesis in proposal.hypotheses
        ],
        "open_questions": list(proposal.open_questions),
    }


def _snapshot_payload(snapshot: InvestigationSnapshot) -> dict[str, Any]:
    return {
        "investigation_id": snapshot.investigation_id,
        "ledger_sequence": snapshot.ledger_sequence,
        "observations": [
            {
                "observation_id": item.observation_id,
                "name": item.name,
                "value": _json_ready(item.value),
                "unit": item.unit,
                "source": item.source,
                "uncertainty": item.uncertainty,
            }
            for item in snapshot.observations
        ],
        "hypotheses": [
            {
                "hypothesis_id": item.hypothesis_id,
                "claim": item.claim,
                "initial_confidence": item.initial_confidence,
                "proposed_by": item.proposed_by,
            }
            for item in snapshot.hypotheses
        ],
        "predictions": [
            {
                "prediction_id": item.prediction_id,
                "hypothesis_id": item.hypothesis_id,
                "statement": item.statement,
                "confidence": item.confidence,
                "conditions": _json_ready(item.conditions),
                "created_by": item.created_by,
            }
            for item in snapshot.predictions
        ],
        "relations": [
            {
                "relation_id": item.relation_id,
                "source_id": item.source_id,
                "target_id": item.target_id,
                "relation_type": item.relation_type.value,
                "rationale": item.rationale,
                "created_by": item.created_by,
            }
            for item in snapshot.relations
        ],
    }


def _finding_payload(finding: ExaminationResult) -> dict[str, Any]:
    return {
        "examination_id": finding.examination_id,
        "examiner_ref": finding.examiner_ref,
        "thinker_ref": finding.thinker_ref,
        "task_id": finding.task_id,
        "disposition": finding.disposition.value,
        "findings_summary": finding.findings_summary,
        "reservations": list(finding.reservations),
        "unresolved_questions": list(finding.unresolved_questions),
        "statistics": _json_ready(finding.statistics),
        "proposal": _proposal_payload(finding.proposal),
    }


def _transcript_payload(
    transcript: tuple[ExaminationExchange, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "challenge": {
                "challenge_id": exchange.challenge.challenge_id,
                "question": exchange.challenge.question,
                "targeted_claim": exchange.challenge.targeted_claim,
                "evidence_refs": list(exchange.challenge.evidence_refs),
            },
            "response": {
                "response_id": exchange.response.response_id,
                "answer": exchange.response.answer,
                "revised_proposal_id": (
                    None
                    if exchange.response.revised_proposal is None
                    else exchange.response.revised_proposal.proposal_id
                ),
            },
        }
        for exchange in transcript
    ]


async def _call_json(
    *,
    backend: ModelBackend,
    model: str,
    system: str,
    payload: Mapping[str, Any],
    schema_name: str,
    schema: Mapping[str, Any],
    usage_tag: str | None = None,
) -> dict[str, Any]:
    response = await backend.generate(
        ModelRequest(
            model=model,
            system=system,
            input_text=json.dumps(
                _json_ready(payload),
                ensure_ascii=False,
                allow_nan=False,
            ),
            response_schema_name=schema_name,
            response_schema=schema,
            usage_tag=usage_tag,
        )
    )
    try:
        parsed = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"{backend.backend_id} returned invalid JSON for {schema_name}"
        ) from exc
    if not isinstance(parsed, dict):
        raise StructuredOutputError(
            f"{backend.backend_id} returned a non-object for {schema_name}"
        )
    return parsed


def _parse_prediction(data: Mapping[str, Any]) -> PredictionProposal:
    conditions = {
        str(item["name"]): item.get("value")
        for item in data.get("conditions", [])
    }
    return PredictionProposal(
        statement=str(data["statement"]),
        confidence=float(data["confidence"]),
        conditions=conditions,
    )


def _parse_hypothesis(data: Mapping[str, Any]) -> HypothesisProposal:
    confidence = data.get("confidence")
    return HypothesisProposal(
        claim=str(data["claim"]),
        confidence=None if confidence is None else float(confidence),
        rationale=(
            None if data.get("rationale") is None else str(data["rationale"])
        ),
        predictions=tuple(
            _parse_prediction(item)
            for item in data.get("predictions", [])
        ),
    )


def _parse_proposal(
    data: Mapping[str, Any],
    *,
    thinker_ref: str,
    task_id: str,
) -> ThinkerProposal:
    return ThinkerProposal(
        thinker_ref=thinker_ref,
        task_id=task_id,
        summary=str(data["summary"]),
        hypotheses=tuple(
            _parse_hypothesis(item)
            for item in data.get("hypotheses", [])
        ),
        open_questions=tuple(
            str(item) for item in data.get("open_questions", [])
        ),
    )


class ModelCoordinator:
    """Professional interface role backed by any ModelBackend."""

    def __init__(
        self,
        *,
        identity: AgentIdentity,
        backend: ModelBackend,
        model: str,
        thinker_catalog: Mapping[str, str],
    ) -> None:
        self.identity = identity
        self.backend = backend
        self.model = model
        self.thinker_catalog = dict(thinker_catalog)

    async def intake(
        self,
        user_input: str,
        snapshot: InvestigationSnapshot,
    ) -> InvestigationPlan:
        refs = tuple(self.thinker_catalog)
        task_schema = _object_schema(
            {
                "assigned_to": {"type": "string", "enum": list(refs)},
                "question": {"type": "string"},
                "scope": {"type": "string"},
                "observation_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            [
                "assigned_to",
                "question",
                "scope",
                "observation_ids",
                "constraints",
            ],
        )
        observation_schema = _object_schema(
            {
                "name": {"type": "string"},
                "value": _SCALAR_SCHEMA,
                "unit": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "uncertainty": {
                    "anyOf": [{"type": "number"}, {"type": "null"}]
                },
            },
            ["name", "value", "unit", "uncertainty"],
        )
        schema = _object_schema(
            {
                "question": {"type": "string"},
                "normalized_input": {"type": "string"},
                "observations": {
                    "type": "array",
                    "items": observation_schema,
                },
                "tasks": {"type": "array", "items": task_schema},
                "ambiguities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            [
                "question",
                "normalized_input",
                "observations",
                "tasks",
                "ambiguities",
            ],
        )
        data = await _call_json(
            backend=self.backend,
            model=self.model,
            schema_name="mosaic_coordinator_intake",
            usage_tag=f"{self.identity.ref}:intake",
            schema=schema,
            system=(
                "You are the MOSAIC Coordinator. Act as a restrained professional "
                "interface, not a scientific reasoner. Convert user input into a "
                "neutral investigation question, extract only direct user-reported "
                "observations, and route neutral tasks to relevant Thinkers. Never "
                "convert an inferred mechanism or interpretation into an observation. "
                "Do not originate hypotheses, suggest likely answers, or contaminate "
                "one specialist with another specialist's view. Return only the "
                "requested JSON."
            ),
            payload={
                "user_input": user_input,
                "investigation": _snapshot_payload(snapshot),
                "available_thinkers": [
                    {"ref": ref, "description": description}
                    for ref, description in self.thinker_catalog.items()
                ],
            },
        )
        return InvestigationPlan(
            coordinator_ref=self.identity.ref,
            question=str(data["question"]),
            normalized_input=str(data["normalized_input"]),
            observations=tuple(
                ObservationDraft(
                    name=str(item["name"]),
                    value=item.get("value"),
                    unit=(
                        None
                        if item.get("unit") is None
                        else str(item["unit"])
                    ),
                    uncertainty=(
                        None
                        if item.get("uncertainty") is None
                        else float(item["uncertainty"])
                    ),
                )
                for item in data.get("observations", [])
            ),
            tasks=tuple(
                ThinkerTask(
                    assigned_to=str(item["assigned_to"]),
                    question=str(item["question"]),
                    scope=str(item["scope"]),
                    observation_ids=tuple(
                        str(value)
                        for value in item.get("observation_ids", [])
                    ),
                    constraints=tuple(
                        str(value)
                        for value in item.get("constraints", [])
                    ),
                )
                for item in data.get("tasks", [])
            ),
            ambiguities=tuple(
                str(item) for item in data.get("ambiguities", [])
            ),
        )

    async def report(
        self,
        snapshot: InvestigationSnapshot,
        findings: Sequence[ExaminationResult],
        synthesis_context: Mapping[str, Any] | None = None,
    ) -> CoordinatorReport:
        finding_ids = [finding.examination_id for finding in findings]
        schema = _object_schema(
            {
                "answer": {"type": "string"},
                "finding_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": finding_ids},
                },
                "observations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "established": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "surviving_hypotheses": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "key_test": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "examiner_status": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "unresolved_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "caveats": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            [
                "answer",
                "finding_ids",
                "observations",
                "established",
                "surviving_hypotheses",
                "key_test",
                "examiner_status",
                "unresolved_questions",
                "caveats",
            ],
        )
        data = await _call_json(
            backend=self.backend,
            model=self.model,
            schema_name="mosaic_coordinator_report",
            usage_tag=f"{self.identity.ref}:report",
            schema=schema,
            system=(
                "You are the MOSAIC Coordinator in reporting mode. Produce a "
                "professional, faithful structured report from examined findings "
                "and cross-domain synthesis. Keep answer to a short current-finding "
                "summary. Put raw reported facts under observations, statements "
                "actually supported by the evidence under established, plausible "
                "surviving mechanisms under surviving_hypotheses, and name the "
                "single most discriminating next test under key_test when one is "
                "available. Do not invent scientific conclusions, alter confidence, "
                "suppress minority findings, or hide unresolved questions. Represent "
                "every supplied finding. Return only the requested JSON."
            ),
            payload={
                "investigation": _snapshot_payload(snapshot),
                "examined_findings": [
                    _finding_payload(finding) for finding in findings
                ],
                "cross_domain_synthesis": _json_ready(
                    synthesis_context or {}
                ),
            },
        )
        return CoordinatorReport(
            coordinator_ref=self.identity.ref,
            answer=str(data["answer"]),
            finding_ids=tuple(
                str(item) for item in data.get("finding_ids", [])
            ),
            observations=tuple(
                str(item) for item in data.get("observations", [])
            ),
            established=tuple(
                str(item) for item in data.get("established", [])
            ),
            surviving_hypotheses=tuple(
                str(item)
                for item in data.get("surviving_hypotheses", [])
            ),
            key_test=(
                None
                if data.get("key_test") is None
                else str(data["key_test"])
            ),
            examiner_status=tuple(
                str(item) for item in data.get("examiner_status", [])
            ),
            unresolved_questions=tuple(
                str(item)
                for item in data.get("unresolved_questions", [])
            ),
            caveats=tuple(str(item) for item in data.get("caveats", [])),
        )


class ModelThinker:
    """Independent specialist Thinker backed by a ModelBackend."""

    def __init__(
        self,
        *,
        identity: AgentIdentity,
        backend: ModelBackend,
        model: str,
        instructions: str,
    ) -> None:
        self.identity = identity
        self.backend = backend
        self.model = model
        self.instructions = instructions.strip()
        if not self.instructions:
            raise ValueError("instructions must be non-empty")

    @property
    def _system(self) -> str:
        return (
            "You are a MOSAIC Thinker. Perform independent domain reasoning. "
            "Generate testable mechanisms, explicit assumptions, falsifiable "
            "predictions, and concise auditable rationale. Do not claim that "
            "another specialist agrees with you unless that information is in "
            "the supplied context. Do not fabricate observations. "
            f"Specialist instructions: {self.instructions} "
            "Return only the requested JSON."
        )

    async def investigate(
        self,
        snapshot: InvestigationSnapshot,
        task: ThinkerTask,
    ) -> ThinkerProposal:
        data = await _call_json(
            backend=self.backend,
            model=self.model,
            schema_name="mosaic_thinker_proposal",
            usage_tag=f"{self.identity.ref}:investigate",
            schema=_proposal_schema(),
            system=self._system,
            payload={
                "investigation": _snapshot_payload(snapshot),
                "task": {
                    "task_id": task.task_id,
                    "question": task.question,
                    "scope": task.scope,
                    "observation_ids": list(task.observation_ids),
                    "constraints": list(task.constraints),
                },
            },
        )
        return _parse_proposal(
            data,
            thinker_ref=self.identity.ref,
            task_id=task.task_id,
        )

    async def answer_examination(
        self,
        snapshot: InvestigationSnapshot,
        task: ThinkerTask,
        proposal: ThinkerProposal,
        challenge: ExaminationChallenge,
    ) -> ThinkerResponse:
        schema = _object_schema(
            {
                "answer": {"type": "string"},
                "revised_proposal": {
                    "anyOf": [_proposal_schema(), {"type": "null"}]
                },
            },
            ["answer", "revised_proposal"],
        )
        data = await _call_json(
            backend=self.backend,
            model=self.model,
            schema_name="mosaic_thinker_examination_response",
            usage_tag=f"{self.identity.ref}:examination_response",
            schema=schema,
            system=(
                self._system
                + " Answer the Examiner's challenge directly. Revise your "
                "proposal when the challenge exposes a material weakness; "
                "otherwise return null for revised_proposal."
            ),
            payload={
                "investigation": _snapshot_payload(snapshot),
                "task": {
                    "task_id": task.task_id,
                    "question": task.question,
                    "scope": task.scope,
                },
                "current_proposal": _proposal_payload(proposal),
                "challenge": {
                    "challenge_id": challenge.challenge_id,
                    "question": challenge.question,
                    "targeted_claim": challenge.targeted_claim,
                    "evidence_refs": list(challenge.evidence_refs),
                },
            },
        )
        revised = data.get("revised_proposal")
        return ThinkerResponse(
            thinker_ref=self.identity.ref,
            challenge_id=challenge.challenge_id,
            answer=str(data["answer"]),
            revised_proposal=(
                None
                if revised is None
                else _parse_proposal(
                    revised,
                    thinker_ref=self.identity.ref,
                    task_id=task.task_id,
                )
            ),
        )


class ModelExaminer:
    """Skeptical analytical gatekeeper backed by a ModelBackend."""

    def __init__(
        self,
        *,
        identity: AgentIdentity,
        backend: ModelBackend,
        model: str,
        instructions: str = (
            "Challenge unsupported claims, hidden assumptions, internal "
            "inconsistency, non-falsifiable reasoning, and unjustified confidence."
        ),
    ) -> None:
        self.identity = identity
        self.backend = backend
        self.model = model
        self.instructions = instructions.strip()

    async def examine(
        self,
        snapshot: InvestigationSnapshot,
        task: ThinkerTask,
        proposal: ThinkerProposal,
        transcript: tuple[ExaminationExchange, ...],
    ) -> ExaminerReview:
        metric_schema = _object_schema(
            {
                "name": {"type": "string"},
                "value": _SCALAR_SCHEMA,
            },
            ["name", "value"],
        )
        schema = _object_schema(
            {
                "action": {
                    "type": "string",
                    "enum": ["challenge", "disposition"],
                },
                "question": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "targeted_claim": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "evidence_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "disposition": {
                    "anyOf": [
                        {
                            "type": "string",
                            "enum": [
                                item.value
                                for item in ExaminationDisposition
                            ],
                        },
                        {"type": "null"},
                    ]
                },
                "findings_summary": {
                    "anyOf": [{"type": "string"}, {"type": "null"}]
                },
                "reservations": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "unresolved_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "statistics": {
                    "type": "array",
                    "items": metric_schema,
                },
            },
            [
                "action",
                "question",
                "targeted_claim",
                "evidence_refs",
                "disposition",
                "findings_summary",
                "reservations",
                "unresolved_questions",
                "statistics",
            ],
        )
        data = await _call_json(
            backend=self.backend,
            model=self.model,
            schema_name="mosaic_examiner_review",
            usage_tag=f"{self.identity.ref}:review",
            schema=schema,
            system=(
                "You are the MOSAIC Examiner. You are an epistemic gatekeeper, "
                "not a summarizer. Attempt to break the supplied Thinker proposal. "
                "Ask one focused challenge when a material weakness remains. Issue "
                "a terminal disposition only when another challenge is not needed. "
                "Do not force agreement when evidence is insufficient; use "
                "unresolved. "
                f"Examiner instructions: {self.instructions} "
                "Return only the requested JSON."
            ),
            payload={
                "investigation": _snapshot_payload(snapshot),
                "task": {
                    "task_id": task.task_id,
                    "question": task.question,
                    "scope": task.scope,
                },
                "proposal": _proposal_payload(proposal),
                "transcript": _transcript_payload(transcript),
            },
        )

        if data.get("action") == "challenge":
            question = data.get("question")
            if not isinstance(question, str) or not question.strip():
                raise StructuredOutputError(
                    "Examiner chose challenge without a question"
                )
            return ExaminationChallenge(
                examiner_ref=self.identity.ref,
                thinker_ref=proposal.thinker_ref,
                proposal_id=proposal.proposal_id,
                question=question,
                targeted_claim=(
                    None
                    if data.get("targeted_claim") is None
                    else str(data["targeted_claim"])
                ),
                evidence_refs=tuple(
                    str(item) for item in data.get("evidence_refs", [])
                ),
            )

        disposition = data.get("disposition")
        summary = data.get("findings_summary")
        if disposition is None or not isinstance(summary, str) or not summary.strip():
            raise StructuredOutputError(
                "Examiner chose disposition without disposition and summary"
            )
        return ExaminationResult(
            examiner_ref=self.identity.ref,
            thinker_ref=proposal.thinker_ref,
            task_id=task.task_id,
            proposal=proposal,
            disposition=ExaminationDisposition(str(disposition)),
            findings_summary=summary,
            reservations=tuple(
                str(item) for item in data.get("reservations", [])
            ),
            unresolved_questions=tuple(
                str(item)
                for item in data.get("unresolved_questions", [])
            ),
            statistics={
                str(item["name"]): item.get("value")
                for item in data.get("statistics", [])
            },
        )


class ModelCrossDomainReviewer:
    """Cross-domain comparison role backed by a ModelBackend."""

    def __init__(
        self,
        *,
        identity: AgentIdentity,
        backend: ModelBackend,
        model: str,
    ) -> None:
        self.identity = identity
        self.backend = backend
        self.model = model

    async def synthesize(
        self,
        snapshot: InvestigationSnapshot,
        findings: Sequence[ExaminationResult],
    ) -> CrossDomainSynthesis:
        finding_ids = [finding.examination_id for finding in findings]
        relation_schema = _object_schema(
            {
                "source_examination_id": {
                    "type": "string",
                    "enum": finding_ids,
                },
                "target_examination_id": {
                    "type": "string",
                    "enum": finding_ids,
                },
                "relation_type": {
                    "type": "string",
                    "enum": [item.value for item in FindingRelationType],
                },
                "rationale": {"type": "string"},
            },
            [
                "source_examination_id",
                "target_examination_id",
                "relation_type",
                "rationale",
            ],
        )
        minority_schema = _object_schema(
            {
                "examination_id": {
                    "type": "string",
                    "enum": finding_ids,
                },
                "rationale": {"type": "string"},
            },
            ["examination_id", "rationale"],
        )
        promotion_schema = _object_schema(
            {
                "examination_id": {
                    "type": "string",
                    "enum": finding_ids,
                },
                "hypothesis_index": {"type": "integer"},
                "rationale": {"type": "string"},
                "minority": {"type": "boolean"},
            },
            [
                "examination_id",
                "hypothesis_index",
                "rationale",
                "minority",
            ],
        )
        metric_schema = _object_schema(
            {
                "name": {"type": "string"},
                "value": _SCALAR_SCHEMA,
            },
            ["name", "value"],
        )
        schema = _object_schema(
            {
                "finding_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": finding_ids},
                },
                "summary": {"type": "string"},
                "relations": {
                    "type": "array",
                    "items": relation_schema,
                },
                "minority_reports": {
                    "type": "array",
                    "items": minority_schema,
                },
                "promotions": {
                    "type": "array",
                    "items": promotion_schema,
                },
                "unresolved_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "statistics": {
                    "type": "array",
                    "items": metric_schema,
                },
            },
            [
                "finding_ids",
                "summary",
                "relations",
                "minority_reports",
                "promotions",
                "unresolved_questions",
                "statistics",
            ],
        )
        data = await _call_json(
            backend=self.backend,
            model=self.model,
            schema_name="mosaic_cross_domain_synthesis",
            usage_tag=f"{self.identity.ref}:synthesis",
            schema=schema,
            system=(
                "You are the MOSAIC cross-domain reviewer. Compare only the "
                "already-examined findings. Identify agreements, contradictions, "
                "overlap, dependencies, minority positions, and unresolved "
                "questions. Preserve meaningful alternatives. Do not promote a "
                "rejected finding. Select explicit hypothesis indices for graph "
                "promotion when warranted. Return only the requested JSON."
            ),
            payload={
                "investigation": _snapshot_payload(snapshot),
                "examined_findings": [
                    _finding_payload(finding) for finding in findings
                ],
            },
        )
        return CrossDomainSynthesis(
            reviewer_ref=self.identity.ref,
            investigation_id=snapshot.investigation_id,
            source_ledger_sequence=snapshot.ledger_sequence,
            finding_ids=tuple(
                str(item) for item in data.get("finding_ids", [])
            ),
            summary=str(data["summary"]),
            relations=tuple(
                FindingRelation(
                    source_examination_id=str(
                        item["source_examination_id"]
                    ),
                    target_examination_id=str(
                        item["target_examination_id"]
                    ),
                    relation_type=FindingRelationType(
                        str(item["relation_type"])
                    ),
                    rationale=str(item["rationale"]),
                )
                for item in data.get("relations", [])
            ),
            minority_reports=tuple(
                MinorityReport(
                    examination_id=str(item["examination_id"]),
                    rationale=str(item["rationale"]),
                )
                for item in data.get("minority_reports", [])
            ),
            promotions=tuple(
                PromotionCandidate(
                    examination_id=str(item["examination_id"]),
                    hypothesis_index=int(item["hypothesis_index"]),
                    rationale=str(item["rationale"]),
                    minority=bool(item["minority"]),
                )
                for item in data.get("promotions", [])
            ),
            unresolved_questions=tuple(
                str(item)
                for item in data.get("unresolved_questions", [])
            ),
            statistics={
                str(item["name"]): item.get("value")
                for item in data.get("statistics", [])
            },
        )
