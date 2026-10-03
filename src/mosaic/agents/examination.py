"""Private Thinker-Examiner examination loop and append-only audit events."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import uuid4

from mosaic.kernel.events import Event
from mosaic.kernel.identity import KERNEL_IDENTITY
from mosaic.kernel.store import SQLiteEventStore
from mosaic.snapshot import InvestigationSnapshot

from .independent import AgentProtocolError
from .progress import ProgressCallback, emit_progress
from .roles import (
    ExaminationChallenge,
    ExaminationExchange,
    ExaminationResult,
    Examiner,
    Thinker,
    ThinkerProposal,
    ThinkerResponse,
    ThinkerTask,
)


class ExaminationLimitError(AgentProtocolError):
    """Raised when an Examiner fails to reach a disposition within the limit."""


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
                        "conditions": dict(prediction.conditions),
                    }
                    for prediction in hypothesis.predictions
                ],
            }
            for hypothesis in proposal.hypotheses
        ],
        "open_questions": list(proposal.open_questions),
    }


def _challenge_payload(challenge: ExaminationChallenge) -> dict[str, Any]:
    return {
        "challenge_id": challenge.challenge_id,
        "examiner_ref": challenge.examiner_ref,
        "thinker_ref": challenge.thinker_ref,
        "proposal_id": challenge.proposal_id,
        "question": challenge.question,
        "targeted_claim": challenge.targeted_claim,
        "evidence_refs": list(challenge.evidence_refs),
        "category": challenge.category.value,
        "decision_impact": challenge.decision_impact,
    }


def _response_payload(response: ThinkerResponse) -> dict[str, Any]:
    return {
        "response_id": response.response_id,
        "thinker_ref": response.thinker_ref,
        "challenge_id": response.challenge_id,
        "answer": response.answer,
        "revised_proposal": (
            None
            if response.revised_proposal is None
            else _proposal_payload(response.revised_proposal)
        ),
    }


class PrivateExamination:
    """Run one Thinker through an isolated Examiner challenge loop."""

    def __init__(
        self,
        *,
        audit_store: SQLiteEventStore | None = None,
        max_rounds: int = 6,
        progress: ProgressCallback | None = None,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self.audit_store = audit_store
        self.max_rounds = max_rounds
        self.progress = progress

    def _audit(
        self,
        *,
        snapshot: InvestigationSnapshot,
        event_type: str,
        actor,
        payload: dict[str, Any],
        examination_id: str,
    ) -> None:
        if self.audit_store is None:
            return
        self.audit_store.append(
            Event(
                event_type=event_type,
                stream_id=snapshot.investigation_id,
                actor=actor,
                payload=payload,
                correlation_id=examination_id,
            )
        )

    @staticmethod
    def _validate_task(task: ThinkerTask, thinker: Thinker) -> None:
        if task.assigned_to != thinker.identity.ref:
            raise AgentProtocolError(
                f"task assigned to {task.assigned_to}, not {thinker.identity.ref}"
            )

    @staticmethod
    def _validate_proposal(
        proposal: ThinkerProposal,
        task: ThinkerTask,
        thinker: Thinker,
    ) -> None:
        if proposal.thinker_ref != thinker.identity.ref:
            raise AgentProtocolError(
                f"Thinker {thinker.identity.ref} returned proposal attributed "
                f"to {proposal.thinker_ref}"
            )
        if proposal.task_id != task.task_id:
            raise AgentProtocolError(
                f"proposal targets task {proposal.task_id}, expected {task.task_id}"
            )

    @staticmethod
    def _validate_challenge(
        challenge: ExaminationChallenge,
        *,
        examiner: Examiner,
        thinker: Thinker,
        proposal: ThinkerProposal,
    ) -> None:
        if challenge.examiner_ref != examiner.identity.ref:
            raise AgentProtocolError(
                f"Examiner {examiner.identity.ref} returned challenge attributed "
                f"to {challenge.examiner_ref}"
            )
        if challenge.thinker_ref != thinker.identity.ref:
            raise AgentProtocolError("Examiner challenged the wrong Thinker")
        if challenge.proposal_id != proposal.proposal_id:
            raise AgentProtocolError("Examiner challenged a stale or foreign proposal")

    @staticmethod
    def _validate_response(
        response: ThinkerResponse,
        *,
        thinker: Thinker,
        challenge: ExaminationChallenge,
        task: ThinkerTask,
    ) -> None:
        if response.thinker_ref != thinker.identity.ref:
            raise AgentProtocolError("Thinker response attribution mismatch")
        if response.challenge_id != challenge.challenge_id:
            raise AgentProtocolError("Thinker answered a different challenge")
        if response.revised_proposal is not None:
            if response.revised_proposal.thinker_ref != thinker.identity.ref:
                raise AgentProtocolError("revised proposal attribution mismatch")
            if response.revised_proposal.task_id != task.task_id:
                raise AgentProtocolError("revised proposal targets a different task")

    @staticmethod
    def _validate_result(
        result: ExaminationResult,
        *,
        examiner: Examiner,
        thinker: Thinker,
        task: ThinkerTask,
    ) -> None:
        if result.examiner_ref != examiner.identity.ref:
            raise AgentProtocolError("Examiner result attribution mismatch")
        if result.thinker_ref != thinker.identity.ref:
            raise AgentProtocolError("Examiner result targets the wrong Thinker")
        if result.task_id != task.task_id:
            raise AgentProtocolError("Examiner result targets the wrong task")

    async def run(
        self,
        snapshot: InvestigationSnapshot,
        task: ThinkerTask,
        thinker: Thinker,
        examiner: Examiner,
    ) -> ExaminationResult:
        """Return only after the Examiner issues a valid terminal disposition."""

        self._validate_task(task, thinker)
        examination_id = f"EX-{uuid4()}"

        emit_progress(
            self.progress,
            stage="thinker",
            message=f"{thinker.identity.ref} is investigating.",
            actor_ref=thinker.identity.ref,
            task_id=task.task_id,
        )
        proposal = await thinker.investigate(snapshot, task)
        self._validate_proposal(proposal, task, thinker)
        initial_proposal_id = proposal.proposal_id
        transcript: list[ExaminationExchange] = []

        self._audit(
            snapshot=snapshot,
            event_type="examination.started",
            actor=KERNEL_IDENTITY,
            payload={
                "examination_id": examination_id,
                "task_id": task.task_id,
                "thinker_ref": thinker.identity.ref,
                "examiner_ref": examiner.identity.ref,
                "initial_proposal": _proposal_payload(proposal),
            },
            examination_id=examination_id,
        )

        for round_number in range(1, self.max_rounds + 1):
            emit_progress(
                self.progress,
                stage="examiner",
                message=(
                    f"{examiner.identity.ref} is reviewing "
                    f"{thinker.identity.ref} (round {round_number})."
                ),
                actor_ref=examiner.identity.ref,
                task_id=task.task_id,
                round=round_number,
                challenge_category=review.category.value,
                decision_impact=review.decision_impact,
            )
            review = await examiner.examine(
                snapshot,
                task,
                proposal,
                tuple(transcript),
            )

            if isinstance(review, ExaminationResult):
                self._validate_result(
                    review,
                    examiner=examiner,
                    thinker=thinker,
                    task=task,
                )
                result = replace(
                    review,
                    examination_id=examination_id,
                    proposal=proposal,
                    transcript=tuple(transcript),
                    initial_proposal_id=initial_proposal_id,
                )
                emit_progress(
                    self.progress,
                    stage="examiner",
                    message=(
                        f"{thinker.identity.ref} examination completed: "
                        f"{result.disposition.value}."
                    ),
                    actor_ref=examiner.identity.ref,
                    task_id=task.task_id,
                    disposition=result.disposition.value,
                    rounds=len(result.transcript),
                )
                self._audit(
                    snapshot=snapshot,
                    event_type="examination.completed",
                    actor=examiner.identity,
                    payload={
                        "examination_id": result.examination_id,
                        "task_id": result.task_id,
                        "thinker_ref": result.thinker_ref,
                        "examiner_ref": result.examiner_ref,
                        "initial_proposal_id": result.initial_proposal_id,
                        "final_proposal": _proposal_payload(result.proposal),
                        "disposition": result.disposition.value,
                        "findings_summary": result.findings_summary,
                        "reservations": list(result.reservations),
                        "unresolved_questions": list(result.unresolved_questions),
                        "statistics": dict(result.statistics),
                        "rounds": len(result.transcript),
                    },
                    examination_id=examination_id,
                )
                return result

            if not isinstance(review, ExaminationChallenge):
                raise AgentProtocolError(
                    "Examiner returned neither a challenge nor a disposition"
                )

            self._validate_challenge(
                review,
                examiner=examiner,
                thinker=thinker,
                proposal=proposal,
            )
            self._audit(
                snapshot=snapshot,
                event_type="examination.challenge",
                actor=examiner.identity,
                payload={
                    "examination_id": examination_id,
                    "round": round_number,
                    **_challenge_payload(review),
                },
                examination_id=examination_id,
            )

            emit_progress(
                self.progress,
                stage="challenge",
                message=(
                    f"{examiner.identity.ref} challenged "
                    f"{thinker.identity.ref} "
                    f"[{review.category.value}]: {review.question}"
                ),
                actor_ref=examiner.identity.ref,
                task_id=task.task_id,
                round=round_number,
            )
            emit_progress(
                self.progress,
                stage="thinker",
                message=(
                    f"{thinker.identity.ref} is answering examiner "
                    f"round {round_number}."
                ),
                actor_ref=thinker.identity.ref,
                task_id=task.task_id,
                round=round_number,
            )
            response = await thinker.answer_examination(
                snapshot,
                task,
                proposal,
                review,
            )
            self._validate_response(
                response,
                thinker=thinker,
                challenge=review,
                task=task,
            )
            self._audit(
                snapshot=snapshot,
                event_type="examination.response",
                actor=thinker.identity,
                payload={
                    "examination_id": examination_id,
                    "round": round_number,
                    **_response_payload(response),
                },
                examination_id=examination_id,
            )

            transcript.append(
                ExaminationExchange(
                    challenge=review,
                    response=response,
                )
            )
            if response.revised_proposal is not None:
                proposal = response.revised_proposal

        self._audit(
            snapshot=snapshot,
            event_type="examination.aborted",
            actor=KERNEL_IDENTITY,
            payload={
                "examination_id": examination_id,
                "task_id": task.task_id,
                "reason": "maximum examination rounds exceeded",
                "rounds": self.max_rounds,
            },
            examination_id=examination_id,
        )
        raise ExaminationLimitError(
            f"Examiner failed to issue a disposition within "
            f"{self.max_rounds} rounds"
        )
