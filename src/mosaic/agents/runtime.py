"""End-to-end MOSAIC research-cycle orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Mapping
from uuid import uuid4

from mosaic.investigation import Investigation
from mosaic.kernel.events import Event

from .coordination import CoordinatorSession
from .examination import PrivateExamination
from .progress import ProgressCallback, emit_progress
from .roles import (
    Coordinator,
    CoordinatorReport,
    ExaminationResult,
    Examiner,
    InvestigationPlan,
    ReviewRequest,
    Thinker,
)
from .synthesis import (
    CrossDomainReviewer,
    CrossDomainSynthesis,
    PromotionExecutor,
    PromotionRecord,
    SynthesisSession,
)


ReviewInputProvider = Callable[[ReviewRequest], Awaitable[str | None]]


@dataclass(frozen=True, slots=True)
class ResearchCycleResult:
    """Complete result of one MOSAIC investigation cycle."""

    plan: InvestigationPlan
    findings: tuple[ExaminationResult, ...]
    synthesis: CrossDomainSynthesis
    promotions: tuple[PromotionRecord, ...]
    report: CoordinatorReport
    review_request: ReviewRequest | None = None
    review_response: str | None = None


class ResearchCycle:
    """Execute the institutional pipeline without delegating orchestration."""

    def __init__(
        self,
        *,
        coordinator: Coordinator,
        thinkers: Mapping[str, Thinker],
        examiner: Examiner,
        reviewer: CrossDomainReviewer,
        max_examination_rounds: int = 6,
        progress: ProgressCallback | None = None,
        review_input_provider: ReviewInputProvider | None = None,
    ) -> None:
        if not thinkers:
            raise ValueError("at least one Thinker is required")
        self.coordinator = coordinator
        self.thinkers = dict(thinkers)
        self.examiner = examiner
        self.reviewer = reviewer
        self.max_examination_rounds = max_examination_rounds
        self.progress = progress
        self.review_input_provider = review_input_provider

        for ref, thinker in self.thinkers.items():
            if ref != thinker.identity.ref:
                raise ValueError(
                    f"Thinker registry key {ref} does not match "
                    f"{thinker.identity.ref}"
                )

    @staticmethod
    def _synthesis_context(
        synthesis: CrossDomainSynthesis,
        promotions: tuple[PromotionRecord, ...],
    ) -> dict:
        return {
            "synthesis_id": synthesis.synthesis_id,
            "summary": synthesis.summary,
            "relations": [
                {
                    "source_examination_id": item.source_examination_id,
                    "target_examination_id": item.target_examination_id,
                    "relation_type": item.relation_type.value,
                    "rationale": item.rationale,
                }
                for item in synthesis.relations
            ],
            "minority_reports": [
                {
                    "examination_id": item.examination_id,
                    "rationale": item.rationale,
                }
                for item in synthesis.minority_reports
            ],
            "unresolved_questions": list(synthesis.unresolved_questions),
            "statistics": dict(synthesis.statistics),
            "promotions": [
                {
                    "examination_id": item.examination_id,
                    "hypothesis_id": item.hypothesis_id,
                    "prediction_ids": list(item.prediction_ids),
                    "minority": item.minority,
                }
                for item in promotions
            ],
        }

    def _record_intake_plan(
        self,
        *,
        investigation: Investigation,
        plan: InvestigationPlan,
        raw_input: str,
        input_kind: str,
    ) -> InvestigationPlan:
        reported_context_count = len(
            {
                draft.context_key
                for draft in plan.observations
                if draft.context_key is not None
            }
        )

        investigation.store.append(
            Event(
                event_type="coordinator.intake_completed",
                stream_id=investigation.investigation_id,
                actor=self.coordinator.identity,
                correlation_id=plan.plan_id,
                payload={
                    "plan_id": plan.plan_id,
                    "input_kind": input_kind,
                    "raw_user_input": raw_input,
                    "normalized_input": plan.normalized_input,
                    "question": plan.question,
                    "ambiguities": list(plan.ambiguities),
                    "clarification_questions": list(
                        plan.clarification_questions
                    ),
                    "task_ids": [task.task_id for task in plan.tasks],
                    "observation_count": len(plan.observations),
                    "reported_event_group_count": reported_context_count,
                },
            )
        )

        new_observation_ids: list[str] = []
        context_ids: dict[str, str] = {}
        for draft in plan.observations:
            context_id = None
            if draft.context_key is not None:
                context_id = context_ids.setdefault(
                    draft.context_key,
                    f"CTX-{uuid4()}",
                )
            observation = investigation.record_observation(
                name=draft.name,
                value=draft.value,
                unit=draft.unit,
                uncertainty=draft.uncertainty,
                context_id=context_id,
                context_label=draft.context_label,
                source=f"{input_kind}:{plan.plan_id}",
                actor=self.coordinator.identity,
            )
            new_observation_ids.append(observation.observation_id)

        if not new_observation_ids:
            return plan

        return replace(
            plan,
            tasks=tuple(
                replace(
                    task,
                    observation_ids=tuple(
                        dict.fromkeys(
                            (
                                *task.observation_ids,
                                *new_observation_ids,
                            )
                        )
                    ),
                )
                for task in plan.tasks
            ),
        )

    @staticmethod
    def _clarification_input(
        request: ReviewRequest,
        response: str,
    ) -> str:
        questions = "\n".join(
            f"{index}. {question}"
            for index, question in enumerate(request.questions, start=1)
        )
        return (
            "MOSAIC CLARIFICATION CONTEXT "
            "(the questions below are not user observations):\n"
            f"{questions}\n\n"
            "USER CLARIFICATION RESPONSE:\n"
            f"{response.strip()}"
        )

    async def run(
        self,
        *,
        investigation: Investigation,
        user_input: str,
    ) -> ResearchCycleResult:
        """Run one complete MOSAIC reasoning cycle."""

        intake_snapshot = investigation.snapshot()
        emit_progress(
            self.progress,
            stage="coordinator",
            message="Coordinator is processing intake.",
            actor_ref=self.coordinator.identity.ref,
        )
        plan = await CoordinatorSession.intake(
            coordinator=self.coordinator,
            user_input=user_input,
            snapshot=intake_snapshot,
            available_thinkers=tuple(self.thinkers.values()),
        )

        reported_context_count = len(
            {
                draft.context_key
                for draft in plan.observations
                if draft.context_key is not None
            }
        )
        emit_progress(
            self.progress,
            stage="coordinator",
            message=(
                f"Coordinator selected {len(plan.tasks)} specialist task(s) "
                f"and extracted {len(plan.observations)} observation(s)"
                + (
                    f" across {reported_context_count} reported event group(s)."
                    if reported_context_count
                    else "."
                )
            ),
            actor_ref=self.coordinator.identity.ref,
            tasks=len(plan.tasks),
            observations=len(plan.observations),
            reported_event_groups=reported_context_count,
        )

        plan = self._record_intake_plan(
            investigation=investigation,
            plan=plan,
            raw_input=user_input,
            input_kind="user_input",
        )

        review_request = None
        review_response = None
        if plan.clarification_questions and self.review_input_provider is not None:
            review_request = ReviewRequest(
                coordinator_ref=self.coordinator.identity.ref,
                plan_id=plan.plan_id,
                questions=plan.clarification_questions,
            )
            investigation.store.append(
                Event(
                    event_type="review.requested",
                    stream_id=investigation.investigation_id,
                    actor=self.coordinator.identity,
                    correlation_id=review_request.review_id,
                    payload={
                        "review_id": review_request.review_id,
                        "plan_id": plan.plan_id,
                        "questions": list(review_request.questions),
                    },
                )
            )
            emit_progress(
                self.progress,
                stage="review",
                message=(
                    f"Coordinator is requesting {len(review_request.questions)} "
                    f"clarification answer(s) before specialist analysis."
                ),
                actor_ref=self.coordinator.identity.ref,
                questions=len(review_request.questions),
            )
            candidate = await self.review_input_provider(review_request)
            if candidate is not None and candidate.strip():
                review_response = candidate.strip()
                investigation.store.append(
                    Event(
                        event_type="review.response_received",
                        stream_id=investigation.investigation_id,
                        actor=self.coordinator.identity,
                        correlation_id=review_request.review_id,
                        payload={
                            "review_id": review_request.review_id,
                            "plan_id": plan.plan_id,
                            "raw_user_response": review_response,
                        },
                    )
                )
                investigation.replay()
                emit_progress(
                    self.progress,
                    stage="coordinator",
                    message="Coordinator is integrating clarification answers.",
                    actor_ref=self.coordinator.identity.ref,
                )
                followup_plan = await CoordinatorSession.intake(
                    coordinator=self.coordinator,
                    user_input=self._clarification_input(
                        review_request,
                        review_response,
                    ),
                    snapshot=investigation.snapshot(),
                    available_thinkers=tuple(self.thinkers.values()),
                )
                plan = self._record_intake_plan(
                    investigation=investigation,
                    plan=replace(
                        followup_plan,
                        clarification_questions=(),
                    ),
                    raw_input=review_response,
                    input_kind="review_input",
                )
                emit_progress(
                    self.progress,
                    stage="review",
                    message=(
                        f"Clarification integrated; proceeding with "
                        f"{len(plan.tasks)} specialist task(s)."
                    ),
                    actor_ref=self.coordinator.identity.ref,
                )
            else:
                investigation.store.append(
                    Event(
                        event_type="review.skipped",
                        stream_id=investigation.investigation_id,
                        actor=self.coordinator.identity,
                        correlation_id=review_request.review_id,
                        payload={
                            "review_id": review_request.review_id,
                            "plan_id": plan.plan_id,
                        },
                    )
                )
                emit_progress(
                    self.progress,
                    stage="review",
                    message="Clarification skipped; proceeding with existing evidence.",
                    actor_ref=self.coordinator.identity.ref,
                )

        # Direct audit writes do not mutate the in-memory Investigation state.
        # Replay once so every Thinker receives the same snapshot containing
        # all observations extracted from the current user input.
        investigation.replay()
        source_snapshot = investigation.snapshot()
        emit_progress(
            self.progress,
            stage="snapshot",
            message=(
                f"Frozen investigation snapshot at ledger sequence "
                f"{source_snapshot.ledger_sequence}."
            ),
            ledger_sequence=source_snapshot.ledger_sequence,
        )

        examination = PrivateExamination(
            audit_store=investigation.store,
            max_rounds=self.max_examination_rounds,
            progress=self.progress,
        )
        findings: list[ExaminationResult] = []

        # Deliberately sequential for a shared local/hosted backend. Every
        # Thinker still receives the same frozen source_snapshot.
        for task in plan.tasks:
            thinker = self.thinkers[task.assigned_to]
            finding = await examination.run(
                source_snapshot,
                task,
                thinker,
                self.examiner,
            )
            findings.append(finding)

        findings_tuple = tuple(findings)
        emit_progress(
            self.progress,
            stage="synthesis",
            message=(
                f"Cross-domain review is comparing "
                f"{len(findings_tuple)} examined finding(s)."
            ),
            actor_ref=self.reviewer.identity.ref,
            findings=len(findings_tuple),
        )
        synthesis = await SynthesisSession(
            audit_store=investigation.store
        ).run(
            source_snapshot,
            findings_tuple,
            self.reviewer,
        )

        emit_progress(
            self.progress,
            stage="synthesis",
            message="Cross-domain synthesis completed.",
            actor_ref=self.reviewer.identity.ref,
            promotions=len(synthesis.promotions),
            unresolved=len(synthesis.unresolved_questions),
        )

        promotions = PromotionExecutor.apply(
            investigation=investigation,
            synthesis=synthesis,
            findings=findings_tuple,
            thinker_identities={
                ref: thinker.identity
                for ref, thinker in self.thinkers.items()
            },
            promoter=self.reviewer.identity,
        )

        emit_progress(
            self.progress,
            stage="promotion",
            message=(
                f"Promoted {len(promotions)} hypothesis candidate(s) "
                f"into the institutional graph."
            ),
            promotions=len(promotions),
        )

        final_snapshot = investigation.snapshot()
        emit_progress(
            self.progress,
            stage="coordinator",
            message="Coordinator is preparing the final report.",
            actor_ref=self.coordinator.identity.ref,
        )
        report = await CoordinatorSession.report(
            coordinator=self.coordinator,
            snapshot=final_snapshot,
            findings=findings_tuple,
            synthesis_context=self._synthesis_context(
                synthesis,
                promotions,
            ),
        )

        emit_progress(
            self.progress,
            stage="complete",
            message="MOSAIC investigation cycle completed.",
            actor_ref=self.coordinator.identity.ref,
        )

        return ResearchCycleResult(
            plan=plan,
            findings=findings_tuple,
            synthesis=synthesis,
            promotions=promotions,
            report=report,
            review_request=review_request,
            review_response=review_response,
        )
