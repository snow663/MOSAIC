"""End-to-end MOSAIC research-cycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from mosaic.investigation import Investigation
from mosaic.kernel.events import Event

from .coordination import CoordinatorSession
from .examination import PrivateExamination
from .roles import (
    Coordinator,
    CoordinatorReport,
    ExaminationResult,
    Examiner,
    InvestigationPlan,
    Thinker,
)
from .synthesis import (
    CrossDomainReviewer,
    CrossDomainSynthesis,
    PromotionExecutor,
    PromotionRecord,
    SynthesisSession,
)


@dataclass(frozen=True, slots=True)
class ResearchCycleResult:
    """Complete result of one MOSAIC investigation cycle."""

    plan: InvestigationPlan
    findings: tuple[ExaminationResult, ...]
    synthesis: CrossDomainSynthesis
    promotions: tuple[PromotionRecord, ...]
    report: CoordinatorReport


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
    ) -> None:
        if not thinkers:
            raise ValueError("at least one Thinker is required")
        self.coordinator = coordinator
        self.thinkers = dict(thinkers)
        self.examiner = examiner
        self.reviewer = reviewer
        self.max_examination_rounds = max_examination_rounds

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

    async def run(
        self,
        *,
        investigation: Investigation,
        user_input: str,
    ) -> ResearchCycleResult:
        """Run one complete MOSAIC reasoning cycle."""

        intake_snapshot = investigation.snapshot()
        plan = await CoordinatorSession.intake(
            coordinator=self.coordinator,
            user_input=user_input,
            snapshot=intake_snapshot,
            available_thinkers=tuple(self.thinkers.values()),
        )

        investigation.store.append(
            Event(
                event_type="coordinator.intake_completed",
                stream_id=investigation.investigation_id,
                actor=self.coordinator.identity,
                correlation_id=plan.plan_id,
                payload={
                    "plan_id": plan.plan_id,
                    "raw_user_input": user_input,
                    "normalized_input": plan.normalized_input,
                    "question": plan.question,
                    "ambiguities": list(plan.ambiguities),
                    "task_ids": [task.task_id for task in plan.tasks],
                    "observation_count": len(plan.observations),
                },
            )
        )

        for draft in plan.observations:
            investigation.record_observation(
                name=draft.name,
                value=draft.value,
                unit=draft.unit,
                uncertainty=draft.uncertainty,
                source=f"user_input:{plan.plan_id}",
                actor=self.coordinator.identity,
            )

        # Direct audit writes do not mutate the in-memory Investigation state.
        # Replay once so every Thinker receives the same snapshot containing
        # all observations extracted from the current user input.
        investigation.replay()
        source_snapshot = investigation.snapshot()

        examination = PrivateExamination(
            audit_store=investigation.store,
            max_rounds=self.max_examination_rounds,
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
        synthesis = await SynthesisSession(
            audit_store=investigation.store
        ).run(
            source_snapshot,
            findings_tuple,
            self.reviewer,
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

        final_snapshot = investigation.snapshot()
        report = await CoordinatorSession.report(
            coordinator=self.coordinator,
            snapshot=final_snapshot,
            findings=findings_tuple,
            synthesis_context=self._synthesis_context(
                synthesis,
                promotions,
            ),
        )

        return ResearchCycleResult(
            plan=plan,
            findings=findings_tuple,
            synthesis=synthesis,
            promotions=promotions,
            report=report,
        )
