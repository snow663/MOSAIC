"""End-to-end MOSAIC research-cycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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
        progress: ProgressCallback | None = None,
    ) -> None:
        if not thinkers:
            raise ValueError("at least one Thinker is required")
        self.coordinator = coordinator
        self.thinkers = dict(thinkers)
        self.examiner = examiner
        self.reviewer = reviewer
        self.max_examination_rounds = max_examination_rounds
        self.progress = progress

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

        emit_progress(
            self.progress,
            stage="coordinator",
            message=(
                f"Coordinator selected {len(plan.tasks)} specialist task(s) "
                f"and extracted {len(plan.observations)} observation(s)."
            ),
            actor_ref=self.coordinator.identity.ref,
            tasks=len(plan.tasks),
            observations=len(plan.observations),
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
        )
