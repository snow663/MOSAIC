"""Coordinator gateway enforcing MOSAIC interface discipline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mosaic.snapshot import InvestigationSnapshot

from .independent import AgentProtocolError
from .roles import (
    Coordinator,
    CoordinatorReport,
    ExaminationResult,
    InvestigationPlan,
    Thinker,
)


class CoordinatorProtocolError(AgentProtocolError):
    """Raised when the Coordinator violates its constrained interface role."""


class CoordinatorSession:
    """Validate Coordinator routing and reporting without granting science authority."""

    @staticmethod
    async def intake(
        *,
        coordinator: Coordinator,
        user_input: str,
        snapshot: InvestigationSnapshot,
        available_thinkers: Sequence[Thinker],
    ) -> InvestigationPlan:
        if not user_input.strip():
            raise ValueError("user_input must be non-empty")

        thinker_refs = {thinker.identity.ref for thinker in available_thinkers}
        if not thinker_refs:
            raise ValueError("at least one available Thinker is required")

        plan = await coordinator.intake(user_input, snapshot)

        if plan.coordinator_ref != coordinator.identity.ref:
            raise CoordinatorProtocolError(
                f"Coordinator {coordinator.identity.ref} returned plan attributed "
                f"to {plan.coordinator_ref}"
            )

        task_ids: set[str] = set()
        for task in plan.tasks:
            if task.assigned_to not in thinker_refs:
                raise CoordinatorProtocolError(
                    f"Coordinator assigned task to unavailable Thinker "
                    f"{task.assigned_to}"
                )
            if task.task_id in task_ids:
                raise CoordinatorProtocolError(
                    f"Coordinator reused task id {task.task_id}"
                )
            task_ids.add(task.task_id)

        return plan

    @staticmethod
    async def report(
        *,
        coordinator: Coordinator,
        snapshot: InvestigationSnapshot,
        findings: Sequence[ExaminationResult],
        synthesis_context: Mapping[str, Any] | None = None,
    ) -> CoordinatorReport:
        if not findings:
            raise ValueError("at least one examined finding is required")

        if synthesis_context is None:
            report = await coordinator.report(snapshot, findings)
        else:
            report = await coordinator.report(
                snapshot,
                findings,
                synthesis_context,
            )

        if report.coordinator_ref != coordinator.identity.ref:
            raise CoordinatorProtocolError(
                f"Coordinator {coordinator.identity.ref} returned report attributed "
                f"to {report.coordinator_ref}"
            )

        expected = {finding.examination_id for finding in findings}
        represented = set(report.finding_ids)

        missing = expected - represented
        unknown = represented - expected
        if missing:
            raise CoordinatorProtocolError(
                "Coordinator report omitted examined findings: "
                + ", ".join(sorted(missing))
            )
        if unknown:
            raise CoordinatorProtocolError(
                "Coordinator report cited unknown findings: "
                + ", ".join(sorted(unknown))
            )

        return report
