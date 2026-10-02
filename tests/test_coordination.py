import asyncio

import pytest

from mosaic import AgentIdentity, Investigation, SQLiteEventStore
from mosaic.agents import (
    CoordinatorProtocolError,
    CoordinatorReport,
    CoordinatorSession,
    ExaminationDisposition,
    ExaminationResult,
    InvestigationPlan,
    ThinkerProposal,
    ThinkerTask,
)


class MinimalThinker:
    def __init__(self, ref="mechanical-01"):
        self.identity = AgentIdentity(ref, "v1", "thinker")

    async def investigate(self, snapshot, task):
        raise NotImplementedError

    async def answer_examination(self, snapshot, task, proposal, challenge):
        raise NotImplementedError


class FakeCoordinator:
    def __init__(self, omit_findings=False, unavailable_target=False):
        self.identity = AgentIdentity("coordinator", "v1", "coordinator")
        self.omit_findings = omit_findings
        self.unavailable_target = unavailable_target

    async def intake(self, user_input, snapshot):
        target = (
            "unavailable-01:v1"
            if self.unavailable_target
            else "mechanical-01:v1"
        )
        return InvestigationPlan(
            coordinator_ref=self.identity.ref,
            question="What explains the transient?",
            normalized_input=user_input.strip(),
            tasks=(
                ThinkerTask(
                    assigned_to=target,
                    question="Analyze mechanical explanations.",
                    scope="Mechanical and fuel-system mechanisms.",
                ),
            ),
        )

    async def report(self, snapshot, findings):
        ids = tuple(f.examination_id for f in findings)
        if self.omit_findings:
            ids = ids[:1]
        return CoordinatorReport(
            coordinator_ref=self.identity.ref,
            answer="The examined findings remain partly unresolved.",
            finding_ids=ids,
        )


def make_finding(name):
    proposal = ThinkerProposal(
        thinker_ref=f"{name}:v1",
        task_id=f"T-{name}",
        summary=f"{name} finding",
    )
    return ExaminationResult(
        examiner_ref="examiner-01:v1",
        thinker_ref=proposal.thinker_ref,
        task_id=proposal.task_id,
        proposal=proposal,
        disposition=ExaminationDisposition.UNRESOLVED,
        findings_summary=f"{name} requires more evidence.",
    )


def test_coordinator_routes_only_to_available_thinkers(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        snapshot = Investigation(store, "coord").snapshot()
        thinker = MinimalThinker()

        plan = asyncio.run(
            CoordinatorSession.intake(
                coordinator=FakeCoordinator(),
                user_input="Engine falls lean near 1.2 ms.",
                snapshot=snapshot,
                available_thinkers=(thinker,),
            )
        )

        assert plan.tasks[0].assigned_to == thinker.identity.ref


def test_coordinator_cannot_route_to_unavailable_thinker(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        snapshot = Investigation(store, "coord").snapshot()

        with pytest.raises(CoordinatorProtocolError, match="unavailable Thinker"):
            asyncio.run(
                CoordinatorSession.intake(
                    coordinator=FakeCoordinator(unavailable_target=True),
                    user_input="Investigate.",
                    snapshot=snapshot,
                    available_thinkers=(MinimalThinker(),),
                )
            )


def test_coordinator_report_must_account_for_every_finding(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        snapshot = Investigation(store, "coord").snapshot()
        findings = (make_finding("mechanical"), make_finding("electrical"))

        report = asyncio.run(
            CoordinatorSession.report(
                coordinator=FakeCoordinator(),
                snapshot=snapshot,
                findings=findings,
            )
        )
        assert set(report.finding_ids) == {
            finding.examination_id for finding in findings
        }

        with pytest.raises(CoordinatorProtocolError, match="omitted examined"):
            asyncio.run(
                CoordinatorSession.report(
                    coordinator=FakeCoordinator(omit_findings=True),
                    snapshot=snapshot,
                    findings=findings,
                )
            )
