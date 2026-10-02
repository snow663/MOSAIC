import asyncio

import pytest

from mosaic import AgentIdentity, Investigation, SQLiteEventStore
from mosaic.agents import (
    ExaminationChallenge,
    ExaminationDisposition,
    ExaminationLimitError,
    ExaminationResult,
    HypothesisProposal,
    PredictionProposal,
    PrivateExamination,
    ThinkerProposal,
    ThinkerResponse,
    ThinkerTask,
)


class RevisingThinker:
    def __init__(self):
        self.identity = AgentIdentity(
            "mechanical-01",
            "v1",
            "thinker",
            ("mechanical", "fuel-systems"),
        )
        self.challenge_count = 0

    async def investigate(self, snapshot, task):
        return ThinkerProposal(
            thinker_ref=self.identity.ref,
            task_id=task.task_id,
            summary="Wall-film depletion is plausible.",
            hypotheses=(
                HypothesisProposal(
                    claim="Wall-film depletion causes the lean transient.",
                    confidence=0.62,
                    predictions=(
                        PredictionProposal(
                            statement="Disabling AE should expose a longer deficit.",
                            confidence=0.7,
                        ),
                    ),
                ),
            ),
        )

    async def answer_examination(self, snapshot, task, proposal, challenge):
        self.challenge_count += 1
        revised = ThinkerProposal(
            thinker_ref=self.identity.ref,
            task_id=task.task_id,
            summary=(
                "Wall-film depletion remains plausible, with AE magnitude "
                "masking part of the observed recovery."
            ),
            hypotheses=proposal.hypotheses,
            open_questions=("Need matched AE-disabled transient data.",),
        )
        return ThinkerResponse(
            thinker_ref=self.identity.ref,
            challenge_id=challenge.challenge_id,
            answer=(
                "A larger throttle event can invoke more AE, shortening the "
                "observed lean period without changing the underlying deficit."
            ),
            revised_proposal=revised,
        )


class OneChallengeExaminer:
    def __init__(self):
        self.identity = AgentIdentity(
            "examiner-01",
            "v1",
            "examiner",
            ("analysis",),
        )
        self.calls = 0

    async def examine(self, snapshot, task, proposal, transcript):
        self.calls += 1
        if not transcript:
            return ExaminationChallenge(
                examiner_ref=self.identity.ref,
                thinker_ref=proposal.thinker_ref,
                proposal_id=proposal.proposal_id,
                question=(
                    "How does this explain faster recovery during the larger "
                    "throttle movement?"
                ),
                targeted_claim=proposal.hypotheses[0].claim,
            )

        return ExaminationResult(
            examiner_ref=self.identity.ref,
            thinker_ref=proposal.thinker_ref,
            task_id=task.task_id,
            proposal=proposal,
            disposition=ExaminationDisposition.ACCEPTED_WITH_RESERVATIONS,
            findings_summary=(
                "The mechanism is internally coherent after revision but "
                "requires an AE-disabled discriminating test."
            ),
            reservations=("Injector nonlinearity remains an alternative.",),
            unresolved_questions=("Does the effect persist with AE disabled?",),
            statistics={"rounds_requested": 1},
        )


class EndlessExaminer(OneChallengeExaminer):
    async def examine(self, snapshot, task, proposal, transcript):
        return ExaminationChallenge(
            examiner_ref=self.identity.ref,
            thinker_ref=proposal.thinker_ref,
            proposal_id=proposal.proposal_id,
            question="Provide another independent discriminator.",
        )


class SpoofingExaminer(OneChallengeExaminer):
    async def examine(self, snapshot, task, proposal, transcript):
        return ExaminationChallenge(
            examiner_ref="other-examiner:v9",
            thinker_ref=proposal.thinker_ref,
            proposal_id=proposal.proposal_id,
            question="Spoofed challenge",
        )


def make_task(thinker):
    return ThinkerTask(
        assigned_to=thinker.identity.ref,
        question="Explain the observed lean transient.",
        scope="Fuel delivery and manifold mechanisms only.",
        observation_ids=("O-1", "O-2"),
    )


def test_private_examination_challenges_then_accepts(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "efi-exam-001")
        investigation.record_observation(
            name="injector_pw",
            value=1.31,
            unit="ms",
            source="ALDL log",
        )
        snapshot = investigation.snapshot()

        thinker = RevisingThinker()
        examiner = OneChallengeExaminer()
        result = asyncio.run(
            PrivateExamination(audit_store=store).run(
                snapshot,
                make_task(thinker),
                thinker,
                examiner,
            )
        )

        assert result.disposition is ExaminationDisposition.ACCEPTED_WITH_RESERVATIONS
        assert result.thinker_ref == thinker.identity.ref
        assert result.examiner_ref == examiner.identity.ref
        assert len(result.transcript) == 1
        assert thinker.challenge_count == 1
        assert result.proposal.proposal_id != result.initial_proposal_id
        assert result.statistics["rounds_requested"] == 1

        audit_types = [
            item.event.event_type
            for item in store.events_for_stream("efi-exam-001")
            if item.event.event_type.startswith("examination.")
        ]
        assert audit_types == [
            "examination.started",
            "examination.challenge",
            "examination.response",
            "examination.completed",
        ]
        assert store.verify_chain()


def test_examination_does_not_commit_hypotheses_to_investigation(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "isolated")
        snapshot = investigation.snapshot()
        thinker = RevisingThinker()
        examiner = OneChallengeExaminer()

        asyncio.run(
            PrivateExamination(audit_store=store).run(
                snapshot,
                make_task(thinker),
                thinker,
                examiner,
            )
        )

        replayed = Investigation(store, "isolated")
        assert replayed.hypotheses == {}
        assert replayed.predictions == {}


def test_examination_enforces_task_assignment(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "assignment")
        thinker = RevisingThinker()
        examiner = OneChallengeExaminer()
        task = ThinkerTask(
            assigned_to="electrical-01:v1",
            question="Investigate.",
            scope="Electrical only.",
        )

        with pytest.raises(Exception, match="task assigned"):
            asyncio.run(
                PrivateExamination(audit_store=store).run(
                    investigation.snapshot(),
                    task,
                    thinker,
                    examiner,
                )
            )


def test_examination_rejects_spoofed_examiner(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "spoof")
        thinker = RevisingThinker()

        with pytest.raises(Exception, match="challenge attributed"):
            asyncio.run(
                PrivateExamination(audit_store=store).run(
                    investigation.snapshot(),
                    make_task(thinker),
                    thinker,
                    SpoofingExaminer(),
                )
            )


def test_examination_round_limit_is_audited(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "limit")
        thinker = RevisingThinker()
        examiner = EndlessExaminer()

        with pytest.raises(ExaminationLimitError, match="failed to issue"):
            asyncio.run(
                PrivateExamination(
                    audit_store=store,
                    max_rounds=2,
                ).run(
                    investigation.snapshot(),
                    make_task(thinker),
                    thinker,
                    examiner,
                )
            )

        audit_types = [
            item.event.event_type
            for item in store.events_for_stream("limit")
            if item.event.event_type.startswith("examination.")
        ]
        assert audit_types[-1] == "examination.aborted"
