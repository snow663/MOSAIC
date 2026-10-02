import asyncio

import pytest

from mosaic import AgentIdentity, Investigation, SQLiteEventStore
from mosaic.agents import (
    CrossDomainSynthesis,
    ExaminationDisposition,
    ExaminationResult,
    FindingRelation,
    FindingRelationType,
    HypothesisProposal,
    MinorityReport,
    PredictionProposal,
    PromotionCandidate,
    PromotionExecutor,
    SynthesisProtocolError,
    SynthesisSession,
    ThinkerProposal,
)


def make_finding(
    *,
    thinker_ref,
    task_id,
    claim,
    disposition=ExaminationDisposition.ACCEPTED,
    confidence=0.65,
):
    proposal = ThinkerProposal(
        thinker_ref=thinker_ref,
        task_id=task_id,
        summary=f"Examined proposal: {claim}",
        hypotheses=(
            HypothesisProposal(
                claim=claim,
                confidence=confidence,
                predictions=(
                    PredictionProposal(
                        statement=f"Prediction for {claim}",
                        confidence=0.7,
                        conditions={"rpm": 875},
                    ),
                ),
            ),
        ),
    )
    return ExaminationResult(
        examiner_ref="examiner-01:v1",
        thinker_ref=thinker_ref,
        task_id=task_id,
        proposal=proposal,
        disposition=disposition,
        findings_summary=f"Finding for {claim}",
    )


class FakeCrossDomainReviewer:
    def __init__(self, *, omit_last=False, promote_rejected=False):
        self.identity = AgentIdentity(
            "cross-domain-reviewer",
            "v1",
            "cross-domain-reviewer",
            ("systems", "adversarial-review"),
        )
        self.omit_last = omit_last
        self.promote_rejected = promote_rejected

    async def synthesize(self, snapshot, findings):
        finding_ids = tuple(f.examination_id for f in findings)
        if self.omit_last:
            finding_ids = finding_ids[:-1]

        promotions = tuple(
            PromotionCandidate(
                examination_id=f.examination_id,
                hypothesis_index=0,
                rationale="Retain as a competing examined hypothesis.",
                minority=(index > 0),
            )
            for index, f in enumerate(findings)
            if self.promote_rejected
            or f.disposition is not ExaminationDisposition.REJECTED
        )

        return CrossDomainSynthesis(
            reviewer_ref=self.identity.ref,
            investigation_id=snapshot.investigation_id,
            source_ledger_sequence=snapshot.ledger_sequence,
            finding_ids=finding_ids,
            summary="Two mechanisms remain materially distinguishable.",
            relations=(
                FindingRelation(
                    source_examination_id=findings[0].examination_id,
                    target_examination_id=findings[1].examination_id,
                    relation_type=FindingRelationType.CONTRADICTS,
                    rationale="They predict different responses to fuel pressure.",
                ),
            )
            if len(findings) >= 2
            else (),
            minority_reports=(
                MinorityReport(
                    examination_id=findings[-1].examination_id,
                    rationale="Retain until the discriminating test is run.",
                ),
            ),
            promotions=promotions,
            unresolved_questions=("Which mechanism survives a pressure sweep?",),
            statistics={"findings_compared": len(findings)},
        )


def test_synthesis_promotes_examined_hypotheses_with_original_attribution(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "efi-synthesis")
        investigation.record_observation(
            name="injector_pw",
            value=1.31,
            unit="ms",
            source="ALDL",
        )
        snapshot = investigation.snapshot()

        mechanical = AgentIdentity(
            "mechanical-01",
            "v1",
            "thinker",
            ("mechanical",),
        )
        electrical = AgentIdentity(
            "electrical-01",
            "v1",
            "thinker",
            ("electrical",),
        )
        findings = (
            make_finding(
                thinker_ref=mechanical.ref,
                task_id="T-mech",
                claim="Wall-film depletion causes the lean transient.",
            ),
            make_finding(
                thinker_ref=electrical.ref,
                task_id="T-elec",
                claim="Injector low-PW nonlinearity causes the lean transient.",
                disposition=ExaminationDisposition.ACCEPTED_WITH_RESERVATIONS,
            ),
        )
        reviewer = FakeCrossDomainReviewer()

        synthesis = asyncio.run(
            SynthesisSession(audit_store=store).run(
                snapshot,
                findings,
                reviewer,
            )
        )
        records = PromotionExecutor.apply(
            investigation=investigation,
            synthesis=synthesis,
            findings=findings,
            thinker_identities={
                mechanical.ref: mechanical,
                electrical.ref: electrical,
            },
            promoter=reviewer.identity,
        )

        assert len(records) == 2
        assert len(investigation.hypotheses) == 2
        assert len(investigation.predictions) == 2
        assert {
            hypothesis.proposed_by
            for hypothesis in investigation.hypotheses.values()
        } == {mechanical.ref, electrical.ref}
        assert records[1].minority is True

        audit_types = [
            item.event.event_type
            for item in store.events_for_stream("efi-synthesis")
            if item.event.event_type.startswith("synthesis.")
        ]
        assert audit_types == [
            "synthesis.completed",
            "synthesis.hypothesis_promoted",
            "synthesis.hypothesis_promoted",
        ]
        assert store.verify_chain()


def test_synthesis_cannot_omit_examined_finding(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "omit")
        snapshot = investigation.snapshot()
        findings = (
            make_finding(
                thinker_ref="mechanical-01:v1",
                task_id="T1",
                claim="A",
            ),
            make_finding(
                thinker_ref="electrical-01:v1",
                task_id="T2",
                claim="B",
            ),
        )

        with pytest.raises(SynthesisProtocolError, match="omitted examined"):
            asyncio.run(
                SynthesisSession(audit_store=store).run(
                    snapshot,
                    findings,
                    FakeCrossDomainReviewer(omit_last=True),
                )
            )


def test_rejected_finding_cannot_be_promoted(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "rejected")
        snapshot = investigation.snapshot()
        findings = (
            make_finding(
                thinker_ref="mechanical-01:v1",
                task_id="T1",
                claim="Rejected mechanism",
                disposition=ExaminationDisposition.REJECTED,
            ),
        )

        with pytest.raises(SynthesisProtocolError, match="rejected"):
            asyncio.run(
                SynthesisSession(audit_store=store).run(
                    snapshot,
                    findings,
                    FakeCrossDomainReviewer(promote_rejected=True),
                )
            )


def test_promotion_refuses_epistemic_drift_after_snapshot(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "drift")
        snapshot = investigation.snapshot()
        thinker = AgentIdentity("mechanical-01", "v1", "thinker")
        finding = make_finding(
            thinker_ref=thinker.ref,
            task_id="T1",
            claim="Candidate mechanism",
        )
        reviewer = FakeCrossDomainReviewer()
        synthesis = asyncio.run(
            SynthesisSession(audit_store=store).run(
                snapshot,
                (finding,),
                reviewer,
            )
        )

        investigation.record_observation(
            name="fuel_pressure",
            value=12.0,
            unit="psi",
            source="gauge",
        )

        with pytest.raises(SynthesisProtocolError, match="changed after"):
            PromotionExecutor.apply(
                investigation=investigation,
                synthesis=synthesis,
                findings=(finding,),
                thinker_identities={thinker.ref: thinker},
                promoter=reviewer.identity,
            )


def test_unresolved_finding_can_be_retained_as_minority_hypothesis(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "minority")
        snapshot = investigation.snapshot()
        thinker = AgentIdentity("physics-01", "v1", "thinker")
        finding = make_finding(
            thinker_ref=thinker.ref,
            task_id="T1",
            claim="Transport delay dominates the apparent transient.",
            disposition=ExaminationDisposition.UNRESOLVED,
        )
        reviewer = FakeCrossDomainReviewer()

        synthesis = asyncio.run(
            SynthesisSession(audit_store=store).run(
                snapshot,
                (finding,),
                reviewer,
            )
        )
        records = PromotionExecutor.apply(
            investigation=investigation,
            synthesis=synthesis,
            findings=(finding,),
            thinker_identities={thinker.ref: thinker},
            promoter=reviewer.identity,
        )

        assert len(records) == 1
        assert len(investigation.hypotheses) == 1
