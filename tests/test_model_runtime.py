import asyncio
import json

from mosaic import AgentIdentity, Investigation, SQLiteEventStore
from mosaic.agents import (
    BackendLocation,
    ModelCoordinator,
    ModelCrossDomainReviewer,
    ModelExaminer,
    ModelResponse,
    ModelThinker,
    ResearchCycle,
)


class ScriptedBackend:
    backend_id = "scripted-hosted"
    location = BackendLocation.REMOTE

    def __init__(self):
        self.requests = []
        self.examiner_calls = 0

    async def generate(self, request):
        self.requests.append(request)
        payload = json.loads(request.input_text)
        name = request.response_schema_name

        if name == "mosaic_coordinator_intake":
            existing_ids = [
                item["observation_id"]
                for item in payload["investigation"]["observations"]
            ]
            data = {
                "question": "What causes the lean transient?",
                "normalized_input": "Lean transient near 1.2 ms PW.",
                "observations": [
                    {
                        "name": "reported_injector_pw",
                        "value": 1.2,
                        "unit": "ms",
                        "uncertainty": None,
                    }
                ],
                "tasks": [
                    {
                        "assigned_to": "mechanical-01:v1",
                        "question": "Analyze fuel-system mechanisms.",
                        "scope": "Mechanical and fuel-delivery mechanisms.",
                        "observation_ids": existing_ids[:1],
                        "constraints": [],
                    }
                ],
                "ambiguities": [],
            }

        elif name == "mosaic_thinker_proposal":
            assert len(payload["investigation"]["observations"]) == 2
            data = {
                "summary": "Wall-film depletion is plausible.",
                "hypotheses": [
                    {
                        "claim": "Wall-film depletion causes the transient.",
                        "confidence": 0.62,
                        "rationale": "Transient fuel storage can create delay.",
                        "predictions": [
                            {
                                "statement": (
                                    "Disabling AE should expose a longer "
                                    "lean deficit."
                                ),
                                "confidence": 0.72,
                                "conditions": [
                                    {"name": "rpm", "value": 875}
                                ],
                            }
                        ],
                    }
                ],
                "open_questions": [],
            }

        elif name == "mosaic_examiner_review":
            assert "predictions" not in payload["investigation"]
            assert "relations" not in payload["investigation"]
            assert "observations" in payload["investigation"]
            self.examiner_calls += 1
            if self.examiner_calls == 1:
                data = {
                    "action": "challenge",
                    "question": "How does the larger throttle event recover faster?",
                    "targeted_claim": (
                        "Wall-film depletion causes the transient."
                    ),
                    "evidence_refs": [],
                    "disposition": None,
                    "findings_summary": None,
                    "reservations": [],
                    "unresolved_questions": [],
                    "statistics": [],
                }
            else:
                data = {
                    "action": "disposition",
                    "question": None,
                    "targeted_claim": None,
                    "evidence_refs": [],
                    "disposition": "accepted_with_reservations",
                    "findings_summary": (
                        "Mechanism is coherent but needs an AE-disabled test."
                    ),
                    "reservations": [
                        "Injector nonlinearity remains an alternative."
                    ],
                    "unresolved_questions": [
                        "Does the deficit persist with AE disabled?"
                    ],
                    "statistics": [
                        {"name": "challenge_rounds", "value": 1}
                    ],
                }

        elif name == "mosaic_thinker_examination_response":
            data = {
                "answer": (
                    "A larger transient can invoke more AE and mask part "
                    "of the underlying deficit."
                ),
                "revised_proposal": None,
            }

        elif name == "mosaic_cross_domain_synthesis":
            assert "predictions" not in payload["investigation"]
            assert "relations" not in payload["investigation"]
            finding = payload["examined_findings"][0]
            assert "proposal" not in finding
            examination_id = finding["examination_id"]
            data = {
                "finding_ids": [examination_id],
                "summary": "One examined mechanism remains viable.",
                "relations": [],
                "minority_reports": [],
                "promotions": [
                    {
                        "examination_id": examination_id,
                        "hypothesis_index": 0,
                        "rationale": (
                            "Retain the examined mechanism for prediction testing."
                        ),
                        "minority": False,
                    }
                ],
                "unresolved_questions": [
                    "Does the deficit persist with AE disabled?"
                ],
                "statistics": [
                    {"name": "findings_compared", "value": 1}
                ],
            }

        elif name == "mosaic_coordinator_report":
            examination_id = payload["examined_findings"][0]["examination_id"]
            assert payload["cross_domain_synthesis"]["summary"]
            assert "hypotheses" not in payload["investigation"]
            assert "predictions" not in payload["investigation"]
            assert "proposal" not in payload["examined_findings"][0]
            data = {
                "answer": (
                    "The examined wall-film mechanism remains viable but "
                    "requires an AE-disabled test."
                ),
                "finding_ids": [examination_id],
                "observations": [
                    "Injector pulse width was reported near 1.2 ms."
                ],
                "established": [
                    "The available evidence does not yet distinguish the cause."
                ],
                "surviving_hypotheses": [
                    "Wall-film depletion.",
                    "Injector low-PW nonlinearity.",
                ],
                "key_test": "Repeat the transient with AE disabled.",
                "unresolved_questions": [
                    "Does the deficit persist with AE disabled?"
                ],
                "caveats": [
                    "Injector nonlinearity remains an alternative."
                ],
            }

        else:
            raise AssertionError(f"unexpected schema: {name}")

        return ModelResponse(
            backend_id=self.backend_id,
            model=request.model,
            output_text=json.dumps(data),
        )


def test_model_backed_research_cycle_runs_end_to_end(tmp_path):
    backend = ScriptedBackend()
    coordinator_identity = AgentIdentity(
        "coordinator",
        "v1",
        "coordinator",
    )
    thinker_identity = AgentIdentity(
        "mechanical-01",
        "v1",
        "thinker",
        ("mechanical", "fuel-systems"),
    )
    examiner_identity = AgentIdentity(
        "examiner-01",
        "v1",
        "examiner",
    )
    reviewer_identity = AgentIdentity(
        "cross-domain-reviewer",
        "v1",
        "cross-domain-reviewer",
    )

    coordinator = ModelCoordinator(
        identity=coordinator_identity,
        backend=backend,
        model="hosted-model",
        thinker_catalog={
            thinker_identity.ref: "Mechanical and fuel-system specialist."
        },
    )
    thinker = ModelThinker(
        identity=thinker_identity,
        backend=backend,
        model="hosted-model",
        instructions="Focus on fuel delivery and manifold physics.",
    )
    examiner = ModelExaminer(
        identity=examiner_identity,
        backend=backend,
        model="hosted-model",
    )
    reviewer = ModelCrossDomainReviewer(
        identity=reviewer_identity,
        backend=backend,
        model="hosted-model",
    )
    progress_events = []
    cycle = ResearchCycle(
        coordinator=coordinator,
        thinkers={thinker.identity.ref: thinker},
        examiner=examiner,
        reviewer=reviewer,
        progress=progress_events.append,
    )

    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "hosted-cycle")
        investigation.record_observation(
            name="injector_pw",
            value=1.2,
            unit="ms",
            source="user",
        )

        result = asyncio.run(
            cycle.run(
                investigation=investigation,
                user_input="It goes lean near 1.2 ms.",
            )
        )

        assert len(result.findings) == 1
        assert len(result.promotions) == 1
        assert len(investigation.observations) == 2
        assert len(investigation.hypotheses) == 1
        assert len(investigation.predictions) == 1
        assert "AE-disabled" in result.report.answer
        assert result.report.key_test == "Repeat the transient with AE disabled."
        assert result.report.surviving_hypotheses == (
            "Wall-film depletion.",
            "Injector low-PW nonlinearity.",
        )
        assert result.report.finding_ids == (
            result.findings[0].examination_id,
        )
        assert result.report.examiner_status == (
            "Mechanical - ACCEPTED WITH RESERVATIONS, 1 challenge",
        )
        assert "EX-" not in result.report.examiner_status[0]
        assert progress_events
        stages = [event.stage for event in progress_events]
        assert stages[0] == "coordinator"
        assert "thinker" in stages
        assert "examiner" in stages
        assert "challenge" in stages
        assert "synthesis" in stages
        assert stages[-1] == "complete"

        assert [r.response_schema_name for r in backend.requests] == [
            "mosaic_coordinator_intake",
            "mosaic_thinker_proposal",
            "mosaic_examiner_review",
            "mosaic_thinker_examination_response",
            "mosaic_examiner_review",
            "mosaic_cross_domain_synthesis",
            "mosaic_coordinator_report",
        ]
        assert store.verify_chain()
