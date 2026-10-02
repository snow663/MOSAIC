import asyncio

import pytest

from mosaic import AgentIdentity, Investigation, SQLiteEventStore
from mosaic.agents import (
    AgentProposal,
    AgentProtocolError,
    HypothesisProposal,
    IndependentPass,
    PredictionProposal,
)


class FakeAgent:
    def __init__(self, identity, claim):
        self.identity = identity
        self.claim = claim
        self.seen_sequences = []

    async def analyze(self, snapshot):
        self.seen_sequences.append(snapshot.ledger_sequence)
        return AgentProposal(
            agent_ref=self.identity.ref,
            summary=f"Proposal from {self.identity.ref}",
            hypotheses=(
                HypothesisProposal(
                    claim=self.claim,
                    confidence=0.6,
                    predictions=(
                        PredictionProposal(
                            statement=f"Prediction for {self.claim}",
                            confidence=0.7,
                        ),
                    ),
                ),
            ),
        )


class SpoofingAgent(FakeAgent):
    async def analyze(self, snapshot):
        return AgentProposal(
            agent_ref="someone-else:v1",
            summary="Wrong attribution",
        )


def test_independent_pass_uses_one_frozen_snapshot(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "efi-001")
        investigation.record_observation(
            name="rpm",
            value=875,
            unit="rpm",
            source="logger",
        )

        mechanical = FakeAgent(
            AgentIdentity("mechanical-01", "v1", "specialist"),
            "Injector nonlinearity",
        )
        electrical = FakeAgent(
            AgentIdentity("electrical-01", "v1", "specialist"),
            "Sensor transport delay",
        )

        snapshot = investigation.snapshot()
        sequence_before = investigation.last_sequence

        proposals = asyncio.run(
            IndependentPass([mechanical, electrical]).run(snapshot)
        )

        assert len(proposals) == 2
        assert mechanical.seen_sequences == [sequence_before]
        assert electrical.seen_sequences == [sequence_before]

        # First-pass proposals are not committed to shared state yet.
        assert investigation.last_sequence == sequence_before
        assert investigation.hypotheses == {}
        assert investigation.predictions == {}


def test_independent_pass_rejects_spoofed_attribution(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "test")
        agent = SpoofingAgent(
            AgentIdentity("mechanical-01", "v1", "specialist"),
            "unused",
        )

        with pytest.raises(AgentProtocolError, match="returned proposal"):
            asyncio.run(IndependentPass([agent]).run(investigation.snapshot()))


def test_agent_identities_must_be_unique():
    identity = AgentIdentity("mechanical-01", "v1", "specialist")
    a = FakeAgent(identity, "A")
    b = FakeAgent(identity, "B")

    with pytest.raises(ValueError, match="must be unique"):
        IndependentPass([a, b])


def test_snapshot_captures_point_in_time_state(tmp_path):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "snapshot-test")
        obs = investigation.record_observation(
            name="map",
            value=42,
            unit="kPa",
            source="logger",
        )

        snapshot = investigation.snapshot()

        assert snapshot.investigation_id == "snapshot-test"
        assert snapshot.ledger_sequence == investigation.last_sequence
        assert snapshot.observations == (obs,)
        assert snapshot.hypotheses == ()
        assert snapshot.predictions == ()
        assert snapshot.relations == ()
