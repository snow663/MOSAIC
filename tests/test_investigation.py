import pytest

from mosaic.investigation import Investigation
from mosaic.kernel.identity import AgentIdentity
from mosaic.kernel.store import SQLiteEventStore
from mosaic.knowledge.relations import RelationType


@pytest.fixture
def mechanical():
    return AgentIdentity(
        agent_id="mechanical-01",
        version="v1",
        role="specialist",
        domains=("mechanical", "fuel-systems"),
    )


def test_investigation_round_trip_replay(tmp_path, mechanical):
    path = tmp_path / "research.db"

    with SQLiteEventStore(path) as store:
        investigation = Investigation(store, "efi-test-001")

        obs = investigation.record_observation(
            name="injector_pw",
            value=1.31,
            unit="ms",
            source="ALDL log",
            uncertainty=0.01,
        )
        hypothesis = investigation.propose_hypothesis(
            claim="Low-pulse-width injector nonlinearity causes the transient",
            actor=mechanical,
            confidence=0.58,
        )
        prediction = investigation.add_prediction(
            hypothesis_id=hypothesis.hypothesis_id,
            statement=(
                "Increasing fuel pressure will shift the effective "
                "pulse-width discontinuity"
            ),
            confidence=0.72,
            actor=mechanical,
            conditions={"rpm": 875, "map_kpa": 42},
        )
        relation = investigation.link(
            obs.observation_id,
            hypothesis.hypothesis_id,
            RelationType.SUPPORTS,
            actor=mechanical,
            rationale="The anomaly appears only in the low-PW region.",
        )

        replayed = Investigation(store, "efi-test-001")

        assert replayed.observations[obs.observation_id] == obs
        assert replayed.hypotheses[hypothesis.hypothesis_id] == hypothesis
        assert replayed.predictions[prediction.prediction_id] == prediction
        assert replayed.relations == [relation]
        assert replayed.predictions_for(hypothesis.hypothesis_id) == (
            prediction,
        )
        assert store.verify_chain()


def test_prediction_requires_existing_hypothesis(tmp_path, mechanical):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "test")

        with pytest.raises(KeyError, match="unknown hypothesis"):
            investigation.add_prediction(
                hypothesis_id="H-does-not-exist",
                statement="Something measurable happens",
                confidence=0.5,
                actor=mechanical,
            )


def test_relation_requires_existing_entities(tmp_path, mechanical):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "test")
        obs = investigation.record_observation(
            name="rpm",
            value=875,
            unit="rpm",
            source="logger",
        )

        with pytest.raises(KeyError, match="unknown target"):
            investigation.link(
                obs.observation_id,
                "H-missing",
                RelationType.SUPPORTS,
                actor=mechanical,
            )


def test_confidence_is_bounded(tmp_path, mechanical):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        investigation = Investigation(store, "test")

        with pytest.raises(ValueError, match="between 0 and 1"):
            investigation.propose_hypothesis(
                claim="Impossible confidence",
                actor=mechanical,
                confidence=1.5,
            )


def test_investigation_streams_are_isolated(tmp_path, mechanical):
    with SQLiteEventStore(tmp_path / "research.db") as store:
        a = Investigation(store, "A")
        b = Investigation(store, "B")

        obs_a = a.record_observation(
            name="rpm",
            value=875,
            unit="rpm",
            source="logger-A",
        )
        obs_b = b.record_observation(
            name="rpm",
            value=1200,
            unit="rpm",
            source="logger-B",
        )

        a_replayed = Investigation(store, "A")
        b_replayed = Investigation(store, "B")

        assert set(a_replayed.observations) == {obs_a.observation_id}
        assert set(b_replayed.observations) == {obs_b.observation_id}
