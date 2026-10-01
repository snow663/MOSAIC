import sqlite3

import pytest

from mosaic.kernel.events import Event
from mosaic.kernel.identity import AgentIdentity
from mosaic.kernel.store import SQLiteEventStore


@pytest.fixture
def actor():
    return AgentIdentity(
        agent_id="mechanical-01",
        version="v1",
        role="specialist",
        domains=("mechanical",),
    )


def test_round_trip_event(tmp_path, actor):
    path = tmp_path / "research.db"

    with SQLiteEventStore(path) as store:
        event = Event(
            event_type="observation.recorded",
            stream_id="investigation-001",
            actor=actor,
            payload={"rpm": 875, "map_kpa": 42},
        )

        stored = store.append(event)
        loaded = store.get(event.event_id)

        assert stored.sequence == 1
        assert loaded is not None
        assert loaded.event.event_id == event.event_id
        assert loaded.event.payload == {"map_kpa": 42, "rpm": 875}
        assert loaded.event.actor == actor
        assert store.verify_chain()


def test_events_form_a_hash_chain(tmp_path, actor):
    path = tmp_path / "research.db"

    with SQLiteEventStore(path) as store:
        first = store.append(
            Event(
                event_type="observation.recorded",
                stream_id="investigation-001",
                actor=actor,
                payload={"rpm": 875},
            )
        )
        second = store.append(
            Event(
                event_type="hypothesis.proposed",
                stream_id="investigation-001",
                actor=actor,
                payload={"claim": "Injector nonlinearity"},
            )
        )

        assert first.previous_hash is None
        assert second.previous_hash == first.event_hash
        assert store.verify_chain()


def test_stream_query_preserves_global_sequence(tmp_path, actor):
    path = tmp_path / "research.db"

    with SQLiteEventStore(path) as store:
        store.append(
            Event(
                event_type="observation.recorded",
                stream_id="A",
                actor=actor,
                payload={"value": 1},
            )
        )
        store.append(
            Event(
                event_type="observation.recorded",
                stream_id="B",
                actor=actor,
                payload={"value": 2},
            )
        )
        third = store.append(
            Event(
                event_type="prediction.created",
                stream_id="A",
                actor=actor,
                payload={"value": 3},
            )
        )

        events = store.events_for_stream("A")

        assert [item.sequence for item in events] == [1, third.sequence]
        assert [item.event.payload["value"] for item in events] == [1, 3]


def test_database_rejects_update_and_delete(tmp_path, actor):
    path = tmp_path / "research.db"

    with SQLiteEventStore(path) as store:
        stored = store.append(
            Event(
                event_type="observation.recorded",
                stream_id="investigation-001",
                actor=actor,
                payload={"rpm": 875},
            )
        )

        outsider = sqlite3.connect(path)
        try:
            with pytest.raises(
                sqlite3.IntegrityError,
                match="append-only",
            ):
                outsider.execute(
                    "UPDATE events SET event_type = ? WHERE event_id = ?",
                    ("rewritten", stored.event.event_id),
                )

            with pytest.raises(
                sqlite3.IntegrityError,
                match="append-only",
            ):
                outsider.execute(
                    "DELETE FROM events WHERE event_id = ?",
                    (stored.event.event_id,),
                )
        finally:
            outsider.close()


def test_independent_event_ids_are_unique(tmp_path, actor):
    path = tmp_path / "research.db"

    with SQLiteEventStore(path) as store:
        one = store.append(
            Event(
                event_type="observation.recorded",
                stream_id="investigation-001",
                actor=actor,
            )
        )
        two = store.append(
            Event(
                event_type="observation.recorded",
                stream_id="investigation-001",
                actor=actor,
            )
        )

        assert one.event.event_id != two.event.event_id
        assert one.event_hash != two.event_hash
        assert store.verify_chain()
