from dataclasses import FrozenInstanceError

import pytest

from mosaic.kernel.events import Event
from mosaic.kernel.identity import AgentIdentity


def test_identity_is_versioned_and_frozen():
    identity = AgentIdentity(
        agent_id="mechanical-01",
        version="v1",
        role="specialist",
        domains=("mechanical", "fluids"),
    )

    assert identity.ref == "mechanical-01:v1"
    assert identity.domains == ("mechanical", "fluids")

    with pytest.raises(FrozenInstanceError):
        identity.version = "v2"


def test_event_requires_aware_timestamp():
    from datetime import datetime

    actor = AgentIdentity("physics-01", "v1", "specialist")

    with pytest.raises(ValueError, match="timezone-aware"):
        Event(
            event_type="hypothesis.proposed",
            stream_id="investigation-1",
            actor=actor,
            occurred_at=datetime(2026, 1, 1),
        )


def test_event_rejects_non_portable_json():
    actor = AgentIdentity("physics-01", "v1", "specialist")

    with pytest.raises(ValueError):
        Event(
            event_type="observation.recorded",
            stream_id="investigation-1",
            actor=actor,
            payload={"value": float("nan")},
        )
