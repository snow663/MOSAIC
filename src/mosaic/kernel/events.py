"""Canonical event envelope used by the MOSAIC event ledger."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from .identity import AgentIdentity


JsonObject = Mapping[str, Any]


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _json_ready(value: Any) -> Any:
    """Convert immutable event data back into ordinary JSON containers."""

    if isinstance(value, MappingABC):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    return value


def _freeze_json(value: Any) -> Any:
    """Recursively freeze JSON containers after validating string keys."""

    if isinstance(value, MappingABC):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)

    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)

    return value


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for storage and hashing.

    NaN and infinities are rejected because they are not portable JSON and
    would undermine deterministic event hashing.
    """

    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class Event:
    """An immutable statement that something happened in MOSAIC.

    Events are facts about system activity, not claims that an interpretation
    is true. Domain objects such as observations and hypotheses will be built
    on top of this envelope.
    """

    event_type: str
    stream_id: str
    actor: AgentIdentity
    payload: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=utc_now)
    causation_id: str | None = None
    correlation_id: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.event_type or not self.event_type.strip():
            raise ValueError("event_type must be non-empty")
        if not self.stream_id or not self.stream_id.strip():
            raise ValueError("stream_id must be non-empty")
        if not self.event_id or not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if self.schema_version < 1:
            raise ValueError("schema_version must be at least 1")
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")

        # Validate before freezing so malformed data never reaches persistence.
        canonical_json(self.payload)
        canonical_json(self.metadata)

        object.__setattr__(self, "event_type", self.event_type.strip())
        object.__setattr__(self, "stream_id", self.stream_id.strip())
        object.__setattr__(
            self,
            "occurred_at",
            self.occurred_at.astimezone(timezone.utc),
        )
        object.__setattr__(self, "payload", _freeze_json(self.payload))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))
