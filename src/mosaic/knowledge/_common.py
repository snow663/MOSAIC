"""Shared helpers for immutable epistemic records."""

from __future__ import annotations

from math import isfinite
from uuid import uuid4


def new_entity_id(prefix: str) -> str:
    return f"{prefix}-{uuid4()}"


def require_text(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def validate_confidence(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be finite and between 0 and 1")
    return value
