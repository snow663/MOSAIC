"""Lightweight progress events for interactive MOSAIC runtimes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

ProgressCallback = Callable[["ProgressEvent"], None]


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    stage: str
    message: str
    actor_ref: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stage = self.stage.strip()
        message = self.message.strip()
        if not stage:
            raise ValueError("stage must be non-empty")
        if not message:
            raise ValueError("message must be non-empty")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "message", message)
        object.__setattr__(
            self,
            "detail",
            MappingProxyType(dict(self.detail)),
        )


def emit_progress(
    callback: ProgressCallback | None,
    *,
    stage: str,
    message: str,
    actor_ref: str | None = None,
    **detail: Any,
) -> None:
    if callback is None:
        return
    callback(
        ProgressEvent(
            stage=stage,
            message=message,
            actor_ref=actor_ref,
            detail=detail,
        )
    )
