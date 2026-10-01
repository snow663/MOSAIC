"""Stable identities for every actor that writes to the MOSAIC ledger."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    """Versioned identity attached to every event-producing actor.

    Credibility belongs to a specific agent/version pair. A future prompt or
    method mutation must therefore create a new version rather than silently
    changing the meaning of an existing identity.
    """

    agent_id: str
    version: str
    role: str
    domains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("agent_id", "version", "role"):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")

        normalized_domains = tuple(
            domain.strip() for domain in self.domains if domain.strip()
        )
        object.__setattr__(self, "domains", normalized_domains)

    @property
    def ref(self) -> str:
        """Human-readable immutable version reference."""

        return f"{self.agent_id}:{self.version}"


KERNEL_IDENTITY = AgentIdentity(
    agent_id="kernel",
    version="1",
    role="institutional-kernel",
    domains=("infrastructure", "audit"),
)
