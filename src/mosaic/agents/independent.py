"""Independent first-pass orchestration for research agents."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from mosaic.snapshot import InvestigationSnapshot

from .contracts import AgentProposal, ResearchAgent


class AgentProtocolError(RuntimeError):
    """Raised when an agent violates the MOSAIC agent contract."""


class IndependentPass:
    """Run specialists against one frozen snapshot before cross-examination."""

    def __init__(self, agents: Sequence[ResearchAgent]) -> None:
        if not agents:
            raise ValueError("at least one agent is required")

        refs = [agent.identity.ref for agent in agents]
        if len(refs) != len(set(refs)):
            raise ValueError("agent identities must be unique within a pass")

        self._agents = tuple(agents)

    async def run(
        self,
        snapshot: InvestigationSnapshot,
    ) -> tuple[AgentProposal, ...]:
        """Collect proposals without mutating the investigation."""

        proposals = await asyncio.gather(
            *(agent.analyze(snapshot) for agent in self._agents)
        )

        validated: list[AgentProposal] = []
        for agent, proposal in zip(self._agents, proposals, strict=True):
            if proposal.agent_ref != agent.identity.ref:
                raise AgentProtocolError(
                    f"agent {agent.identity.ref} returned proposal attributed "
                    f"to {proposal.agent_ref}"
                )
            validated.append(proposal)

        return tuple(validated)
