"""Read-only investigation snapshots for analysis and agent execution."""

from __future__ import annotations

from dataclasses import dataclass

from mosaic.knowledge import Hypothesis, Observation, Prediction, Relation


@dataclass(frozen=True, slots=True)
class InvestigationSnapshot:
    """A point-in-time view of an investigation.

    Agent passes consume snapshots rather than a live Investigation object.
    This prevents an agent from seeing proposals appended by another agent
    during the same independent reasoning round.
    """

    investigation_id: str
    ledger_sequence: int
    observations: tuple[Observation, ...]
    hypotheses: tuple[Hypothesis, ...]
    predictions: tuple[Prediction, ...]
    relations: tuple[Relation, ...]
