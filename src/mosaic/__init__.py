"""MOSAIC: Multi-agent Observational Synthesis, Adversarial Inquiry, and Calibration."""

from .investigation import Investigation
from .kernel.events import Event
from .kernel.identity import AgentIdentity, KERNEL_IDENTITY
from .kernel.store import LedgerIntegrityError, SQLiteEventStore, StoredEvent
from .knowledge import Hypothesis, Observation, Prediction, Relation, RelationType

__all__ = [
    "AgentIdentity",
    "Event",
    "Hypothesis",
    "Investigation",
    "KERNEL_IDENTITY",
    "LedgerIntegrityError",
    "Observation",
    "Prediction",
    "Relation",
    "RelationType",
    "SQLiteEventStore",
    "StoredEvent",
]

__version__ = "0.1.0"
