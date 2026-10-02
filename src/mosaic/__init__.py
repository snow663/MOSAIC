"""MOSAIC: Multi-agent Observational Synthesis, Adversarial Inquiry, and Calibration."""

from .kernel.events import Event
from .kernel.identity import AgentIdentity, KERNEL_IDENTITY
from .kernel.store import LedgerIntegrityError, SQLiteEventStore, StoredEvent

__all__ = [
    "AgentIdentity",
    "Event",
    "KERNEL_IDENTITY",
    "LedgerIntegrityError",
    "SQLiteEventStore",
    "StoredEvent",
]

__version__ = "0.1.0"
