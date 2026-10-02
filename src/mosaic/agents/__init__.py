"""Provider-independent research agent contracts and orchestration."""

from .contracts import (
    AgentDefinition,
    AgentProposal,
    BackendLocation,
    HypothesisProposal,
    ModelBackend,
    ModelRequest,
    ModelResponse,
    PredictionProposal,
    ResearchAgent,
)
from .independent import AgentProtocolError, IndependentPass

__all__ = [
    "AgentDefinition",
    "AgentProposal",
    "AgentProtocolError",
    "BackendLocation",
    "HypothesisProposal",
    "IndependentPass",
    "ModelBackend",
    "ModelRequest",
    "ModelResponse",
    "PredictionProposal",
    "ResearchAgent",
]
