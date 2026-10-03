"""Contracts separating MOSAIC research agents from inference backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from mosaic.kernel.identity import AgentIdentity
from mosaic.snapshot import InvestigationSnapshot


def _require_text(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _confidence(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be finite and between 0 and 1")
    return value


class BackendLocation(StrEnum):
    """Where inference is performed."""

    LOCAL = "local"
    REMOTE = "remote"
    DETERMINISTIC = "deterministic"


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Versioned research-role configuration independent of any provider SDK."""

    identity: AgentIdentity
    instructions: str
    backend_id: str
    model: str
    memory_namespace: str | None = None
    allowed_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instructions",
            _require_text(self.instructions, "instructions"),
        )
        object.__setattr__(
            self,
            "backend_id",
            _require_text(self.backend_id, "backend_id"),
        )
        object.__setattr__(self, "model", _require_text(self.model, "model"))
        if self.memory_namespace is not None:
            object.__setattr__(
                self,
                "memory_namespace",
                _require_text(self.memory_namespace, "memory_namespace"),
            )


@dataclass(frozen=True, slots=True)
class PredictionProposal:
    """A falsifiable prediction proposed by an agent."""

    statement: str
    confidence: float
    conditions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "statement",
            _require_text(self.statement, "statement"),
        )
        normalized = _confidence(self.confidence)
        assert normalized is not None
        object.__setattr__(self, "confidence", normalized)
        object.__setattr__(
            self,
            "conditions",
            MappingProxyType(dict(self.conditions)),
        )


@dataclass(frozen=True, slots=True)
class HypothesisProposal:
    """A proposed interpretation with explicit predictions."""

    claim: str
    confidence: float | None = None
    rationale: str | None = None
    predictions: tuple[PredictionProposal, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim", _require_text(self.claim, "claim"))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if self.rationale is not None:
            object.__setattr__(
                self,
                "rationale",
                _require_text(self.rationale, "rationale"),
            )


@dataclass(frozen=True, slots=True)
class AgentProposal:
    """Structured first-pass output.

    Rationale is intended to be a concise, auditable justification, not hidden
    chain-of-thought. Agents cannot create observations through this object.
    """

    agent_ref: str
    summary: str
    hypotheses: tuple[HypothesisProposal, ...] = ()
    questions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_ref",
            _require_text(self.agent_ref, "agent_ref"),
        )
        object.__setattr__(
            self,
            "summary",
            _require_text(self.summary, "summary"),
        )
        normalized_questions = tuple(
            _require_text(question, "question") for question in self.questions
        )
        object.__setattr__(self, "questions", normalized_questions)


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Provider-neutral request passed to an inference backend."""

    model: str
    system: str
    input_text: str
    response_schema_name: str | None = None
    response_schema: Mapping[str, Any] | None = None
    usage_tag: str | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _require_text(self.model, "model"))
        object.__setattr__(self, "system", _require_text(self.system, "system"))
        object.__setattr__(
            self,
            "input_text",
            _require_text(self.input_text, "input_text"),
        )
        if self.response_schema_name is not None:
            object.__setattr__(
                self,
                "response_schema_name",
                _require_text(
                    self.response_schema_name,
                    "response_schema_name",
                ),
            )
        if self.response_schema is not None:
            object.__setattr__(
                self,
                "response_schema",
                MappingProxyType(dict(self.response_schema)),
            )
        if self.usage_tag is not None:
            object.__setattr__(
                self,
                "usage_tag",
                _require_text(self.usage_tag, "usage_tag"),
            )
        if self.max_output_tokens is not None:
            value = int(self.max_output_tokens)
            if value < 1:
                raise ValueError("max_output_tokens must be positive")
            object.__setattr__(self, "max_output_tokens", value)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Provider-neutral raw model response plus provenance and token usage."""

    backend_id: str
    model: str
    output_text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_id",
            _require_text(self.backend_id, "backend_id"),
        )
        object.__setattr__(self, "model", _require_text(self.model, "model"))
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            value = int(value)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)


@runtime_checkable
class ModelBackend(Protocol):
    """Inference adapter implemented by local, remote, or deterministic backends."""

    backend_id: str
    location: BackendLocation

    async def generate(self, request: ModelRequest) -> ModelResponse:
        ...


@runtime_checkable
class ResearchAgent(Protocol):
    """A specialist that analyzes a read-only investigation snapshot."""

    identity: AgentIdentity

    async def analyze(
        self,
        snapshot: InvestigationSnapshot,
    ) -> AgentProposal:
        ...
