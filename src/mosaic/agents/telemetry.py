"""Token and cost telemetry for model-backed MOSAIC runs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

from .contracts import BackendLocation, ModelBackend, ModelRequest, ModelResponse


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Standard text-token prices in USD per one million tokens."""

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "input_per_million",
            "output_per_million",
            "cached_input_per_million",
        ):
            value = getattr(self, field_name)
            if value is not None and float(value) < 0:
                raise ValueError(f"{field_name} must be non-negative")


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """Token usage and estimated cost for one backend request."""

    tag: str
    backend_id: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    reasoning_tokens: int | None
    estimated_cost_usd: float | None


@dataclass(frozen=True, slots=True)
class UsageTotal:
    """Aggregated usage for a role/tag or complete investigation."""

    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    estimated_cost_usd: float | None
    requests: int


def openai_standard_pricing(model: str) -> ModelPricing | None:
    """Return an explicit convenience pricing snapshot for common models."""

    prices = {
        "gpt-6-luna": ModelPricing(
            input_per_million=0.10,
            cached_input_per_million=0.01,
            output_per_million=0.50,
        ),
        "gpt-6.1-sol": ModelPricing(
            input_per_million=2.00,
            cached_input_per_million=0.10,
            output_per_million=10.00,
        ),
    }
    return prices.get(model)


class UsageTracker:
    """Collect per-request usage without coupling it to the event ledger."""

    def __init__(
        self,
        *,
        pricing: Mapping[str, ModelPricing] | None = None,
    ) -> None:
        self.pricing = dict(pricing or {})
        self._records: list[UsageRecord] = []

    @property
    def records(self) -> tuple[UsageRecord, ...]:
        return tuple(self._records)

    def _cost(
        self,
        *,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_input_tokens: int | None,
    ) -> float | None:
        price = self.pricing.get(model)
        if price is None or input_tokens is None or output_tokens is None:
            return None

        cached = min(cached_input_tokens or 0, input_tokens)
        uncached = max(input_tokens - cached, 0)
        cached_rate = (
            price.input_per_million
            if price.cached_input_per_million is None
            else price.cached_input_per_million
        )

        return (
            uncached * price.input_per_million
            + cached * cached_rate
            + output_tokens * price.output_per_million
        ) / 1_000_000.0

    def record(self, request: ModelRequest, response: ModelResponse) -> UsageRecord:
        record = UsageRecord(
            tag=request.usage_tag or request.response_schema_name or "model_call",
            backend_id=response.backend_id,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_input_tokens=response.cached_input_tokens,
            reasoning_tokens=response.reasoning_tokens,
            estimated_cost_usd=self._cost(
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cached_input_tokens=response.cached_input_tokens,
            ),
        )
        self._records.append(record)
        return record

    @staticmethod
    def _aggregate(records: list[UsageRecord]) -> UsageTotal:
        known_costs = [
            record.estimated_cost_usd
            for record in records
            if record.estimated_cost_usd is not None
        ]
        all_costs_known = len(known_costs) == len(records)

        return UsageTotal(
            input_tokens=sum(record.input_tokens or 0 for record in records),
            output_tokens=sum(record.output_tokens or 0 for record in records),
            cached_input_tokens=sum(
                record.cached_input_tokens or 0 for record in records
            ),
            reasoning_tokens=sum(
                record.reasoning_tokens or 0 for record in records
            ),
            estimated_cost_usd=(
                sum(known_costs) if records and all_costs_known else None
            ),
            requests=len(records),
        )

    def total(self) -> UsageTotal:
        return self._aggregate(list(self._records))

    def by_tag(self) -> dict[str, UsageTotal]:
        grouped: dict[str, list[UsageRecord]] = defaultdict(list)
        for record in self._records:
            grouped[record.tag].append(record)
        return {
            tag: self._aggregate(records)
            for tag, records in grouped.items()
        }


class TrackedBackend:
    """Wrap any ModelBackend and record its token usage."""

    def __init__(self, backend: ModelBackend, tracker: UsageTracker) -> None:
        self.backend = backend
        self.tracker = tracker
        self.backend_id = backend.backend_id
        self.location = BackendLocation(backend.location)

    async def generate(self, request: ModelRequest) -> ModelResponse:
        response = await self.backend.generate(request)
        self.tracker.record(request, response)
        return response
