from mosaic.agents import (
    BackendLocation,
    ModelPricing,
    ModelRequest,
    ModelResponse,
    TrackedBackend,
    UsageTracker,
    openai_standard_pricing,
)


class FakeBackend:
    backend_id = "fake"
    location = BackendLocation.REMOTE

    async def generate(self, request):
        return ModelResponse(
            backend_id=self.backend_id,
            model=request.model,
            output_text='{"ok": true}',
            input_tokens=1000,
            output_tokens=200,
            cached_input_tokens=400,
            reasoning_tokens=50,
        )


def test_usage_tracker_calculates_cached_and_uncached_cost():
    tracker = UsageTracker(
        pricing={
            "gpt-6-luna": ModelPricing(
                input_per_million=0.10,
                cached_input_per_million=0.01,
                output_per_million=0.50,
            )
        }
    )
    request = ModelRequest(
        model="gpt-6-luna",
        system="system",
        input_text="input",
        usage_tag="mechanical:v1:investigate",
    )
    response = ModelResponse(
        backend_id="fake",
        model="gpt-6-luna",
        output_text="{}",
        input_tokens=1000,
        output_tokens=200,
        cached_input_tokens=400,
        reasoning_tokens=50,
    )

    record = tracker.record(request, response)

    assert record.tag == "mechanical:v1:investigate"
    assert record.estimated_cost_usd == 0.000164
    total = tracker.total()
    assert total.input_tokens == 1000
    assert total.output_tokens == 200
    assert total.cached_input_tokens == 400
    assert total.reasoning_tokens == 50
    assert total.estimated_cost_usd == 0.000164


def test_unknown_model_keeps_exact_tokens_without_cost():
    tracker = UsageTracker()
    request = ModelRequest(
        model="provider-model",
        system="system",
        input_text="input",
        usage_tag="coordinator:v1:intake",
    )
    response = ModelResponse(
        backend_id="fake",
        model="provider-model",
        output_text="{}",
        input_tokens=12,
        output_tokens=7,
    )

    tracker.record(request, response)
    total = tracker.total()

    assert total.input_tokens == 12
    assert total.output_tokens == 7
    assert total.estimated_cost_usd is None


def test_openai_pricing_snapshot_includes_luna():
    pricing = openai_standard_pricing("gpt-6-luna")

    assert pricing is not None
    assert pricing.input_per_million == 0.10
    assert pricing.cached_input_per_million == 0.01
    assert pricing.output_per_million == 0.50
