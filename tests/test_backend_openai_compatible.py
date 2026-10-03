import json

from mosaic.agents import BackendLocation, ModelRequest
from mosaic.backends import OpenAICompatibleBackend


def test_openai_compatible_backend_builds_strict_schema_payload():
    backend = OpenAICompatibleBackend(
        backend_id="hosted",
        base_url="https://example.invalid/v1/",
        api_key="secret",
    )
    request = ModelRequest(
        model="model-x",
        system="Return structured output.",
        input_text="input",
        response_schema_name="test_schema",
        response_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    payload = backend._payload(request)

    assert backend.endpoint == "https://example.invalid/v1/chat/completions"
    assert payload["model"] == "model-x"
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["response_format"]["json_schema"]["name"] == "test_schema"


def test_openai_compatible_backend_can_fallback_to_json_mode():
    backend = OpenAICompatibleBackend(
        backend_id="local",
        base_url="http://127.0.0.1:8080/v1",
        api_key_env=None,
        location=BackendLocation.LOCAL,
        structured_outputs=False,
    )
    request = ModelRequest(
        model="local-model",
        system="Be concise.",
        input_text="input",
        response_schema_name="test",
        response_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )

    payload = backend._payload(request)

    assert payload["response_format"] == {"type": "json_object"}
    assert "JSON" in payload["messages"][0]["content"]


def test_backend_extracts_chat_completion_text():
    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"answer": "ok"}),
                }
            }
        ]
    }

    assert OpenAICompatibleBackend._extract_text(data) == '{"answer": "ok"}'


def test_backend_extracts_usage_metadata():
    data = {
        "usage": {
            "prompt_tokens": 123,
            "completion_tokens": 45,
            "prompt_tokens_details": {"cached_tokens": 67},
            "completion_tokens_details": {"reasoning_tokens": 8},
        }
    }

    assert OpenAICompatibleBackend._extract_usage(data) == (123, 45, 67, 8)


def test_backend_sends_provider_configurable_completion_budget():
    backend = OpenAICompatibleBackend(
        backend_id="hosted",
        base_url="https://example.invalid/v1",
        api_key="secret",
        completion_token_field="max_completion_tokens",
    )
    request = ModelRequest(
        model="model-x",
        system="Return JSON.",
        input_text="input",
        max_output_tokens=777,
    )

    payload = backend._payload(request)

    assert payload["max_completion_tokens"] == 777


def test_backend_can_use_legacy_max_tokens_field():
    backend = OpenAICompatibleBackend(
        backend_id="local",
        base_url="http://127.0.0.1:8080/v1",
        api_key_env=None,
        completion_token_field="max_tokens",
    )
    request = ModelRequest(
        model="local-model",
        system="Return JSON.",
        input_text="input",
        max_output_tokens=555,
    )

    payload = backend._payload(request)

    assert payload["max_tokens"] == 555
    assert "max_completion_tokens" not in payload
