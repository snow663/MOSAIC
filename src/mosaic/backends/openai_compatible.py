"""OpenAI-compatible Chat Completions backend.

The adapter intentionally depends only on the Python standard library so the
MOSAIC core remains lightweight. It can point at a hosted provider or a local
server exposing a compatible endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping as MappingABC
from typing import Any
from urllib import error, request

from mosaic.agents.contracts import (
    BackendLocation,
    ModelRequest,
    ModelResponse,
)


class BackendHTTPError(RuntimeError):
    """HTTP-level failure from an inference backend."""


class BackendProtocolError(RuntimeError):
    """Backend returned a response MOSAIC could not interpret."""


class ModelRefusalError(RuntimeError):
    """Backend reported an explicit model refusal."""


def _plain_json(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    if isinstance(value, list):
        return [_plain_json(item) for item in value]
    return value


class OpenAICompatibleBackend:
    """Use an OpenAI-compatible chat-completions HTTP endpoint."""

    def __init__(
        self,
        *,
        backend_id: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
        location: BackendLocation = BackendLocation.REMOTE,
        timeout_seconds: float = 120.0,
        structured_outputs: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.backend_id = backend_id.strip()
        if not self.backend_id:
            raise ValueError("backend_id must be non-empty")

        self.base_url = base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("base_url must be non-empty")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.location = BackendLocation(location)
        self.timeout_seconds = float(timeout_seconds)
        self.structured_outputs = bool(structured_outputs)
        self.extra_headers = dict(extra_headers or {})
        self._api_key = api_key
        self._api_key_env = api_key_env

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _resolve_api_key(self) -> str | None:
        if self._api_key:
            return self._api_key
        if self._api_key_env:
            value = os.getenv(self._api_key_env)
            if value:
                return value
        return None

    def _payload(self, model_request: ModelRequest) -> dict[str, Any]:
        system = model_request.system
        payload: dict[str, Any] = {
            "model": model_request.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": model_request.input_text},
            ],
        }
        if model_request.max_output_tokens is not None:
            payload["max_completion_tokens"] = model_request.max_output_tokens

        if model_request.response_schema is not None:
            schema = _plain_json(model_request.response_schema)
            if self.structured_outputs:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": (
                            model_request.response_schema_name
                            or "mosaic_response"
                        ),
                        "strict": True,
                        "schema": schema,
                    },
                }
            else:
                if "json" not in system.lower():
                    payload["messages"][0]["content"] = (
                        system
                        + "\nReturn the response as a JSON object only."
                    )
                payload["response_format"] = {"type": "json_object"}

        return payload

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendProtocolError(
                "backend response did not contain choices[0].message"
            ) from exc

        refusal = message.get("refusal")
        if refusal:
            raise ModelRefusalError(str(refusal))

        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)

        raise BackendProtocolError(
            "backend response did not contain assistant text content"
        )

    @staticmethod
    def _extract_usage(
        data: dict[str, Any],
    ) -> tuple[int | None, int | None, int | None, int | None]:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return None, None, None, None

        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")

        prompt_details = usage.get("prompt_tokens_details")
        cached_input_tokens = None
        if isinstance(prompt_details, dict):
            cached_input_tokens = prompt_details.get("cached_tokens")

        completion_details = usage.get("completion_tokens_details")
        reasoning_tokens = None
        if isinstance(completion_details, dict):
            reasoning_tokens = completion_details.get("reasoning_tokens")

        def normalized(value: Any) -> int | None:
            if value is None:
                return None
            try:
                result = int(value)
            except (TypeError, ValueError):
                return None
            return result if result >= 0 else None

        return (
            normalized(input_tokens),
            normalized(output_tokens),
            normalized(cached_input_tokens),
            normalized(reasoning_tokens),
        )

    def _generate_sync(self, model_request: ModelRequest) -> ModelResponse:
        body = json.dumps(
            self._payload(model_request),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **self.extra_headers,
        }
        api_key = self._resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        http_request = request.Request(
            self.endpoint,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(
                http_request,
                timeout=self.timeout_seconds,
            ) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise BackendHTTPError(
                f"backend returned HTTP {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise BackendHTTPError(f"backend connection failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BackendProtocolError(
                "backend returned invalid JSON"
            ) from exc

        if not isinstance(data, dict):
            raise BackendProtocolError("backend response root must be an object")

        output_text = self._extract_text(data)
        response_model = data.get("model")
        if not isinstance(response_model, str) or not response_model.strip():
            response_model = model_request.model

        (
            input_tokens,
            output_tokens,
            cached_input_tokens,
            reasoning_tokens,
        ) = self._extract_usage(data)

        return ModelResponse(
            backend_id=self.backend_id,
            model=response_model,
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    async def generate(self, model_request: ModelRequest) -> ModelResponse:
        return await asyncio.to_thread(self._generate_sync, model_request)
