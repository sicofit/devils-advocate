"""Thin client for OpenAI-compatible endpoints (Ollama, llama.cpp, vLLM, ...).

Standard library only: chat completions with JSON-schema output, and embeddings.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Config


class LLMError(RuntimeError):
    """The model endpoint could not be reached, returned an error, or returned unusable output."""


@dataclass(slots=True)
class ChatResult:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration: float = 0.0


class ChatModel(Protocol):
    """What the pipeline needs from a model. Tests substitute a scripted fake."""

    def json(self, messages: list[dict], schema: dict, *, name: str) -> dict[str, Any]: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMClient:
    def __init__(self, config: Config) -> None:
        self.config = config

    def health(self) -> bool:
        request = urllib.request.Request(self.config.endpoint + "/v1/models", headers=self._auth())
        try:
            with urllib.request.urlopen(request, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
            return False

    def chat(
        self,
        messages: list[dict],
        *,
        schema: dict | None = None,
        name: str = "output",
        max_tokens: int | None = None,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": False,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            }
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        started = time.perf_counter()
        data = self._post("/v1/chat/completions", payload)
        try:
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            return ChatResult(
                text=choice["message"].get("content") or "",
                finish_reason=choice.get("finish_reason"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                duration=time.perf_counter() - started,
            )
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(f"unexpected reply from the chat endpoint: {str(data)[:200]}") from exc

    def json(
        self, messages: list[dict], schema: dict, *, name: str, attempts: int = 2
    ) -> dict[str, Any]:
        """Ask for a JSON object matching `schema`; retry once if the reply does not parse."""
        error = ""
        for _ in range(attempts):
            result = self.chat(messages, schema=schema, name=name)
            try:
                value = json.loads(_strip_fences(result.text))
            except json.JSONDecodeError as exc:
                error = f"{exc}: {result.text[:200]!r}"
                continue
            if isinstance(value, dict):
                return value
            error = f"expected a JSON object, got {type(value).__name__}"
        raise LLMError(f"model did not return valid JSON for {name!r} ({error})")

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post(
            "/v1/embeddings",
            {"model": self.config.embed_model, "input": texts},
            timeout=self.config.embed_timeout,
        )
        try:
            rows = sorted(data["data"], key=lambda row: row["index"])
            vectors = [row["embedding"] for row in rows]
        except (KeyError, TypeError) as exc:
            raise LLMError(
                f"unexpected reply from the embedding endpoint: {str(data)[:200]}"
            ) from exc
        if len(vectors) != len(texts):
            raise LLMError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            self.config.endpoint + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **self._auth()},
            method="POST",
        )
        try:
            limit = timeout or self.config.request_timeout
            with urllib.request.urlopen(request, timeout=limit) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise LLMError(f"model endpoint returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as exc:
            raise LLMError(
                f"cannot reach model at {self.config.endpoint} ({exc}). "
                "Is the local server running?"
            ) from exc
        try:
            data = json.loads(body.decode("utf-8"))
        except ValueError as exc:  # also UnicodeDecodeError
            raise LLMError(f"model endpoint did not return JSON: {body[:200]!r}") from exc
        if not isinstance(data, dict):
            raise LLMError(f"model endpoint returned {type(data).__name__}, not an object")
        return data


def _strip_fences(text: str) -> str:
    """Remove a ```json fence some servers add even in JSON mode."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()
