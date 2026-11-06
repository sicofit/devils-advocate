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
    """The model endpoint returned an error or unusable output."""


class Truncated(LLMError):
    """The reply was cut off: it reached the token limit or the end of the context window.

    Sending the same request again would cut it off at the same place.
    """


class ModelUnavailable(LLMError):
    """No usable model behind the endpoint: not running, too slow, unknown model, refused.

    Unlike other errors this one ends the whole turn: the next call would fail the same way.
    """


class ModelNotFound(ModelUnavailable):
    """The server answers, but has no model of that name (HTTP 404)."""


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
        """Whether the server answers at all."""
        try:
            return self._models() is not None
        except ModelUnavailable:
            return True  # it answers, with a refusal

    def problem(self) -> str | None:
        """Why the chat model cannot be used, or None if nothing is known to be wrong."""
        try:
            models = self._models()
        except ModelUnavailable as exc:
            return str(exc)
        if models is None:
            return self._unreachable()
        model = self.config.model
        if models and not {model, model + ":latest"} & set(models):
            return (
                f"The server at {self.config.endpoint} does not list the model {model!r}. "
                f"Pull it (Ollama: `ollama pull {model}`), or set DA_MODEL."
            )
        return None

    def chat(
        self, messages: list[dict], *, schema: dict | None = None, name: str = "output"
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
            if result.finish_reason == "length":
                # Asking again would give the same cut-off reply.
                raise Truncated(
                    f"the model's reply for {name!r} was cut off before it was complete: "
                    "it reached the model's context limit. On Ollama, start the server with "
                    "OLLAMA_CONTEXT_LENGTH=8192."
                )
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
        if any(not isinstance(vector, list) or not vector for vector in vectors):
            raise LLMError(
                "the embedding endpoint returned an empty vector; "
                f"is {self.config.embed_model!r} an embedding model?"
            )
        return vectors

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}

    def _models(self) -> list[str] | None:
        """The model names the server lists, [] if it lists none, None if it does not answer.

        Raises ModelUnavailable when the server refuses the API key.
        """
        request = urllib.request.Request(self.config.endpoint + "/v1/models", headers=self._auth())
        try:
            with urllib.request.urlopen(request, timeout=5) as resp:
                if resp.status != 200:
                    return []
                body = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise ModelUnavailable(self._refused(exc.code)) from exc
            return []  # it answers, but lists no models
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException):
            return None
        try:
            rows = json.loads(body.decode("utf-8")).get("data")
            return [str(row["id"]) for row in rows if isinstance(row, dict) and "id" in row]
        except (ValueError, AttributeError, TypeError):
            return []  # it answers, but not with a list of models

    def _refused(self, code: int) -> str:
        return (
            f"The server at {self.config.endpoint} refused the request (HTTP {code}). "
            "Check DA_API_KEY."
        )

    def _unreachable(self) -> str:
        return (
            f"No model server answers at {self.config.endpoint}. "
            "Start Ollama (or your server), or set DA_BASE_URL."
        )

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            self.config.endpoint + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **self._auth()},
            method="POST",
        )
        limit = self.config.request_timeout if timeout is None else timeout
        variable = "DA_REQUEST_TIMEOUT" if timeout is None else "DA_EMBED_TIMEOUT"
        try:
            with urllib.request.urlopen(request, timeout=limit) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = " ".join(exc.read().decode("utf-8", "replace").split())[:300]
            if exc.code in (401, 403):
                raise ModelUnavailable(self._refused(exc.code)) from exc
            if exc.code == 404:
                model = payload.get("model")
                raise ModelNotFound(
                    f"The server at {self.config.endpoint} has no model {model!r} "
                    f"(HTTP 404: {detail}). Pull it (Ollama: `ollama pull {model}`), "
                    "or set DA_MODEL / DA_EMBED_MODEL."
                ) from exc
            raise LLMError(f"model endpoint returned HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            if isinstance(reason, TimeoutError):
                raise ModelUnavailable(
                    f"The model at {self.config.endpoint} did not answer within {limit:.0f} s. "
                    f"Try again (a model that is still loading is slow), or raise {variable}."
                ) from exc
            # The operating system's wording ("[WinError 10061] ...") is left out: it
            # says the same thing in the system's language.
            raise ModelUnavailable(self._unreachable()) from exc
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise ModelUnavailable(
                f"The connection to the model server at {self.config.endpoint} failed "
                f"({type(exc).__name__}). Try again."
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
