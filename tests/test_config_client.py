from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from devils_advocate import config
from devils_advocate.client import LLMClient, LLMError, ModelUnavailable, Truncated, _strip_fences


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(__import__("os").environ):
        if name.startswith(config.ENV_PREFIX):
            monkeypatch.delenv(name)


def test_defaults_env_and_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert config.load().model == "qwen3.5:9b"
    monkeypatch.setenv("DA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("DA_TOP_K", "5")
    monkeypatch.setenv("DA_TEMPERATURE", "0.5")
    monkeypatch.setenv("DA_DATA_DIR", str(tmp_path))
    cfg = config.load(model=None, base_url="http://host:8080/v1")
    assert cfg.model == "llama3.1:8b"  # an unset flag falls through to the environment
    assert (cfg.top_k, cfg.temperature) == (5, 0.5)
    assert cfg.db_path == tmp_path / "index.sqlite"
    assert cfg.endpoint == "http://host:8080"
    assert config.load(model="flag-wins").model == "flag-wins"


def test_empty_env_switches_an_optional_setting_off(monkeypatch: pytest.MonkeyPatch) -> None:
    assert config.load().reasoning_effort == "none"
    monkeypatch.setenv("DA_REASONING_EFFORT", "")
    assert config.load().reasoning_effort is None
    # PowerShell deletes a variable set to "", so there is a word for it too.
    monkeypatch.setenv("DA_REASONING_EFFORT", "omit")
    assert config.load().reasoning_effort is None


def test_a_bad_number_names_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DA_TOP_K", "acht")
    with pytest.raises(SystemExit, match="DA_TOP_K must be a whole number, not 'acht'"):
        config.load()


@pytest.mark.parametrize(
    ("base_url", "endpoint"),
    [
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://host:8080/v1/", "http://host:8080"),
        ("localhost:11434", "http://localhost:11434"),
        (" https://api.example.org/v1 ", "https://api.example.org"),
    ],
)
def test_endpoint(base_url: str, endpoint: str) -> None:
    assert config.load(base_url=base_url).endpoint == endpoint


def test_strip_fences() -> None:
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences(' {"a": 1} ') == '{"a": 1}'


class Recorder(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict, dict]] = []
    replies: list[dict] = []
    status = 200  # of every POST reply
    get_status = 200  # of GET /v1/models
    models: list[str] | None = None  # what GET /v1/models lists; None: an empty body

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Recorder.requests.append((self.path, dict(self.headers), body))
        reply = json.dumps(Recorder.replies.pop(0) if Recorder.replies else {}).encode()
        self.send_response(Recorder.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        Recorder.requests.append((self.path, dict(self.headers), {}))
        body = b""
        if Recorder.models is not None:
            body = json.dumps({"data": [{"id": m} for m in Recorder.models]}).encode()
        self.send_response(Recorder.get_status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    Recorder.requests, Recorder.replies = [], []
    Recorder.status, Recorder.get_status, Recorder.models = 200, 200, None
    httpd = HTTPServer(("127.0.0.1", 0), Recorder)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()
    httpd.server_close()


def chat_reply(content: str) -> dict:
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


def test_json_call_sends_schema_and_retries_bad_output(server: str) -> None:
    Recorder.replies = [chat_reply("not json"), chat_reply('```json\n{"ok": true}\n```')]
    client = LLMClient(config.load(base_url=server, api_key="secret"))
    assert client.json([{"role": "user", "content": "hi"}], {"type": "object"}, name="t") == {
        "ok": True
    }
    path, headers, body = Recorder.requests[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert body["response_format"]["json_schema"]["name"] == "t"
    assert body["reasoning_effort"] == "none"
    assert len(Recorder.requests) == 2


def test_json_call_gives_up(server: str) -> None:
    Recorder.replies = [chat_reply("[1, 2]"), chat_reply("still no object")]
    client = LLMClient(config.load(base_url=server))
    with pytest.raises(LLMError, match="did not return valid JSON"):
        client.json([], {}, name="t")


def test_embeddings_come_back_in_input_order(server: str) -> None:
    Recorder.replies = [
        {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}
    ]
    client = LLMClient(config.load(base_url=server, embed_model="e"))
    assert client.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert Recorder.requests[0][2] == {"model": "e", "input": ["a", "b"]}


def test_unreachable_endpoint_is_an_llm_error() -> None:
    with HTTPServer(("127.0.0.1", 0), Recorder) as closed:
        port = closed.server_address[1]
    client = LLMClient(config.load(base_url=f"http://127.0.0.1:{port}"))
    assert not client.health()
    message = f"No model server answers at http://127.0.0.1:{port}. Start Ollama"
    assert (client.problem() or "").startswith(message)
    # The operating system's own wording ("[WinError 10061] ...") is not passed on.
    with pytest.raises(ModelUnavailable, match=f"^{message}") as error:
        client.chat([])
    assert "Errno" not in str(error.value) and "WinError" not in str(error.value)


def test_a_model_the_server_does_not_list_is_named(server: str) -> None:
    Recorder.models = ["llama3.1:8b", "qwen3.5:latest"]
    assert "does not list the model 'qwen3.5:9b'" in (
        LLMClient(config.load(base_url=server)).problem() or ""
    )
    assert LLMClient(config.load(base_url=server, model="qwen3.5")).problem() is None
    Recorder.models = []  # a server that lists nothing may still serve the model
    assert LLMClient(config.load(base_url=server)).problem() is None


@pytest.mark.parametrize("code", [401, 403])
def test_a_refused_api_key_is_named_before_the_first_call(server: str, code: int) -> None:
    Recorder.get_status = code
    client = LLMClient(config.load(base_url=server))
    assert client.health()
    assert client.problem() == (
        f"The server at {server.removesuffix('/v1')} refused the request (HTTP {code}). "
        "Check DA_API_KEY."
    )


def test_a_server_without_a_model_list_still_answers(server: str) -> None:
    Recorder.get_status = 404
    client = LLMClient(config.load(base_url=server))
    assert client.health() and client.problem() is None


def test_an_unknown_model_ends_the_turn(server: str) -> None:
    Recorder.status = 404
    with pytest.raises(ModelUnavailable, match="has no model 'qwen3.5:9b'"):
        LLMClient(config.load(base_url=server)).chat([])


def test_a_server_error_is_an_ordinary_llm_error(server: str) -> None:
    Recorder.status = 500
    with pytest.raises(LLMError, match="HTTP 500") as error:
        LLMClient(config.load(base_url=server)).chat([])
    assert not isinstance(error.value, ModelUnavailable)


def test_a_cut_off_reply_is_not_asked_again(server: str) -> None:
    reply = {"choices": [{"message": {"content": '{"a": '}, "finish_reason": "length"}]}
    Recorder.replies = [reply, reply]
    with pytest.raises(Truncated, match="OLLAMA_CONTEXT_LENGTH"):
        LLMClient(config.load(base_url=server)).json([], {}, name="t")
    assert len(Recorder.requests) == 1


def test_an_empty_embedding_is_an_llm_error(server: str) -> None:
    Recorder.replies = [{"data": [{"index": 0, "embedding": []}]}]
    with pytest.raises(LLMError, match="empty vector"):
        LLMClient(config.load(base_url=server)).embed(["a"])


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("DA_MAX_CLAIMS", "-1", "DA_MAX_CLAIMS must be between 1 and 10, not -1"),
        ("DA_TOP_K", "0", "DA_TOP_K must be between 1 and 50, not 0"),
        ("DA_TEMPERATURE", "5", "DA_TEMPERATURE must be between 0 and 2, not 5.0"),
        ("DA_REQUEST_TIMEOUT", "0", "DA_REQUEST_TIMEOUT must be between 1 and 86400"),
        ("DA_SINCE", "20", "DA_SINCE must be a year between 1875 and 2100, not 20"),
    ],
)
def test_a_number_out_of_range_names_the_setting(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, message: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(SystemExit, match=message):
        config.load()


def test_health_check_sends_the_api_key(server: str) -> None:
    assert LLMClient(config.load(base_url=server, api_key="secret")).health()
    path, headers, _ = Recorder.requests[0]
    assert path == "/v1/models"
    assert headers["Authorization"] == "Bearer secret"


@pytest.mark.parametrize("reply", [{}, {"choices": []}, {"choices": [{"message": None}]}])
def test_a_malformed_reply_is_an_llm_error(server: str, reply: dict) -> None:
    Recorder.replies = [reply]
    with pytest.raises(LLMError, match="unexpected reply"):
        LLMClient(config.load(base_url=server)).chat([])


def test_missing_embeddings_are_an_llm_error(server: str) -> None:
    Recorder.replies = [{"data": [{"index": 0, "embedding": [1.0]}]}]
    with pytest.raises(LLMError, match="asked for 2 embeddings, got 1"):
        LLMClient(config.load(base_url=server)).embed(["a", "b"])
