from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from devils_advocate import config
from devils_advocate.client import LLMClient, LLMError, _strip_fences


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


def test_strip_fences() -> None:
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences(' {"a": 1} ') == '{"a": 1}'


class Recorder(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict, dict]] = []
    replies: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Recorder.requests.append((self.path, dict(self.headers), body))
        reply = json.dumps(Recorder.replies.pop(0)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    Recorder.requests, Recorder.replies = [], []
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
    with pytest.raises(LLMError, match="cannot reach model"):
        client.chat([])
