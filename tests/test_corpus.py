from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from devils_advocate import corpus

BODY = bytes(range(256)) * 40
REVISION = "6336cd66331427f65f7f18379a7f7a0d0105963b"


class Response(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200) -> None:
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))}


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Serves BODY with byte ranges, like Hugging Face; records each Range header."""
    ranges: list[str | None] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> Response:
        assert REVISION in request.full_url
        header = request.get_header("Range")
        ranges.append(header)
        if header is None:
            return Response(BODY)
        start = int(header.removeprefix("bytes=").rstrip("-"))
        if start >= len(BODY):
            raise urllib.error.HTTPError(request.full_url, 416, "Range Not Satisfiable", {}, None)
        return Response(BODY[start:], status=206)

    monkeypatch.setattr(corpus.urllib.request, "urlopen", urlopen)
    return ranges


def test_download_resumes_the_same_revision(tmp_path: Path, server: list[str | None]) -> None:
    target = tmp_path / "bge.parquet"
    target.with_name(f"bge.parquet.{REVISION[:12]}.part").write_bytes(BODY[:1000])
    corpus.download(target, revision=REVISION)
    assert server == ["bytes=1000-"]
    assert target.read_bytes() == BODY
    assert not list(tmp_path.glob("*.part"))


def test_download_drops_a_part_of_another_revision(
    tmp_path: Path, server: list[str | None]
) -> None:
    target = tmp_path / "bge.parquet"
    (tmp_path / "bge.parquet.0123456789ab.part").write_bytes(b"older file")
    (tmp_path / "bge.parquet.part").write_bytes(b"from before revisions were named")
    corpus.download(target, revision=REVISION)
    assert server == [None]
    assert target.read_bytes() == BODY
    assert not list(tmp_path.glob("*.part"))


def test_download_finishes_a_complete_part(tmp_path: Path, server: list[str | None]) -> None:
    target = tmp_path / "bge.parquet"
    target.with_name(f"bge.parquet.{REVISION[:12]}.part").write_bytes(BODY)
    corpus.download(target, revision=REVISION)  # the server answers 416
    assert target.read_bytes() == BODY


def test_expected_sha256_reads_the_lfs_oid(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = [
        {"type": "file", "path": "data/bger.parquet", "lfs": {"oid": "other"}},
        {"type": "file", "path": corpus.FILE, "lfs": {"oid": "fd363e6c", "size": 219481107}},
    ]

    def urlopen(url: str, timeout: float) -> Response:
        assert url.endswith(f"/tree/{REVISION}/data")
        return Response(json.dumps(listing).encode())

    monkeypatch.setattr(corpus.urllib.request, "urlopen", urlopen)
    assert corpus.expected_sha256(REVISION) == "fd363e6c"
