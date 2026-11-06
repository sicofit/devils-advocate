"""Download and read the BGE corpus published by OpenCaseLaw.

OpenCaseLaw (https://opencaselaw.ch) publishes Swiss court decisions on Hugging
Face as `voilaj/swiss-caselaw` under CC0. This module only needs the leading
decisions of the Federal Supreme Court (BGE), a single ~220 MB Parquet file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

DATASET = "voilaj/swiss-caselaw"
FILE = "data/bge.parquet"
COLUMNS = ["decision_id", "decision_date", "language", "full_text", "source_url"]


@dataclass(frozen=True, slots=True)
class Decision:
    reference: str  # "BGE 145 III 72"
    date: str  # ISO date of the judgment
    language: str  # "de", "fr" or "it"
    text: str
    url: str

    @property
    def year(self) -> int:
        return int(self.date[:4]) if self.date[:4].isdigit() else 0


def dataset_revision(revision: str = "main") -> str:
    """Resolve a branch name to the dataset's commit hash, for reproducible indexes."""
    url = f"https://huggingface.co/api/datasets/{DATASET}/revision/{revision}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))["sha"]


def expected_sha256(revision: str) -> str | None:
    """The Git LFS object id of the BGE file at `revision`, which is its SHA-256."""
    folder = FILE.rsplit("/", 1)[0]
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{revision}/{folder}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        entries = json.loads(resp.read().decode("utf-8"))
    for entry in entries:
        if entry.get("path") == FILE:
            return (entry.get("lfs") or {}).get("oid")
    return None


class DamagedDownload(ValueError):
    """The downloaded file does not have the published SHA-256."""


def download(
    target: Path,
    *,
    revision: str,
    expected: str | None = None,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download the BGE Parquet file at commit `revision` to `target`.

    A partial download is resumed only for the same revision, so an interrupted
    download is never completed with the bytes of a newer file. With
    `expected`, the file is checked before it replaces `target`: a damaged
    download is deleted and the file already at `target` is kept.
    """
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{FILE}"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.{revision[:12]}.part")
    for stale in target.parent.glob(f"{target.name}*.part"):
        if stale != partial:
            stale.unlink()
    done = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    try:
        with urllib.request.urlopen(request, timeout=60) as resp:
            if done and resp.status != 206:  # server ignored the range: start over
                done = 0
            length = resp.headers.get("Content-Length")
            total = done + int(length) if length else None
            with partial.open("ab" if done else "wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if on_progress:
                        on_progress(done, total)
    except urllib.error.HTTPError as exc:
        # 416: the range starts at the end of the file, so the last run had got all of it.
        if exc.code != 416 or not done:
            raise
    if expected and (digest := sha256(partial)) != expected:
        partial.unlink()
        raise DamagedDownload(
            f"The download is damaged: SHA-256 {digest[:12]}, published {expected[:12]}."
        )
    shutil.move(partial, target)
    return target


def sha256(path: Path) -> str:
    """Hash of the downloaded file; it equals the dataset's Git LFS object id."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path, *, since: int = 0) -> Iterator[Decision]:
    """Yield the decisions in `path` from year `since` on, skipping texts under 500 characters."""
    import pyarrow.parquet as pq

    from .segment import citation_of

    table = pq.read_table(path, columns=COLUMNS)
    for row in table.to_pylist():
        decision = Decision(
            reference=citation_of(row["decision_id"]),
            date=row["decision_date"] or "",
            language=row["language"] or "",
            text=row["full_text"] or "",
            url=row["source_url"] or "",
        )
        if decision.year >= since and len(decision.text) >= 500:
            yield decision
