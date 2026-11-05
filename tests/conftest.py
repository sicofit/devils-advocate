from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from devils_advocate import store
from devils_advocate.search import Retriever

from fakes import DECISIONS


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "index.sqlite"
    store.build(path, DECISIONS, {"since": "1990", "revision": "test"})
    return path


@pytest.fixture
def db(db_path: Path) -> Iterator[store.Store]:
    with store.Store(db_path) as opened:
        yield opened


@pytest.fixture
def retriever(db: store.Store) -> Retriever:
    return Retriever(db)


def passage_id(db: store.Store, reference: str, pinpoint: str) -> int:
    row = db.db.execute(
        "SELECT id FROM passages WHERE reference = ? AND pinpoint = ?", (reference, pinpoint)
    ).fetchone()
    assert row is not None, f"{reference} {pinpoint} not indexed"
    return row[0]
