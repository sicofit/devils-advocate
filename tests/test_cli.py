from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from devils_advocate import cli, store
from devils_advocate.search import build_vectors, vectors_path

from fakes import DECISIONS, FakeModel, HashEmbedder, dumps, passages_in

HOLDING = "Die fristlose Kündigung ist nur bei besonders schweren Verfehlungen gerechtfertigt."


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.LLMClient, "health", lambda self: False)


def respond(messages: list[dict], name: str) -> dict:
    if name == "analysis":
        return {
            "language": "de",
            "claims": [{"claim": messages[-1]["content"], "queries": ["Verfehlungen"]}],
        }
    label = next((k for k, text in passages_in(messages).items() if HOLDING in text), None)
    if label is None:
        return {"counterarguments": []}
    return {"counterarguments": [{"passage": label, "quote": HOLDING, "argument": "a"}]}


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    model = FakeModel(respond)
    monkeypatch.setattr(cli, "LLMClient", lambda cfg: model)
    return model


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    store.build(tmp_path / "index.sqlite", DECISIONS, {"since": "1990", "revision": "test"})
    return tmp_path


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert "devils-advocate 0.1.0" in capsys.readouterr().out


def test_status_on_an_empty_data_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--data-dir", str(tmp_path), "status"]) == 0
    out = capsys.readouterr().out
    assert "missing" in out and "model reachable" in out


def test_commands_need_their_inputs(tmp_path: Path) -> None:
    empty = ["--data-dir", str(tmp_path)]
    assert cli.main([*empty, "argue", "Eine Behauptung."]) == 1
    assert cli.main([*empty, "index"]) == 1
    assert cli.main([*empty, "embed"]) == 1
    assert cli.main([*empty, "bench"]) == 1
    assert cli.main([*empty, "eval"]) == 1


def test_index_reads_the_parquet_corpus(tmp_path: Path) -> None:
    rows = [
        {
            "decision_id": "bge_" + d.reference.removeprefix("BGE "),
            "decision_date": d.date,
            "language": d.language,
            "full_text": d.text,
            "source_url": d.url,
        }
        for d in DECISIONS
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "bge.parquet")
    assert cli.main(["--data-dir", str(tmp_path), "index", "--since", "2000"]) == 0
    with store.Store(tmp_path / "index.sqlite") as db:
        assert db.has_decision("BGE 150 III 100")
        assert not db.has_decision("BGE 120 II 50")  # decided in 1994
        assert db.meta()["since"] == "2000"


def test_argue_once_as_json(
    data_dir: Path, fake_model: FakeModel, capsys: pytest.CaptureFixture[str]
) -> None:
    position = "Eine fristlose Kündigung ist immer zulässig."
    code = cli.main(["--data-dir", str(data_dir), "argue", "--json", position])
    assert code == 0
    result = json.loads(capsys.readouterr().out)  # stdout holds the JSON and nothing else
    [claim] = result["claims"]
    [counter] = claim["counterarguments"]
    assert counter == {
        "argument": "a",
        "quote": HOLDING,
        "citation": "BGE 150 III 100 E. 2.2 S. 101",
        "match": "exact",
    }
    assert result["position"] == position


def test_argue_renders_for_humans(
    data_dir: Path, fake_model: FakeModel, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--data-dir", str(data_dir), "argue", "Kündigung immer zulässig."]) == 0
    out = capsys.readouterr().out
    assert "BGE 150 III 100 E. 2.2 S. 101" in out
    assert "besonders schweren Verfehlungen" in out


def type_lines(monkeypatch: pytest.MonkeyPatch, *lines: str) -> None:
    """Feed the conversation these lines, then end it like Ctrl+D would."""
    queue = list(lines)

    def fake_input(prompt: str = "") -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)


def test_devil_without_arguments_opens_the_conversation(
    data_dir: Path,
    fake_model: FakeModel,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    export = data_dir / "talk.md"
    type_lines(monkeypatch, "/help", "Kündigung immer zulässig.", f"/export {export}", "/quit")
    assert cli.main(["--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "State your position" in out and "No model reachable" not in out
    assert "/sources" in out  # the help text
    assert "BGE 150 III 100 E. 2.2 S. 101" in out
    assert "besonders schweren Verfehlungen" in export.read_text(encoding="utf-8")
    assert fake_model.names() == ["analysis", "counterarguments"]


def test_devil_with_a_position_answers_it_and_keeps_talking(
    data_dir: Path,
    fake_model: FakeModel,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    type_lines(monkeypatch, "Und bei Verfehlungen?")
    assert cli.main(["--data-dir", str(data_dir), "Kündigung", "immer", "zulässig."]) == 0
    assert "BGE 150 III 100 E. 2.2 S. 101" in capsys.readouterr().out
    assert fake_model.names() == ["analysis", "counterarguments"] * 2
    first, second = (messages for name, messages in fake_model.calls if name == "analysis")
    assert "Kündigung immer zulässig." in str(first)
    assert "Und bei Verfehlungen?" in str(second)
    # The second turn knows what was cited and is not shown that passage again.
    assert "BGE 150 III 100 E. 2.2 S. 101" in str(second)
    _, again = (messages for name, messages in fake_model.calls if name == "counterarguments")
    assert HOLDING not in str(again)


def test_export_takes_quoted_paths_and_reports_failures(
    data_dir: Path,
    fake_model: FakeModel,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    good = data_dir / "mit Leerzeichen.md"
    bad = data_dir / "missing" / "dir" / "x.md"
    type_lines(
        monkeypatch,
        "/export",  # nothing to export yet
        "\ufeffKündigung immer zulässig.",  # piped in with a byte order mark
        f'/export "{good}"',
        f"/export {bad}",
    )
    assert cli.main(["--data-dir", str(data_dir)]) == 0
    captured = capsys.readouterr()
    assert "Nothing to export yet." in captured.out
    assert good.exists()
    assert "Could not save" in captured.err
    analysis = next(messages for name, messages in fake_model.calls if name == "analysis")
    assert "\ufeff" not in str(analysis)


def test_ctrl_c_cancels_the_turn_not_the_conversation(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(messages: list[dict], name: str) -> dict:
        if "abbrechen" in str(messages):
            raise KeyboardInterrupt
        return respond(messages, name)

    model = FakeModel(interrupted)
    monkeypatch.setattr(cli, "LLMClient", lambda cfg: model)
    type_lines(monkeypatch, "Bitte abbrechen.", "Kündigung immer zulässig.")
    assert cli.main(["--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    assert "Cancelled." in out
    assert "BGE 150 III 100 E. 2.2 S. 101" in out


def test_one_shot_answer_does_not_mention_conversation_commands(
    data_dir: Path, fake_model: FakeModel, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--data-dir", str(data_dir), "argue", "Kündigung immer zulässig."]) == 0
    assert "/sources" not in capsys.readouterr().out


def test_eval_without_cases_explains_itself(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = data_dir / "nope.jsonl"
    assert cli.main(["--data-dir", str(data_dir), "eval", "--cases", str(missing)]) == 1
    assert "pass --cases" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ([], "")),
        (["--version"], (["--version"], "")),
        (["status"], (["status"], "")),
        (["--model", "m", "argue", "x"], (["--model", "m", "argue", "x"], "")),
        (["Ein", "Vertrag"], ([], "Ein Vertrag")),
        (["--model", "m", "Ein", "Vertrag"], (["--model", "m"], "Ein Vertrag")),
        (["--data-dir=d", "Vertrag", "--json"], (["--data-dir=d"], "Vertrag --json")),
    ],
)
def test_split_opening(argv: list[str], expected: tuple[list[str], str]) -> None:
    assert cli._split_opening(argv) == expected


def test_conversation_warns_when_the_model_is_down(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DA_BASE_URL", raising=False)
    type_lines(monkeypatch)  # Ctrl+D right away
    assert cli.main(["--data-dir", str(data_dir)]) == 0
    assert "No model reachable at http://127.0.0.1:11434" in capsys.readouterr().err


def test_dense_retriever_needs_vectors(data_dir: Path, fake_model: FakeModel) -> None:
    with pytest.raises(SystemExit, match="No vectors"):
        cli.main(["--data-dir", str(data_dir), "argue", "--retriever", "dense", "x"])


def test_vectors_from_an_earlier_index_are_not_used(
    data_dir: Path, fake_model: FakeModel, capsys: pytest.CaptureFixture[str]
) -> None:
    with store.Store(data_dir / "index.sqlite") as db:
        build_vectors(
            db, HashEmbedder(), "nomic-embed-text", vectors_path(data_dir, "nomic-embed-text")
        )
    store.build(data_dir / "index.sqlite", DECISIONS, {"revision": "test"})
    base = ["--data-dir", str(data_dir), "argue"]
    with pytest.raises(SystemExit, match="earlier index"):
        cli.main([*base, "--retriever", "hybrid", "Kündigung immer zulässig."])
    assert cli.main([*base, "Kündigung immer zulässig."]) == 0  # auto falls back to BM25
    captured = capsys.readouterr()
    assert "Using BM25 for now." in captured.out
    assert "BGE 150 III 100" in captured.out  # FakeModel cannot embed, so this was BM25


def test_eval_writes_report_and_badge(data_dir: Path, fake_model: FakeModel) -> None:
    cases = data_dir / "cases.jsonl"
    cases.write_text(
        dumps({"id": "c1", "position": "Kündigung immer zulässig.", "expect": ["BGE 150 III 100"]})
        + "\n",
        encoding="utf-8",
    )
    out = data_dir / "results"
    args = ["--data-dir", str(data_dir), "eval", "--cases", str(cases), "--out", str(out)]
    assert cli.main(args) == 0
    report = json.loads((out / "end_to_end.json").read_text(encoding="utf-8"))
    assert report["summary"]["found_expected"] == 1
    assert report["retriever"] == "bm25"
    badge = json.loads((out / "badge.json").read_text(encoding="utf-8"))
    assert badge["message"] == "1/1" and badge["schemaVersion"] == 1


def test_bench_writes_retrieval_report(data_dir: Path, fake_model: FakeModel) -> None:
    out = data_dir / "results"
    args = ["--data-dir", str(data_dir), "bench", "--modes", "bm25,hybrid", "--out", str(out)]
    assert cli.main(args) == 0
    report = json.loads((out / "retrieval.json").read_text(encoding="utf-8"))
    assert list(report["modes"]) == ["bm25"]  # no vectors, so hybrid is skipped
    assert report["queries"] == 1
    assert report["index"]["revision"] == "test"


def test_corpus_hash_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    from devils_advocate import corpus

    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 3_000_000)
    assert corpus.sha256(path) == hashlib.sha256(b"x" * 3_000_000).hexdigest()
