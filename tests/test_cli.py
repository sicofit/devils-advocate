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
    """No test talks to a real model server, even if one runs on this machine."""
    monkeypatch.setattr(cli.LLMClient, "_models", lambda self: None)


def respond(messages: list[dict], name: str) -> dict:
    if name == "analysis":
        return {
            "language": "de",
            "claims": [{"claim": messages[-1]["content"], "queries": ["Verfehlungen"]}],
        }
    if name == "translation":
        return {"de": "", "fr": ""}  # a model that cannot translate
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
    assert "missing" in out
    assert "model server     No model server answers at http://127.0.0.1:11434" in out


def test_status_on_a_damaged_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "index.sqlite").write_bytes(b"not a database" * 100)
    assert cli.main(["--data-dir", str(tmp_path), "status"]) == 0
    assert "unreadable" in capsys.readouterr().out


def test_commands_need_their_inputs(tmp_path: Path) -> None:
    empty = ["--data-dir", str(tmp_path)]
    assert cli.main([*empty, "argue", "Eine Behauptung."]) == 1
    assert cli.main([*empty, "index"]) == 1
    assert cli.main([*empty, "embed"]) == 1
    assert cli.main([*empty, "bench"]) == 1
    assert cli.main([*empty, "eval"]) == 1


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["argue", "--json"], "--json needs a position"),
        (["embed", "--batch", "0"], "--batch must be at least 1"),
        (["bench", "--modes", "bm42"], "Unknown mode 'bm42'"),
        (["bench", "--modes", "bm25+translate"], "needs --translate"),
        (["bench", "--n", "0"], "--n of at least 1"),
    ],
)
def test_bad_options_are_explained_not_raised(
    data_dir: Path, argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--data-dir", str(data_dir), *argv]) == 1
    assert message in capsys.readouterr().err


def test_index_rejects_an_impossible_year(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "bge.parquet").write_bytes(b"")
    assert cli.main(["--data-dir", str(tmp_path), "index", "--since", "20"]) == 1
    assert "--since must be a year between 1875 and 2100" in capsys.readouterr().err


def test_fetch_offline_is_explained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import urllib.error

    from devils_advocate import corpus

    def offline(revision: str) -> str:
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(corpus, "dataset_revision", offline)
    assert cli.main(["--data-dir", str(tmp_path), "fetch"]) == 1
    assert "Could not reach Hugging Face" in capsys.readouterr().err


def write_parquet(path: Path) -> None:
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
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_index_reads_the_parquet_corpus(tmp_path: Path) -> None:
    from devils_advocate import corpus

    write_parquet(tmp_path / "bge.parquet")
    digest = corpus.sha256(tmp_path / "bge.parquet")
    source = {"revision": "6336cd66331427f6", "sha256": digest}
    (tmp_path / "corpus.json").write_text(json.dumps(source), encoding="utf-8")
    assert cli.main(["--data-dir", str(tmp_path), "index", "--since", "2000"]) == 0
    with store.Store(tmp_path / "index.sqlite") as db:
        assert db.has_decision("BGE 150 III 100")
        assert not db.has_decision("BGE 120 II 50")  # decided in 1994
        assert db.meta()["since"] == "2000"
        assert db.meta()["revision"] == "6336cd66331427f6"


def test_index_records_the_file_it_indexed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from devils_advocate import corpus

    write_parquet(tmp_path / "bge.parquet")
    stale = {"revision": "aaaaaaaaaaaa", "sha256": "b7ece62f228f"}  # of an earlier download
    (tmp_path / "corpus.json").write_text(json.dumps(stale), encoding="utf-8")
    assert cli.main(["--data-dir", str(tmp_path), "index"]) == 0
    assert "does not describe this corpus file" in capsys.readouterr().err
    with store.Store(tmp_path / "index.sqlite") as db:
        meta = db.meta()
    assert meta["revision"] == "unknown"
    assert meta["sha256"] == corpus.sha256(tmp_path / "bge.parquet")


def test_an_index_without_decisions_does_not_replace_the_old_one(
    data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_parquet(data_dir / "bge.parquet")
    assert cli.main(["--data-dir", str(data_dir), "index", "--since", "2100"]) == 1
    assert "the old index is kept" in capsys.readouterr().err
    with store.Store(data_dir / "index.sqlite") as db:
        assert db.decision_count() == 3


def test_an_index_of_an_earlier_version_asks_for_a_rebuild(
    data_dir: Path, fake_model: FakeModel, capsys: pytest.CaptureFixture[str]
) -> None:
    with store.Store(data_dir / "index.sqlite") as db:
        db.db.execute("DELETE FROM meta WHERE key = 'schema'")
        db.db.commit()
    assert cli.main(["--data-dir", str(data_dir), "argue", "Kündigung immer zulässig."]) == 1
    # Rich wraps the message at the terminal width.
    assert "Run `devil index` again" in " ".join(capsys.readouterr().err.split())
    assert fake_model.names() == []


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
        "starts_mid_sentence": False,
        "ends_mid_sentence": False,
        "reports_lower_court_or_party": False,
    }
    assert claim["skipped"] == []
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
    out, err = capsys.readouterr()
    assert "State your position" in err and "No model server" not in err
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
    assert "Nothing to export yet." in captured.err
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
    out, err = capsys.readouterr()
    assert "Cancelled." in err
    assert "BGE 150 III 100 E. 2.2 S. 101" in out


def test_discarded_quotes_are_shown_on_request(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def invented(messages: list[dict], name: str) -> dict:
        if name == "analysis":
            return respond(messages, name)
        label = next(iter(passages_in(messages)))
        return {
            "counterarguments": [
                {
                    "passage": label,
                    "quote": "Das Gericht hat dies so nie entschieden.",
                    "argument": "a",
                }
            ]
        }

    model = FakeModel(invented)
    monkeypatch.setattr(cli, "LLMClient", lambda cfg: model)
    type_lines(monkeypatch, "Kündigung immer zulässig.", "/discarded")
    assert cli.main(["--data-dir", str(data_dir)]) == 0
    out = capsys.readouterr().out
    # The answer says why nothing is shown; /discarded then shows the last answer's discards.
    assert "bestanden die Prüfung am zitierten Text nicht" in out
    assert "(/discarded shows them)" in out
    assert "✗ BGE " in out and "not found in the cited passage" in out and "„Das Gericht" in out
    assert "P1" not in out  # labels are internal

    type_lines(monkeypatch, "Kündigung immer zulässig.")
    assert cli.main(["--data-dir", str(data_dir), "argue", "--show-discarded"]) == 0
    assert "not found in the cited passage" in capsys.readouterr().out


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
        (["--data-dir=d", "Vertrag", "-", "Kauf"], (["--data-dir=d"], "Vertrag - Kauf")),
    ],
)
def test_split_opening(argv: list[str], expected: tuple[list[str], str]) -> None:
    assert cli._split_opening(argv) == expected


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["Vertrag", "--json"], "--json after the position .* Use devil argue --json <position>"),
        (["Vertrag", "--js"], "--json after the position"),
        (["Vertrag", "--retriever=bm25"], "Use devil argue --retriever <name> <position>"),
        (["Vertrag", "--model", "m"], "--model after the position .* go before the position"),
        (["Vertrag", "--model=m"], "Options go before the position"),
        (["Vertrag", "-h"], "Options go before the position"),
    ],
)
def test_an_option_after_the_position_is_refused(argv: list[str], message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        cli._split_opening(argv)


def test_an_abbreviated_option_before_the_position_takes_its_value() -> None:
    assert cli._split_opening(["--mod", "m", "Vertrag"]) == (["--mod", "m"], "Vertrag")


def test_ctrl_c_at_the_prompt_drops_the_unsent_lines(
    data_dir: Path, fake_model: FakeModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = ["Kündigung \\"]

    def fake_input(prompt: str = "") -> str:
        if not queue:
            raise KeyboardInterrupt
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli.main(["--data-dir", str(data_dir)]) == 0
    assert fake_model.names() == []


def test_the_default_export_name_does_not_overwrite(
    data_dir: Path,
    fake_model: FakeModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(data_dir)
    type_lines(monkeypatch, "Kündigung immer zulässig.", "/export", "/export")
    assert cli.main(["--data-dir", str(data_dir)]) == 0
    assert len(list(data_dir.glob("devils-advocate-*.md"))) == 2
    assert list(data_dir.glob("devils-advocate-*-2.md"))


def test_a_backslash_continues_the_line(
    data_dir: Path, fake_model: FakeModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    type_lines(monkeypatch, "Kündigung \\", "immer zulässig.", "/quit")
    assert cli.main(["--data-dir", str(data_dir)]) == 0
    assert fake_model.names() == ["analysis", "counterarguments"]
    assert fake_model.calls[0][1][-1]["content"] == "Kündigung immer zulässig."


def test_a_pasted_block_is_one_turn(
    data_dir: Path, fake_model: FakeModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    type_lines(monkeypatch, "Kündigung", "immer zulässig.", "/quit")
    waiting = iter([True, False, False])  # the second line was pasted with the first
    monkeypatch.setattr(cli, "_more_waiting", lambda: next(waiting))
    assert cli.main(["--data-dir", str(data_dir)]) == 0
    assert fake_model.names() == ["analysis", "counterarguments"]
    assert fake_model.calls[0][1][-1]["content"] == "Kündigung immer zulässig."


def test_conversation_warns_when_the_model_is_down(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DA_BASE_URL", raising=False)
    type_lines(monkeypatch)  # Ctrl+D right away
    assert cli.main(["--data-dir", str(data_dir)]) == 0
    assert "No model server answers at http://127.0.0.1:11434" in capsys.readouterr().err


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
    assert "Using BM25 for now." in captured.err
    assert "BGE 150 III 100" in captured.out  # FakeModel cannot embed, so this was BM25


def test_damaged_vectors_fall_back_to_bm25(
    data_dir: Path, fake_model: FakeModel, capsys: pytest.CaptureFixture[str]
) -> None:
    path = vectors_path(data_dir, "nomic-embed-text")
    with store.Store(data_dir / "index.sqlite") as db:
        build_vectors(db, HashEmbedder(), "nomic-embed-text", path)
    path.write_bytes(path.read_bytes()[:200])  # cut off
    base = ["--data-dir", str(data_dir), "argue"]
    with pytest.raises(SystemExit, match="cannot be read"):
        cli.main([*base, "--retriever", "hybrid", "Kündigung immer zulässig."])
    assert cli.main([*base, "Kündigung immer zulässig."]) == 0
    assert "cannot be read" in capsys.readouterr().err


def test_the_retriever_is_named_as_it_searches(db: store.Store, tmp_path: Path) -> None:
    from devils_advocate.search import Retriever, VectorIndex

    vectors = VectorIndex.load(build_vectors(db, HashEmbedder(), "hash", tmp_path / "v.npy"))
    assert cli.retrieval_name(Retriever(db)) == "bm25"
    assert cli.retrieval_name(Retriever(db, HashEmbedder(), vectors)) == "hybrid"
    assert cli.retrieval_name(cli._DenseOnly(db, HashEmbedder(), vectors)) == "dense"


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
    settings = report["settings"]
    assert settings["top_k"] == 8 and settings["temperature"] == 0.2
    assert len(settings["prompts_sha256"]) == 12
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
    assert "translations_missing" not in report


def test_bench_reports_missing_translations(
    data_dir: Path, fake_model: FakeModel, capsys: pytest.CaptureFixture[str]
) -> None:
    out = data_dir / "results"
    args = ["--data-dir", str(data_dir), "bench", "--modes", "bm25", "--translate"]
    assert cli.main([*args, "--out", str(out)]) == 0
    report = json.loads((out / "retrieval.json").read_text(encoding="utf-8"))
    assert report["translations_missing"] == 1  # the fake model does not translate
    assert report["translations_partial"] == 0
    assert "have no translation" in capsys.readouterr().err


@pytest.mark.parametrize("published", [None, "match", "fd363e6c59d4"])
def test_fetch_checks_the_published_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    published: str | None,
) -> None:
    import hashlib

    from devils_advocate import corpus

    body = b"parquet bytes"
    digest = hashlib.sha256(body).hexdigest()

    def download(target: Path, *, expected: str | None, **_: object) -> None:
        if expected and expected != digest:
            raise corpus.DamagedDownload("The download is damaged.")
        target.write_bytes(body)

    monkeypatch.setattr(corpus, "dataset_revision", lambda revision: "6336cd66331427f6")
    monkeypatch.setattr(corpus, "download", download)
    monkeypatch.setattr(
        corpus, "expected_sha256", lambda revision: digest if published == "match" else published
    )
    (tmp_path / "data").mkdir()
    args = ["--data-dir", str(tmp_path / "data"), "fetch"]
    if published == "fd363e6c59d4":
        assert cli.main(args) == 1
        assert "damaged" in capsys.readouterr().err
        return
    assert cli.main(args) == 0
    source = json.loads((tmp_path / "data" / "corpus.json").read_text(encoding="utf-8"))
    assert source["sha256"] == digest


def test_corpus_hash_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    from devils_advocate import corpus

    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 3_000_000)
    assert corpus.sha256(path) == hashlib.sha256(b"x" * 3_000_000).hexdigest()
