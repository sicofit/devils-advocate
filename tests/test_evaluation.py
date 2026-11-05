from __future__ import annotations

from pathlib import Path

import pytest

from devils_advocate import evaluation
from devils_advocate.devil import Devil
from devils_advocate.search import Retriever

from fakes import FakeModel, dumps, passages_in

HOLDING = "Die fristlose Kündigung ist nur bei besonders schweren Verfehlungen gerechtfertigt."


def write_cases(path: Path) -> Path:
    path.write_text(
        "// comment lines and blank lines are skipped\n\n"
        + dumps(
            {
                "id": "leichte-verfehlung",
                "position": "Eine fristlose Kündigung ist bei jeder Verfehlung zulässig.",
                "expect": ["BGE 150 III 100"],
            }
        )
        + "\n"
        + dumps({"id": "smalltalk", "position": "Guten Morgen!"})
        + "\n",
        encoding="utf-8",
    )
    return path


def test_load_cases(tmp_path: Path) -> None:
    cases = evaluation.load_cases(write_cases(tmp_path / "cases.jsonl"))
    assert [c.id for c in cases] == ["leichte-verfehlung", "smalltalk"]
    assert cases[0].expect == ("BGE 150 III 100",)
    assert cases[1].should_abstain


def test_load_cases_reports_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "ok", "position": "x"}\n{broken\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        evaluation.load_cases(path)


def respond(messages: list[dict], name: str) -> dict:
    if name == "analysis":
        position = messages[-1]["content"]
        if "Kündigung" not in position:
            return {"language": "de", "claims": []}
        return {"language": "de", "claims": [{"claim": position, "queries": ["Verfehlungen"]}]}
    label = next(k for k, text in passages_in(messages).items() if HOLDING in text)
    return {"counterarguments": [{"passage": label, "quote": HOLDING, "argument": "a"}]}


def test_run_scores_and_summarises(tmp_path: Path, retriever: Retriever) -> None:
    cases = evaluation.load_cases(write_cases(tmp_path / "cases.jsonl"))
    seen: list[str] = []
    report = evaluation.run(
        cases,
        lambda: Devil(FakeModel(respond), retriever),
        on_progress=lambda i, n, case_id: seen.append(f"{i}/{n} {case_id}"),
    )
    assert seen == ["1/2 leichte-verfehlung", "2/2 smalltalk"]

    first, second = report["cases"]
    assert first["found_expected"] and first["retrieved_expected"] and not first["abstained"]
    assert first["cited"] == ["BGE 150 III 100"]
    assert second["abstained"] and second["claims"] == 0

    summary = report["summary"]
    assert summary["contradicting_positions"] == 1 and summary["out_of_scope"] == 1
    assert summary["found_expected"] == summary["answered"] == 1
    assert summary["abstained_out_of_scope"] == 1
    assert summary["quotes_proposed"] == summary["quotes_verified"] == summary["quotes_exact"] == 1
    assert "1/1" in evaluation.table(report)


@pytest.mark.parametrize(
    ("found", "total", "color"),
    [(18, 22, "brightgreen"), (14, 22, "green"), (5, 22, "yellow"), (0, 0, "yellow")],
)
def test_badge(found: int, total: int, color: str) -> None:
    report = {"summary": {"found_expected": found, "contradicting_positions": total}}
    assert evaluation.badge(report) == {
        "schemaVersion": 1,
        "label": "eval: counter-precedent found",
        "message": f"{found}/{total}",
        "color": color,
    }


def test_shipped_cases_are_well_formed() -> None:
    path = Path(__file__).resolve().parents[1] / "evals" / "positions.jsonl"
    cases = evaluation.load_cases(path)
    assert len({c.id for c in cases}) == len(cases)
    assert sum(c.should_abstain for c in cases) >= 3
    for case in cases:
        assert case.position.strip()
        assert all(ref.startswith("BGE ") for ref in case.expect)
