from __future__ import annotations

from devils_advocate.segment import (
    Passage,
    _chunk,
    citation_of,
    ends_with_abbreviation,
    first_page,
    segment,
)

from fakes import ALTE_PRAXIS, CONSIDERANTS, KUENDIGUNG


def by_pinpoint(reference: str, text: str) -> dict[str, Passage]:
    return {p.pinpoint: p for p in segment(reference, text)}


def test_erwaegung_numbers_become_pinpoints() -> None:
    passages = segment("BGE 150 III 100", KUENDIGUNG)
    assert [p.pinpoint for p in passages] == ["Regeste", "E. 2", "E. 2.1", "E. 2.2", "E. 3"]


def test_page_markers_are_removed_and_recorded() -> None:
    passages = by_pinpoint("BGE 150 III 100", KUENDIGUNG)
    assert passages["E. 2.1"].page == 100
    assert passages["E. 2.2"].page == 101
    assert all("S. 101" not in p.text.split("(")[0] for p in passages.values())
    assert "BGE 150 III 100 S. 101" not in passages["E. 2.1"].text


def test_hyperlink_line_breaks_are_joined() -> None:
    passages = by_pinpoint("BGE 150 III 100", KUENDIGUNG)
    assert passages["E. 2"].text == "Die Beschwerdeführerin rügt eine Verletzung von Art. 337 OR."
    assert passages["E. 2.1"].text.startswith("Nach Art. 337 Abs. 1 OR kann der Arbeitgeber")
    assert passages["E. 2.1"].text.endswith(
        "(BGE 142 III 579 E. 4.2 S. 579 f.; vgl. BGE 137 III 303 E. 2.1.1)."
    )


def test_numbered_list_inside_an_erwaegung_is_not_a_heading() -> None:
    passages = by_pinpoint("BGE 150 III 100", KUENDIGUNG)
    assert "1. ob die Verfehlung" in passages["E. 2.2"].text
    assert "2. ob eine Verwarnung" in passages["E. 2.2"].text
    assert "E. 1" not in passages


def test_introductory_line_is_skipped() -> None:
    passages = by_pinpoint("BGE 149 II 200", CONSIDERANTS)
    assert set(passages) == {"Regeste", "E. 4", "E. 4.1"}
    assert not any("considérants" in p.text for p in passages.values())


def test_letter_subdivisions_of_older_decisions() -> None:
    passages = by_pinpoint("BGE 120 II 50", ALTE_PRAXIS)
    assert list(passages) == ["Regeste", "E. 3a", "E. 3b", "E. 3b/aa", "E. 3b/bb", "E. 4"]
    assert passages["E. 3b/aa"].text == "Massgebend ist der Wille der ursprünglichen Parteien."


def test_citations() -> None:
    passages = by_pinpoint("BGE 150 III 100", KUENDIGUNG)
    assert passages["E. 2.2"].citation == "BGE 150 III 100 E. 2.2 S. 101"
    # The headnote is the court's summary, not its reasoning, and is marked as such.
    assert passages["Regeste"].citation == "BGE 150 III 100 (Regeste)"
    assert Passage("BGE 1 I 1", "E. 2", None, "x").citation == "BGE 1 I 1 E. 2"


def test_text_before_the_first_number_is_cited_by_page() -> None:
    text = "Regeste\nKurz.\nErwägungen\nDie Beschwerde ist offensichtlich unbegründet.\n"
    passages = segment("BGE 140 I 10", text)
    assert passages[-1].pinpoint == "E."
    assert passages[-1].citation == "BGE 140 I 10 S. 10"


def test_decision_without_erwaegungen_keeps_its_regeste() -> None:
    passages = segment("BGE 140 I 10", "Regeste\nNur ein Leitsatz.\nSachverhalt\nA.\n")
    assert [(p.pinpoint, p.text) for p in passages] == [("Regeste", "Nur ein Leitsatz.")]


def test_long_erwaegung_is_split_at_sentence_breaks() -> None:
    sentence = "Der Arbeitgeber muss die Kündigung schriftlich begründen, wenn dies verlangt wird. "
    text = sentence * 40
    spans = _chunk(text.strip(), limit=500)
    assert len(spans) > 1
    pieces = [text[s:e] for s, e in spans]
    assert all(len(piece) <= 500 for piece in pieces)
    assert all(piece.endswith("verlangt wird.") for piece in pieces)
    assert all(piece.startswith("Der Arbeitgeber") for piece in pieces)


def test_reference_helpers() -> None:
    assert citation_of("bge_145 III 72") == "BGE 145 III 72"
    assert first_page("BGE 145 III 72") == 72


def test_abbreviations_are_whole_words() -> None:
    assert ends_with_abbreviation("vgl. Art.")
    assert ends_with_abbreviation("(BGE 140 III 86 E.")
    assert ends_with_abbreviation("n.")
    assert not ends_with_abbreviation("sa décision.")
    assert not ends_with_abbreviation("BGE.")


def test_paragraphs_survive_a_sentence_ending_in_an_abbreviation_letter() -> None:
    text = "Erwägungen\n2.\nDas Gericht muss entscheiden.\nDie Frage ist offen.\n"
    [passage] = segment("BGE 140 I 10", text)
    assert passage.text == "Das Gericht muss entscheiden.\nDie Frage ist offen."
