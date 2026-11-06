from __future__ import annotations

from devils_advocate.segment import (
    Passage,
    _chunk,
    citation_of,
    ends_with_abbreviation,
    first_page,
    names_another_decision,
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
    assert citation_of("bge_120 IA 1") == "BGE 120 Ia 1"
    assert citation_of("bge_119 IB 254") == "BGE 119 Ib 254"
    assert citation_of("bge_1_I_179") == "BGE 1 I 179"
    assert citation_of("bge_145 IV 1") == "BGE 145 IV 1"
    assert first_page("BGE 145 III 72") == 72
    # The page markers of such a decision say "Ia", so it is not mistaken for another one.
    assert not names_another_decision(citation_of("bge_120 IA 1"), "BGE 120 Ia 1 S. 2\n")


def test_every_part_of_a_regeste_is_a_passage() -> None:
    nbsp = "\xa0"
    text = (
        f"Urteilskopf\nRegeste{nbsp}a\nErster Leitsatz zur Zuständigkeit.\n"
        f"Regeste{nbsp}b\nZweiter Leitsatz zur Verhältnismässigkeit.\n"
        "Sachverhalt\nA.\nErwägungen\n3.\nDer Eingriff ist schwer.\n"
    )
    passages = segment("BGE 151 I 137", text)
    assert [(p.pinpoint, p.text) for p in passages] == [
        ("Regeste", "Erster Leitsatz zur Zuständigkeit."),
        ("Regeste", "Zweiter Leitsatz zur Verhältnismässigkeit."),
        ("E. 3", "Der Eingriff ist schwer."),
    ]
    assert passages[1].citation == "BGE 151 I 137 (Regeste)"


def test_abbreviations_are_whole_words() -> None:
    assert ends_with_abbreviation("vgl. Art.")
    assert ends_with_abbreviation("(BGE 140 III 86 E.")
    assert ends_with_abbreviation("n.")
    assert not ends_with_abbreviation("sa décision.")
    assert not ends_with_abbreviation("BGE.")
    assert ends_with_abbreviation("entre D.")
    assert ends_with_abbreviation("Kanton St.")
    assert not ends_with_abbreviation("Das trifft nicht zu.")
    assert not ends_with_abbreviation("Das Gericht folgt dem Antrag des Klägers.")


def test_an_abbreviation_or_a_date_does_not_end_a_chunk() -> None:
    sentence = "Das Gericht hat die Frage bisher offengelassen und prüft sie hier. "
    for inside in (
        "Denn entgegen dem zit. Urteil 4A_384/2016 gilt die Frist.",
        "Das Urteil vom 31. Dezember 2019 ist rechtskräftig.",
        "Der Kanton St. Gallen hat das Eidg. Departement angerufen.",
        "Das Eidg. Versicherungsgericht hat die Frage offengelassen.",
        "Das Urteil vom 23. April 1920 ist rechtskräftig.",
        "Le contrat conclu entre D. SA et la banque est nul.",
        "Der Schaden beträgt 2 Mio. Franken und Fr. 5'000.- Zins.",
        "So entschied Prof. Müller in seinem Gutachten.",
    ):
        text = sentence * 20 + inside + " " + sentence * 20
        spans = _chunk(text, limit=len(sentence) * 20 + 30)
        assert any(inside in text[start:end] for start, end in spans), inside


def test_a_day_of_the_month_does_not_end_a_paragraph() -> None:
    text = "Erwägungen\n2.\nDas Urteil erging am 31.\nDezember 2019 und ist rechtskräftig.\n"
    [passage] = segment("BGE 140 I 10", text)
    assert passage.text == "Das Urteil erging am 31. Dezember 2019 und ist rechtskräftig."


def test_paragraphs_survive_a_sentence_ending_in_an_abbreviation_letter() -> None:
    text = "Erwägungen\n2.\nDas Gericht muss entscheiden.\nDie Frage ist offen.\n"
    [passage] = segment("BGE 140 I 10", text)
    assert passage.text == "Das Gericht muss entscheiden.\nDie Frage ist offen."


SEITENWECHSEL = """\
Regeste
Kurz.
Sachverhalt
BGE 141 III 200 S. 200
A. klagte.
Erwägungen
Aus den Erwägungen:
5.
Die Vorinstanz hat die Frist richtig berechnet. Sie begann mit der Zustellung
BGE 141 III 200 S. 201
und endete zehn Tage später.
5.1.1
Nach Art. 3 gilt:
3
Die Frist steht still.
6.
Die Beschwerde ist abzuweisen.
Dispositiv
1.
Die Beschwerde wird abgewiesen.
"""


def test_a_passage_records_where_a_new_page_begins() -> None:
    passages = by_pinpoint("BGE 141 III 200", SEITENWECHSEL)
    five = passages["E. 5"]
    assert five.page == 200
    assert "S. 201" not in five.text
    later = five.text.index("und endete")
    assert five.breaks == ((later, 201),)
    assert five.citation == "BGE 141 III 200 E. 5 S. 200"
    assert five.citation_of_span(0, 20) == "BGE 141 III 200 E. 5 S. 200"
    assert five.citation_of_span(0, len(five.text)) == "BGE 141 III 200 E. 5 S. 200 f."
    assert five.citation_of_span(later, len(five.text)) == "BGE 141 III 200 E. 5 S. 201"


def test_quotes_over_several_pages_are_cited_ff() -> None:
    passage = Passage("BGE 1 I 1", "E. 2", 5, "a b c d", ((2, 6), (4, 7)))
    assert passage.citation_of_span(0, 6) == "BGE 1 I 1 E. 2 S. 5 ff."
    assert passage.citation_of_span(2, 4) == "BGE 1 I 1 E. 2 S. 6"


def test_excerpts_may_start_late_and_skip_a_level() -> None:
    # An excerpt ("Aus den Erwägungen") begins at E. 5; E. 5.1.1 has no 5.1 above it.
    assert [p.pinpoint for p in segment("BGE 141 III 200", SEITENWECHSEL)] == [
        "Regeste",
        "E. 5",
        "E. 5.1.1",
        "E. 6",
    ]


def test_a_bare_number_is_a_statute_paragraph_not_a_heading() -> None:
    passages = by_pinpoint("BGE 141 III 200", SEITENWECHSEL)
    assert "E. 3" not in passages
    assert passages["E. 5.1.1"].text.endswith("Die Frist steht still.")


def test_the_dispositiv_is_not_part_of_the_reasoning() -> None:
    passages = by_pinpoint("BGE 141 III 200", SEITENWECHSEL)
    assert "abgewiesen" not in " ".join(p.text for p in passages.values())
    assert "E. 1" not in passages


def test_a_decision_filed_under_another_reference_is_recognised() -> None:
    assert names_another_decision("BGE 141 III 300", SEITENWECHSEL)
    assert not names_another_decision("BGE 141 III 200", SEITENWECHSEL)
    assert not names_another_decision("BGE 141 III 300", "Regeste\nOhne Seitenzahlen.\n")
    french = "ATF 141 III 200 S. 201\n"
    assert not names_another_decision("BGE 141 III 200", french)


def test_chunks_do_not_end_at_an_abbreviation_or_inside_brackets() -> None:
    text = (
        "Die Frist ist gewahrt (vgl. BGE 140 III 86 E. 2. Dort wurde es offengelassen). "
        "Der Arbeitgeber muss die Kündigung begründen, wenn dies verlangt wird. " * 12
    )
    spans = _chunk(text.strip(), limit=400)
    pieces = [text[s:e] for s, e in spans]
    assert len(pieces) > 1
    assert all(not piece.endswith("E. 2.") for piece in pieces)
    assert all(piece.count("(") == piece.count(")") for piece in pieces)


def test_a_numbered_decision_has_no_letter_subdivisions() -> None:
    # The layout of BGE 144 V 299: decimal headings, and a list "a) ... b) ..." inside one.
    text = (
        "Erwägungen\n5.\nZu prüfen ist zweierlei:\na) ob die Frist lief;\n"
        "b) ob sie gewahrt ist.\n5.1\nDie Frist lief.\n"
    )
    passages = by_pinpoint("BGE 144 V 299", text)
    assert list(passages) == ["E. 5", "E. 5.1"]
    assert "a) ob die Frist lief;" in passages["E. 5"].text


def test_the_operative_part_ends_the_reasoning_in_every_language() -> None:
    for closing in (
        "Demnach erkennt das Bundesgericht:",
        "Demnach erkennt das Eidg. Versicherungsgericht:",
        "Par ces motifs, le Tribunal fédéral prononce:",
        "Per questi motivi, il Tribunale federale pronuncia:",
    ):
        text = f"Erwägungen\n6.\nDie Beschwerde ist abzuweisen.\n{closing}\n1.\nAbgewiesen.\n"
        passages = by_pinpoint("BGE 141 III 200", text)
        assert list(passages) == ["E. 6"], closing
        assert passages["E. 6"].text == "Die Beschwerde ist abzuweisen."


def test_reasoning_that_begins_with_par_ces_motifs_goes_on() -> None:
    # The layout of BGE 120 Ib 224: the words open a sentence of reasoning.
    text = (
        "Erwägungen\n1.\nPer questi motivi la decisione impugnata deve essere confermata.\n"
        "2.\nIl ricorso è respinto.\nPer questi motivi,\n1.\nRespinto.\n"
    )
    passages = by_pinpoint("BGE 120 Ib 224", text)
    assert list(passages) == ["E. 1", "E. 2"]
    assert passages["E. 2"].text == "Il ricorso è respinto."


def test_a_stray_number_line_keeps_the_letters_of_the_other_erwaegungen() -> None:
    # The layout of BGE 124 II 146: letters, and one quoted line "3.1" under E. 3.
    text = (
        "Erwägungen\n2.\na) Die Frist lief.\nb) Sie ist gewahrt.\n3.\nZitiert wird:\n3.1\n"
        "Die Norm gilt.\n4.\na) Der Antrag ist zulässig.\n"
    )
    passages = by_pinpoint("BGE 124 II 146", text)
    assert {"E. 2a", "E. 2b", "E. 4a"} <= set(passages)
