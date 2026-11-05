from __future__ import annotations

from devils_advocate.verify import CLOSE_THRESHOLD, Status, normalize, verify

PASSAGE = (
    "Die fristlose Kündigung ist nur bei besonders schweren Verfehlungen gerechtfertigt. "
    "Sind die Verfehlungen weniger schwerwiegend, so müssen sie trotz Verwarnung wiederholt "
    "vorgekommen sein. Massgebend ist der «Grundsatz von Treu und Glauben» (Art. 2 ZGB)."
)


def test_exact_quote() -> None:
    verdict = verify("Die fristlose Kündigung ist nur bei besonders schweren", PASSAGE)
    assert verdict.status is Status.EXACT
    assert verdict.text == "Die fristlose Kündigung ist nur bei besonders schweren"
    assert verdict.score == 1.0


def test_typography_case_and_whitespace_do_not_matter() -> None:
    quote = '"massgebend ist der  "Grundsatz von Treu und Glauben" (Art. 2 ZGB)."'
    verdict = verify(quote, PASSAGE)
    assert verdict.status is Status.EXACT
    # What is shown is the passage's wording, not the model's.
    assert verdict.text == "Massgebend ist der «Grundsatz von Treu und Glauben» (Art. 2 ZGB)."


def test_ellipsis_fragments_must_appear_in_order() -> None:
    quote = "Die fristlose Kündigung ist nur [...] trotz Verwarnung wiederholt vorgekommen sein."
    verdict = verify(quote, PASSAGE)
    assert verdict.status is Status.EXACT
    assert verdict.text == (
        "Die fristlose Kündigung ist nur […] trotz Verwarnung wiederholt vorgekommen sein."
    )

    swapped = "trotz Verwarnung wiederholt vorgekommen sein … Die fristlose Kündigung ist nur"
    assert verify(swapped, PASSAGE).status is Status.REJECTED


def test_ellipsis_before_punctuation_has_no_space() -> None:
    verdict = verify("Sind die Verfehlungen weniger schwerwiegend … , so müssen sie", PASSAGE)
    assert verdict.ok
    assert verdict.text == "Sind die Verfehlungen weniger schwerwiegend […], so müssen sie"


def test_near_miss_is_close_and_shows_the_original() -> None:
    quote = "Die fristlose Kündigung ist nur bei besonders schwerer Verfehlungen gerechtfertigt."
    verdict = verify(quote, PASSAGE)
    assert verdict.status is Status.CLOSE
    assert CLOSE_THRESHOLD <= verdict.score < 1.0
    assert verdict.text == (
        "Die fristlose Kündigung ist nur bei besonders schweren Verfehlungen gerechtfertigt."
    )


def test_invented_quote_is_rejected() -> None:
    quote = "Eine fristlose Kündigung ist bei jeder Verfehlung ohne Weiteres zulässig."
    verdict = verify(quote, PASSAGE)
    assert verdict.status is Status.REJECTED
    assert verdict.text == ""
    assert verdict.reason == "not found in the cited passage"


def test_paraphrase_is_rejected() -> None:
    quote = "Nur schwere Verfehlungen rechtfertigen eine fristlose Entlassung."
    assert not verify(quote, PASSAGE).ok


def test_too_short_quote_is_rejected() -> None:
    verdict = verify("fristlose Kündigung", PASSAGE)
    assert verdict.status is Status.REJECTED
    assert "shorter than" in verdict.reason


def test_normalize_keeps_an_index_into_the_original() -> None:
    text = "A  «B» ,C"
    normal, index = normalize(text)
    assert normal == "a b,c"  # quotation marks dropped, space before "," too
    assert len(normal) == len(index)
    assert [text[i] for i in index] == ["A", " ", "B", ",", "C"]


def test_a_quote_must_not_start_or_end_inside_a_word() -> None:
    passage = "Die Kündigung ist unzulässig, wenn sie missbräuchlich erfolgt ist."
    assert not verify("Kündigung ist zulässig, wenn sie missbräuchlich", passage).ok
    assert verify("Kündigung ist unzulässig, wenn sie missbräuchlich", passage).ok
    # A clipped last word is a near miss: the passage's whole word is shown.
    clipped = verify("Kündigung ist unzulässig, wenn sie missbräuch", passage)
    assert clipped.status is Status.CLOSE
    assert clipped.text == "Kündigung ist unzulässig, wenn sie missbräuchlich"


def test_ellipses_do_not_count_as_words() -> None:
    verdict = verify("Die fristlose […] Kündigung […] ist nur", PASSAGE)
    assert verdict.status is Status.REJECTED
    assert "between ellipses" in verdict.reason
    assert not verify("… […] ... … […]", PASSAGE).ok
    assert not verify('"" … ""', PASSAGE).ok


def test_an_ellipsis_must_not_leave_out_a_negation() -> None:
    passage = "Eine Verwarnung ist nicht in jedem Fall erforderlich, um fristlos zu kündigen."
    quote = "Eine Verwarnung ist […] in jedem Fall erforderlich, um fristlos zu kündigen."
    verdict = verify(quote, passage)
    assert verdict.status is Status.REJECTED
    assert verdict.reason == "the ellipsis leaves out a negation"


def test_a_close_match_must_not_drop_or_add_a_word() -> None:
    passage = "Der Arbeitgeber muss die Kündigung nicht schriftlich begründen, wenn verlangt."
    dropped = "Der Arbeitgeber muss die Kündigung schriftlich begründen, wenn verlangt."
    verdict = verify(dropped, passage)
    assert verdict.status is Status.REJECTED
    assert verdict.reason == "changes a word of the cited passage"
    changed = "Der Arbeitgeber muss die Kündigung nicht schriftlich begründet, wenn verlangt."
    assert verify(changed, passage).status is Status.CLOSE


def test_french_quotation_marks_with_spaces() -> None:
    passage = "Le juge applique le principe « in dubio pro reo » de manière stricte."
    verdict = verify("applique le principe «in dubio pro reo» de manière stricte", passage)
    assert verdict.status is Status.EXACT
    assert verdict.text == "applique le principe « in dubio pro reo » de manière stricte"


def test_shown_quotation_marks_are_balanced() -> None:
    verdict = verify("der «Grundsatz von Treu und Glauben", PASSAGE)
    assert verdict.ok
    assert verdict.text == "der «Grundsatz von Treu und Glauben»"


def test_only_enclosing_quotation_marks_are_stripped() -> None:
    passage = 'Der Begriff "wichtiger Grund" und der Begriff "Verfehlung" sind auszulegen.'
    verdict = verify('"wichtiger Grund" und der Begriff "Verfehlung"', passage)
    assert verdict.status is Status.EXACT
    assert verdict.text == '"wichtiger Grund" und der Begriff "Verfehlung"'


def test_spans_point_into_the_passage() -> None:
    quote = "Die fristlose Kündigung ist nur [...] trotz Verwarnung wiederholt vorgekommen sein."
    verdict = verify(quote, PASSAGE)
    assert [PASSAGE[a:b] for a, b in verdict.spans] == [
        "Die fristlose Kündigung ist nur",
        "trotz Verwarnung wiederholt vorgekommen sein.",
    ]
