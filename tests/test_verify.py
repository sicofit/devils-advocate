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
    assert normal == 'a "b",c'
    assert len(normal) == len(index)
    assert [text[i] for i in index] == ["A", " ", "«", "B", "»", ",", "C"]
