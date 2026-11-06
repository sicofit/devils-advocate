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


def test_a_close_match_must_not_change_a_reference_or_a_party() -> None:
    passage = (
        "Nach Art. 257d Abs. 2 OR kann der Vermieter fristlos kündigen, wenn der "
        "Beschwerdeführer die Frist zur Zahlung des Mietzinses verpasst."
    )
    article = passage.replace("257d", "257a").replace("verpasst", "verpasste")
    party = passage.replace("Beschwerdeführer", "Beschwerdegegner").replace("Frist", "Fristen")
    worker = (
        "Der Arbeitgeber muss die Kündigung auf Verlangen schriftlich begründen, sagt das Gesetz."
    )
    for quote, source in (
        (article, passage),
        (party, passage),
        (worker.replace("Arbeitgeber", "Arbeitnehmer").replace("sagt", "sagte"), worker),
    ):
        verdict = verify(quote, source)
        assert verdict.status is Status.REJECTED, quote
        assert verdict.reason == "changes a word of the cited passage"


def test_a_close_match_must_not_turn_a_negation_into_another_word() -> None:
    passage = "Le bailleur ne peut pas résilier le bail sans avertissement écrit préalable."
    quote = "Le bailleur ne peut par résilier le bail sans avertissement écrit préalable."
    assert verify(quote, passage).status is Status.REJECTED


def test_only_the_last_word_of_a_quote_may_be_clipped() -> None:
    passage = "Die Kündigung ist unzulässig, wenn sie missbräuchlich erfolgt ist."
    assert not verify("zulässig, wenn sie missbräuchlich erfolgt", passage).ok


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


def test_an_ellipsis_must_not_leave_out_any_kind_of_negation() -> None:
    cases = [
        ("Die Beschwerde ist keineswegs begründet und abzuweisen.", "de"),
        ("Der Anspruch besteht ohne Zweifel nur für die Zukunft und Vergangenheit.", "de"),
        ("Le recours ne saurait guère être admis en cette matière précise.", "fr"),
        ("La décision est rendue sans motivation suffisante pour les parties.", "fr"),
        ("Il ricorso non è mai ammissibile in questa materia specifica.", "it"),
        ("Il ricorso è stato respinto senza alcuna motivazione dalla corte.", "it"),
        ("Die Vorinstanz hat den Anspruch zu Unrecht verneint und die Klage abgewiesen.", "de"),
    ]
    for passage, language in cases:
        words = passage.split()
        quote = " ".join(words[:3]) + " ... " + " ".join(words[-3:])
        verdict = verify(quote, passage, language)
        assert verdict.status is Status.REJECTED, passage
        assert verdict.reason == "the ellipsis leaves out a negation", passage


def test_a_word_that_only_looks_like_a_negation_may_be_left_out() -> None:
    passage = "Der Entscheid vom 23. Mai 1989 wurde von der Vorinstanz bestätigt und eröffnet."
    quote = "Der Entscheid vom ... von der Vorinstanz bestätigt und eröffnet."
    assert verify(quote, passage, "de").ok
    passage = "L'autorité entend la personne concernée avant que la décision soit rendue."
    assert verify("L'autorité entend ... avant que la décision soit rendue.", passage, "fr").ok
    passage = "Secondo la giurisprudenza ne va tenuto conto in ogni caso concreto."
    assert verify("Secondo la giurisprudenza ... in ogni caso concreto.", passage, "it").ok
    passage = "Siehe dazu insbesondere N. 248 zu Art. 8 ZGB, wonach der Beweis dem Kläger obliegt."
    quote = "Siehe dazu insbesondere ... zu Art. 8 ZGB, wonach der Beweis dem Kläger obliegt."
    assert verify(quote, passage, "de").ok


def test_the_language_decides_which_words_negate() -> None:
    passage = "Questa norma non trova mai applicazione nei rapporti tra privati."
    quote = "Questa norma non trova ... applicazione nei rapporti tra privati."
    assert not verify(quote, passage, "it").ok
    passage = "La décision du 24 mai 2023 a été confirmée par la cour cantonale."
    assert verify(
        "La décision du 24 ... 2023 a été confirmée par la cour cantonale.", passage, "fr"
    ).ok
    passage = "Le délai n'a pas été respecté par la partie recourante en l'espèce."
    passage = passage.replace("Le délai", "Le délai de recours")
    quote = "Le délai de recours ... été respecté par la partie recourante en l'espèce."
    assert verify(quote, passage, "fr").reason == "the ellipsis leaves out a negation"


def test_an_ellipsis_must_not_leave_out_a_qualifier() -> None:
    passage = (
        "Eine Verlängerung der Frist ist nur ausnahmsweise zulässig, "
        "wenn wichtige Gründe vorliegen."
    )
    quote = "Eine Verlängerung der Frist ist ... zulässig, wenn wichtige Gründe vorliegen."
    verdict = verify(quote, passage, "de")
    assert verdict.status is Status.REJECTED
    assert verdict.reason == "the ellipsis leaves out a qualifier"
    passage = "Le délai peut en principe être prolongé lorsque des motifs importants existent."
    quote = "Le délai peut ... être prolongé lorsque des motifs importants existent."
    assert verify(quote, passage, "fr").reason == "the ellipsis leaves out a qualifier"


def test_an_ellipsis_may_leave_out_whole_sentences() -> None:
    passage = (
        "Die Frist beginnt mit der Zustellung des Entscheids. Sie gilt nur für die Parteien, "
        "nicht für Dritte. Wird die Frist versäumt, so ist auf die Beschwerde nicht einzutreten."
    )
    quote = (
        "Die Frist beginnt mit der Zustellung des Entscheids. [...] Wird die Frist versäumt, "
        "so ist auf die Beschwerde nicht einzutreten."
    )
    assert verify(quote, passage, "de").status is Status.EXACT


def test_a_quote_must_not_leave_out_the_denial_it_depends_on() -> None:
    passage = "Es trifft nicht zu, dass die Frist während der Gerichtsferien stillsteht."
    verdict = verify("die Frist während der Gerichtsferien stillsteht.", passage, "de")
    assert verdict.reason == "the quote leaves out the denial its sentence starts with"
    passage = "On ne saurait admettre que le délai soit suspendu pendant les féries."
    verdict = verify("le délai soit suspendu pendant les féries.", passage, "fr")
    assert verdict.reason == "the quote leaves out the denial its sentence starts with"
    # A negation in an earlier clause does not deny what the quote says.
    passage = "Da die Frist nicht eingehalten wurde, ist auf die Beschwerde nicht einzutreten."
    verdict = verify("ist auf die Beschwerde nicht einzutreten.", passage, "de")
    assert verdict.status is Status.EXACT
    assert verdict.opens_inside


def test_a_quote_cut_from_a_sentence_is_marked() -> None:
    verdict = verify("bei besonders schweren Verfehlungen gerechtfertigt.", PASSAGE)
    assert verdict.opens_inside and not verdict.ends_inside
    assert verdict.shown == "[…] bei besonders schweren Verfehlungen gerechtfertigt."
    verdict = verify("Die fristlose Kündigung ist nur bei besonders schweren", PASSAGE)
    assert verdict.ends_inside and not verdict.opens_inside
    assert verdict.shown == "Die fristlose Kündigung ist nur bei besonders schweren […]"
    whole = "Die fristlose Kündigung ist nur bei besonders schweren Verfehlungen gerechtfertigt."
    assert verify(whole, PASSAGE).shown == whole


def test_a_numbered_item_starts_a_sentence() -> None:
    passage = (
        "Die Kurse werden veröffentlicht. (6) In den übrigen Fällen gilt der Kurs des Vortags."
    )
    verdict = verify("In den übrigen Fällen gilt der Kurs des Vortags.", passage)
    assert not verdict.opens_inside


def test_a_close_match_must_keep_the_word() -> None:
    pairs = [
        ("Arbeitsfähigkeit", "Arbeitsunfähigkeit"),
        ("urteilsfähig", "urteilsunfähig"),
        ("Erwerbsfähigkeit", "Erwerbsunfähigkeit"),
        ("verfassungsmässig", "verfassungswidrig"),
        ("rechtsgleich", "rechtsungleich"),
        ("strafbar", "straflos"),
    ]
    for quoted, original in pairs:
        passage = (
            f"Das Gericht hielt fest, die Person sei {original} gewesen, wie der Bericht zeigt."
        )
        quote = f"Das Gericht hielt fest, die Person sei {quoted} gewesen, wie der Bericht zeigt."
        assert not verify(quote, passage).ok, quoted


def test_a_close_match_may_change_an_ending() -> None:
    passage = (
        "Bei schwerer Verfehlung ist die fristlose Kündigung zulässige Massnahme des Arbeitgebers."
    )
    quote = (
        "Bei schweren Verfehlung ist die fristlose Kündigung zulässig Massnahme des Arbeitgebers."
    )
    verdict = verify(quote, passage)
    assert verdict.status is Status.CLOSE
    assert verdict.text == passage


def test_sharp_s_and_raised_suffixes() -> None:
    passage = "Gemäss Art. 4 bis Abs. 1 ist die Massnahme zu prüfen, bevor sie angeordnet wird."
    assert (
        verify("Gemäß Art. 4bis Abs. 1 ist die Maßnahme zu prüfen", passage).status is Status.EXACT
    )
    passage = "La loi est entrée en vigueur le 1 er janvier 2011 pour tous les cantons."
    assert verify("entrée en vigueur le 1er janvier 2011 pour tous", passage).ok
    # "bis" stays a word: "Art. 4 bis 6" is a range, and a quote may start with it.
    passage = "Die Regel galt von 2010 bis 2015 für alle Verfahren vor dem Gericht."
    assert verify("bis 2015 für alle Verfahren vor dem Gericht.", passage).ok


def test_an_ellipsis_in_the_passage_itself_is_matched_whole() -> None:
    passage = (
        "Der Kanton verweist auf § 3d PolV/LU. (...) [Es] handle (...) sich nicht um eine "
        "Bewilligung im Sinne des Gesetzes."
    )
    quote = (
        "Der Kanton verweist auf § 3d PolV/LU. (...) [Es] handle (...) sich nicht um eine "
        "Bewilligung"
    )
    assert verify(quote, passage, "de").status is Status.EXACT


def test_a_restated_view_is_marked_reported() -> None:
    passage = (
        "Die Vorinstanz erwog, die Frist stehe während der Gerichtsferien nicht still. "
        "Die Ausnahme gelte auch für das Rechtsmittelverfahren."
    )
    verdict = verify("Die Ausnahme gelte auch für das Rechtsmittelverfahren.", passage, "de")
    assert verdict.reported
    verdict = verify("die Frist stehe während der Gerichtsferien nicht still.", passage, "de")
    assert verdict.reported
    assert verdict.text.startswith("Die Vorinstanz erwog,")
    assert not verdict.opens_inside


def test_the_court_answering_is_not_reported() -> None:
    passage = (
        "Der Beschwerdeführer macht geltend, die Frist sei eingehalten. "
        "Dies trifft nicht zu. Die Frist begann mit der Zustellung des Entscheids zu laufen."
    )
    verdict = verify("Die Frist begann mit der Zustellung des Entscheids zu laufen.", passage, "de")
    assert verdict.ok and not verdict.reported
    passage = (
        "Dies setzt, wie die Beschwerdegegnerinnen zutreffend einwenden, voraus, dass die "
        "Ansprüche trotz Löschung durchgesetzt werden können."
    )
    assert not verify(
        "dass die Ansprüche trotz Löschung durchgesetzt werden können.", passage
    ).reported
    passage = (
        "Entgegen der Auffassung der Vorinstanz ist die Frist während der Gerichtsferien "
        "nicht stillgestanden."
    )
    assert not verify(passage, passage, "de").reported


def test_a_quote_that_begins_with_the_complementizer_keeps_the_denial() -> None:
    passage = "Es trifft nicht zu, dass die Frist während der Gerichtsferien stillsteht."
    verdict = verify("dass die Frist während der Gerichtsferien stillsteht.", passage, "de")
    assert verdict.reason == "the quote leaves out the denial its sentence starts with"
    passage = "On ne saurait admettre que le délai soit suspendu pendant les féries."
    verdict = verify("que le délai soit suspendu pendant les féries.", passage, "fr")
    assert verdict.reason == "the quote leaves out the denial its sentence starts with"


def test_a_reported_quote_in_quotation_marks_keeps_its_attribution() -> None:
    passage = "Die Vorinstanz erwog, «die Frist stehe während der Gerichtsferien nicht still»."
    verdict = verify("die Frist stehe während der Gerichtsferien nicht still.", passage, "de")
    assert verdict.reported
    assert verdict.text == passage


def test_a_quote_that_stops_at_the_day_of_a_date_ends_inside() -> None:
    passage = "Die Frist lief am 31. Mai 2020 ab und wurde nicht gewahrt."
    assert verify("Die Frist lief am 31.", passage, "de").ends_inside


def test_bis_zu_is_a_range() -> None:
    passage = "Die Strafe beträgt 3 bis zu 6 Monate Freiheitsstrafe nach dem Gesetz."
    assert verify("bis zu 6 Monate Freiheitsstrafe nach dem Gesetz.", passage, "de").ok


def test_a_decomposed_accent_is_shown_with_its_letter() -> None:
    passage = "Le recours est donc rejeté dans la mesure où il est recevable."
    verdict = verify("Le recours est donc rejeté", passage, "fr")
    assert verdict.text == "Le recours est donc rejeté"


def test_an_ellipsis_tries_a_later_occurrence_of_the_fragment_before() -> None:
    passage = (
        "Die Kündigung ist zulässig. Die Kündigung ist nicht ohne Weiteres wirksam, sondern "
        "die Kündigung ist im Ergebnis erst nach Ablauf der Frist wirksam."
    )
    verdict = verify("die Kündigung ist […] erst nach Ablauf der Frist wirksam.", passage, "de")
    assert verdict.ok
    assert verdict.text == "die Kündigung ist […] erst nach Ablauf der Frist wirksam."


def test_german_single_quotation_marks() -> None:
    passage = "Massgebend ist der ‚Grundsatz von Treu und Glauben‘ nach Art. 2 ZGB."
    verdict = verify('der "Grundsatz von Treu und Glauben" nach Art. 2 ZGB.', passage, "de")
    assert verdict.status is Status.EXACT
    assert verdict.text == "der ‚Grundsatz von Treu und Glauben‘ nach Art. 2 ZGB."
    passage = "Selon l’art. 8 CC, le fardeau de la preuve incombe à celui qui allègue."
    assert verify("Selon l'art. 8 CC, le fardeau de la preuve", passage, "fr").ok
