from __future__ import annotations

from collections.abc import Callable

import pytest

from devils_advocate import devil as devil_module
from devils_advocate.client import LLMError, ModelUnavailable, Truncated
from devils_advocate.devil import Devil
from devils_advocate.search import Retriever
from devils_advocate.verify import Status

from fakes import FakeModel, passages_in

POSITION = "Eine fristlose Kündigung ist schon bei einer leichten Verfehlung zulässig."
HOLDING = "Die fristlose Kündigung ist nur bei besonders schweren Verfehlungen gerechtfertigt."


def analysis(*claims: str, language: str = "de") -> dict:
    return {
        "language": language,
        "claims": [
            {"claim": c, "queries": ["fristlose Kündigung schwere Verfehlung"]} for c in claims
        ],
    }


def label_of(messages: list[dict], needle: str) -> str:
    return next(label for label, text in passages_in(messages).items() if needle in text)


def scripted(counter: Callable[[list[dict]], list[dict]]) -> FakeModel:
    def respond(messages: list[dict], name: str) -> dict:
        if name == "analysis":
            return analysis(POSITION)
        return {"counterarguments": counter(messages)}

    return FakeModel(respond)


def test_verified_quote_is_kept_and_cut_from_the_passage(retriever: Retriever) -> None:
    model = scripted(
        lambda m: [
            {
                "passage": label_of(m, HOLDING),
                # The model's copy differs in case and quote marks; the passage wins.
                "quote": "«die fristlose Kündigung ist nur bei besonders schweren Verfehlungen»",
                "argument": "BGE 150 III 100 verlangt eine besonders schwere Verfehlung.",
            }
        ]
    )
    rebuttal = Devil(model, retriever).argue(POSITION)

    assert model.names() == ["analysis", "counterarguments"]
    [claim] = rebuttal.claims
    [counter] = claim.counters
    assert counter.quote == "Die fristlose Kündigung ist nur bei besonders schweren Verfehlungen"
    assert counter.status is Status.EXACT
    assert counter.citation == "BGE 150 III 100 E. 2.2 S. 101"
    assert rebuttal.verified == rebuttal.proposed == 1


def test_hallucinations_are_discarded_with_a_reason(retriever: Retriever) -> None:
    model = scripted(
        lambda m: [
            {
                "passage": label_of(m, HOLDING),
                "quote": "Eine fristlose Kündigung ist stets ohne Verwarnung zulässig.",
                "argument": "invented",
            },
            {
                "passage": "P99",
                "quote": HOLDING,
                "argument": "cites a passage that was never shown",
            },
        ]
    )
    rebuttal = Devil(model, retriever).argue(POSITION)

    [claim] = rebuttal.claims
    assert claim.counters == []
    assert claim.abstained
    assert [d.reason for d in claim.discarded] == [
        "not found in the cited passage",
        "cites a passage it was not given",
    ]
    assert rebuttal.proposed == 2 and rebuttal.verified == 0


def test_one_counter_per_passage(retriever: Retriever) -> None:
    def twice(m: list[dict]) -> list[dict]:
        label = label_of(m, HOLDING)
        quotes = [HOLDING, "Sind die Verfehlungen weniger schwerwiegend, so müssen sie"]
        return [{"passage": label, "quote": q, "argument": "a"} for q in quotes]

    [claim] = Devil(scripted(twice), retriever).argue(POSITION).claims
    assert len(claim.counters) == 1


def everything(m: list[dict]) -> list[dict]:
    """A verbatim quote from every passage shown."""
    return [
        {"passage": label, "quote": " ".join(text.split()[:30]), "argument": "a"}
        for label, text in passages_in(m).items()
        if len(text.split()) >= 5
    ]


def test_one_counter_per_decision(retriever: Retriever) -> None:
    [claim] = Devil(scripted(everything), retriever).argue(POSITION).claims
    shown = {hit.passage.reference for hit in claim.hits}
    assert len(shown) >= 2 and len(claim.hits) > len(shown)  # several passages per decision
    references = [counter.passage.reference for counter in claim.counters]
    assert sorted(references) == sorted(shown)


def test_at_most_max_counters(retriever: Retriever, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devil_module, "MAX_COUNTERS", 1)
    [claim] = Devil(scripted(everything), retriever).argue(POSITION).claims
    assert len(claim.counters) == 1


def test_the_language_of_the_position_wins_over_the_models_guess(retriever: Retriever) -> None:
    model = FakeModel(
        lambda m, name: (
            analysis(POSITION, language="en") if name == "analysis" else {"counterarguments": []}
        )
    )
    assert Devil(model, retriever).argue(POSITION).language == "de"


def test_an_unusable_reply_costs_one_claim_not_the_turn(retriever: Retriever) -> None:
    replies = iter([LLMError("did not return valid JSON"), None])

    def respond(messages: list[dict], name: str) -> dict:
        if name == "analysis":
            return analysis("Erste Behauptung.", POSITION)
        if isinstance(failure := next(replies), Exception):
            raise failure
        return {
            "counterarguments": [
                {"passage": label_of(messages, HOLDING), "quote": HOLDING, "argument": "a"}
            ]
        }

    first, second = Devil(FakeModel(respond), retriever).argue(POSITION).claims
    assert first.error == "did not return valid JSON" and first.abstained
    assert len(second.counters) == 1 and not second.error


def test_a_model_that_is_gone_ends_the_turn(retriever: Retriever) -> None:
    def respond(messages: list[dict], name: str) -> dict:
        if name == "analysis":
            return analysis("Erste Behauptung.", POSITION)
        raise ModelUnavailable("No model server answers")

    model = FakeModel(respond)
    with pytest.raises(ModelUnavailable):
        Devil(model, retriever).argue(POSITION)
    assert model.names() == ["analysis", "counterarguments"]  # the second claim is not tried


@pytest.mark.parametrize(
    ("argument", "position", "kept"),
    [
        ("BGE 145 III 303 verlangt eine schwere Verfehlung.", POSITION, False),
        ("Gemäss Urteil 4A_384/2016 genügt das nicht.", POSITION, False),
        ("BGE 150 III 100 verlangt eine schwere Verfehlung.", POSITION, True),
        # The cited passage itself names BGE 142 III 579 (E. 2.1, the passage before).
        ("Wie BGE 137 III 303 zeigt, genügt das nicht.", f"{POSITION} Vgl. BGE 137 III 303.", True),
        ("Urteil 4A_384/2016 hilft nicht.", f"{POSITION} So Urteil 4A_384/2016.", True),
    ],
)
def test_an_argument_may_not_pin_the_quote_on_another_decision(
    retriever: Retriever, argument: str, position: str, kept: bool
) -> None:
    model = scripted(
        lambda m: [{"passage": label_of(m, HOLDING), "quote": HOLDING, "argument": argument}]
    )
    [claim] = Devil(model, retriever).argue(position).claims
    assert bool(claim.counters) is kept
    if not kept:
        [discarded] = claim.discarded
        assert discarded.reason == "the argument names a decision the cited passage does not"
        assert discarded.citation == "BGE 150 III 100 E. 2.2 S. 101"


def test_a_passage_quoted_for_an_earlier_claim_is_not_quoted_again(retriever: Retriever) -> None:
    def respond(messages: list[dict], name: str) -> dict:
        if name == "analysis":
            return analysis("Erste Behauptung.", POSITION)
        label = label_of(messages, HOLDING)
        return {"counterarguments": [{"passage": label, "quote": HOLDING, "argument": name}]}

    rebuttal = Devil(FakeModel(respond), retriever).argue(POSITION)
    first, second = rebuttal.claims
    assert len(first.counters) == 1 and second.counters == []
    [(counter, claim_number)] = second.repeated
    assert claim_number == 1 and counter.passage.id == first.counters[0].passage.id
    assert not second.abstained
    assert rebuttal.verified == 1


def test_another_quote_from_a_passage_already_quoted_is_a_new_counter(
    retriever: Retriever,
) -> None:
    other = "Sind die Verfehlungen weniger schwerwiegend, so müssen sie trotz Verwarnung"

    def respond(messages: list[dict], name: str) -> dict:
        if name == "analysis":
            return analysis("Erste Behauptung.", POSITION)
        quote = HOLDING if "Erste" in messages[-1]["content"] else other
        label = label_of(messages, HOLDING)
        return {"counterarguments": [{"passage": label, "quote": quote, "argument": "x"}]}

    first, second = Devil(FakeModel(respond), retriever).argue(POSITION).claims
    assert len(first.counters) == 1 and len(second.counters) == 1
    assert second.repeated == []
    assert second.counters[0].passage.id == first.counters[0].passage.id


def test_a_bge_the_user_did_not_name_is_taken_out_of_the_queries(retriever: Retriever) -> None:
    raw = {
        "language": "de",
        "claims": [
            {
                "claim": "Kündigung nach BGE 999 II 1 und BGE 150 III 100.",
                "queries": ["BGE 145 III 303 fristlose Kündigung", "BGE 150 III 100", "BGE 1 I 1"],
            }
        ],
    }
    model = FakeModel(lambda m, name: raw if name == "analysis" else {"counterarguments": []})
    [claim] = Devil(model, retriever).argue(f"{POSITION} Siehe BGE 150 III 100.").claims
    assert claim.queries == ["fristlose Kündigung", "BGE 150 III 100"]


def test_labels_are_normalised_and_never_shown(retriever: Retriever) -> None:
    def sloppy(m: list[dict]) -> list[dict]:
        label = label_of(m, HOLDING)
        return [
            {
                "passage": label.lower(),
                "quote": HOLDING,
                "argument": f"{label} verlangt eine besonders schwere Verfehlung, das heißt mehr.",
            }
        ]

    [claim] = Devil(scripted(sloppy), retriever).argue(POSITION).claims
    [counter] = claim.counters
    assert counter.argument == (
        "BGE 150 III 100 verlangt eine besonders schwere Verfehlung, das heisst mehr."
    )


def test_citation_names_the_page_of_the_quote(retriever: Retriever) -> None:
    model = scripted(
        lambda m: [{"passage": label_of(m, HOLDING), "quote": HOLDING, "argument": "a"}]
    )
    [claim] = Devil(model, retriever).argue(POSITION).claims
    [counter] = claim.counters
    start, end = counter.spans[0]
    assert counter.passage.text[start:end] == HOLDING
    assert counter.citation == "BGE 150 III 100 E. 2.2 S. 101"


def test_no_claim_means_no_search_and_no_counter(retriever: Retriever) -> None:
    model = FakeModel(lambda m, name: {"language": "en", "claims": []})
    rebuttal = Devil(model, retriever).argue("What a lovely day.")
    assert model.names() == ["analysis"]
    assert rebuttal.claims == [] and rebuttal.verified == 0


def test_claims_are_capped_and_cleaned(retriever: Retriever) -> None:
    raw = {
        "language": "de",
        "claims": [
            {"claim": "  ", "queries": []},
            "not an object",
            *({"claim": f"Behauptung {i}", "queries": ["Kündigung", "", 3]} for i in range(6)),
        ],
    }
    model = FakeModel(lambda m, name: raw if name == "analysis" else {"counterarguments": []})
    rebuttal = Devil(model, retriever, max_claims=2).argue(POSITION)
    assert [c.claim for c in rebuttal.claims] == ["Behauptung 0", "Behauptung 1"]
    assert rebuttal.claims[0].queries == ["Kündigung", "3"]


def test_follow_up_turn_sees_the_earlier_position(retriever: Retriever) -> None:
    model = FakeModel(lambda m, name: {"language": "de", "claims": []})
    devil = Devil(model, retriever)
    devil.argue(POSITION)
    devil.argue("Aber nicht bei leitenden Angestellten.")
    first, second = (messages[-1]["content"] for _, messages in model.calls)
    assert "Earlier in this conversation" not in first
    assert POSITION in second

    devil.reset()
    devil.argue("Neue Frage.")
    assert POSITION not in model.calls[-1][1][-1]["content"]


def test_stages_are_reported(retriever: Retriever) -> None:
    stages: list[str] = []
    devil = Devil(scripted(lambda m: []), retriever, on_stage=stages.append)
    devil.argue(POSITION)
    assert stages == [
        "Splitting your position into claims",
        "Searching precedent against claim 1/1",
        "Arguing against claim 1/1",
    ]


def test_a_quote_cut_inside_a_sentence_is_shown_with_marks(retriever: Retriever) -> None:
    quote = "nur bei besonders schweren Verfehlungen gerechtfertigt."
    model = scripted(lambda m: [{"passage": label_of(m, HOLDING), "quote": quote, "argument": "a"}])
    [counter] = Devil(model, retriever).argue(POSITION).claims[0].counters
    assert counter.quote == quote
    assert counter.shown == "[…] " + quote


def test_every_proposal_is_accounted_for(retriever: Retriever) -> None:
    def many(m: list[dict]) -> list[dict]:
        label = label_of(m, HOLDING)
        second = "Sind die Verfehlungen weniger schwerwiegend, so müssen sie"
        return [
            {"passage": label, "quote": HOLDING, "argument": "a", "effect": "contradicts"},
            {"passage": label, "quote": second, "argument": "a", "effect": "limits"},
            {"passage": label, "quote": HOLDING, "argument": "   ", "effect": "contradicts"},
            {"passage": label, "quote": HOLDING, "argument": "b", "effect": "supports"},
            {"passage": "P99", "quote": HOLDING, "argument": "c", "effect": "contradicts"},
        ]

    rebuttal = Devil(scripted(many), retriever).argue(POSITION)
    [claim] = rebuttal.claims
    assert len(claim.counters) + len(claim.discarded) + len(claim.skipped) == 5
    assert [d.reason for d in claim.discarded] == [
        "no argument given",
        "cites a passage it was not given",
    ]
    assert [d.reason for d in claim.skipped] == [
        "same decision as an earlier counter",
        "its argument says the passage supports the claim",
    ]
    # What was set aside did not fail a check: it is neither proposed nor discarded.
    assert rebuttal.proposed == 3 and rebuttal.verified == 1


def test_proposals_over_the_limit_are_skipped(
    retriever: Retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(devil_module, "MAX_COUNTERS", 1)
    model = scripted(everything)
    [claim] = Devil(model, retriever).argue(POSITION).claims
    offered = len(model.respond(model.calls[-1][1], "counterarguments")["counterarguments"])
    assert len(claim.counters) == 1
    assert len(claim.counters) + len(claim.discarded) + len(claim.skipped) == offered
    assert any(d.reason == "over the limit of 1" for d in claim.skipped)


@pytest.mark.parametrize(
    ("argument", "supports"),
    [
        ("BGE 150 III 100 hält fest, dass die Frist läuft, was die Behauptung bestätigt.", True),
        ("Dies untermauert die Behauptung direkt.", True),
        ("BGE 150 III 100 stützt die Behauptung.", True),
        ("This supports the claim directly.", True),
        ("L'arrêt confirme l'affirmation.", True),
        ("BGE 150 III 100 stützt die Behauptung nicht.", False),
        ("Das Urteil bestätigt die Behauptung keineswegs.", False),
        ("This does not support the claim.", False),
        ("This doesn't support the claim.", False),
        ("The passage cannot support the claim.", False),
        ("Nothing here supports the claim.", False),
        ("BGE 150 III 100 stützt die Behauptung nur teilweise.", False),
        ("It supports the claim only partly: the deadline differs.", False),
        ("Il confirme l'affirmation, mais seulement pour les baux.", False),
        ("BGE 150 III 100 bestätigt ausdrücklich, dass dies nicht zulässig ist.", False),
        ("BGE 150 III 100 widerlegt die Behauptung.", False),
    ],
)
def test_an_argument_that_agrees_with_the_claim_is_set_aside(
    retriever: Retriever, argument: str, supports: bool
) -> None:
    model = scripted(
        lambda m: [{"passage": label_of(m, HOLDING), "quote": HOLDING, "argument": argument}]
    )
    [claim] = Devil(model, retriever).argue(POSITION).claims
    assert bool(claim.skipped) is supports
    assert bool(claim.counters) is not supports


def test_a_cut_off_reply_is_retried_with_fewer_passages(retriever: Retriever) -> None:
    shown: list[int] = []

    def respond(messages: list[dict], name: str) -> dict:
        if name == "analysis":
            return analysis(POSITION)
        shown.append(len(passages_in(messages)))
        if len(shown) == 1:
            raise Truncated("cut off")
        return {"counterarguments": []}

    [claim] = Devil(FakeModel(respond), retriever, top_k=4).argue(POSITION).claims
    assert len(shown) == 2 and shown[1] == shown[0] // 2
    assert not claim.error


def test_a_reply_cut_off_twice_costs_the_claim(retriever: Retriever) -> None:
    def respond(messages: list[dict], name: str) -> dict:
        if name == "analysis":
            return analysis(POSITION)
        raise Truncated("the reply was cut off")

    model = FakeModel(respond)
    [claim] = Devil(model, retriever, top_k=4).argue(POSITION).claims
    assert claim.error == "the reply was cut off"
    assert model.names() == ["analysis", "counterarguments", "counterarguments"]


def test_a_claim_without_counter_is_not_called_answered(retriever: Retriever) -> None:
    model = FakeModel(
        lambda m, name: analysis(POSITION) if name == "analysis" else {"counterarguments": []}
    )
    devil = Devil(model, retriever)
    devil.argue(POSITION)
    devil.argue("Auch bei leitenden Angestellten.")
    prompt = model.calls[-2][1][-1]["content"]  # the second turn's analysis
    assert "Claims answered" not in prompt
    assert f"Claims without counter-precedent: {POSITION}" in prompt


def test_an_argument_with_a_label_it_was_not_shown_is_discarded(retriever: Retriever) -> None:
    model = scripted(
        lambda m: [
            {"passage": label_of(m, HOLDING), "quote": HOLDING, "argument": "Wie P99 zeigt."}
        ]
    )
    [claim] = Devil(model, retriever).argue(POSITION).claims
    assert claim.counters == []
    [discarded] = claim.discarded
    assert discarded.reason == "the argument cites a passage it was not given"


def test_a_decision_the_passage_lists_without_prefix_may_be_named() -> None:
    from devils_advocate.devil import _stray_reference
    from devils_advocate.store import StoredPassage

    text = "(ATF 145 I 183 consid. 4.1.2; ATF 143 I 388 consid. 2.2.2; 133 I 259 consid. 2.2)"
    passage = StoredPassage(1, "BGE 151 I 177", "E. 2.1", 180, text, "fr", 2025)
    assert _stray_reference("L'ATF 133 I 259 le confirme.", passage, "") is None
    assert _stray_reference("L'ATF 133 I 25 le confirme.", passage, "") == "BGE 133 I 25"
    assert _stray_reference("L'ATF 259 I 133 le confirme.", passage, "") == "BGE 259 I 133"
