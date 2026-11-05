from __future__ import annotations

from collections.abc import Callable

from devils_advocate.devil import MAX_COUNTERS, Devil
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


def test_at_most_max_counters(retriever: Retriever) -> None:
    def everything(m: list[dict]) -> list[dict]:
        return [
            {"passage": label, "quote": text[:200], "argument": "a"}
            for label, text in passages_in(m).items()
            if len(text.split()) >= 5
        ]

    [claim] = Devil(scripted(everything), retriever).argue(POSITION).claims
    assert len(claim.counters) == MAX_COUNTERS


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
