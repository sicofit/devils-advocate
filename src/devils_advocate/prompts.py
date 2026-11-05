"""Prompts and output schemas for the two model calls of a turn."""

from __future__ import annotations

from typing import NamedTuple

LANGUAGES = {"de": "German", "fr": "French", "it": "Italian", "en": "English"}

ANALYSE_SYSTEM = """\
You are the analysis step of a tool that argues against a user's legal position \
using decisions of the Swiss Federal Supreme Court (BGE).

Break the user's position into the separate legal claims it rests on, at most \
{max_claims}. A claim is one assertion about Swiss law or its application that \
a court could reject. Skip rhetoric and facts nobody disputes.

For each claim write search queries that would find BGE passages taking the \
OPPOSITE view. Write them the way the court writes: Swiss legal terminology, \
statute abbreviations (OR, ZGB, StGB, BV, SchKG, ...), article numbers where they \
are obvious. Give one query in German and one in French, since the court writes \
in both.

Set "language" to the language the user wrote in (de, fr, it or en).
If the input contains no legal claim at all, return an empty "claims" list."""

ANALYSE_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {"type": "string", "enum": ["de", "fr", "it", "en"]},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "queries": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["claim", "queries"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["language", "claims"],
    "additionalProperties": False,
}

COUNTER_SYSTEM = """\
You are a Swiss litigator assigned to argue against the claim below. You may \
only use the numbered BGE passages provided. You have no other sources.

Pick the passages that contradict, limit or qualify the claim, strongest first, \
at most {max_counters}. For each:
- "passage": its label, e.g. "P3".
- "quote": one to three consecutive sentences copied character for character \
from that passage. Do not translate, shorten, paraphrase or fix the quote. It \
will be checked against the passage and discarded if it does not match.
- "argument": two or three sentences in {language} explaining why this passage \
undermines the claim. Name the decision by its reference (e.g. "BGE 142 III \
579"), never by its label: the reader does not see the labels.

A passage that merely mentions the same statute is not a counterargument. If no \
passage argues against the claim, return an empty list. An empty list is a \
correct answer; an invented one is not."""


def counter_schema(labels: list[str]) -> dict:
    """The schema of the counterarguments reply; "passage" must be one of `labels`."""
    passage = {"type": "string", "enum": labels} if labels else {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "counterarguments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "passage": passage,
                        "quote": {"type": "string"},
                        "argument": {"type": "string"},
                    },
                    "required": ["passage", "quote", "argument"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["counterarguments"],
        "additionalProperties": False,
    }


COUNTER_SCHEMA = counter_schema([])


class Turn(NamedTuple):
    """An earlier turn of the conversation, as the analysis step sees it."""

    argument: str
    claims: list[str]
    citations: list[str]  # the precedent already quoted against those claims


def analyse_messages(argument: str, history: list[Turn], max_claims: int) -> list[dict]:
    content = argument
    if history:
        earlier = "\n".join(_turn(turn) for turn in history[-3:])
        content = (
            "Earlier in this conversation:\n"
            f"{earlier}\n\n"
            "The user now writes (this may defend, narrow or extend the earlier "
            "position; analyse the position as it stands now, and list a claim that "
            "was already answered only if the user now adds something to it):\n"
            f"{argument}"
        )
    return [
        {"role": "system", "content": ANALYSE_SYSTEM.format(max_claims=max_claims)},
        {"role": "user", "content": content},
    ]


def _turn(turn: Turn) -> str:
    lines = [f"- The user said: {turn.argument}"]
    if turn.claims:
        lines.append("  Claims answered: " + "; ".join(turn.claims))
    if turn.citations:
        lines.append("  Precedent already cited: " + "; ".join(turn.citations))
    return "\n".join(lines)


def counter_messages(
    claim: str, passages: list[tuple[str, str, str]], language: str, max_counters: int
) -> list[dict]:
    """`passages` are (label, citation, text) triples."""
    blocks = "\n\n".join(f"[{label}] {citation}\n{text}" for label, citation, text in passages)
    system = COUNTER_SYSTEM.format(
        max_counters=max_counters, language=LANGUAGES.get(language, "English")
    )
    user = f"Claim to argue against:\n{claim}\n\nPassages:\n\n{blocks}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# Used by the retrieval benchmark to measure what the bilingual queries of the
# analysis step are worth, without the rest of the pipeline.
TRANSLATE_SYSTEM = """\
Translate the sentence from a Swiss Federal Supreme Court decision into German \
and into French. Use the terminology the court itself uses in each language \
(for example "Rechtsöffnung" / "mainlevée", "Erwägung" / "considérant"). If the \
sentence already is in one of the two languages, return it unchanged for that \
language."""

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {"de": {"type": "string"}, "fr": {"type": "string"}},
    "required": ["de", "fr"],
    "additionalProperties": False,
}


def translate_messages(sentence: str) -> list[dict]:
    return [
        {"role": "system", "content": TRANSLATE_SYSTEM},
        {"role": "user", "content": sentence},
    ]
