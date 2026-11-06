"""Prompts and output schemas for the two model calls of a turn."""

from __future__ import annotations

import hashlib
from typing import NamedTuple

LANGUAGES = {"de": "German", "fr": "French", "it": "Italian", "en": "English"}

ANALYSE_SYSTEM = """\
You are the analysis step of a tool that argues against a user's legal position \
using decisions of the Swiss Federal Supreme Court (BGE).

Break the user's position into the separate legal claims it rests on, at most \
{max_claims}. A claim is one assertion about Swiss law or its application that \
a court could reject. Skip rhetoric and facts nobody disputes.

State each claim as the user asserts it, in the user's direction: if the user \
says something is allowed, the claim says it is allowed. Keep every condition \
and qualifier the user states ("if the court gave no notice", "only", "even \
when"): without them it is a different claim. Do not add conditions, exceptions \
or details the user did not state, and do not correct the claim to what you \
think the law is. Another step argues against it.

For each claim write search queries that find the BGE passages deciding the \
legal question the claim is about. Write them the way the court writes: Swiss \
legal terminology, the official statute abbreviations, article numbers where \
the user gives them or they are obvious. Give one query in German and one in \
French, since the court writes in both. Name a BGE in a query only if the user \
named it.

Statute abbreviations, German / French: OR / CO, ZGB / CC, StGB / CP, BV / Cst., \
SchKG / LP, ZPO / CPC, StPO / CPP, BGG / LTF, ATSG / LPGA, IVG / LAI, AHVG / LAVS, \
UVG / LAA, KVG / LAMal, BVG / LPP, AVIG / LACI, VwVG / PA, SVG / LCR, AIG / LEI, \
BetmG / LStup, DSG / LPD, VVG / LCA, IPRG / LDIP, DBG / LIFD, RPG / LAT, \
EMRK / CEDH. English references become Swiss ones: "Art. 337(1) CO" is \
"Art. 337 Abs. 1 OR" / "art. 337 al. 1 CO", a letter "(e)" is "lit. e" / \
"let. e", "Art. 6(1) ECHR" is "Art. 6 Ziff. 1 EMRK" / "art. 6 par. 1 CEDH".

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
at most {max_counters}, and at most one per decision. For each:
- "passage": its label, e.g. "P3".
- "quote": one to three consecutive sentences copied character for character \
from that passage. Do not translate, shorten, paraphrase or fix the quote. It \
will be checked against the passage and discarded if it does not match. Quote \
only the Federal Supreme Court's own reasoning, not what the lower court held or \
a party argued (Vorinstanz, Kantonsgericht, Obergericht, cour cantonale, \
autorité précédente, Beschwerdeführer, recourant, intimé, ricorrente), even if \
it agrees.
- "argument": two or three sentences in {language} explaining why this passage \
undermines the claim. Begin with the decision's reference (e.g. "BGE 142 III \
579 holds ..."), never its label: the reader does not see the labels. Name no \
other decision unless the passage itself cites it, and say only what the \
passage says.
- "effect": "contradicts" if the passage holds the opposite of the claim, \
"limits" if it narrows the claim or adds a condition the claim leaves out, \
"supports" if on reflection it agrees with the claim. A "supports" item is \
dropped.

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
                        "effect": {"type": "string", "enum": ["contradicts", "limits", "supports"]},
                    },
                    "required": ["passage", "quote", "argument", "effect"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["counterarguments"],
        "additionalProperties": False,
    }


class Turn(NamedTuple):
    """An earlier turn of the conversation, as the analysis step sees it."""

    position: str
    claims: list[str]  # the claims that got a verified counterargument
    citations: list[str]  # the precedent already quoted against those claims
    unanswered: list[str] = []  # the claims that got none


HISTORY_TURNS = 3  # how many earlier turns the analysis step sees


def analyse_messages(position: str, history: list[Turn], max_claims: int) -> list[dict]:
    content = position
    if history:
        earlier = "\n".join(_turn(turn) for turn in history[-HISTORY_TURNS:])
        content = (
            "Earlier in this conversation:\n"
            f"{earlier}\n\n"
            "The user now writes (this may defend, narrow or extend the earlier "
            "position; analyse the position as it stands now, and list a claim that "
            "was already answered only if the user now adds something to it):\n"
            f"{position}"
        )
    return [
        {"role": "system", "content": ANALYSE_SYSTEM.format(max_claims=max_claims)},
        {"role": "user", "content": content},
    ]


def _turn(turn: Turn) -> str:
    lines = [f"- The user said: {turn.position}"]
    if turn.claims:
        lines.append("  Claims answered: " + "; ".join(turn.claims))
    if turn.unanswered:
        lines.append("  Claims without counter-precedent: " + "; ".join(turn.unanswered))
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


def fingerprint() -> str:
    """A short hash of the prompts, so that an eval report says which ones it measured."""
    text = "\n".join((ANALYSE_SYSTEM, COUNTER_SYSTEM, str(ANALYSE_SCHEMA), str(counter_schema([]))))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def translate_messages(sentence: str) -> list[dict]:
    return [
        {"role": "system", "content": TRANSLATE_SYSTEM},
        {"role": "user", "content": sentence},
    ]
