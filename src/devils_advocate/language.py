"""Which language a position is written in, and which BGE it names.

Both are decided here, without the model: a German position must get a German
answer even when the model's own guess is English, and a BGE the model adds to
a search query or an argument must be one that is really there.
"""

from __future__ import annotations

import re
from collections import Counter

WORD = re.compile(r"[^\W\d_]+")
# Short words that are frequent in one of the four languages and rare in the
# others. Words shared by two of them ("la", "il", "se", "du", "on") are left out,
# and so is "or", which in a German position is the Code of Obligations.
MARKERS = {
    "de": """der die das dem den und ist nicht ein eine einen einem einer kein keine
        mit von für auf auch wenn kann darf muss wird werden sich zu im sind oder nur
        bei nach aus als wer ohne gegen""",
    "fr": """les et est pas une au aux pour sur dans qui que peut doit être sont
        avec ou elle sans contre lorsque même""",
    "it": """lo gli è dei del della dello di per con che può deve essere sono nel
        nella senza contro quando anche""",
    "en": """the and is not of to for can may must be are with if he she it that
        this without against when even""",
}
WORDS = {language: frozenset(words.split()) for language, words in MARKERS.items()}

REFERENCE = re.compile(r"\b(?:BGE|ATF|DTF)\s+(\d{1,3})\s+([IVX]+[a-z]?)\s+(\d{1,4})\b")


def detect(text: str, fallback: str = "en") -> str:
    """The language of `text`: "de", "fr", "it" or "en"; `fallback` when it is unclear."""
    tally: Counter[str] = Counter()
    for word in WORD.findall(text.lower()):
        for language, words in WORDS.items():
            if word in words:
                tally[language] += 1
    ranked = tally.most_common(2)
    if not ranked or (len(ranked) == 2 and ranked[0][1] == ranked[1][1]):
        return fallback if fallback in WORDS else "en"
    return ranked[0][0]


def references(text: str) -> set[str]:
    """The BGE named in `text`, as "BGE 145 III 72"; "ATF" and "DTF" count as "BGE"."""
    return {f"BGE {volume} {part} {page}" for volume, part, page in REFERENCE.findall(text)}


def mentions(text: str, reference: str) -> bool:
    """Whether `text` names `reference`, also without "BGE" before it.

    After one prefix the court often lists more decisions without it:
    "ATF 143 I 388 consid. 2.2.2; 133 I 259 consid. 2.2".
    """
    volume, part, page = reference.split()[1:]
    return re.search(rf"(?<![\d/.])\b{volume}\s+{part}\s+{page}\b(?![./]\d)", text) is not None


def without_references(text: str, keep: set[str]) -> str:
    """`text` with every BGE reference removed that is not in `keep`."""

    def replace(match: re.Match[str]) -> str:
        reference = f"BGE {match.group(1)} {match.group(2)} {match.group(3)}"
        return match.group(0) if reference in keep else ""

    return " ".join(REFERENCE.sub(replace, text).split())
