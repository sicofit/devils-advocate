"""Check that a quote the model attributes to a passage is really in that passage.

The check runs on a normalised form of both texts (typographic quotes and
dashes unified, whitespace collapsed, case folded), but what the user sees is
always cut from the original passage, never the model's wording. A quote may
skip text with an ellipsis ("...", "[…]"); every fragment must then appear in
the passage in order.

A quote that is almost but not exactly in the passage (a changed inflection, a
dropped comma) is accepted as "close" when at least `CLOSE_THRESHOLD` of its
characters line up, and the passage's own wording is shown instead.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum

MIN_WORDS = 5
CLOSE_THRESHOLD = 0.92

ELLIPSIS = re.compile(r"\s*(?:\[\s*(?:\.\.\.|…)\s*\]|\(\s*(?:\.\.\.|…)\s*\)|\.\.\.|…)\s*")
TRANSLATE = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "„": '"',
        "“": '"',
        "”": '"',
        "‟": '"',
        "″": '"',
        "‹": "'",
        "›": "'",
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "`": "'",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        " ": " ",
        " ": " ",
        " ": " ",
    }
)
# Spaces the scrape or the model may add or drop around punctuation.
TIGHT_BEFORE = set(",.;:)]!?")
TIGHT_AFTER = set("([")


class Status(StrEnum):
    EXACT = "exact"
    CLOSE = "close"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Verdict:
    status: Status
    text: str  # the passage's own wording of the quote; empty when rejected
    score: float  # share of the quote's characters found in the passage
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is not Status.REJECTED


def normalize(text: str) -> tuple[str, list[int]]:
    """Return the normalised text and, for each of its characters, the index in `text`."""
    out: list[str] = []
    index: list[int] = []
    for position, raw in enumerate(text):
        for char in unicodedata.normalize("NFKC", raw).translate(TRANSLATE):
            if char == "­":  # soft hyphen
                continue
            if char.isspace():
                if not out or out[-1] == " " or out[-1] in TIGHT_AFTER:
                    continue
                out.append(" ")
                index.append(position)
                continue
            if char in TIGHT_BEFORE and out and out[-1] == " ":
                out.pop()
                index.pop()
            lowered = char.lower()
            out.append(lowered if len(lowered) == 1 else char)
            index.append(position)
    if out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def verify(quote: str, passage: str) -> Verdict:
    quote = quote.strip().strip("\"'«»„“”").strip()
    if len(quote.split()) < MIN_WORDS:
        return Verdict(Status.REJECTED, "", 0.0, f"quote shorter than {MIN_WORDS} words")
    fragments = [fragment for fragment in ELLIPSIS.split(quote) if fragment.strip()]
    haystack, index = normalize(passage)

    spans: list[tuple[int, int]] = []
    matched = 0
    total = 0
    exact = True
    cursor = 0
    for fragment in fragments:
        needle, _ = normalize(fragment)
        total += len(needle)
        found = haystack.find(needle, cursor)
        if found >= 0:
            spans.append((found, found + len(needle)))
            matched += len(needle)
            cursor = found + len(needle)
            continue
        exact = False
        span, hits = _closest(haystack, needle, cursor)
        if span is None:
            return Verdict(Status.REJECTED, "", 0.0, "not found in the cited passage")
        spans.append(span)
        matched += hits
        cursor = span[1]

    score = matched / total if total else 0.0
    if not exact and score < CLOSE_THRESHOLD:
        return Verdict(Status.REJECTED, "", score, "not found in the cited passage")
    text = ""
    for start, end in spans:
        piece = _original(passage, index, start, end)
        if text:
            text += " […]" if piece.startswith(tuple(",;.:)]")) else " […] "
        text += piece
    return Verdict(Status.EXACT if exact else Status.CLOSE, text, score)


def _closest(haystack: str, needle: str, cursor: int) -> tuple[tuple[int, int] | None, int]:
    """Find the window of `haystack` after `cursor` that best matches `needle`."""
    region = haystack[cursor:]
    matcher = SequenceMatcher(None, region, needle, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size >= 4]
    if not blocks:
        return None, 0
    # Keep the densest run of blocks: those whose positions in the passage stay
    # within a window not much longer than the quote itself.
    best: tuple[int, int, int] | None = None
    for i, first in enumerate(blocks):
        hits = 0
        for last in blocks[i:]:
            if last.a + last.size - first.a > len(needle) * 1.3:
                break
            hits += last.size
            if best is None or hits > best[2]:
                best = (first.a, last.a + last.size, hits)
    assert best is not None
    start, end, hits = best
    # Widen to whole words so the shown quote never starts or ends mid-word.
    while start > 0 and region[start - 1].isalnum():
        start -= 1
    while end < len(region) and region[end].isalnum():
        end += 1
    return (cursor + start, cursor + end), hits


def _original(passage: str, index: list[int], start: int, end: int) -> str:
    return passage[index[start] : index[end - 1] + 1].strip()
