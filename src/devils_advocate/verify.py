"""Check that a quote the model attributes to a passage is really in that passage.

The check runs on a normalised form of both texts (quotation marks dropped,
dashes unified, whitespace collapsed, case folded), but what the user sees is
always cut from the original passage, never the model's wording. Matches
respect word boundaries: "zulässig" is not found inside "unzulässig".

A quote may skip text with an ellipsis ("...", "[…]"). Every fragment must then
appear in the passage in order, have at least `MIN_FRAGMENT_WORDS` words, and
the skipped text must not contain a negation: "ist […] gerechtfertigt" is not a
quote of "ist nicht gerechtfertigt".

A quote that is almost but not exactly in the passage (a changed inflection, a
dropped comma) is accepted as "close" when at least `CLOSE_THRESHOLD` of the
characters of quote and passage line up and no word was added, dropped or
swapped for a different one. The passage's own wording is shown instead.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum

MIN_WORDS = 5
MIN_FRAGMENT_WORDS = 3
CLOSE_THRESHOLD = 0.92

ELLIPSIS = re.compile(r"\s*(?:\[\s*(?:\.\.\.|…)\s*\]|\(\s*(?:\.\.\.|…)\s*\)|\.\.\.|…)\s*")
WORD = re.compile(r"\w+")
TRANSLATE = str.maketrans(
    {
        # Quotation marks all become '"', which normalize() then drops: whether
        # and how a quotation is marked ("…", «…», « … ») does not change what it says.
        "«": '"',
        "»": '"',
        "„": '"',
        "“": '"',
        "”": '"',
        "‟": '"',
        "‹": '"',
        "›": '"',
        # Apostrophes stay, they hold words together ("l'art.").
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "`": "'",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)
INVISIBLE = {"\u00ad", "\u200b", "\u2060", "\ufeff"}  # soft hyphen, zero-width space, ...
QUOTE_MARKS = set('"«»„“”‟‹›')
# Spaces the scrape or the model may add or drop around punctuation.
TIGHT_BEFORE = set(",.;:)]!?")
TIGHT_AFTER = set("([")
# Words whose omission turns a statement into its opposite.
NEGATIONS = frozenset(
    """
    nicht kein keine keinem keinen keiner keines nie niemals weder nirgends
    ne n pas non aucun aucune jamais ni nullement
    nessun nessuno nessuna mai né
    not no never
    """.split()
)


class Status(StrEnum):
    EXACT = "exact"
    CLOSE = "close"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class Verdict:
    status: Status
    text: str  # the passage's own wording of the quote; empty when rejected
    score: float  # how closely quote and passage line up, 1.0 for an exact match
    reason: str = ""
    spans: tuple[tuple[int, int], ...] = ()  # where in the passage the quoted text is

    @property
    def ok(self) -> bool:
        return self.status is not Status.REJECTED


def normalize(text: str) -> tuple[str, list[int]]:
    """Return the normalised text and, for each of its characters, the index in `text`."""
    out: list[str] = []
    index: list[int] = []
    for position, raw in enumerate(text):
        if unicodedata.combining(raw) and out:
            # A decomposed accent ("e" + U+0301): fold it into the letter before it.
            composed = unicodedata.normalize("NFC", out[-1] + raw)
            if len(composed) == 1:
                out[-1] = composed.lower()
            continue
        for char in unicodedata.normalize("NFKC", raw).translate(TRANSLATE):
            if char in INVISIBLE or char == '"':
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
    quote = _unwrap(quote)
    fragments = [fragment for fragment in ELLIPSIS.split(quote) if fragment.strip()]
    words = [len(WORD.findall(fragment)) for fragment in fragments]
    if sum(words) < MIN_WORDS:
        return _rejected(f"quote shorter than {MIN_WORDS} words")
    if min(words) < MIN_FRAGMENT_WORDS:
        return _rejected(f"a part between ellipses is shorter than {MIN_FRAGMENT_WORDS} words")
    haystack, index = normalize(passage)

    spans: list[tuple[int, int]] = []
    matched = 0
    compared = 0
    exact = True
    cursor = 0
    for fragment in fragments:
        needle, _ = normalize(fragment)
        found = _find_words(haystack, needle, cursor)
        if found >= 0:
            span = (found, found + len(needle))
            hits = len(needle)
        else:
            exact = False
            closest, hits = _closest(haystack, needle, cursor)
            if closest is None:
                return _rejected("not found in the cited passage")
            span = closest
            if 2 * hits / (len(needle) + span[1] - span[0]) < CLOSE_THRESHOLD:
                return _rejected("not found in the cited passage")
            if not _same_words(needle, haystack[span[0] : span[1]]):
                return _rejected("changes a word of the cited passage")
        if spans and _negation_between(haystack[spans[-1][1] : span[0]]):
            return _rejected("the ellipsis leaves out a negation")
        spans.append(span)
        matched += hits
        compared += len(needle) + span[1] - span[0]
        cursor = span[1]

    # Every fragment reached CLOSE_THRESHOLD, so their weighted mean does too.
    score = 2 * matched / compared
    originals = tuple(
        _original(passage, index, start, end, fragment.strip())
        for (start, end), fragment in zip(spans, fragments, strict=True)
    )
    text = ""
    for start, end in originals:
        piece = passage[start:end]
        if text:
            text += " […]" if piece.startswith(tuple(",;.:)]")) else " […] "
        text += piece
    return Verdict(Status.EXACT if exact else Status.CLOSE, text, score, spans=originals)


def _rejected(reason: str) -> Verdict:
    return Verdict(Status.REJECTED, "", 0.0, reason)


def _unwrap(quote: str) -> str:
    """Strip whitespace and quotation marks that enclose the whole quote, but no others."""
    quote = quote.strip()
    while len(quote) >= 2 and quote[0] in QUOTE_MARKS and quote[-1] in QUOTE_MARKS:
        inner = quote[1:-1]
        if any(mark in inner for mark in QUOTE_MARKS):
            break  # '"a" und "b"': the marks belong to the text
        quote = inner.strip()
    return quote


def _find_words(haystack: str, needle: str, start: int) -> int:
    """`haystack.find(needle, start)`, but only at word boundaries."""
    if not needle:
        return -1
    while (found := haystack.find(needle, start)) >= 0:
        end = found + len(needle)
        cut_before = needle[0].isalnum() and found > 0 and haystack[found - 1].isalnum()
        cut_after = needle[-1].isalnum() and end < len(haystack) and haystack[end].isalnum()
        if not cut_before and not cut_after:
            return found
        start = found + 1
    return -1


def _closest(haystack: str, needle: str, cursor: int) -> tuple[tuple[int, int] | None, int]:
    """Find the window of `haystack` after `cursor` that best matches `needle`."""
    region = haystack[cursor:]
    if not needle:
        return None, 0
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


def _same_words(needle: str, window: str) -> bool:
    """Whether `window` says what `needle` says, up to inflection and punctuation.

    Every word must have a counterpart: no word added or dropped ("nicht"), no
    number changed, and a changed word must keep its stem ("schwerer" for
    "schweren", but not "unzulässig" for "zulässig").
    """
    quoted, source = WORD.findall(needle), WORD.findall(window)
    matcher = SequenceMatcher(None, quoted, source, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or i2 - i1 != j2 - j1:
            return False
        for a, b in zip(quoted[i1:i2], source[j1:j2], strict=True):
            if a.isdigit() or b.isdigit() or a[:3] != b[:3]:
                return False
            if SequenceMatcher(None, a, b).ratio() < 0.75:
                return False
    return True


def _negation_between(skipped: str) -> bool:
    return any(word in NEGATIONS for word in WORD.findall(skipped))


def _original(
    passage: str, index: list[int], start: int, end: int, fragment: str
) -> tuple[int, int]:
    """The span of `passage` behind normalised positions `start`..`end`.

    Normalising dropped the quotation marks, so they are put back: those the
    quoted `fragment` begins or ends with, and one more if the span would
    otherwise open a quotation without closing it, or the other way round.
    """
    first, last = index[start], index[end - 1] + 1
    while first < last and passage[first].isspace():
        first += 1
    while last > first and passage[last - 1].isspace():
        last -= 1
    if fragment[:1] in QUOTE_MARKS and first > 0 and passage[first - 1] in QUOTE_MARKS:
        first -= 1
    if fragment[-1:] in QUOTE_MARKS and last < len(passage) and passage[last] in QUOTE_MARKS:
        last += 1
    if sum(passage[i] in QUOTE_MARKS for i in range(first, last)) % 2:
        if last < len(passage) and passage[last] in QUOTE_MARKS:
            last += 1
        elif first > 0 and passage[first - 1] in QUOTE_MARKS:
            first -= 1
    return first, last
