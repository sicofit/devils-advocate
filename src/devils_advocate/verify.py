"""Check that a quote the model attributes to a passage is really in that passage.

The check runs on a normalised form of both texts (quotation marks dropped,
dashes unified, whitespace collapsed, case folded), but what the user sees is
always cut from the original passage, never the model's wording. Matches
respect word boundaries: "zulässig" is not found inside "unzulässig".

A quote may skip text with an ellipsis ("...", "[…]"). Every fragment must then
appear in the passage in order and have at least `MIN_FRAGMENT_WORDS` words, and
the skipped text must not contain a negation or a qualifier: "ist […]
gerechtfertigt" is not a quote of "ist nicht gerechtfertigt", nor "ist […]
zulässig" of "ist nur ausnahmsweise zulässig". A quote that begins inside a
sentence must not leave out a denial it depends on: "X" is not a quote of "Es
trifft nicht zu, dass X".

A quote that is almost but not exactly in the passage (a changed inflection, a
dropped comma) is accepted as "close" when at least `CLOSE_THRESHOLD` of the
characters of quote and passage line up and no word was added, dropped or
swapped for a different one. The passage's own wording is shown instead.

Courts restate what the lower court held or a party argued before they answer
it. A quote from such a restatement is marked `reported`, and when the
attribution is in the quote's own sentence, the shown text starts with it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum

from .segment import SENTENCE_BREAK, ends_with_abbreviation, is_date

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
        "‚": '"',
        "‛": '"',
        # Apostrophes stay, they hold words together ("l'art."). A curly one that
        # does not stand between two letters closes a quotation (‚…‘) and is dropped.
        "‘": "'",
        "’": "'",
        "′": "'",
        "`": "'",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
    }
)
INVISIBLE = {"­", "​", "⁠", "﻿"}  # soft hyphen, zero-width space, ...
QUOTE_MARKS = set('"«»„“”‟‹›‚‛‘')
CURLY_APOSTROPHES = {"‘", "’"}
# Spaces the scrape or the model may add or drop around punctuation.
TIGHT_BEFORE = set(",.;:)]!?")
TIGHT_AFTER = set("([")
# "Abs. 4bis", "art. 4 quater", "1er janvier": the scrape often splits a raised
# suffix from its number. Not before a number: "Art. 4 bis 6" is a range.
SUFFIX_SPACE = re.compile(
    r"(?<![\d.])(?:\d{1,3}) (?=(?:bis|ter|quater|quinquies|sexies|septies|octies)\b"
    r"(?! ?(?:\d|zu\b|zum\b|zur\b|au\b|à\b|a\b)))"
    r"|(?<![\d.])1 (?=(?:er|re)\b)"
)
# Words whose omission turns a statement into its opposite.
NEGATIONS = frozenset(
    """
    nicht kein keine keinem keinen keiner keines keinerlei nie niemals nimmer weder
    nirgends keineswegs keinesfalls nichts niemand niemandem niemanden kaum mitnichten ohne
    ne pas non aucun aucune aucunement jamais ni nullement rien guère sans
    nessun nessuno nessuna mai né nulla niente nemmeno neppure neanche senza mica
    not no never nothing none nor without
    """.split()
)
# These negate only in some languages: "Mai" is a month in German and French,
# "ne" a pronoun in Italian, "né" means "born" in French, "No" a number ("N° 7").
NEGATES_IN = {"mai": {"it"}, "ne": {"fr"}, "né": {"it"}, "no": {"it", "en"}}
ELIDED_NEGATION = re.compile(r"\bn'")  # French "n'est", "n'a"
NEGATING_PHRASE = re.compile(r"\b(?:zu unrecht|à tort|a torto|wrongly)\b")
# Words that limit a statement; leaving one out overstates it.
QUALIFIERS = frozenset(
    """
    nur lediglich bloss einzig ausschliesslich grundsätzlich ausnahmsweise allenfalls
    seulement uniquement exclusivement exceptionnellement
    solo soltanto unicamente esclusivamente eccezionalmente
    only solely merely
    """.split()
)
QUALIFYING_PHRASE = re.compile(
    r"\b(?:in der regel|en principe|en règle générale|in linea di principio|di regola|"
    r"in principle|as a rule)\b"
)
# A numbered item that opens a sentence: "(6) ", "b) ", "2. ".
ITEM = re.compile(r"(?:^|[.!?;:]\s+)(?:\(\w{1,3}\)|\w{1,2}\)|\d{1,2}\.)\s*$")
# A clause that the quote's start depends on: "Es trifft nicht zu, dass ...".
COMPLEMENTIZER = re.compile(r"\b(?:dass|ob|inwiefern|que|qu'?|en quoi|che|se|that|whether)$")
# ... and the same word when the quote itself begins with it: "dass X".
LEADING_COMPLEMENTIZER = re.compile(r"(?:dass|ob|inwiefern|que|qu'|en quoi|che|se|that|whether)\b")

# Who the court reports, and the verbs it reports them with.
SPEAKER = re.compile(
    r"\b(?:vorinstanz\w*|kantonsgericht\w*|obergericht\w*|verwaltungsgericht\w*|"
    r"appellationsgericht\w*|kantonale\w* gericht\w*|beschwerdeführer\w*|beschwerdegegner\w*|"
    r"gesuchsteller\w*|kläger\w*|beklagte\w*|gemeinde|cour cantonale|"
    r"autorité (?:précédente|cantonale|inférieure|intimée)|instance (?:précédente|cantonale|"
    r"inférieure)|juridiction cantonale|tribunal cantonal|recourant\w*|intimé\w*|"
    r"corte cantonale|autorità (?:precedente|cantonale|inferiore)|istanza precedente|"
    r"tribunale cantonale|ricorrent\w*|resistent\w*)"
)
REPORTING = re.compile(
    r"\b(?:erwog|erwogen|erwägt|gelangt\w*|schluss|geltend|vorgebracht|vorbringt|"
    r"bringt\w* (?:\w+ )?vor|rügt\w*|gerügt\w*|beanstandet\w*|argumentiert\w*|hielt\w* fest|"
    r"festgehalten|befand|befunden|ansicht|auffassung|standpunkt|wendet\w* ein|"
    r"wandte\w* ein|"
    r"eingewendet|behauptet\w*|fait valoir|faisait valoir|soutient|soutenait|soutenu|estime|"
    r"estimait|invoque\w*|allègue\w*|reproche\w*|considéré|considérait|retenu|retenait|jugé|"
    r"jugeait|fa valere|faceva valere|sostiene|sosteneva|ritiene|riteneva|lamenta\w*|"
    r"censura\w*|considerato|ritenuto)\b"
)
NEAR = 10  # words between speaker and verb
SUBJECT = 2  # a speaker this close to the sentence start is its subject
# "wie der Beschwerdeführer geltend macht": the court refers to the view, it does
# not restate it.
REFERRED = re.compile(r"\b(?:wie|comme|come|as)\W+(?:\w+\W+){0,2}$")
# The court answering in the same sentence: "Entgegen der Vorinstanz ...".
COURT_IN_SENTENCE = re.compile(
    r"\b(?:entgegen|zu unrecht|zu recht|zutreffend|nicht gefolgt|nicht zu folgen|"
    r"nach dem gesagten|vielmehr|das bundesgericht|contrairement|à tort|à juste titre|"
    r"le tribunal fédéral|il tribunale federale|contrariamente|a torto|a ragione)\b"
)
# A sentence with which the court turns to its own answer.
COURT_TURN = re.compile(
    r"^(?:indes|indessen|dem ist nicht so|dies trifft (?:nicht )?zu|das trifft (?:nicht )?zu|"
    r"or|toutefois|cependant|il est constant|ce n'est pas le cas|tuttavia)\b"
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
    opens_inside: bool = False  # the quoted text starts inside a sentence
    ends_inside: bool = False  # ... or stops before its end
    reported: bool = False  # it restates the lower court's or a party's view

    @property
    def ok(self) -> bool:
        return self.status is not Status.REJECTED

    @property
    def shown(self) -> str:
        """The text as shown to the reader: a cut sentence is marked "[…]"."""
        if not self.text:
            return ""
        return ("[…] " if self.opens_inside else "") + self.text + (" […]" * self.ends_inside)


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
        if raw in CURLY_APOSTROPHES and not (
            out and out[-1].isalnum() and text[position + 1 : position + 2].isalnum()
        ):
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
            # casefold, not lower: "ß" is "ss", as the Swiss court writes it.
            for folded in char.casefold():
                out.append(folded)
                index.append(position)
    if out and out[-1] == " ":
        out.pop()
        index.pop()
    joined = "".join(out)
    dropped = {m.end() - 1 for m in SUFFIX_SPACE.finditer(joined)}
    if dropped:
        joined = "".join(c for i, c in enumerate(joined) if i not in dropped)
        index = [p for i, p in enumerate(index) if i not in dropped]
    return joined, index


def verify(quote: str, passage: str, language: str | None = None) -> Verdict:
    """Check `quote` against `passage`, written in `language` ("de", "fr", "it").

    Without a language, every word that negates in any of them counts as a negation.
    """
    quote = _unwrap(quote)
    whole = [quote] if quote.strip() else []
    if sum(len(WORD.findall(part)) for part in whole) < MIN_WORDS:
        return _rejected(f"quote shorter than {MIN_WORDS} words")
    haystack, index = normalize(passage)

    # The whole quote first: the court's own text may contain "(...)".
    needle, _ = normalize(quote)
    found = _find_words(haystack, needle, 0) if ELLIPSIS.search(quote) else -1
    if found >= 0:
        fragments = whole
        spans = [(found, found + len(needle))]
        exact, score = True, 1.0
    else:
        fragments = [fragment for fragment in ELLIPSIS.split(quote) if fragment.strip()]
        if min(len(WORD.findall(fragment)) for fragment in fragments) < MIN_FRAGMENT_WORDS:
            return _rejected(f"a part between ellipses is shorter than {MIN_FRAGMENT_WORDS} words")
        matched = _match(passage, haystack, index, fragments, language)
        if isinstance(matched, Verdict):
            return matched
        spans, exact, score = matched

    head = _sentence_head(passage, index[spans[0][0]])
    lead = LEADING_COMPLEMENTIZER.match(normalize(quote)[0])
    if head and _denied(head + (lead.group(0) if lead else ""), language):
        return _rejected("the quote leaves out the denial its sentence starts with")
    originals = [
        _original(passage, index, start, end, fragment.strip())
        for (start, end), fragment in zip(spans, fragments, strict=True)
    ]
    reported = _reported(passage, originals[0][0])
    if head and _reporting(head):
        # Show who said it: the attribution is part of the court's sentence.
        originals[0] = (index[spans[0][0]] - len(head), originals[0][1])
        head = ""
    text = ""
    for start, end in originals:
        piece = passage[start:end]
        if text:
            text += " […]" if piece.startswith(tuple(",;.:)]")) else " […] "
        text += piece
    return Verdict(
        Status.EXACT if exact else Status.CLOSE,
        text,
        score,
        spans=tuple(originals),
        opens_inside=bool(head.strip()),
        ends_inside=not _ends_sentence(passage, originals[-1][1]),
        reported=reported,
    )


def _match(
    passage: str, haystack: str, index: list[int], fragments: list[str], language: str | None
) -> tuple[list[tuple[int, int]], bool, float] | Verdict:
    """Find the fragments of a quote in order: their spans, whether all are exact, the score."""
    starts = _sentence_starts(passage)
    starts += [i + 1 for i, char in enumerate(passage) if char == "\n"]

    def gap(a: tuple[int, int], b: tuple[int, int]) -> str:
        return _omission(passage, index[a[1] - 1] + 1, index[b[0]], starts, language)

    spans: list[tuple[int, int]] = []
    needles: list[str | None] = []  # each fragment's needle, None for a close match
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
            needles.append(needle)
        else:
            needles.append(None)
            exact = False
            closest, hits = _closest(haystack, needle, cursor)
            if closest is None:
                return _rejected("not found in the cited passage")
            span = closest
            if 2 * hits / (len(needle) + span[1] - span[0]) < CLOSE_THRESHOLD:
                return _rejected("not found in the cited passage")
            if not _same_words(needle, haystack[span[0] : span[1]]):
                return _rejected("changes a word of the cited passage")
        if spans:
            omitted = gap(spans[-1], span)
            previous = needles[-2]
            if omitted and previous is not None:
                # The fragment before may occur again, closer to this one.
                later = _find_words(haystack, previous, spans[-1][0] + 1)
                while later >= 0 and later + len(previous) <= span[0]:
                    moved = (later, later + len(previous))
                    if not gap(moved, span) and (len(spans) < 2 or not gap(spans[-2], moved)):
                        spans[-1], omitted = moved, ""
                        break
                    later = _find_words(haystack, previous, later + 1)
            if omitted:
                return _rejected(f"the ellipsis leaves out {omitted}")
        spans.append(span)
        matched += hits
        compared += len(needle) + span[1] - span[0]
        cursor = span[1]
    # Every fragment reached CLOSE_THRESHOLD, so their weighted mean does too.
    return spans, exact, 2 * matched / compared


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
    number or reference changed ("257d" for "257a"), and a changed word may
    differ only in an inflected ending: "schwerer" for "schweren", but not
    "unzulässig" for "zulässig", "Arbeitsfähigkeit" for "Arbeitsunfähigkeit" or
    "Beschwerdegegner" for "Beschwerdeführer".
    """
    quoted, source = WORD.findall(needle), WORD.findall(window)
    if not quoted or len(quoted) != len(source):
        return False
    matcher = SequenceMatcher(None, quoted, source, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace" or i2 - i1 != j2 - j1:
            return False
        for position, a, b in zip(range(i1, i2), quoted[i1:i2], source[j1:j2], strict=True):
            last = position == len(quoted) - 1
            if not (_same_stem(a, b) or (last and _clipped(a, b))):
                return False
    return True


# Inflected endings in German, French and Italian. A word may lose at most one.
ENDINGS = sorted(
    {"e", "en", "em", "er", "es", "n", "s", "ns", "nen", "ern", "t", "et", "te", "ten", "st"}
    | {"x", "a", "i", "o"}
)
MIN_STEM = 4
# Articles may change with the case of the noun: "dem Arbeitnehmer" for "den Arbeitnehmer".
ARTICLES = (
    frozenset("der die das den dem des".split()),
    frozenset("ein eine einen einem einer eines".split()),
    frozenset("le la les".split()),
    frozenset("il lo la i gli le".split()),
)
MIN_CLIPPED = 4  # characters a clipped last word must keep: "missbräuch" for "missbräuchlich"


def _same_stem(a: str, b: str) -> bool:
    """Whether two different words are inflections of one word."""
    if _digits(a + b) or a in NEGATIONS or b in NEGATIONS:
        return False  # "pas" for "par" drops a negation
    if any(a in group and b in group for group in ARTICLES):
        return True
    return bool(_stems(a) & _stems(b))


def _stems(word: str) -> set[str]:
    stems = {word}
    for ending in ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= MIN_STEM:
            stems.add(word.removesuffix(ending))
    if word.endswith("aux") and len(word) > MIN_STEM + 1:
        stems.add(word[:-3] + "al")  # "légaux" for "légal"
    return {stem for stem in stems if len(stem) >= MIN_STEM}


def _clipped(a: str, b: str) -> bool:
    """Whether the quote's last word `a` is the start of the passage's word `b`.

    The passage's whole word is what is shown. Only the end of a word may be
    clipped: a clipped start could hide a prefix that negates ("zulässig" in
    "unzulässig"), a clipped end cannot.
    """
    return len(a) >= MIN_CLIPPED and b.startswith(a) and not _digits(a + b)


def _digits(text: str) -> bool:
    return any(char.isdigit() for char in text)


def _negates(word: str, language: str | None) -> bool:
    if word in NEGATES_IN:
        return language is None or language in NEGATES_IN[word]
    return word in NEGATIONS


def _negated(text: str, language: str | None) -> bool:
    """Whether the normalised `text` contains a negation."""
    return bool(ELIDED_NEGATION.search(text) or NEGATING_PHRASE.search(text)) or any(
        _negates(word, language) for word in WORD.findall(text)
    )


def _omission(passage: str, a: int, b: int, starts: list[int], language: str | None) -> str:
    """What the text `passage[a:b]` an ellipsis skips must not contain, if it does.

    Only the parts of sentences the quote cuts count: leaving out a whole sentence
    changes no quoted sentence.
    """
    inside = sorted(start for start in starts if a < start <= b)
    pieces = [(a, inside[0]), (inside[-1], b)] if inside else [(a, b)]
    for start, end in pieces:
        skipped = normalize(passage[start:end])[0]
        if _negated(skipped, language):
            return "a negation"
        words = WORD.findall(skipped)
        if QUALIFYING_PHRASE.search(skipped) or any(word in QUALIFIERS for word in words):
            return "a qualifier"
    return ""


def _sentence_starts(text: str) -> list[int]:
    """Where the sentences of a paragraph begin: not after "vgl." or in "vom 31. Mai"."""
    starts = [0]
    for m in SENTENCE_BREAK.finditer(text):
        before = text[: m.start()]
        if (
            not ends_with_abbreviation(before)
            and not is_date(before, text[m.end() :])
            and before.count("(") <= before.count(")")  # not inside a parenthesis
        ):
            starts.append(m.end())
    return starts


def _sentence_head(passage: str, first: int) -> str:
    """The part of the sentence at `first` that comes before it; "" at a sentence start.

    Opening quotation marks and parentheses in front of the quote do not count.
    """
    paragraph = passage.rfind("\n", 0, first) + 1
    end = passage.find("\n", first)
    text = passage[paragraph : len(passage) if end < 0 else end]
    offset = first - paragraph
    start = max(s for s in _sentence_starts(text) if s <= offset)
    head = text[start:offset]
    bare = head.rstrip(" («„“\"'‹")
    if not bare.strip(" -–—") or bare.endswith(":") or text[offset : offset + 2] in ("- ", "– "):
        return ""  # a sentence start, "... wie folgt: Die Kündigung ...", or a list item
    if ITEM.search(head):
        return ""  # "... veröffentlicht. (6) In den ..."
    return head


def _denied(head: str, language: str | None) -> bool:
    """Whether `head` denies what follows it.

    "Es trifft nicht zu, dass ...", "On ne saurait admettre que ...".
    """
    text = normalize(head)[0].rstrip(" ,")
    found = COMPLEMENTIZER.search(text)
    if not found:
        return False
    clause = re.split(r"[,;:]", text[: found.start()].rstrip(" ,"))[-1]
    return _negated(clause, language)


def _ends_sentence(passage: str, last: int) -> bool:
    """Whether the quoted text that ends at `last` ends its sentence."""
    quoted = passage[:last].rstrip(" )»“”\"'›")
    if (
        quoted.endswith((".", "!", "?", ":"))
        and not ends_with_abbreviation(quoted)
        and not is_date(quoted, passage[last:].lstrip())
    ):
        return True
    rest = passage[last:].lstrip(" )»“”\"'›")
    return not rest or rest[0] in ".!?:\n" or passage[last:].lstrip(" ").startswith("\n")


def _reporting(sentence: str) -> bool:
    """Whether `sentence` restates what the lower court held or a party argued."""
    text = normalize(sentence)[0]
    if COURT_IN_SENTENCE.search(text):
        return False
    speakers = [m.start() for m in SPEAKER.finditer(text) if not REFERRED.search(text[: m.start()])]
    for verb in REPORTING.finditer(text):
        for speaker in speakers:
            if len(WORD.findall(text[:speaker])) <= SUBJECT:
                return True  # "Die Vorinstanz hat [...] festgehalten"
            a, b = sorted((speaker, verb.start()))
            if len(WORD.findall(text[a:b])) <= NEAR:
                return True
    return False


def _reported(passage: str, first: int) -> bool:
    """Whether the text at `first` is part of a restated view.

    It is when its own sentence restates one, or when the first sentence of its
    paragraph does and the court has not turned to its own answer since.
    """
    paragraph = passage.rfind("\n", 0, first) + 1
    end = passage.find("\n", first)
    text = passage[paragraph : len(passage) if end < 0 else end]
    starts = _sentence_starts(text)
    bounds = [*starts[1:], len(text)]
    offset = first - paragraph
    current = max(i for i, start in enumerate(starts) if start <= offset)
    if _reporting(text[starts[current] : min(bounds[current], offset + 200)]):
        return True
    if current == 0 or not _reporting(text[: bounds[0]]):
        return False
    for start, stop in zip(starts[1 : current + 1], bounds[1 : current + 1], strict=True):
        sentence = normalize(text[start:stop])[0]
        if COURT_TURN.search(sentence) or COURT_IN_SENTENCE.search(sentence):
            return False
    return True


def _original(
    passage: str, index: list[int], start: int, end: int, fragment: str
) -> tuple[int, int]:
    """The span of `passage` behind normalised positions `start`..`end`.

    Normalising dropped the quotation marks, so they are put back: those the
    quoted `fragment` begins or ends with, and one more if the span would
    otherwise open a quotation without closing it, or the other way round.
    """
    first, last = index[start], index[end - 1] + 1
    while last < len(passage) and unicodedata.combining(passage[last]):
        last += 1
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
