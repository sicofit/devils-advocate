"""Split a BGE full text into citable passages.

The OpenCaseLaw texts are scraped from the Federal Supreme Court's HTML. Three
artifacts of that scrape matter here:

* Page markers ("BGE 145 III 72 S. 75") stand on their own lines, sometimes in
  the middle of a sentence. They are removed from the text, and the offsets at
  which a new page begins are kept with the passage, so a quote is cited with
  the page it is on.
* Hyperlinked statute and case references ("Art. 19 Abs. 1 lit. a URG") are
  broken onto their own lines. Those lines are joined back into the sentence.
* Erwägung numbers ("2.", "2.1", "4.3.2") stand alone on a line. Older
  decisions subdivide with "a)", "b)" and "aa)" at the start of a paragraph.
  A bare number without a dot ("2") is not a heading: it is a statute
  paragraph, a table value or the "2" of "m2" broken onto its own line.

Each passage carries a pinpoint ("E. 2.1", "E. 3b/aa") and its pages, so a
quote can be cited the way a Swiss lawyer cites it: "BGE 145 III 72 E. 2.2.1 S. 76".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PAGE_MARKER = re.compile(r"^((?:BGE|ATF|DTF) \d+ [IVX]+[a-z]? \d+) S\. (\d+)$")
NUMBER_LINE = re.compile(r"^(?:(\d{1,2})\.|(\d{1,2}(?:\.\d{1,2}){1,4})\.?)$")
LETTER_START = re.compile(r"^([a-z])\1?\) ")
# Artifact lines without content ("ab Seite 2") that the scrape sometimes leaves behind.
NOISE_LINE = re.compile(r"^ab Seite \d+$")
SECTION_LINE = re.compile(r"^(Regeste|Sachverhalt|Erwägungen|Dispositiv)$")
# Lines ending in a colon right after the "Erwägungen" heading introduce the
# reasons ("Aus den Erwägungen:", "Extrait des considérants:", "Das Eidg.
# Versicherungsgericht zieht in Erwägung:"). They are not part of any Erwägung.
INTRO_LINE = re.compile(r"^.{0,120}:$")
# A short line holding only a hyperlinked reference is a scrape artifact, never
# the start of a paragraph.
LINK_LINE = re.compile(r"^(?:(?:BGE|ATF|DTF) \d|[Aa]rt\. ?\d|§ ?\d|Ziff\. ?\d|ch\. ?\d)")
LINK_LINE_MAX = 70
ABBREVIATIONS = (
    "vgl.",
    "Vgl.",
    "z.B.",
    "Abs.",
    "lit.",
    "Art.",
    "art.",
    "S.",
    "E.",
    "N.",
    "Rz.",
    "Nr.",
    "Ziff.",
    "bzw.",
    "f.",
    "ff.",
    "a.a.O.",
    "i.V.m.",
    "cf.",
    "Cf.",
    "consid.",
    "al.",
    "let.",
    "ch.",
    "p.",
    "n.",
    "op. cit.",
    "v.",
    "consid",
    "cfr.",
    "Cfr.",
    "pag.",
)
NO_SPACE_BEFORE = tuple(",;.:)]")
NO_SPACE_AFTER = ("(", "[", "'", "’", "«")
SENTENCE_END = tuple(".:!?»")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜÀÉÈÊ«])")

MAX_PASSAGE_CHARS = 1500

PageBreaks = tuple[tuple[int, int], ...]  # (offset in the text, page that begins there)


@dataclass(frozen=True, slots=True)
class Passage:
    decision: str  # "BGE 145 III 72"
    pinpoint: str  # "E. 2.1", or "Regeste"
    page: int | None  # page the passage starts on, when known
    text: str
    breaks: PageBreaks = ()  # where in `text` a later page begins

    @property
    def citation(self) -> str:
        """Full citation, e.g. 'BGE 145 III 72 E. 2.2.1 S. 76'.

        The Regeste is the court's headnote, not part of the reasoning, so a
        quote from it is marked as such.
        """
        return self.citation_of_span(0, 0)

    def citation_of_span(self, start: int, end: int) -> str:
        """Citation of the text between offsets `start` and `end`, with the pages it is on.

        "S. 76" when it is on one page, "S. 76 f." when it runs onto the next,
        "S. 76 ff." when it runs further.
        """
        if self.pinpoint == "Regeste":
            return f"{self.decision} (Regeste)"
        # Text before the first numbered Erwägung is cited by page alone.
        cite = self.decision if self.pinpoint == "E." else f"{self.decision} {self.pinpoint}"
        first = self.page_at(start)
        if first is None:
            return cite
        last = self.page_at(max(start, end - 1))
        suffix = "" if last is None or last <= first else " f." if last == first + 1 else " ff."
        return f"{cite} S. {first}{suffix}"

    def page_at(self, offset: int) -> int | None:
        page = self.page
        for start, value in self.breaks:
            if start > offset:
                break
            page = value
        return page


def ends_with_abbreviation(text: str) -> bool:
    """Whether `text` ends in an abbreviation ("vgl.", "E.") rather than a sentence.

    The abbreviation has to be a word of its own: "décision." ends in "n." but
    is the end of a sentence, "n." alone (French for "note") is not.
    """
    for abbreviation in ABBREVIATIONS:
        if text.endswith(abbreviation):
            before = text[: -len(abbreviation)][-1:]
            if not before.isalnum():
                return True
    return False


def citation_of(decision_id: str) -> str:
    """'bge_145 III 72' -> 'BGE 145 III 72'."""
    return "BGE " + decision_id.removeprefix("bge_").strip()


def first_page(reference: str) -> int | None:
    """'BGE 145 III 72' -> 72, the page the decision starts on."""
    tail = reference.rsplit(" ", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _page_of(decision: str, line: str) -> int | None:
    """The page a marker line starts, if it is a page marker of `decision`."""
    marker = PAGE_MARKER.match(line)
    if marker is None or "BGE" + marker.group(1)[3:] != decision:
        return None
    return int(marker.group(2))


def names_another_decision(decision: str, full_text: str) -> bool:
    """Whether every page marker in the text belongs to a different decision.

    The dataset has a few rows whose text is that of another decision (the row
    "152 I 2" holds the text of BGE 152 I 20). Indexing them would cite the
    wrong reference.
    """
    markers = [m for line in full_text.splitlines() if (m := PAGE_MARKER.match(line.strip()))]
    return bool(markers) and all("BGE" + m.group(1)[3:] != decision for m in markers)


def segment(decision: str, full_text: str) -> list[Passage]:
    """Return the Regeste and the Erwägungen of one decision as passages."""
    lines = [line.strip() for line in full_text.splitlines()]
    passages: list[Passage] = []

    regeste = _section(lines, "Regeste")
    if regeste:
        text, _ = _join([(line, None) for line in regeste])
        if text:
            passages.append(Passage(decision, "Regeste", first_page(decision), text))

    try:
        start = lines.index("Erwägungen") + 1
    except ValueError:
        return passages
    page = first_page(decision)
    for line in lines[:start]:
        page = _page_of(decision, line) or page
    passages.extend(_erwaegungen(decision, lines[start:], page))
    return passages


def _section(lines: list[str], name: str) -> list[str]:
    try:
        start = lines.index(name) + 1
    except ValueError:
        return []
    end = start
    while end < len(lines) and not SECTION_LINE.match(lines[end]):
        end += 1
    return [line for line in lines[start:end] if not PAGE_MARKER.match(line)]


def _erwaegungen(decision: str, lines: list[str], page: int | None) -> list[Passage]:
    number: tuple[int, ...] | None = None
    letters: list[str] = []
    buffer: list[tuple[str, int | None]] = []
    passages: list[Passage] = []

    def flush() -> None:
        text, pages = _join(buffer)
        buffer.clear()
        if not text:
            return
        pinpoint = _pinpoint(number, letters)
        for start, end in _chunk(text):
            breaks = tuple((at - start, p) for at, p in pages if start < at < end and p is not None)
            passages.append(
                Passage(decision, pinpoint, _page_at(pages, start), text[start:end], breaks)
            )

    seen_content = False
    for line in lines:
        if not line or NOISE_LINE.match(line):
            continue
        if PAGE_MARKER.match(line):
            page = _page_of(decision, line) or page
            continue
        if line == "Dispositiv":  # the operative part follows; it is not reasoning
            break
        if not seen_content and number is None and INTRO_LINE.match(line):
            continue
        seen_content = True

        found = NUMBER_LINE.match(line)
        if found:
            candidate = tuple(int(part) for part in (found.group(1) or found.group(2)).split("."))
            if _follows(number, candidate):
                flush()
                number, letters = candidate, []
                continue

        lettered = LETTER_START.match(line)
        if lettered and number is not None and len(number) == 1:
            label = lettered.group(0)[:-2]
            new_letters = _next_letters(letters, label)
            if new_letters is not None:
                flush()
                letters = new_letters
                buffer.append((line[len(label) + 2 :], page))
                continue

        buffer.append((line, page))
    flush()
    return passages


def _follows(current: tuple[int, ...] | None, candidate: tuple[int, ...]) -> bool:
    """Whether `candidate` plausibly is the next Erwägung heading after `current`.

    Published decisions are often excerpts: "Aus den Erwägungen" may print 2, 5
    and 7, start at 2.5, or go from 2 straight to 2.3.3 or from 1.1 to 2.4. So
    a heading may jump forward, at any level. A number that goes backwards, or
    jumps implausibly far, is a numbered list item inside the text.
    """
    if current is None:
        return 1 <= candidate[0] <= 40 and all(1 <= part <= 20 for part in candidate[1:])
    common = 0
    while common < min(len(current), len(candidate)) and current[common] == candidate[common]:
        common += 1
    if common == len(candidate):  # the same heading again, or a level above it
        return False
    if common == len(current):  # a subdivision of the current heading
        return all(1 <= part <= 20 for part in candidate[common:])
    return current[common] < candidate[common] <= current[common] + 20 and all(
        1 <= part <= 20 for part in candidate[common + 1 :]
    )


def _next_letters(letters: list[str], label: str) -> list[str] | None:
    """Advance the a)/aa) subdivision, or return None if `label` does not continue it."""
    letter = label[0]
    if len(label) == 1:
        if not letters:
            return [letter] if letter == "a" else None
        return [letter] if ord(letter) == ord(letters[0]) + 1 else None
    if not letters:
        return None
    if len(letters) == 1:
        return [letters[0], label] if letter == "a" else None
    return [letters[0], label] if ord(letter) == ord(letters[1][0]) + 1 else None


def _pinpoint(number: tuple[int, ...] | None, letters: list[str]) -> str:
    if number is None:
        return "E."
    pinpoint = "E. " + ".".join(str(part) for part in number)
    if letters:
        pinpoint += letters[0] + "".join("/" + extra for extra in letters[1:])
    return pinpoint


def _is_link(line: str) -> bool:
    return len(line) <= LINK_LINE_MAX and bool(LINK_LINE.match(line))


def _join(items: list[tuple[str, int | None]]) -> tuple[str, list[tuple[int, int | None]]]:
    """Rejoin scraped lines into paragraphs, undoing the hyperlink line breaks.

    Returns the text and the offsets at which a new page begins.
    """
    out = ""
    pages: list[tuple[int, int | None]] = []
    previous = ""
    for line, page in items:
        if not line:
            continue
        if not out or line.startswith(NO_SPACE_BEFORE) or out.endswith(NO_SPACE_AFTER):
            separator = ""
        elif (
            out.endswith(SENTENCE_END)
            and not ends_with_abbreviation(out)
            and line[:1].isupper()
            and not _is_link(line)
            and not _is_link(previous)
        ):
            separator = "\n"
        else:
            separator = " "
        if not pages or pages[-1][1] != page:
            pages.append((len(out) + len(separator), page))
        out += separator + line
        previous = line
    return out, pages


def _page_at(pages: list[tuple[int, int | None]], offset: int) -> int | None:
    page = pages[0][1] if pages else None
    for start, value in pages:
        if start > offset:
            break
        page = value
    return page


def _chunk(text: str, limit: int = MAX_PASSAGE_CHARS) -> list[tuple[int, int]]:
    """Split a long Erwägung into (start, end) spans at paragraph or sentence breaks."""
    if len(text) <= limit:
        return [(0, len(text))]
    breaks = sorted(
        {m.start() for m in re.finditer("\n", text)}
        | {m.start() for m in SENTENCE_BREAK.finditer(text) if _sentence_ends(text, m.start())}
    )
    spans: list[tuple[int, int]] = []
    start = 0
    last_ok = None
    for position in [*breaks, len(text)]:
        if position - start > limit and last_ok is not None:
            spans.append((start, last_ok))
            start = _skip_space(text, last_ok)
        last_ok = position
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _sentence_ends(text: str, position: int) -> bool:
    """Whether the break before `position` ends a sentence.

    Not after an abbreviation ("i.V.m. Art. 40") and not inside parentheses
    ("(cf. Message ...)").
    """
    before = text[text.rfind("\n", 0, position) + 1 : position]  # the paragraph so far
    if ends_with_abbreviation(before):
        return False
    return before.count("(") <= before.count(")")


def _skip_space(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position
