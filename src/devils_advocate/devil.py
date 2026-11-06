"""The pipeline: split the position into claims, retrieve, argue against each, verify."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import language, prompts
from .client import ChatModel, LLMError, ModelUnavailable, Truncated
from .search import Hit, Retriever
from .store import StoredPassage
from .verify import Status, verify

MAX_COUNTERS = 3
PER_DECISION = 2
LABEL = re.compile(r"\bP(\d+)\b")
# Unpublished decisions are cited by docket number: "Urteil 4A_384/2016".
DOCKET = re.compile(r"\b\d[A-Z]_\d+/\d{4}\b")
# An argument that says its passage backs the claim is no counterargument, whatever
# "effect" the model gave it: "was die Behauptung bestätigt", "supports the claim".
SUPPORTS = re.compile(
    r"\b(?:die|der|diese|dieser) Behauptung\W+(?:\w+\W+){0,2}"
    r"(?:bestätigt|stützt|untermauert|bekräftigt)\b"
    r"|\b(?:bestätigt|stützt|untermauert|bekräftigt)\W+(?:\w+\W+)?(?:die|diese) Behauptung\b"
    r"|\b(?:supports?|confirms?|backs up)\W+(?:\w+\W+)?(?:the|this) claim\b"
    r"|\b(?:confirme|conforte|appuie|corrobore)\W+(?:\w+\W+)?(?:l'|cette )affirmation\b"
    r"|\b(?:conferma|sostiene|avvalora)\W+(?:\w+\W+)?(?:l'|questa )affermazione\b",
    re.IGNORECASE,
)
# Words near such a phrase that deny it ("does not support the claim") or limit it
# ("stützt die Behauptung nur teilweise", "confirms the claim, but ..."): the
# argument may still be a counterargument.
NOT = set(
    """
    nicht kein keine keineswegs nie niemals nichts weder
    not no never nothing none neither nor
    pas ne n non jamais rien ni
    nessun nessuna nulla mai
    nur lediglich teilweise teils aber jedoch
    only partly partially rather than but
    seulement partiellement mais
    solo parzialmente ma
    """.split()
)


@dataclass(slots=True)
class Counter:
    argument: str
    quote: str  # cut from the passage, never the model's wording
    status: Status  # EXACT or CLOSE
    passage: StoredPassage
    spans: tuple[tuple[int, int], ...] = ()  # where the quote is in passage.text
    opens_inside: bool = False  # the quote starts inside a sentence of the court
    ends_inside: bool = False  # ... or stops before the sentence ends
    # The quote restates the lower court's or a party's view, not the court's own.
    reported: bool = False

    @property
    def shown(self) -> str:
        """The quote with "[…]" where it starts or stops inside a sentence."""
        head = "[…] " if self.opens_inside else ""
        return head + self.quote + (" […]" if self.ends_inside else "")

    @property
    def citation(self) -> str:
        """The passage's citation, with the page(s) the quote itself is on."""
        if not self.spans:
            return self.passage.citation
        return self.passage.citation_of_span(self.spans[0][0], self.spans[-1][1])


@dataclass(slots=True)
class Discarded:
    label: str  # the passage label the model gave; internal, never shown
    quote: str  # what the model claimed the passage says
    reason: str
    citation: str = ""  # the cited passage, when the label named one


@dataclass(slots=True)
class ClaimResult:
    claim: str
    queries: list[str]  # as searched: BGE the user did not name are taken out
    counters: list[Counter] = field(default_factory=list)
    discarded: list[Discarded] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    error: str = ""  # why the model's reply for this claim could not be used
    # Verified counters whose passage was already quoted for an earlier claim of the
    # same turn, with that claim's number: the argument is new, the quote is not.
    repeated: list[tuple[Counter, int]] = field(default_factory=list)
    # Proposals set aside without failing a check: a second one from the same decision,
    # one over the limit, or one whose argument agrees with the claim.
    skipped: list[Discarded] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return not self.counters and not self.repeated


@dataclass(slots=True)
class Rebuttal:
    position: str
    language: str
    claims: list[ClaimResult]
    seconds: float = 0.0

    @property
    def proposed(self) -> int:
        return sum(len(c.counters) + len(c.discarded) for c in self.claims)

    @property
    def verified(self) -> int:
        return sum(len(c.counters) for c in self.claims)


class Devil:
    def __init__(
        self,
        model: ChatModel,
        retriever: Retriever,
        *,
        top_k: int = 8,
        max_claims: int = 4,
        on_stage: Callable[[str], None] | None = None,
    ) -> None:
        self.model = model
        self.retriever = retriever
        self.top_k = top_k
        self.max_claims = max_claims
        self.on_stage = on_stage or (lambda _: None)
        self.history: list[prompts.Turn] = []
        self.cited: set[int] = set()  # passages quoted earlier in this conversation

    def reset(self) -> None:
        self.history.clear()
        self.cited.clear()

    def argue(self, position: str) -> Rebuttal:
        started = time.perf_counter()
        self.on_stage("Splitting your position into claims")
        analysis = self.model.json(
            prompts.analyse_messages(position, self.history, self.max_claims),
            prompts.ANALYSE_SCHEMA,
            name="analysis",
        )
        # The model's guess only decides when the words of the position do not.
        lang = language.detect(position, fallback=str(analysis.get("language") or ""))
        named = language.references(position)
        items = analysis.get("claims")
        claims = [
            ClaimResult(
                str(item.get("claim", "")).strip(),
                _queries(item.get("queries"), named),
            )
            for item in (items if isinstance(items, list) else [])
            if isinstance(item, dict) and str(item.get("claim", "")).strip()
        ][: self.max_claims]

        # passage id -> the claims of this turn it was quoted for, with the quoted spans
        quoted: dict[int, list[tuple[int, tuple[tuple[int, int], ...]]]] = {}
        for number, claim in enumerate(claims, 1):
            self.on_stage(f"Searching precedent against claim {number}/{len(claims)}")
            searches = [language.without_references(claim.claim, named), *claim.queries]
            hits = self.retriever.search(
                [s for s in searches if s],
                self.top_k + len(self.cited),
                per_decision=PER_DECISION,
            )
            # A follow-up turn gets new precedent, not the passages already quoted.
            claim.hits = [h for h in hits if h.passage.id not in self.cited][: self.top_k]
            if not claim.hits:
                continue
            self.on_stage(f"Arguing against claim {number}/{len(claims)}")
            try:
                self._counter(claim, number, lang, position, quoted)
            except ModelUnavailable:
                raise  # the next claim would fail the same way
            except LLMError as exc:
                claim.error = str(exc)  # one unusable reply costs this claim, not the turn

        self.history.append(
            prompts.Turn(
                position,
                [c.claim for c in claims if not c.abstained],
                [counter.citation for c in claims for counter in c.counters],
                [c.claim for c in claims if c.abstained],
            )
        )
        self.cited.update(counter.passage.id for c in claims for counter in c.counters)
        return Rebuttal(position, lang, claims, time.perf_counter() - started)

    def _counter(
        self,
        claim: ClaimResult,
        number: int,
        lang: str,
        position: str,
        quoted: dict[int, list[tuple[int, tuple[tuple[int, int], ...]]]],
    ) -> None:
        hits = claim.hits
        while True:
            labelled = {f"P{i}": hit.passage for i, hit in enumerate(hits, 1)}
            try:
                reply = self.model.json(
                    prompts.counter_messages(
                        claim.claim,
                        [(label, p.citation, p.text) for label, p in labelled.items()],
                        lang,
                        MAX_COUNTERS,
                    ),
                    prompts.counter_schema(list(labelled)),
                    name="counterarguments",
                )
                break
            except Truncated:
                # The prompt left too little room for the reply: once more with the
                # better-ranked half of the passages, then give the claim up.
                if len(hits) < 2 or len(hits) < len(claim.hits):
                    raise
                hits = hits[: len(hits) // 2]
        decisions: set[str] = set()
        items = reply.get("counterarguments")
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            label = _label(item.get("passage"))
            quote = str(item.get("quote", ""))
            passage = labelled.get(label)
            if len(claim.counters) + len(claim.repeated) >= MAX_COUNTERS:
                reason = f"over the limit of {MAX_COUNTERS}"
                claim.skipped.append(Discarded(label, quote, reason, _citation(passage)))
                continue
            if passage is None:
                claim.discarded.append(Discarded(label, quote, "cites a passage it was not given"))
                continue
            argument = _readable(str(item.get("argument", "")).strip(), labelled, lang)
            if LABEL.search(argument):
                # A label left over names a passage the model was not shown.
                reason = "the argument cites a passage it was not given"
                claim.discarded.append(Discarded(label, quote, reason, passage.citation))
                continue
            if item.get("effect") == "supports" or _supports(argument):
                reason = "its argument says the passage supports the claim"
                claim.skipped.append(Discarded(label, quote, reason, passage.citation))
                continue
            if not argument:
                reason = "no argument given"
                claim.discarded.append(Discarded(label, quote, reason, passage.citation))
                continue
            # The Regeste of a French or Italian decision is often in German.
            text_language = (
                language.detect(passage.text, passage.language)
                if passage.pinpoint == "Regeste"
                else passage.language
            )
            verdict = verify(quote, passage.text, text_language)
            if not verdict.ok:
                claim.discarded.append(Discarded(label, quote, verdict.reason, passage.citation))
                continue
            if _stray_reference(argument, passage, position):
                reason = "the argument names a decision the cited passage does not"
                claim.discarded.append(Discarded(label, quote, reason, passage.citation))
                continue
            # One counter per decision and claim: a Regeste and an Erwägung of the same
            # decision make the same point twice.
            if passage.reference in decisions:
                reason = "same decision as an earlier counter"
                claim.skipped.append(Discarded(label, quote, reason, passage.citation))
                continue
            decisions.add(passage.reference)
            counter = Counter(
                argument,
                verdict.text,
                verdict.status,
                passage,
                verdict.spans,
                verdict.opens_inside,
                verdict.ends_inside,
                verdict.reported,
            )
            earlier = next(
                (
                    claim_number
                    for claim_number, spans in quoted.get(passage.id, [])
                    if _within(verdict.spans, spans)
                ),
                None,
            )
            if earlier is not None:
                claim.repeated.append((counter, earlier))
            else:
                quoted.setdefault(passage.id, []).append((number, verdict.spans))
                claim.counters.append(counter)


def _within(spans: tuple[tuple[int, int], ...], outer: tuple[tuple[int, int], ...]) -> bool:
    """Whether every span lies inside one of the `outer` spans: the same text quoted again."""
    return all(any(a <= start and end <= b for a, b in outer) for start, end in spans)


def _citation(passage: StoredPassage | None) -> str:
    return passage.citation if passage is not None else ""


def _supports(argument: str) -> bool:
    """Whether the argument says its passage backs the claim ("bestätigt die Behauptung").

    Not when a negation or a limiting word stands within two words before or three
    words after it.
    """
    argument = re.sub(r"\bcannot\b", "can not", argument, flags=re.IGNORECASE)
    argument = re.sub(r"n['’]t\b", " not", argument, flags=re.IGNORECASE)
    for found in SUPPORTS.finditer(argument):
        before = re.findall(r"\w+", argument[: found.start()])[-2:]
        inside = re.findall(r"\w+", found.group(0))
        after = re.findall(r"\w+", argument[found.end() :])[:3]
        if not {word.casefold() for word in before + inside + after} & NOT:
            return True
    return False


def _label(value: object) -> str:
    """'P3' from 'P3', 'p3', '[P3]' or 3."""
    label = str(value if value is not None else "").strip().strip("[]").strip().upper()
    return f"P{label}" if label.isdigit() else label


def _readable(argument: str, labelled: dict[str, StoredPassage], lang: str) -> str:
    """Replace passage labels the reader never sees by references; Swiss spelling."""
    argument = LABEL.sub(
        lambda m: labelled[m.group(0)].reference if m.group(0) in labelled else m.group(0),
        argument,
    )
    return argument.replace("ß", "ss") if lang == "de" else argument


def _stray_reference(argument: str, passage: StoredPassage, position: str) -> str | None:
    """A decision the argument names that is neither the cited one nor named in it.

    The quote is checked against the passage; this keeps the argument around it
    from pinning the quote on another decision ("BGE 145 III 303 clarifies ...").
    A decision the user named may be mentioned too.
    """
    allowed = (
        {passage.reference} | language.references(passage.text) | language.references(position)
    )
    stray = sorted(
        reference
        for reference in language.references(argument) - allowed
        if not language.mentions(passage.text, reference)
    )
    if stray:
        return stray[0]
    for docket in DOCKET.findall(argument):
        if docket not in passage.text and docket not in position:
            return docket
    return None


def _queries(value: object, named: set[str]) -> list[str]:
    """Up to four search queries, without BGE the user did not name: those the model adds
    are often invented, and an invented reference finds nothing or the wrong decision."""
    if not isinstance(value, list):
        return []
    cleaned = (language.without_references(str(v), named) for v in value)
    return [q for q in cleaned if q][:4]
