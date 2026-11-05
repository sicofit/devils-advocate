"""The pipeline: split the position into claims, retrieve, argue against each, verify."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import prompts
from .client import ChatModel
from .search import Hit, Retriever
from .store import StoredPassage
from .verify import Status, verify

MAX_COUNTERS = 3
PER_DECISION = 2
LABEL = re.compile(r"\bP(\d+)\b")


@dataclass(slots=True)
class Counter:
    argument: str
    quote: str  # cut from the passage, never the model's wording
    status: Status  # EXACT or CLOSE
    passage: StoredPassage
    spans: tuple[tuple[int, int], ...] = ()  # where the quote is in passage.text

    @property
    def citation(self) -> str:
        """The passage's citation, with the page(s) the quote itself is on."""
        if not self.spans:
            return self.passage.citation
        return self.passage.citation_of_span(self.spans[0][0], self.spans[-1][1])


@dataclass(slots=True)
class Discarded:
    label: str
    quote: str  # what the model claimed the passage says
    reason: str


@dataclass(slots=True)
class ClaimResult:
    claim: str
    queries: list[str]
    counters: list[Counter] = field(default_factory=list)
    discarded: list[Discarded] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return not self.counters


@dataclass(slots=True)
class Rebuttal:
    argument: str
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

    def argue(self, argument: str) -> Rebuttal:
        started = time.perf_counter()
        self.on_stage("Splitting your position into claims")
        analysis = self.model.json(
            prompts.analyse_messages(argument, self.history, self.max_claims),
            prompts.ANALYSE_SCHEMA,
            name="analysis",
        )
        language = str(analysis.get("language") or "en")
        items = analysis.get("claims")
        claims = [
            ClaimResult(str(item.get("claim", "")).strip(), _strings(item.get("queries")))
            for item in (items if isinstance(items, list) else [])
            if isinstance(item, dict) and str(item.get("claim", "")).strip()
        ][: self.max_claims]

        for number, claim in enumerate(claims, 1):
            self.on_stage(f"Searching precedent against claim {number}/{len(claims)}")
            hits = self.retriever.search(
                [claim.claim, *claim.queries],
                self.top_k + len(self.cited),
                per_decision=PER_DECISION,
            )
            # A follow-up turn gets new precedent, not the passages already quoted.
            claim.hits = [h for h in hits if h.passage.id not in self.cited][: self.top_k]
            if not claim.hits:
                continue
            self.on_stage(f"Arguing against claim {number}/{len(claims)}")
            self._counter(claim, language)

        self.history.append(
            prompts.Turn(
                argument,
                [c.claim for c in claims],
                [counter.citation for c in claims for counter in c.counters],
            )
        )
        self.cited.update(counter.passage.id for c in claims for counter in c.counters)
        return Rebuttal(argument, language, claims, time.perf_counter() - started)

    def _counter(self, claim: ClaimResult, language: str) -> None:
        labelled = {f"P{i}": hit.passage for i, hit in enumerate(claim.hits, 1)}
        reply = self.model.json(
            prompts.counter_messages(
                claim.claim,
                [(label, p.citation, p.text) for label, p in labelled.items()],
                language,
                MAX_COUNTERS,
            ),
            prompts.counter_schema(list(labelled)),
            name="counterarguments",
        )
        used: set[int] = set()
        items = reply.get("counterarguments")
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            label = _label(item.get("passage"))
            quote = str(item.get("quote", ""))
            argument = _readable(str(item.get("argument", "")).strip(), labelled, language)
            passage = labelled.get(label)
            if passage is None:
                claim.discarded.append(Discarded(label, quote, "cites a passage it was not given"))
                continue
            verdict = verify(quote, passage.text)
            if not verdict.ok:
                claim.discarded.append(Discarded(label, quote, verdict.reason))
                continue
            if passage.id in used:
                continue
            used.add(passage.id)
            claim.counters.append(
                Counter(argument, verdict.text, verdict.status, passage, verdict.spans)
            )
            if len(claim.counters) == MAX_COUNTERS:
                break


def _label(value: object) -> str:
    """'P3' from 'P3', 'p3', '[P3]' or 3."""
    label = str(value if value is not None else "").strip().strip("[]").strip().upper()
    return f"P{label}" if label.isdigit() else label


def _readable(argument: str, labelled: dict[str, StoredPassage], language: str) -> str:
    """Replace passage labels the reader never sees by references; Swiss spelling."""
    argument = LABEL.sub(
        lambda m: labelled[m.group(0)].reference if m.group(0) in labelled else m.group(0),
        argument,
    )
    return argument.replace("ß", "ss") if language == "de" else argument


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:4]
