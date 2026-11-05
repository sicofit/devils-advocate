"""The pipeline: split the position into claims, retrieve, argue against each, verify."""

from __future__ import annotations

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


@dataclass(slots=True)
class Counter:
    argument: str
    quote: str  # cut from the passage, never the model's wording
    status: Status  # EXACT or CLOSE
    passage: StoredPassage

    @property
    def citation(self) -> str:
        return self.passage.citation


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
        self.history: list[str] = []

    def reset(self) -> None:
        self.history.clear()

    def argue(self, argument: str) -> Rebuttal:
        started = time.perf_counter()
        self.on_stage("Splitting your position into claims")
        analysis = self.model.json(
            prompts.analyse_messages(argument, self.history, self.max_claims),
            prompts.ANALYSE_SCHEMA,
            name="analysis",
        )
        language = str(analysis.get("language") or "en")
        claims = [
            ClaimResult(str(item.get("claim", "")).strip(), _strings(item.get("queries")))
            for item in analysis.get("claims") or []
            if isinstance(item, dict) and str(item.get("claim", "")).strip()
        ][: self.max_claims]

        for number, claim in enumerate(claims, 1):
            self.on_stage(f"Searching precedent against claim {number}/{len(claims)}")
            claim.hits = self.retriever.search(
                [claim.claim, *claim.queries], self.top_k, per_decision=PER_DECISION
            )
            if not claim.hits:
                continue
            self.on_stage(f"Arguing against claim {number}/{len(claims)}")
            self._counter(claim, language)

        self.history.append(argument)
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
            prompts.COUNTER_SCHEMA,
            name="counterarguments",
        )
        used: set[int] = set()
        for item in reply.get("counterarguments") or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("passage", "")).strip().strip("[]")
            quote = str(item.get("quote", ""))
            argument = str(item.get("argument", "")).strip()
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
            claim.counters.append(Counter(argument, verdict.text, verdict.status, passage))
            if len(claim.counters) == MAX_COUNTERS:
                break


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:4]
