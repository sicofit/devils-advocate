"""End-to-end evaluation on hand-written positions.

Each case in `evals/positions.jsonl` is either

* a position that contradicts the holding of a known BGE, written from its
  Regeste. The devil should come back with a verified quote from that decision.
  Other decisions may well be valid counterarguments too; the metric only checks
  whether the one we know about was found, so it is a lower bound.
* an input the devil should not argue with (no legal claim, foreign law). It
  should come back without any verified quote.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter as Tally
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .devil import Devil, Rebuttal


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    position: str
    expect: tuple[str, ...]  # decisions that hold the opposite; empty = should abstain
    note: str = ""

    @property
    def should_abstain(self) -> bool:
        return not self.expect


def load_cases(path: Path) -> list[Case]:
    cases = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        cases.append(
            Case(raw["id"], raw["position"], tuple(raw.get("expect", [])), raw.get("note", ""))
        )
    return cases


def run(
    cases: list[Case],
    new_devil: Callable[[], Devil],
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Run every case through a fresh devil (no shared history). Returns a JSON-ready report."""
    rows = []
    for i, case in enumerate(cases, 1):
        if on_progress:
            on_progress(i, len(cases), case.id)
        rows.append(score(case, new_devil().argue(case.position)))
    return {"summary": summarise(rows), "cases": rows}


def score(case: Case, rebuttal: Rebuttal) -> dict:
    cited = [c.passage.reference for claim in rebuttal.claims for c in claim.counters]
    retrieved = {h.passage.reference for claim in rebuttal.claims for h in claim.hits}
    reasons = Tally(d.reason for claim in rebuttal.claims for d in claim.discarded)
    return {
        "id": case.id,
        "position": case.position,
        "expect": list(case.expect),
        "language": rebuttal.language,
        "claims": len(rebuttal.claims),
        "proposed": rebuttal.proposed,
        "verified": rebuttal.verified,
        "exact": sum(
            1 for claim in rebuttal.claims for c in claim.counters if c.status.value == "exact"
        ),
        "discarded": dict(reasons),
        "cited": cited,
        "retrieved_expected": any(ref in retrieved for ref in case.expect),
        "found_expected": any(ref in cited for ref in case.expect),
        "abstained": rebuttal.verified == 0,
        "seconds": round(rebuttal.seconds, 1),
    }


def summarise(rows: list[dict]) -> dict:
    positive = [r for r in rows if r["expect"]]
    negative = [r for r in rows if not r["expect"]]
    proposed = sum(r["proposed"] for r in rows)
    verified = sum(r["verified"] for r in rows)
    reasons: Tally[str] = Tally()
    for r in rows:
        reasons.update(r["discarded"])
    seconds = [r["seconds"] for r in rows]
    return {
        "cases": len(rows),
        "contradicting_positions": len(positive),
        "out_of_scope": len(negative),
        # Positions where the decision we know holds the opposite was quoted.
        "found_expected": sum(r["found_expected"] for r in positive),
        # ... or at least retrieved and shown to the model.
        "retrieved_expected": sum(r["retrieved_expected"] for r in positive),
        # Positions that got at least one verified counterargument.
        "answered": sum(not r["abstained"] for r in positive),
        "abstained_out_of_scope": sum(r["abstained"] for r in negative),
        "quotes_proposed": proposed,
        "quotes_verified": verified,
        "quotes_exact": sum(r["exact"] for r in rows),
        "discarded_by_reason": dict(reasons),
        "seconds_median": round(statistics.median(seconds), 1) if seconds else 0.0,
        "seconds_max": max(seconds, default=0.0),
    }


def table(report: dict) -> str:
    s = report["summary"]
    p, n = s["contradicting_positions"], s["out_of_scope"]
    rate = s["quotes_verified"] / s["quotes_proposed"] if s["quotes_proposed"] else 0.0
    lines = [
        f"{s['cases']} cases: {p} positions that contradict a known BGE, {n} out of scope",
        f"  known counter-precedent quoted     {s['found_expected']}/{p}",
        f"  known counter-precedent retrieved  {s['retrieved_expected']}/{p}",
        f"  positions with a verified counter  {s['answered']}/{p}",
        f"  out-of-scope inputs abstained      {s['abstained_out_of_scope']}/{n}",
        f"  quotes verified / proposed         {s['quotes_verified']}/{s['quotes_proposed']}"
        f" ({rate:.0%}; {s['quotes_exact']} exact, rest within normalisation)",
        f"  discarded                          {s['discarded_by_reason'] or 'none'}",
        f"  seconds per position               median {s['seconds_median']},"
        f" max {s['seconds_max']}",
    ]
    return "\n".join(lines)


def badge(report: dict) -> dict:
    """A shields.io endpoint badge: how often the known counter-precedent was quoted."""
    s = report["summary"]
    found, total = s["found_expected"], s["contradicting_positions"]
    share = found / total if total else 0.0
    color = "brightgreen" if share >= 0.8 else "green" if share >= 0.6 else "yellow"
    return {
        "schemaVersion": 1,
        "label": "eval: counter-precedent found",
        "message": f"{found}/{total}",
        "color": color,
    }
