"""End-to-end evaluation on hand-written positions.

Each case in `evals/positions.jsonl` is either

* a position that contradicts the holding of a known BGE, written from its
  Regeste. The devil should come back with a verified quote from that decision.
  Other decisions may well be valid counterarguments too, so a position where
  the known one was not quoted is not necessarily answered wrongly. The metric
  checks which decision was quoted, not whether the argument built on the quote
  is right: that is left to reading the outputs.
* an input the devil should not argue with (no legal claim, foreign law). It
  should come back without any verified quote.

A case whose model reply could not be used is counted as an error, neither as
an answer nor as an abstention.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter as Tally
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .client import LLMError, ModelUnavailable
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
    # utf-8-sig: a file saved by Notepad or PowerShell may begin with a byte order mark.
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            raw = json.loads(line)
            case = Case(
                raw["id"], raw["position"], tuple(raw.get("expect", [])), raw.get("note", "")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{number}: a case needs an id and a position") from exc
        cases.append(case)
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
        try:
            rebuttal = new_devil().argue(case.position)
        except ModelUnavailable:
            raise  # every further case would fail the same way
        except LLMError as exc:
            rows.append(failed(case, str(exc)))
            continue
        rows.append(score(case, rebuttal))
    return {"summary": summarise(rows), "cases": rows}


def failed(case: Case, error: str) -> dict:
    """The row of a case whose analysis reply could not be used."""
    return {
        "id": case.id,
        "position": case.position,
        "expect": list(case.expect),
        "error": error,
        "language": None,
        "claims": 0,
        "proposed": 0,
        "verified": 0,
        "exact": 0,
        "discarded": {},
        "skipped": {},
        "claim_errors": 0,
        "cited": [],
        "retrieved_expected": False,
        "found_expected": False,
        "abstained": False,
        "seconds": 0.0,
    }


def score(case: Case, rebuttal: Rebuttal) -> dict:
    cited = [c.passage.reference for claim in rebuttal.claims for c in claim.counters]
    retrieved = {h.passage.reference for claim in rebuttal.claims for h in claim.hits}
    reasons = Tally(d.reason for claim in rebuttal.claims for d in claim.discarded)
    skipped = Tally(d.reason for claim in rebuttal.claims for d in claim.skipped)
    return {
        "id": case.id,
        "position": case.position,
        "expect": list(case.expect),
        "error": None,
        "language": rebuttal.language,
        "claims": len(rebuttal.claims),
        "proposed": rebuttal.proposed,
        "verified": rebuttal.verified,
        "exact": sum(
            1 for claim in rebuttal.claims for c in claim.counters if c.status.value == "exact"
        ),
        "discarded": dict(reasons),
        # Set aside without failing a check; not part of the proposed count.
        "skipped": dict(skipped),
        # Claims whose counterargument reply could not be used.
        "claim_errors": sum(1 for claim in rebuttal.claims if claim.error),
        "cited": cited,
        "retrieved_expected": any(ref in retrieved for ref in case.expect),
        "found_expected": any(ref in cited for ref in case.expect),
        # No verified quote and no unusable reply: a claim that failed is not an abstention.
        "abstained": rebuttal.verified == 0 and not any(claim.error for claim in rebuttal.claims),
        "seconds": round(rebuttal.seconds, 1),
    }


def summarise(rows: list[dict]) -> dict:
    positive = [r for r in rows if r["expect"]]
    negative = [r for r in rows if not r["expect"]]
    proposed = sum(r["proposed"] for r in rows)
    verified = sum(r["verified"] for r in rows)
    reasons: Tally[str] = Tally()
    skipped: Tally[str] = Tally()
    for r in rows:
        reasons.update(r["discarded"])
        skipped.update(r.get("skipped", {}))
    # Timed over the positions only: an out-of-scope input usually ends after one call.
    seconds = [r["seconds"] for r in positive if not r["error"]]
    return {
        "cases": len(rows),
        # Cases whose analysis reply could not be used: no answer, no abstention.
        "errors": sum(1 for r in rows if r["error"]),
        "claim_errors": sum(r["claim_errors"] for r in rows),
        "contradicting_positions": len(positive),
        "out_of_scope": len(negative),
        # Positions where the decision we know holds the opposite was quoted. Whether
        # the argument around the quote is right is not graded.
        "found_expected": sum(r["found_expected"] for r in positive),
        # ... or at least retrieved and shown to the model.
        "retrieved_expected": sum(r["retrieved_expected"] for r in positive),
        # Positions that got at least one verified counterargument.
        "answered": sum(r["verified"] > 0 for r in positive),
        "abstained_out_of_scope": sum(r["abstained"] for r in negative),
        # ... of which the analysis still found a legal claim in: nothing verified, but
        # the devil did argue.
        "out_of_scope_with_claims": sum(r["abstained"] and r["claims"] > 0 for r in negative),
        "quotes_proposed": proposed,
        "quotes_verified": verified,
        "quotes_exact": sum(r["exact"] for r in rows),
        "discarded_by_reason": dict(reasons),
        "skipped_by_reason": dict(skipped),
        "seconds_median": round(statistics.median(seconds), 1) if seconds else 0.0,
        "seconds_max": max(seconds, default=0.0),
    }


def table(report: dict) -> str:
    s = report["summary"]
    p, n = s["contradicting_positions"], s["out_of_scope"]
    rate = s["quotes_verified"] / s["quotes_proposed"] if s["quotes_proposed"] else 0.0
    lines = [
        f"{s['cases']} cases: {p} positions that contradict a known BGE, {n} out of scope",
        f"  known precedent quoted             {s['found_expected']}/{p}",
        f"  known precedent retrieved          {s['retrieved_expected']}/{p}",
        f"  positions with a verified counter  {s['answered']}/{p}",
        f"  out-of-scope inputs abstained      {s['abstained_out_of_scope']}/{n}"
        f" ({s.get('out_of_scope_with_claims', 0)} with a claim extracted)",
        f"  quotes verified / proposed         {s['quotes_verified']}/{s['quotes_proposed']}"
        f" ({rate:.0%}; {s['quotes_exact']} verbatim, the rest close matches)",
        f"  discarded                          {s['discarded_by_reason'] or 'none'}",
        f"  set aside (not counted above)      {s.get('skipped_by_reason') or 'none'}",
        f"  seconds per position               median {s['seconds_median']},"
        f" max {s['seconds_max']}",
    ]
    if s["errors"] or s["claim_errors"]:
        lines.append(
            f"  unusable model replies             {s['errors']} cases, {s['claim_errors']} claims"
        )
    return "\n".join(lines)


def badge(report: dict) -> dict:
    """A shields.io endpoint badge: how often the known contradicting precedent was quoted."""
    s = report["summary"]
    found, total = s["found_expected"], s["contradicting_positions"]
    bands = ((0.8, "brightgreen"), (0.6, "green"), (0.4, "yellow"), (0.2, "orange"))
    if total:
        color = next((name for low, name in bands if found / total >= low), "red")
    else:
        color = "lightgrey"  # nothing was measured
    return {
        "schemaVersion": 1,
        "label": "eval: known precedent quoted",
        "message": f"{found}/{total}",
        "color": color,
    }
