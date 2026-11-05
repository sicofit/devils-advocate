"""Terminal rendering with rich. Dynamic text is always rendered literally, never as markup."""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from .devil import Rebuttal
from .verify import Status

THEME = Theme(
    {
        "claim": "bold",
        "cite": "cyan",
        "quote": "italic",
        "ok": "green",
        "close": "yellow",
        "abstain": "yellow",
        "error": "bold red",
        "meta": "dim",
    }
)

console = Console(theme=THEME, highlight=False)

NO_COUNTER = {
    "de": "Keine BGE-Passage gefunden, die dieser Behauptung widerspricht.",
    "fr": "Aucun passage ATF trouvé qui contredise cette affirmation.",
    "it": "Nessun passaggio DTF trovato che contraddica questa affermazione.",
    "en": "No BGE passage found that contradicts this claim.",
}
NO_CLAIMS = {
    "de": "Darin steckt keine rechtliche Behauptung, gegen die ich argumentieren könnte.",
    "fr": "Je ne trouve aucune affirmation juridique à contester.",
    "it": "Non trovo alcuna affermazione giuridica da contestare.",
    "en": "I found no legal claim in that to argue against.",
}


def render(rebuttal: Rebuttal, *, show_discarded: bool = False) -> None:
    language = rebuttal.language if rebuttal.language in NO_COUNTER else "en"
    if not rebuttal.claims:
        console.print(Text(NO_CLAIMS[language], style="abstain"))
        return
    for number, claim in enumerate(rebuttal.claims, 1):
        parts: list[Text] = []
        if claim.abstained:
            parts.append(Text(NO_COUNTER[language], style="abstain"))
        for counter in claim.counters:
            mark, style = (
                ("✓ verbatim", "ok")
                if counter.status is Status.EXACT
                else ("≈ wording taken from the source", "close")
            )
            parts.append(Text(counter.argument))
            quote = Text("  „", style="quote")
            quote.append(counter.quote, style="quote")
            quote.append("“", style="quote")
            parts.append(quote)
            cite = Text("  — ", style="meta")
            cite.append(counter.citation, style="cite")
            cite.append(f"  {mark}", style=style)
            parts.append(cite)
            parts.append(Text())
        if show_discarded:
            for item in claim.discarded:
                line = Text(f"  ✗ {item.label}, {item.reason}: ", style="meta")
                line.append(item.quote[:160], style="meta")
                parts.append(line)
        title = Text(f"{number}. ", style="claim")
        title.append(claim.claim, style="claim")
        console.print(Panel(Group(*parts), title=title, title_align="left", border_style="meta"))
    dropped = rebuttal.proposed - rebuttal.verified
    summary = f"{rebuttal.verified} verified quote(s)"
    if dropped:
        summary += f", {dropped} discarded because the cited passage does not contain them"
    summary += f" · {rebuttal.seconds:.0f}s · /sources for full passages"
    console.print(Text(summary, style="meta"))


def render_sources(rebuttal: Rebuttal) -> None:
    seen: set[int] = set()
    for claim in rebuttal.claims:
        for counter in claim.counters:
            passage = counter.passage
            if passage.id in seen:
                continue
            seen.add(passage.id)
            body = Text(passage.text)
            start = passage.text.find(counter.quote)
            if start >= 0:
                body.stylize("bold", start, start + len(counter.quote))
            console.print(
                Panel(body, title=Text(passage.citation, style="cite"), title_align="left")
            )
    if not seen:
        console.print(Text("No sources in the last answer.", style="meta"))


def to_markdown(turns: list[Rebuttal]) -> str:
    """The conversation as a Markdown memo, quotes and citations included."""
    lines = ["# devils-advocate", ""]
    for turn in turns:
        lines += ["## Position", "", turn.argument, ""]
        language = turn.language if turn.language in NO_COUNTER else "en"
        for number, claim in enumerate(turn.claims, 1):
            lines += [f"### {number}. {claim.claim}", ""]
            if claim.abstained:
                lines += [f"_{NO_COUNTER[language]}_", ""]
            for counter in claim.counters:
                lines += [
                    counter.argument,
                    "",
                    f"> {counter.quote}",
                    ">",
                    f"> — {counter.citation}",
                    "",
                ]
    return "\n".join(lines).rstrip() + "\n"


def error(message: str) -> None:
    console.print(Text(message, style="error"))


def meta(message: str) -> None:
    console.print(Text(message, style="meta"))
