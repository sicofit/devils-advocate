"""Terminal rendering with rich. Dynamic text is always rendered literally, never as markup.

The interface itself (labels, summaries, commands) is in English. What answers the
user (claims, arguments, and the message when a claim gets no counterargument)
is in the language the position was written in.
"""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

from .devil import ClaimResult, Counter, Rebuttal
from .store import StoredPassage
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
errors = Console(theme=THEME, highlight=False, stderr=True)

# No passage was retrieved, or the model found none that argues against the claim.
NO_COUNTER = {
    "de": "Keine BGE-Passage gefunden, die dieser Behauptung widerspricht.",
    "fr": "Aucun passage ATF trouvé qui contredise cette affirmation.",
    "it": "Nessun passaggio DTF trovato che contraddica questa affermazione.",
    "en": "No BGE passage found that contradicts this claim.",
}
# Counterarguments were proposed, but every one failed verification.
NONE_VERIFIED = {
    "de": "Vorgeschlagene Gegenargumente bestanden die Prüfung am zitierten Text nicht "
    "und werden daher nicht gezeigt.",
    "fr": "Les contre-arguments proposés n'ont pas passé la vérification sur le texte cité ; "
    "ils ne sont donc pas affichés.",
    "it": "I controargomenti proposti non hanno superato la verifica sul testo citato "
    "e quindi non vengono mostrati.",
    "en": "Counterarguments were proposed, but none passed verification against its cited "
    "passage, so none is shown.",
}
# The model's reply for the claim could not be used.
FAILED = {
    "de": "Die Antwort des Modells zu dieser Behauptung war unbrauchbar; sie wurde übersprungen.",
    "fr": "La réponse du modèle pour cette affirmation était inutilisable ; elle a été ignorée.",
    "it": "La risposta del modello per questa affermazione era inutilizzabile ed è stata saltata.",
    "en": "The model's reply for this claim could not be used, so the claim was skipped.",
}
NO_CLAIMS = {
    "de": "Darin steckt keine rechtliche Behauptung, gegen die ich argumentieren könnte.",
    "fr": "Je ne trouve aucune affirmation juridique à contester.",
    "it": "Non trovo alcuna affermazione giuridica da contestare.",
    "en": "I found no legal claim in that to argue against.",
}
MARKS = {
    Status.EXACT: ("✓ verbatim", "ok"),
    # The model's quote differed by an inflection or a clipped last word; what is
    # shown is the court's text, not the model's.
    Status.CLOSE: ("≈ close match, shown in the court's wording", "close"),
}
# The quote restates what the lower court held or a party argued, not the court's view.
REPORTED = "reports the lower court's or a party's view"


def render(rebuttal: Rebuttal, *, show_discarded: bool = False, conversation: bool = True) -> None:
    language = _language(rebuttal)
    if not rebuttal.claims:
        console.print(Text(NO_CLAIMS[language], style="abstain"))
        return
    for number, claim in enumerate(rebuttal.claims, 1):
        # The claim heads the panel's body, not its title: a title is cut to one line.
        parts: list[Text] = [Text(f"{number}. {claim.claim}", style="claim"), Text()]
        if claim.abstained:
            parts += [Text(no_counter(claim, language), style="abstain"), Text()]
        for counter in claim.counters:
            parts += _counter(counter)
        for counter, earlier in claim.repeated:
            cite = Text(counter.citation, style="cite")
            cite.append(f"  quoted under claim {earlier}", style="meta")
            parts += [cite, Text(counter.argument), Text()]
        if show_discarded:
            parts += _discarded(claim)
        while parts and not parts[-1].plain:
            parts.pop()
        console.print(Panel(Group(*parts), border_style="meta"))
    console.print(Text(_summary(rebuttal, show_discarded, conversation), style="meta"))


def render_discarded(rebuttal: Rebuttal) -> None:
    """Only the discarded and set-aside proposals of `rebuttal`, by claim."""
    shown = False
    for number, claim in enumerate(rebuttal.claims, 1):
        lines = _discarded(claim)
        if lines:
            console.print(Text(f"{number}. {claim.claim}", style="claim"))
            for line in lines:
                console.print(line)
            shown = True
    if not shown:
        console.print(Text("Nothing was discarded in the last answer.", style="meta"))


def render_sources(rebuttal: Rebuttal) -> None:
    """Each passage quoted in `rebuttal` in full, with the quoted parts in bold."""
    quoted: dict[int, tuple[StoredPassage, list[tuple[int, int]]]] = {}
    for claim in rebuttal.claims:
        for counter in claim.counters:
            _, spans = quoted.setdefault(counter.passage.id, (counter.passage, []))
            spans.extend(counter.spans)
    for passage, spans in quoted.values():
        body = Text(passage.text)
        for start, end in spans:
            body.stylize("bold", start, end)
        console.print(Panel(body, title=Text(passage.citation, style="cite"), title_align="left"))
    if not quoted:
        console.print(Text("No sources in the last answer.", style="meta"))


def no_counter(claim: ClaimResult, language: str) -> str:
    """Why `claim` has no counterargument, in the user's language."""
    if claim.error:
        return FAILED[language]
    if claim.discarded:
        return NONE_VERIFIED[language]
    return NO_COUNTER[language]


def to_markdown(turns: list[Rebuttal]) -> str:
    """The conversation as a Markdown memo, quotes and citations included."""
    lines = ["# devils-advocate", ""]
    for turn in turns:
        lines += ["## Position", "", turn.position, ""]
        language = _language(turn)
        if not turn.claims:
            lines += [f"_{NO_CLAIMS[language]}_", ""]
        for number, claim in enumerate(turn.claims, 1):
            lines += [f"### {number}. {claim.claim}", ""]
            if claim.abstained:
                lines += [f"_{no_counter(claim, language)}_", ""]
            for counter in claim.counters:
                mark, _ = MARKS[counter.status]
                if counter.reported:
                    mark += f"; {REPORTED}"
                lines += [
                    f"**{counter.citation}** ({mark})",
                    "",
                    f"> {one_line(counter.shown)}",
                    "",
                    counter.argument,
                    "",
                ]
            for counter, earlier in claim.repeated:
                lines += [
                    f"**{counter.citation}** (quoted under claim {earlier})",
                    "",
                    counter.argument,
                    "",
                ]
    return "\n".join(lines).rstrip() + "\n"


def one_line(text: str) -> str:
    """A quote as one line: the court's paragraph breaks are not part of the sentence."""
    return " ".join(text.split())


def error(message: str) -> None:
    errors.print(Text(message, style="error"))


def meta(message: str) -> None:
    """A status line. It goes to stderr, so that stdout carries only the answer."""
    errors.print(Text(message, style="meta"))


def _counter(counter: Counter) -> list[Text]:
    """Citation first, then the court's words, then the argument built on them."""
    mark, style = MARKS[counter.status]
    cite = Text(counter.citation, style="cite")
    cite.append(f"  {mark}", style=style)
    if counter.reported:
        cite.append(f"  ! {REPORTED}", style="close")
    quote = Text("„" + one_line(counter.shown) + "“", style="quote")
    return [cite, quote, Text(counter.argument), Text()]


def _discarded(claim: ClaimResult) -> list[Text]:
    """The proposals that failed a check (✗), then those set aside without failing one (–)."""
    lines = []
    for mark, items in (("✗", claim.discarded), ("–", claim.skipped)):
        for item in items:
            where = f"{item.citation}: " if item.citation else ""
            line = Text(f"  {mark} {where}{item.reason}. ", style="meta")
            quote = one_line(item.quote)
            line.append(f"„{quote[:160]}{'…' if len(quote) > 160 else ''}“", style="meta")
            lines.append(line)
    if claim.error:
        lines.append(Text(f"  ✗ {claim.error}", style="meta"))
    return lines


def _summary(rebuttal: Rebuttal, show_discarded: bool, conversation: bool) -> str:
    verified = rebuttal.verified
    summary = f"{verified} verified {'quote' if verified == 1 else 'quotes'}"
    dropped = rebuttal.proposed - verified
    if dropped:
        summary += f", {dropped} discarded on verification"
        if not show_discarded:
            summary += " (/discarded shows them)" if conversation else " (--show-discarded)"
    summary += f" · {rebuttal.seconds:.0f} s"
    if conversation and verified:
        summary += " · /sources for the full passages"
    return summary


def _language(rebuttal: Rebuttal) -> str:
    return rebuttal.language if rebuttal.language in NO_COUNTER else "en"
