"""Command line: `devil` opens the conversation, `devil <position>` opens it with a first position.

Subcommands fetch data, build indexes and run the evaluations.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import sqlite3
import sys
import urllib.error
from collections.abc import Iterator
from datetime import date
from itertools import chain, count
from pathlib import Path

from rich.text import Text

from . import __version__, config, corpus, prompts, store, ui
from .client import LLMClient, LLMError
from .devil import Devil, Rebuttal
from .search import Retriever, VectorIndex, build_vectors, vectors_path

HELP = """\
Type a legal position and the devil argues against it with BGE precedent.
Reply to its objections to continue the argument. End a line with \\ to
continue it on the next.

  /sources        full text of the passages behind the last answer
  /discarded      toggle showing proposals that failed a check or were set aside
  /export [file]  save the conversation as Markdown
  /new            forget the conversation so far
  /help           this text
  /quit           leave (also /exit, /q)"""


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    argv, opening = _split_opening(sys.argv[1:] if argv is None else list(argv))
    parser = _parser()
    args = parser.parse_args(argv)
    args.opening = opening
    cfg = config.load(
        data_dir=args.data_dir,
        model=args.model,
        embed_model=args.embed_model,
        base_url=args.base_url,
    )
    command = args.command or "argue"
    try:
        return COMMANDS[command](cfg, args)
    except LLMError as exc:
        ui.error(str(exc))
        return 2
    except store.IndexProblem as exc:
        ui.error(str(exc))
        return 1
    except KeyboardInterrupt:
        ui.meta("interrupted")
        return 130


def _utf8_stdio() -> None:
    """Read and write UTF-8 even where the console or a pipe defaults to cp1252.

    Quotes are full of „“«»≈✓, which would otherwise crash `devil argue --json > out.json`
    on Windows. Characters that still cannot be written become "?" instead of an error.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        # Skipped where a test harness replaced the stream or it is already in use.
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


_GLOBAL_OPTIONS_WITH_VALUE = {"--data-dir", "--model", "--embed-model", "--base-url"}
_GLOBAL_OPTIONS = _GLOBAL_OPTIONS_WITH_VALUE | {"--version", "--help"}
# Options of `devil argue`: typed after an opening position, they would become part of it.
_ARGUE_OPTIONS = {"--json": "--json", "--show-discarded": "--show-discarded"}
_ARGUE_OPTIONS["--retriever"] = "--retriever <name>"


def _option(word: str) -> str:
    """The option `word` names: "--model" for "--model=m" or for the abbreviation "--mod"."""
    name = word.split("=", 1)[0]
    if not name.startswith("--") or len(name) < 3:
        return name
    known = _GLOBAL_OPTIONS | set(_ARGUE_OPTIONS)
    if name in known:
        return name
    matches = [option for option in known if option.startswith(name)]
    return matches[0] if len(matches) == 1 else name


def _split_opening(argv: list[str]) -> tuple[list[str], str]:
    """`devil Ein Vertrag braucht nur Konsens.` opens the conversation with that position.

    Returns the arguments for the parser and the opening position, if the first
    word after the global options is not a subcommand.
    """
    i = 0
    while i < len(argv) and argv[i].startswith("-"):
        takes_value = "=" not in argv[i] and _option(argv[i]) in _GLOBAL_OPTIONS_WITH_VALUE
        i += 2 if takes_value else 1
    if i >= len(argv) or argv[i] in COMMANDS:
        return argv, ""
    for word in argv[i:]:
        option = _option(word)
        if option in _ARGUE_OPTIONS:
            raise SystemExit(
                f"devil: {option} after the position would be read as part of it. "
                f"Use devil argue {_ARGUE_OPTIONS[option]} <position>."
            )
        if option in _GLOBAL_OPTIONS or option == "-h":
            raise SystemExit(
                f"devil: {option} after the position would be read as part of it. "
                "Options go before the position."
            )
    return argv[:i], " ".join(argv[i:]).lstrip("\ufeff").strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devil",
        usage="devil [options] [position ... | command ...]",
        description="Argues against your legal position with Swiss Federal Supreme Court "
        "precedent, and shows a quote only if it is verbatim in the passage it cites.",
        epilog="`devil` opens the conversation. `devil <position>` opens it and answers that "
        "position first. `devil argue <position>` answers once and exits.",
    )
    parser.add_argument("--version", action="version", version=f"devils-advocate {__version__}")
    parser.add_argument("--data-dir", help="where the corpus and indexes live")
    parser.add_argument("--model", help="chat model name")
    parser.add_argument("--embed-model", help="embedding model name")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint")
    sub = parser.add_subparsers(dest="command")

    argue = sub.add_parser("argue", help="argue against a position (default)")
    argue.add_argument("position", nargs="*", help="answer once instead of opening a conversation")
    argue.add_argument("--retriever", choices=["auto", "bm25", "dense", "hybrid"], default="auto")
    argue.add_argument("--json", action="store_true", help="print the result as JSON")
    argue.add_argument("--show-discarded", action="store_true")

    fetch = sub.add_parser("fetch", help="download the BGE corpus from OpenCaseLaw")
    fetch.add_argument("--revision", default="main", help="dataset branch or commit")

    index = sub.add_parser("index", help="split decisions into passages and build the index")
    index.add_argument("--since", type=int, help="earliest decision year (default 2000)")

    embed = sub.add_parser("embed", help="embed all passages for dense retrieval (optional)")
    embed.add_argument("--batch", type=int, default=8)

    sub.add_parser("status", help="show configuration and what is installed")

    bench = sub.add_parser("bench", help="retrieval benchmark built from the court's own citations")
    bench.add_argument("--n", type=int, default=1000, help="number of queries")
    bench.add_argument("--modes", default="bm25,dense,hybrid")
    bench.add_argument("--out", type=Path, default=Path("evals/results"))
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument(
        "--translate",
        action="store_true",
        help="also search with German and French translations of each query (uses the chat model)",
    )

    evaluate = sub.add_parser("eval", help="end-to-end evaluation on hand-written positions")
    evaluate.add_argument("--cases", type=Path, default=Path("evals/positions.jsonl"))
    evaluate.add_argument(
        "--retriever", choices=["auto", "bm25", "dense", "hybrid"], default="auto"
    )
    evaluate.add_argument("--out", type=Path, default=Path("evals/results"))
    return parser


# -- data --------------------------------------------------------------------


def cmd_fetch(cfg: config.Config, args: argparse.Namespace) -> int:
    try:
        revision = corpus.dataset_revision(args.revision)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            ui.error(f"Hugging Face has no revision {args.revision!r} of {corpus.DATASET}.")
        else:
            ui.error(
                f"Hugging Face answered HTTP {exc.code} for {corpus.DATASET}. Try again later."
            )
        return 1
    except (OSError, ValueError, KeyError) as exc:  # offline, or an unexpected reply
        ui.error(f"Could not reach Hugging Face to look up the dataset ({exc}).")
        return 1
    ui.meta(f"OpenCaseLaw {corpus.DATASET} @ {revision[:12]} (CC0)")
    try:
        expected = corpus.expected_sha256(revision)
    except (OSError, ValueError, KeyError) as exc:  # an unexpected reply
        expected = None
        ui.meta(f"Could not look up the published hash ({exc}); the download is not checked.")
    else:
        if expected is None:
            ui.meta("Hugging Face publishes no hash for the file; the download is not checked.")
    with ui.console.status("downloading") as status:

        def progress(done: int, total: int | None) -> None:
            mb = done / 1e6
            status.update(
                f"downloading {mb:.0f} MB" + (f" of {total / 1e6:.0f} MB" if total else "")
            )

        try:
            corpus.download(
                cfg.corpus_path, revision=revision, expected=expected, on_progress=progress
            )
        except corpus.DamagedDownload as exc:
            ui.error(f"{exc} Run `devil fetch` again.")
            return 1
        except (OSError, http.client.HTTPException) as exc:  # resumed by the next fetch
            ui.error(f"The download stopped ({exc}). Run `devil fetch` again to resume it.")
            return 1
        status.update("checking the download")
        digest = corpus.sha256(cfg.corpus_path)
    source = {
        "dataset": corpus.DATASET,
        "revision": revision,
        "file": corpus.FILE,
        "sha256": digest,
    }
    (cfg.data_dir / "corpus.json").write_text(json.dumps(source), encoding="utf-8")
    ui.console.print(Text(f"saved {cfg.corpus_path}", style="ok"), soft_wrap=True)
    return 0


def cmd_index(cfg: config.Config, args: argparse.Namespace) -> int:
    if not cfg.corpus_path.exists():
        ui.error("No corpus yet. Run `devil fetch` first.")
        return 1
    since = args.since if args.since is not None else cfg.since
    low, high, meaning = config.LIMITS["since"]
    if not low <= since <= high:
        ui.error(f"--since must be {meaning}, not {since}.")
        return 1
    source = _corpus_meta(cfg)
    with ui.console.status("checking the corpus"):
        digest = corpus.sha256(cfg.corpus_path)
    revision = source.get("revision", "unknown")
    if source.get("sha256") != digest:
        ui.meta("corpus.json does not describe this corpus file; its revision is unknown.")
        revision = "unknown"
    with ui.console.status("indexing") as status:
        decisions, passages = store.build(
            cfg.db_path,
            corpus.read(cfg.corpus_path, since=since),
            {
                "since": str(since),
                "revision": revision,
                "sha256": digest,
                "built": date.today().isoformat(),
            },
            on_progress=lambda d, p: status.update(f"indexing: {d} decisions, {p} passages"),
        )
    ui.console.print(Text(f"indexed {decisions} decisions, {passages} passages", style="ok"))
    return 0


def cmd_embed(cfg: config.Config, args: argparse.Namespace) -> int:
    if not cfg.db_path.exists():
        ui.error("No index yet. Run `devil index` first.")
        return 1
    if args.batch < 1:
        ui.error(f"--batch must be at least 1, not {args.batch}.")
        return 1
    client = LLMClient(cfg)
    path = vectors_path(cfg.data_dir, cfg.embed_model)
    with store.Store(cfg.db_path) as db, ui.console.status("embedding") as status:
        db.check()
        build_vectors(
            db,
            client,
            cfg.embed_model,
            path,
            batch=args.batch,
            on_progress=lambda d, t: status.update(f"embedding {d}/{t} passages"),
        )
    ui.console.print(Text(f"saved {path}", style="ok"), soft_wrap=True)
    return 0


def cmd_status(cfg: config.Config, args: argparse.Namespace) -> int:
    ui.console.print(f"devils-advocate {__version__}")
    rows = [
        ("endpoint", cfg.endpoint),
        ("model", cfg.model),
        ("embedding model", cfg.embed_model),
        ("data dir", str(cfg.data_dir)),
        ("corpus", _present(cfg.corpus_path)),
        ("index", _present(cfg.db_path)),
        ("vectors", _present(vectors_path(cfg.data_dir, cfg.embed_model))),
    ]
    if cfg.db_path.exists():
        try:
            with store.Store(cfg.db_path) as db:
                meta = db.meta()
        except sqlite3.DatabaseError:
            rows.append(("indexed", "unreadable (damaged or not an index); run `devil index`"))
        else:
            rows.append(
                (
                    "indexed",
                    f"{meta.get('decisions')} decisions, {meta.get('passages')} passages "
                    f"since {meta.get('since')}, dataset {meta.get('revision', '?')[:12]}",
                )
            )
    problem = LLMClient(cfg).problem()
    rows.append(("model server", "ready" if problem is None else problem))
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        ui.console.print(Text(f"  {name:<{width}}  ", style="meta") + Text(value))
    return 0


# -- conversation ------------------------------------------------------------


def cmd_argue(cfg: config.Config, args: argparse.Namespace) -> int:
    if not cfg.db_path.exists():
        ui.error("No index yet. Run `devil fetch` and `devil index` first.")
        return 1
    mode = getattr(args, "retriever", "auto")
    with store.Store(cfg.db_path) as db:
        db.check()
        client = LLMClient(cfg)
        retriever = open_retriever(cfg, db, client, mode)
        position = " ".join(getattr(args, "position", None) or []).strip()
        if position:
            devil = Devil(client, retriever, top_k=cfg.top_k, max_claims=cfg.max_claims)
            if args.json:  # stdout carries the JSON; status lines go to stderr
                rebuttal = devil.argue(position)
                print(json.dumps(rebuttal_to_dict(rebuttal), ensure_ascii=False, indent=2))
            else:
                rebuttal = _with_status(devil, position)
                ui.render(rebuttal, show_discarded=args.show_discarded, conversation=False)
            return 0
        if getattr(args, "json", False):
            ui.error("--json needs a position: devil argue --json <position>")
            return 1
        return _conversation(
            cfg,
            db,
            client,
            retriever,
            opening=getattr(args, "opening", ""),
            show_discarded=getattr(args, "show_discarded", False),
        )


def open_retriever(cfg: config.Config, db: store.Store, client: LLMClient, mode: str) -> Retriever:
    if mode == "bm25":
        return Retriever(db)
    vectors = _vectors(cfg, db, required=mode != "auto")
    if vectors is None:
        return Retriever(db)
    if mode == "dense":
        return _DenseOnly(db, client, vectors)
    if mode == "auto":
        return Retriever(db, client, vectors, on_dense_error=_dense_failed)
    return Retriever(db, client, vectors)


def _dense_failed(problem: str) -> None:
    ui.meta(f"The embedding model failed ({problem}). Using BM25 from now on.")


def retrieval_name(retriever: Retriever) -> str:
    if isinstance(retriever, _DenseOnly):
        return "dense"
    return "hybrid" if retriever.has_dense else "bm25"


def _vectors(cfg: config.Config, db: store.Store, *, required: bool) -> VectorIndex | None:
    """The vectors of `cfg.embed_model`, if they exist and fit the current index."""
    path = vectors_path(cfg.data_dir, cfg.embed_model)
    if not path.exists():
        if required:
            raise SystemExit(
                f"No vectors for {cfg.embed_model}. Run `devil embed` or use --retriever bm25."
            )
        return None
    try:
        vectors = VectorIndex.load(path)
    except (OSError, ValueError, KeyError) as exc:  # a missing, truncated or foreign file
        message = (
            f"The vectors for {cfg.embed_model} cannot be read ({exc}). "
            "Run `devil embed` again to use dense retrieval."
        )
        if required:
            raise SystemExit(message) from None
        ui.meta(message + " Using BM25 for now.")
        return None
    if vectors.matches(db):
        return vectors
    message = (
        f"The vectors for {cfg.embed_model} belong to an earlier index. "
        "Run `devil embed` again to use dense retrieval."
    )
    if required:
        raise SystemExit(message)
    ui.meta(message + " Using BM25 for now.")
    return None


class _DenseOnly(Retriever):
    def bm25(self, query, k, exclude=frozenset()):  # noqa: ANN001, ANN201 - same signature as base
        return []


def _typed_lines() -> Iterator[str]:
    """What the user types, one turn at a time.

    A line that ends in a backslash continues on the next. A Windows console hands
    over a multi-line paste one line at a time; the lines already waiting are joined
    into the same turn instead of becoming a turn each.
    """
    while True:
        parts: list[str] = []
        try:
            # A line piped in from PowerShell or Notepad may begin with a byte order mark.
            parts.append(ui.console.input("[bold]you › [/bold]").lstrip("\ufeff"))
            while parts[-1].rstrip().endswith("\\") or _more_waiting():
                if parts[-1].rstrip().endswith("\\"):
                    parts[-1] = parts[-1].rstrip()[:-1]
                parts.append(input())
        except EOFError:
            if parts:
                yield " ".join(" ".join(parts).split())
            ui.console.print()
            return
        except KeyboardInterrupt:
            ui.console.print()
            return
        yield " ".join(" ".join(parts).split())


def _more_waiting() -> bool:
    """Whether the console already holds more input: the rest of a pasted block."""
    if sys.platform != "win32" or not sys.stdin.isatty():
        return False
    import msvcrt
    import time

    deadline = time.monotonic() + 0.05  # a paste arrives in quick bursts
    while time.monotonic() < deadline:
        if msvcrt.kbhit():
            return True
        time.sleep(0.005)
    return False


def _conversation(
    cfg: config.Config,
    db: store.Store,
    client: LLMClient,
    retriever: Retriever,
    *,
    opening: str = "",
    show_discarded: bool = False,
) -> int:
    meta = db.meta()
    ui.console.print(
        Text("devils-advocate", style="bold")
        + Text(
            f"  ·  {meta.get('decisions')} BGE decisions since {meta.get('since')}"
            f"  ·  {cfg.model}  ·  {retrieval_name(retriever)} retrieval",
            style="meta",
        )
    )
    problem = client.problem()
    if problem:
        ui.error(problem)
    ui.meta("State your position. /help for commands.")
    devil = Devil(client, retriever, top_k=cfg.top_k, max_claims=cfg.max_claims)
    turns: list[Rebuttal] = []
    for line in chain([opening] if opening else [], _typed_lines()):
        if not line:
            continue
        if line.startswith("/"):
            command, _, rest = line.partition(" ")
            if command in ("/quit", "/exit", "/q"):
                return 0
            if command == "/help":
                ui.console.print(Text(HELP))
            elif command == "/new":
                devil.reset()
                turns.clear()
                ui.meta("New conversation.")
            elif command == "/sources":
                if turns:
                    ui.render_sources(turns[-1])
                else:
                    ui.meta("Nothing yet.")
            elif command == "/discarded":
                show_discarded = not show_discarded
                ui.meta(f"Showing discarded quotes: {'on' if show_discarded else 'off'}")
                if show_discarded and turns:
                    ui.render_discarded(turns[-1])
            elif command == "/export":
                _export(turns, rest)
            else:
                ui.meta(f"Unknown command {command}. /help lists them.")
            continue
        try:
            rebuttal = _with_status(devil, line)
        except LLMError as exc:
            ui.error(str(exc))
            continue
        except KeyboardInterrupt:  # Ctrl+C while the model works cancels this turn only
            ui.meta("Cancelled. /quit or Ctrl+C at the prompt leaves.")
            continue
        turns.append(rebuttal)
        ui.render(rebuttal, show_discarded=show_discarded)
    return 0


def _export(turns: list[Rebuttal], name: str) -> None:
    if not turns:
        ui.meta("Nothing to export yet.")
        return
    name = name.strip().strip("\"'")  # a path pasted from Explorer comes in quotes
    if name:
        target = Path(name).expanduser()
    else:
        # The default name never overwrites an earlier export.
        stem = f"devils-advocate-{date.today().isoformat()}"
        target = next(
            path
            for n in chain([1], count(2))
            if not (path := Path(stem + ("" if n == 1 else f"-{n}") + ".md")).exists()
        )
    try:
        target.write_text(ui.to_markdown(turns), encoding="utf-8")
    except OSError as exc:
        ui.error(f"Could not save {target}: {exc.strerror or exc}")
        return
    ui.meta(f"Saved {target.resolve()}")


def _with_status(devil: Devil, position: str) -> Rebuttal:
    with ui.console.status("thinking") as status:
        devil.on_stage = lambda message: status.update(message)
        return devil.argue(position)


def rebuttal_to_dict(rebuttal: Rebuttal) -> dict:
    return {
        "position": rebuttal.position,
        "language": rebuttal.language,
        "seconds": round(rebuttal.seconds, 1),
        "claims": [
            {
                "claim": claim.claim,
                "queries": claim.queries,
                "counterarguments": [
                    {
                        "argument": c.argument,
                        "quote": c.quote,
                        "citation": c.citation,
                        "match": c.status.value,
                        # The quote starts or stops inside a sentence of the court.
                        "starts_mid_sentence": c.opens_inside,
                        "ends_mid_sentence": c.ends_inside,
                        "reports_lower_court_or_party": c.reported,
                    }
                    for c in claim.counters
                ],
                # Verified, but the passage is quoted under an earlier claim already.
                "same_passage_as_claim": [
                    {"argument": c.argument, "citation": c.citation, "claim": earlier}
                    for c, earlier in claim.repeated
                ],
                "discarded": [
                    {"reason": d.reason, "quote": d.quote, "citation": d.citation}
                    for d in claim.discarded
                ],
                # Set aside without failing a check: not counted as discarded.
                "skipped": [
                    {"reason": d.reason, "quote": d.quote, "citation": d.citation}
                    for d in claim.skipped
                ],
                "error": claim.error or None,
                "retrieved": [hit.passage.citation for hit in claim.hits],
            }
            for claim in rebuttal.claims
        ],
    }


# -- evaluation --------------------------------------------------------------


def cmd_bench(cfg: config.Config, args: argparse.Namespace) -> int:
    from . import bench

    if not cfg.db_path.exists():
        ui.error("No index yet. Run `devil index` first.")
        return 1
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        base, plus, extra = mode.partition("+")
        if base not in BENCH_MODES or (plus and extra != "translate"):
            ui.error(
                f"Unknown mode {mode!r}. Modes: {', '.join(BENCH_MODES)}, each optionally "
                "with +translate."
            )
            return 1
        if plus and not args.translate:
            ui.error(f"Mode {mode!r} needs --translate.")
            return 1
    if not modes or args.n < 1:
        ui.error("Give at least one mode and --n of at least 1.")
        return 1
    client = LLMClient(cfg)
    with store.Store(cfg.db_path) as db:
        db.check()
        vectors = _vectors(cfg, db, required=False)
    if vectors is None and any(m.split("+")[0] != "bm25" for m in modes):
        ui.meta(f"No usable vectors for {cfg.embed_model}; skipping dense and hybrid.")
        modes = [m for m in modes if m.split("+")[0] == "bm25"]
        if not modes:
            ui.error("No mode left to run: dense and hybrid need vectors (run `devil embed`).")
            return 1
    if args.translate:
        modes = list(dict.fromkeys(modes + [f"{m}+translate" for m in modes if "+" not in m]))
    with store.Store(cfg.db_path) as db, ui.console.status("building queries") as status:
        retriever = Retriever(db, client, vectors) if vectors is not None else Retriever(db)
        queries = bench.build_queries(db, n=args.n, seed=args.seed)
        translations = None
        if args.translate:
            translations = bench.translate(
                client,
                queries,
                args.out / "translations.json",
                on_progress=lambda i, n: status.update(f"translating query {i}/{n}"),
                model_name=cfg.model,
            )
        report = bench.run(
            retriever,
            queries,
            modes,
            on_progress=lambda mode, i, n: status.update(f"{mode}: query {i}/{n}"),
            translations=translations,
        )
        report["index"] = db.meta()
    report["seed"] = args.seed
    report["embed_model"] = cfg.embed_model if retriever.has_dense else None
    report["translation_model"] = cfg.model if args.translate else None
    if translations is not None:
        # A query the model could not translate is searched as is, which makes its
        # +translate score that of the plain mode; one with a single translation is
        # searched with that one only.
        missing = sum(not found for found in translations.values())
        partial = sum(len(found) == 1 for found in translations.values())
        report["translations_missing"] = missing
        report["translations_partial"] = partial
        n = len(translations)
        if missing:
            ui.meta(
                f"{missing} of {n} queries have no translation; their +translate scores "
                "are those of the plain search. Run bench again to retry."
            )
        if partial:
            ui.meta(
                f"{partial} of {n} queries have only a German or only a French translation "
                "and were searched with that one. Run bench again to retry."
            )
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "retrieval.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    ui.console.print(Text(bench.table(report)), soft_wrap=True)
    ui.errors.print(Text(f"saved {target}", style="meta"), soft_wrap=True)
    return 0


def cmd_eval(cfg: config.Config, args: argparse.Namespace) -> int:
    from . import evaluation

    if not cfg.db_path.exists():
        ui.error("No index yet. Run `devil index` first.")
        return 1
    if not args.cases.exists():
        ui.error(
            f"No cases at {args.cases}. Run this in a clone of the repository or pass --cases."
        )
        return 1
    try:
        cases = evaluation.load_cases(args.cases)
    except ValueError as exc:
        ui.error(str(exc))
        return 1
    with store.Store(cfg.db_path) as db:
        db.check()
        client = LLMClient(cfg)
        retriever = open_retriever(cfg, db, client, args.retriever)
        with ui.console.status("evaluating") as status:
            report = evaluation.run(
                cases,
                lambda: Devil(client, retriever, top_k=cfg.top_k, max_claims=cfg.max_claims),
                on_progress=lambda i, n, case_id: status.update(Text(f"case {i}/{n}: {case_id}")),
            )
        report["index"] = db.meta()
    report["model"] = cfg.model
    report["retriever"] = retrieval_name(retriever)
    # What else decides the result. One run at temperature > 0 is one sample.
    report["settings"] = {
        "version": __version__,
        "temperature": cfg.temperature,
        "reasoning_effort": cfg.reasoning_effort,
        "top_k": cfg.top_k,
        "max_claims": cfg.max_claims,
        "prompts_sha256": prompts.fingerprint(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "end_to_end.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "badge.json").write_text(
        json.dumps(evaluation.badge(report), indent=2), encoding="utf-8"
    )
    ui.console.print(Text(evaluation.table(report)), soft_wrap=True)
    ui.errors.print(Text(f"saved {target}", style="meta"), soft_wrap=True)
    return 0


# -- helpers -----------------------------------------------------------------


def _corpus_meta(cfg: config.Config) -> dict[str, str]:
    path = cfg.data_dir / "corpus.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _present(path: Path) -> str:
    if not path.exists():
        return "missing"
    return f"{path.stat().st_size / 1e6:.0f} MB"


BENCH_MODES = ("bm25", "dense", "hybrid")

COMMANDS = {
    "argue": cmd_argue,
    "fetch": cmd_fetch,
    "index": cmd_index,
    "embed": cmd_embed,
    "status": cmd_status,
    "bench": cmd_bench,
    "eval": cmd_eval,
}

if __name__ == "__main__":
    sys.exit(main())
