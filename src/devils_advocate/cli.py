"""Command line: `devil` opens the conversation.

Subcommands fetch data, build indexes and run the evaluations.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from rich.text import Text

from . import __version__, config, corpus, store, ui
from .client import LLMClient, LLMError
from .devil import Devil, Rebuttal
from .search import Retriever, VectorIndex, build_vectors, vectors_path

HELP = """\
Type a legal position and the devil argues against it with BGE precedent.
Reply to its objections to continue the argument.

  /sources        full text of the passages behind the last answer
  /discarded      toggle showing quotes that failed verification
  /export [file]  save the conversation as Markdown
  /new            forget the conversation so far
  /help           this text
  /quit           leave"""


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
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
    except KeyboardInterrupt:
        ui.meta("interrupted")
        return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devil",
        description="Argues against your legal position with Swiss Federal Supreme Court "
        "precedent, and shows a quote only if it is verbatim in the cited Erwägung.",
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
    revision = corpus.dataset_revision(args.revision)
    ui.meta(f"OpenCaseLaw {corpus.DATASET} @ {revision[:12]} (CC0)")
    with ui.console.status("downloading") as status:

        def progress(done: int, total: int | None) -> None:
            mb = done / 1e6
            status.update(
                f"downloading {mb:.0f} MB" + (f" of {total / 1e6:.0f} MB" if total else "")
            )

        corpus.download(cfg.corpus_path, revision=revision, on_progress=progress)
    source = {
        "dataset": corpus.DATASET,
        "revision": revision,
        "file": corpus.FILE,
        "sha256": corpus.sha256(cfg.corpus_path),
    }
    (cfg.data_dir / "corpus.json").write_text(json.dumps(source), encoding="utf-8")
    ui.console.print(Text(f"saved {cfg.corpus_path}", style="ok"))
    return 0


def cmd_index(cfg: config.Config, args: argparse.Namespace) -> int:
    if not cfg.corpus_path.exists():
        ui.error("No corpus yet. Run `devil fetch` first.")
        return 1
    since = args.since if args.since is not None else cfg.since
    source = _corpus_meta(cfg)
    with ui.console.status("indexing") as status:
        decisions, passages = store.build(
            cfg.db_path,
            corpus.read(cfg.corpus_path, since=since),
            {
                "since": str(since),
                "revision": source.get("revision", "unknown"),
                "sha256": source.get("sha256", "unknown"),
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
    client = LLMClient(cfg)
    path = vectors_path(cfg.data_dir, cfg.embed_model)
    with store.Store(cfg.db_path) as db, ui.console.status("embedding") as status:
        build_vectors(
            db,
            client,
            cfg.embed_model,
            path,
            batch=args.batch,
            on_progress=lambda d, t: status.update(f"embedding {d}/{t} passages"),
        )
    ui.console.print(Text(f"saved {path}", style="ok"))
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
        with store.Store(cfg.db_path) as db:
            meta = db.meta()
        rows.append(
            (
                "indexed",
                f"{meta.get('decisions')} decisions, {meta.get('passages')} passages "
                f"since {meta.get('since')}, dataset {meta.get('revision', '?')[:12]}",
            )
        )
    rows.append(("model reachable", "yes" if LLMClient(cfg).health() else "no"))
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
        client = LLMClient(cfg)
        retriever = open_retriever(cfg, db, client, mode)
        position = " ".join(getattr(args, "position", None) or []).strip()
        if position:
            devil = Devil(client, retriever, top_k=cfg.top_k, max_claims=cfg.max_claims)
            rebuttal = _with_status(devil, position)
            if args.json:
                print(json.dumps(rebuttal_to_dict(rebuttal), ensure_ascii=False, indent=2))
            else:
                ui.render(rebuttal, show_discarded=args.show_discarded)
            return 0
        return _conversation(cfg, db, client, retriever)


def open_retriever(cfg: config.Config, db: store.Store, client: LLMClient, mode: str) -> Retriever:
    path = vectors_path(cfg.data_dir, cfg.embed_model)
    wants_dense = mode in ("dense", "hybrid") or (mode == "auto" and path.exists())
    if not wants_dense:
        return Retriever(db)
    if not path.exists():
        raise SystemExit(
            f"No vectors for {cfg.embed_model}. Run `devil embed` or use --retriever bm25."
        )
    vectors = VectorIndex.load(path)
    if mode == "dense":
        return _DenseOnly(db, client, vectors)
    return Retriever(db, client, vectors)


class _DenseOnly(Retriever):
    def bm25(self, query, k, exclude=frozenset()):  # noqa: ANN001, ANN201 - same signature as base
        return []


def _conversation(
    cfg: config.Config, db: store.Store, client: LLMClient, retriever: Retriever
) -> int:
    meta = db.meta()
    ui.console.print(
        Text("devils-advocate", style="bold")
        + Text(
            f"  ·  {meta.get('decisions')} BGE decisions since {meta.get('since')}"
            f"  ·  {cfg.model}  ·  {'hybrid' if retriever.has_dense else 'bm25'} retrieval",
            style="meta",
        )
    )
    ui.meta("State your position. /help for commands.")
    devil = Devil(client, retriever, top_k=cfg.top_k, max_claims=cfg.max_claims)
    turns: list[Rebuttal] = []
    show_discarded = False
    while True:
        try:
            line = ui.console.input("[bold]you › [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            ui.console.print()
            return 0
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
            elif command == "/export":
                target = Path(rest.strip() or f"devils-advocate-{date.today().isoformat()}.md")
                target.write_text(ui.to_markdown(turns), encoding="utf-8")
                ui.meta(f"Saved {target}")
            else:
                ui.meta(f"Unknown command {command}. /help lists them.")
            continue
        try:
            rebuttal = _with_status(devil, line)
        except LLMError as exc:
            ui.error(str(exc))
            continue
        turns.append(rebuttal)
        ui.render(rebuttal, show_discarded=show_discarded)


def _with_status(devil: Devil, position: str) -> Rebuttal:
    with ui.console.status("thinking") as status:
        devil.on_stage = lambda message: status.update(message)
        return devil.argue(position)


def rebuttal_to_dict(rebuttal: Rebuttal) -> dict:
    return {
        "position": rebuttal.argument,
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
                    }
                    for c in claim.counters
                ],
                "discarded": [asdict(d) for d in claim.discarded],
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
    path = vectors_path(cfg.data_dir, cfg.embed_model)
    if not path.exists() and any(m.split("+")[0] != "bm25" for m in modes):
        ui.meta(f"No vectors for {cfg.embed_model}; skipping dense and hybrid.")
        modes = [m for m in modes if m.split("+")[0] == "bm25"]
    if args.translate:
        modes += [f"{m}+translate" for m in modes if "+" not in m]
    client = LLMClient(cfg)
    with store.Store(cfg.db_path) as db, ui.console.status("building queries") as status:
        retriever = (
            Retriever(db, client, VectorIndex.load(path)) if path.exists() else Retriever(db)
        )
        queries = bench.build_queries(db, n=args.n, seed=args.seed)
        translations = None
        if args.translate:
            translations = bench.translate(
                client,
                queries,
                args.out / "translations.json",
                on_progress=lambda i, n: status.update(f"translating query {i}/{n}"),
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
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "retrieval.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    ui.console.print(Text(bench.table(report)))
    ui.meta(f"saved {target}")
    return 0


def cmd_eval(cfg: config.Config, args: argparse.Namespace) -> int:
    from . import evaluation

    if not cfg.db_path.exists():
        ui.error("No index yet. Run `devil index` first.")
        return 1
    cases = evaluation.load_cases(args.cases)
    with store.Store(cfg.db_path) as db:
        client = LLMClient(cfg)
        retriever = open_retriever(cfg, db, client, args.retriever)
        with ui.console.status("evaluating") as status:
            report = evaluation.run(
                cases,
                lambda: Devil(client, retriever, top_k=cfg.top_k, max_claims=cfg.max_claims),
                on_progress=lambda i, n, case_id: status.update(f"case {i}/{n}: {case_id}"),
            )
        report["index"] = db.meta()
    report["model"] = cfg.model
    report["retriever"] = (
        "dense"
        if isinstance(retriever, _DenseOnly)
        else "hybrid"
        if retriever.has_dense
        else "bm25"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "end_to_end.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "badge.json").write_text(
        json.dumps(evaluation.badge(report), indent=2), encoding="utf-8"
    )
    ui.console.print(Text(evaluation.table(report)))
    ui.meta(f"saved {target}")
    return 0


# -- helpers -----------------------------------------------------------------


def _corpus_meta(cfg: config.Config) -> dict[str, str]:
    path = cfg.data_dir / "corpus.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _present(path: Path) -> str:
    if not path.exists():
        return "missing"
    return f"{path.stat().st_size / 1e6:.0f} MB"


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
