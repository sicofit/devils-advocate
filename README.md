# devils-advocate

[![CI](https://github.com/sicofit/devils-advocate/actions/workflows/ci.yml/badge.svg)](https://github.com/sicofit/devils-advocate/actions/workflows/ci.yml)
[![eval](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/sicofit/devils-advocate/main/evals/results/badge.json)](#end-to-end)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Data: CC0](https://img.shields.io/badge/data-OpenCaseLaw%20CC0-lightgrey)](https://huggingface.co/datasets/voilaj/swiss-caselaw)

You state a legal position. The devil argues against it with decisions of the
Swiss Federal Supreme Court, and it shows a quote only if that quote is
verbatim in the Erwägung it cites. If no quote holds up, it says so instead of
making one up.

A real run (the position: *the landlord may terminate the lease immediately
after the first late rent payment*):

```text
$ devil argue "Le bailleur peut résilier le bail immédiatement dès le premier retard de paiement du loyer."
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Le bailleur dispose d'un droit à la résiliation immédiate du bail dès le premier retard │
│ de paiement du loyer.                                                                      │
│                                                                                            │
│ Cet arrêt établit que la résiliation immédiate n'est pas possible dès le premier retard.   │
│ La loi exige impérativement un délai de grâce d'au moins trente jours accordé au locataire │
│ pour payer l'arriéré avant que le bailleur ne puisse invoquer la résiliation.              │
│   „L'art. 257d al. 1 CO prévoit ainsi que, lorsque, après la réception de la chose, le     │
│ locataire a du retard pour s'acquitter d'un terme ou de frais accessoires échus, le        │
│ bailleur, s'agissant de baux d'habitations ou de locaux commerciaux, peut lui fixer par    │
│ écrit un délai de paiement d'au moins trente jours, avec l'avis comminatoire qu'à défaut   │
│ de paiement dans ce délai il résiliera le bail.“                                           │
│   — BGE 127 III 548 E. 3 S. 550 f.  ✓ verbatim                                             │
│                                                                                            │
│ Ces passages démontrent que la résiliation ne peut pas être immédiate. La loi impose un    │
│ délai de préavis spécifique (trois mois) et des formalités précises (notification avec     │
│ motif dix jours avant), ce qui contredit directement l'idée d'une résiliation effective    │
│ dès le premier retard.                                                                     │
│   „L'avis de majoration du loyer, avec indication des motifs, doit parvenir au locataire   │
│ dix jours au moins avant le début du délai de résiliation et être effectué au moyen d'une  │
│ formule agréée par le canton.                                                              │
│ Le préavis de résiliation est de trois mois pour les habitations (art. 266c in initio      │
│ CO).“                                                                                      │
│   — BGE 131 III 566 E. 3.3 S. 570 f.  ✓ verbatim                                           │
│                                                                                            │
│ Ce passage précise que pour obtenir l'expulsion suite à un défaut de paiement, il faut     │
│ d'abord que le bail ait été résilié valablement. Cela implique que la simple constatation  │
│ d'un retard ne suffit pas pour une résiliation immédiate ; celle-ci doit suivre les        │
│ conditions légales (comme le délai de grâce) pour être valide.                             │
│   „L'action en expulsion pour défaut de paiement du loyer au sens de l'art. 257d CO, comme │
│ celle pour défaut de paiement du fermage au sens de l'art. 282 CO, selon la procédure de   │
│ protection dans les cas clairs (art. 257 CPC), présuppose que le bail ait valablement pris │
│ fin, puisque l'extinction du bail est une condition du droit à la restitution des locaux“  │
│   — BGE 144 III 462 E. 3.3.1 S. 466  ≈ wording taken from the source                       │
└────────────────────────────────────────────────────────────────────────────────────────────┘
3 verified quote(s) · 9s
```

The arguments are the model's. The quotes and citations are cut from the
decisions after a string match, so `✓ verbatim` is a check, not a claim; `≈`
marks a quote the model copied with small changes, shown in the court's
wording. The check does not judge the argument: the second quote is real, but
it is about rent increases and the ordinary notice period, not about late
payment.

## Why

Lawyers test an argument by handing it to a colleague whose job is to attack
it. A language model can play that colleague, but a legal objection is only as
good as its sources, and language models invent sources. Here the model may
argue but may not quote: every quote it proposes is checked against the text of
the passage it names, and what is shown is cut from the court's text, never
copied from the model's output.

## Quickstart

You need Python 3.11+, about 450 MB of disk and a chat model. The default is
[Ollama](https://ollama.com) with `qwen3.5:9b`; any OpenAI-compatible endpoint
works.

```sh
pip install git+https://github.com/sicofit/devils-advocate
ollama pull qwen3.5:9b

devil fetch    # BGE corpus from OpenCaseLaw (220 MB), pinned to a commit hash
devil index    # ~6,800 decisions since 2000 -> ~126k passages in SQLite FTS5
devil          # opens the conversation
devil Ein Vertrag braucht nur Konsens.   # opens it with this position
```

Reply to the devil's objections to continue the argument. In the conversation:

| command | |
| --- | --- |
| `/sources` | full text of the passages behind the last answer, quote in bold |
| `/discarded` | also show the quotes that failed verification, and why |
| `/export [file]` | save the conversation as a Markdown memo |
| `/new` | start over |
| `/help`, `/quit` | list the commands; leave (or Ctrl+C at the prompt) |

A follow-up turn sees the claims and citations of the earlier ones and is
answered from passages not cited yet. Ctrl+C while the devil is working
cancels that turn, not the conversation.

One question without a conversation, for scripts:

```sh
devil argue --json "Eine fristlose Kündigung ist bei jeder Verfehlung zulässig."
```

With `--json`, stdout carries the JSON and nothing else; messages and errors
go to stderr.

### Configuration

Flags override `DA_*` environment variables, which override the defaults.
`--model`, `--embed-model`, `--base-url` and `--data-dir` go before the
command: `devil --model qwen3:8b argue …`.

| variable | default | |
| --- | --- | --- |
| `DA_BASE_URL` | `http://127.0.0.1:11434` | any OpenAI-compatible endpoint, with or without `/v1` |
| `DA_MODEL` | `qwen3.5:9b` | chat model |
| `DA_EMBED_MODEL` | `nomic-embed-text` | embedding model for `devil embed` and dense retrieval |
| `DA_API_KEY` | – | sent as a bearer token if set |
| `DA_DATA_DIR` | `~/.devils-advocate` | corpus, index and vectors |
| `DA_SINCE` | `2000` | earliest decision year for `devil index` |
| `DA_TOP_K` | `8` | passages shown to the model per claim |
| `DA_MAX_CLAIMS` | `4` | claims attacked per position |
| `DA_TEMPERATURE` | `0.2` | sampling temperature of the chat model |
| `DA_REQUEST_TIMEOUT` | `600` | seconds to wait for a chat reply |
| `DA_EMBED_TIMEOUT` | `60` | seconds to wait for an embedding batch before retrying it |
| `DA_REASONING_EFFORT` | `none` | passed through; `none` turns off the thinking phase on Ollama, `omit` sends nothing |

Nothing leaves your machine unless `DA_BASE_URL` points elsewhere. `devil status`
shows what is configured and installed.

## How it works

```text
position
  │
  ├─ 1. analysis      model splits it into claims (max. 4), each with search
  │                   queries in German and French            structured JSON
  ├─ 2. retrieval     BM25 for the claim and each query, fused by
  │                   reciprocal rank; top 8 passages, max. 2 per decision
  ├─ 3. objection     model sees the passages as P1…P8, returns
  │                   (passage, quote, argument)              structured JSON
  └─ 4. verification  is the quote in that passage?           no model
                      yes  -> show the court's wording with its citation
                      no   -> discard; no quote left -> "no counter-precedent found"
```

- **Passages, not pages.** Each decision is split at its Erwägung numbers
  (`E. 2.1`, `E. 3b/aa`) into 126,497 passages. Page markers are removed from
  the text but recorded, so a quote is cited with the page it is on,
  `BGE 146 III 441 E. 2.5.3 S. 451`, or `S. 451 f.` when it runs onto the next.
  Long Erwägungen are split at sentence breaks; the Dispositiv is left out.
  ([`segment.py`](src/devils_advocate/segment.py))
- **Verification** normalises quotation marks, dashes and whitespace and keeps
  an index back into the original, so the quote shown is always the court's
  text. Matches respect word boundaries. An ellipsis splits a quote into
  fragments that must appear in order, and the text it skips must not contain
  a negation. A fragment that differs only in inflection or punctuation (92 %
  of the characters line up, no word added, dropped or replaced) is accepted
  as close and marked `≈` instead of `✓`. Quotes under five words are rejected.
  ([`verify.py`](src/devils_advocate/verify.py))
- **The Regeste** (headnote) is searchable but cited as `(Regeste)`, because it
  summarises the reasoning rather than being part of it.
- **Dense and hybrid retrieval** are implemented (`devil embed`, then
  `devil argue --retriever hybrid`) but not measured; once vectors exist that
  fit the current index, `auto` uses hybrid. See [the decision record](docs/DECISIONS.md#2-bm25-is-the-default-retriever-dense-retrieval-is-optional).

The reasoning behind each of these choices, and what would change it, is in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## Evaluation

Both evaluations run from this repository and write their results, with the
dataset commit and the model, to [`evals/results/`](evals/results).

### Retrieval: the court's own citations

When a decision states a proposition and cites `(BGE 137 IV 105 E. 2.3.3)`,
that sentence, with the citation removed, becomes a query and the cited
Erwägung its answer. That gives 1,000 labelled queries without anyone writing
labels. The citing decision is excluded from the results.

Recall@10: is the cited decision among the first ten decisions, and is a
passage of the cited Erwägung among the first ten passages?

| recall@10 | all | same language | cross language |
| --- | --- | --- | --- |
| BM25, decision | 0.421 | 0.576 | 0.108 |
| BM25 + translations, decision | **0.435** | 0.512 | **0.280** |
| BM25, Erwägung | 0.356 | 0.505 | 0.056 |
| BM25 + translations, Erwägung | **0.369** | 0.446 | **0.213** |
| queries (decision / Erwägung) | 1,000 / 909 | 668 / 608 | 332 / 301 |

A third of the citations cross languages: a German decision citing a French
one, or the reverse. Lexical search does not match *Kündigung* to *résiliation*,
so BM25 finds the cited decision for 58 % of same-language queries but for only
11 % of cross-language ones. Searching with a German and a French translation
of each query as well (written by `qwen3.5:9b`) raises cross-language recall to
28 %, but costs 6 points on same-language queries, because the translated
rankings carry the same weight in the fusion as the original. On balance it
helps, which is why the devil writes its search queries in both languages.

BM25 takes 109 ms per query, 245 ms with the two translations. Recall@1/5/20,
MRR and every query with its ranks are in
[`retrieval.json`](evals/results/retrieval.json). Dense retrieval is
implemented but not measured, for the reason given in
[the decision record](docs/DECISIONS.md#2-bm25-is-the-default-retriever-dense-retrieval-is-optional).

### End to end

[`evals/positions.jsonl`](evals/positions.jsonl) holds 22 positions written to
contradict the holding of one specific BGE, and 4 inputs the devil should not
argue with (small talk, foreign law). A position counts as found only if that
specific decision comes back with a verified quote.

| `qwen3.5:9b`, BM25 | |
| --- | --- |
| expected decision quoted with a verified quote | **17 / 22** |
| expected decision retrieved | 19 / 22 |
| positions with at least one verified counter-quote | 21 / 22 |
| out-of-scope inputs where the devil abstained | 4 / 4 |
| proposed quotes that verified | 77 / 82 (64 exact, 13 close) |
| seconds per position | median 8.1, max 22.4 |

Other decisions may contradict a position just as well, so 17 / 22 is a lower
bound. In the other direction, the positions were written from the Regeste of
the expected decision, so their wording is close to the court's. Read the
table as a regression test of the whole pipeline, not as recall on questions
from practice. Per-case results:
[`end_to_end.json`](evals/results/end_to_end.json).

```sh
devil bench --translate   # retrieval benchmark
devil eval                # end to end, updates the badge above
```

## Limitations

- Verification proves that the court wrote the quote, not that it supports the
  argument built on it. In testing, the model sometimes built an argument
  against an employment-law claim on a verbatim quote from a tenancy decision:
  the quote was real, the argument weak. That judgement stays with the reader,
  who gets the full passage with `/sources`.
- The corpus is the published leading decisions (BGE) from 2000 onwards, not
  the unpublished judgments of the Federal Supreme Court or cantonal courts.
- Passages are found by the Erwägung numbers in scraped text. Where the
  numbering in the source is irregular, a pinpoint can be off even though the
  quote is verbatim.
- This is a tool for testing arguments, not legal advice.

## Data

The decisions come from [OpenCaseLaw](https://huggingface.co/datasets/voilaj/swiss-caselaw)
(`voilaj/swiss-caselaw`, file `bge.parquet`), released under CC0. Thanks to its
maintainers for making Swiss case law available in bulk. The data is not part
of this repository; `devil fetch` downloads it, checks it against the SHA-256
that Hugging Face publishes for the file, and records the commit and the hash.
The results in this README were produced from revision `6336cd6` (SHA-256
`fd363e6c…4931179`, 6,817 decisions, 126,497 passages); to index exactly that
version:

```sh
devil fetch --revision 6336cd66331427f65f7f18379a7f7a0d0105963b
```

## How this was built

The design decisions and their trade-offs are written down in
[`docs/DECISIONS.md`](docs/DECISIONS.md). The tests run without a model or
the corpus, on synthetic decisions and a scripted model
([`tests/fakes.py`](tests/fakes.py)). The numbers above come from runs on my
machine, and their result files are committed.

## License

Code: [MIT](LICENSE). Data: CC0, from OpenCaseLaw.
