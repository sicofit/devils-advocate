# devils-advocate

[![CI](https://github.com/sicofit/devils-advocate/actions/workflows/ci.yml/badge.svg)](https://github.com/sicofit/devils-advocate/actions/workflows/ci.yml)
[![eval](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/sicofit/devils-advocate/main/evals/results/badge.json)](#end-to-end)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Data: CC0](https://img.shields.io/badge/data-OpenCaseLaw%20CC0-lightgrey)](https://huggingface.co/datasets/voilaj/swiss-caselaw)

You state a legal position. The devil argues against it with decisions of the
Swiss Federal Supreme Court, and it shows a quote only if that quote is
verbatim in the passage it cites, an Erwägung or the Regeste. If no quote
holds up, it says so instead of making one up.

A real run. The position: *an employer may dismiss an employee without notice
for any misconduct, even without a prior warning*.

```text
$ devil argue --show-discarded "Der Arbeitgeber darf das Arbeitsverhältnis wegen jeder Verfehlung des Arbeitnehmers fristlos kündigen, auch ohne vorherige Verwarnung."
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Der Arbeitgeber kann das Arbeitsverhältnis aus jedem Grund (jeder Verfehlung)        │
│ fristlos kündigen.                                                                      │
│                                                                                         │
│ BGE 130 III 213 E. 3.1 S. 220 f.  ✓ verbatim                                            │
│ „Eine fristlose Entlassung ist nur bei besonders schweren Verfehlungen des              │
│ Arbeitnehmers gerechtfertigt. Diese müssen einerseits objektiv geeignet sein, die für   │
│ das Arbeitsverhältnis wesentliche Vertrauensgrundlage zu zerstören oder zumindest so    │
│ tief greifend zu erschüttern, dass dem Arbeitgeber die Fortsetzung des Vertrags nicht   │
│ mehr zuzumuten ist.“                                                                    │
│ BGE 130 III 213 hält fest, dass eine fristlose Entlassung nur bei besonders schweren    │
│ Verfehlungen gerechtfertigt ist und diese das Vertrauen im Arbeitsverhältnis zerstören  │
│ müssen. Dies widerspricht der Behauptung, der Arbeitgeber könne aus jedem Grund         │
│ fristlos kündigen, da nicht jede Verfehlung diesen hohen Schweregrad erreicht.          │
│                                                                                         │
│ BGE 130 III 353 E. 2.2.2 S. 360  ✓ verbatim                                             │
│ „Grundsätzlich kann es nur darauf ankommen, aus welchem Grund das Arbeitsverhältnis     │
│ beendet worden ist. Das nachvertragliche Konkurrenzverbot rechtfertigt sich nur, wenn   │
│ die Arbeitgeberin bei einem korrekten Verhalten des Arbeitnehmers bereit gewesen wäre,  │
│ das Arbeitsverhältnis fortzusetzen bzw. wenn nicht die Arbeitgeberin die Beendigung des │
│ Arbeitsverhältnisses ohne begründeten Anlass verursacht hat.“                           │
│ BGE 130 III 353 betont, dass entscheidend ist, aus welchem Grund das Arbeitsverhältnis  │
│ beendet wurde, und dass eine fristlose Entlassung ohne begründeten Anlass die           │
│ Fortsetzung des Verhältnisses rechtfertigen würde. Dies qualifiziert die Behauptung,    │
│ indem es klarstellt, dass nicht jede Kündigung zulässig ist, sondern der tatsächliche   │
│ Beendigungsgrund massgeblich sein muss.                                                 │
│                                                                                         │
│   ✗ BGE 134 III 354 E. 2.1 S. 358: a part between ellipses is shorter than 3 words.     │
│ „Art. 336c OR mit der Marginalie "Kündigung zur Unzeit durch den Arbeitgeber" bestimmt: │
│ "1 Nach Ablauf der Probezeit darf der Arbeitgeber das Arbeitsverhältnis n…“             │
└─────────────────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Eine vorherige Verwarnung ist für eine fristlose Kündigung nicht erforderlich.       │
│                                                                                         │
│ BGE 152 II 19 E. 3.3.1 S. 23  ✓ verbatim                                                │
│ „Wiegen die Verfehlungen weniger schwer, ist die fristlose Auflösung wie im privaten    │
│ Arbeitsrecht nur gerechtfertigt, wenn sie sich trotz Verwarnung wiederholt haben […]“   │
│ BGE 152 II 19 beschränkt den Anwendungsbereich der fristlosen Kündigung auf Fälle, in   │
│ denen weniger schwere Verstösse trotz einer Verwarnung wiederholt wurden. Damit         │
│ qualifiziert das Urteil die Behauptung, indem es eine vorherige Verwarnung als          │
│ notwendige Bedingung für weniger schwere, sich wiederholende Verstösse festlegt.        │
│                                                                                         │
│   ✗ BGE 142 III 579 E. 4.2 S. 579: not found in the cited passage. „Nach der            │
│ Rechtsprechung zu Art. 337 OR ist eine fristlose Kündigung durch den Arbeitgeber nur    │
│ bei besonders schweren Verfehlungen des Arbeitnehmers gerechtferti…“                    │
│   – BGE 127 III 153 (Regeste): its argument says the passage supports the claim. „Das   │
│ Gericht entscheidet unter Würdigung sämtlicher Umstände über das Vorliegen wichtiger    │
│ Gründe. Die Rechtsprechung kann deshalb nicht starre Regeln über Anzahl…“               │
└─────────────────────────────────────────────────────────────────────────────────────────┘
3 verified quotes, 2 discarded on verification · 18 s
```

The arguments are the model's. The quotes and citations are cut from the
decisions after a string match, so `✓ verbatim` is a check, not a claim. A
quote that stops before its sentence ends is marked `[…]`. With
`--show-discarded`, the proposals that failed a check are listed with the
reason (`✗`), and those set aside without failing one (`–`).

The check does not judge the argument. The first and third quotes are on
point. The second is real, but BGE 130 III 353 is about when a non-compete
clause survives the end of the employment. The argument built on it misstates
the passage and does not answer the claim.

## Why

Lawyers test an argument by handing it to a colleague whose job is to attack
it. A language model can play that colleague, but a legal objection is only as
good as its sources, and language models invent sources. Here the model may
argue but may not quote. Every quote it proposes is checked against the text
of the passage it names, and what is shown is cut from the court's text, never
copied from the model's output.

## Quickstart

You need Python 3.11+ and a chat model. The default is
[Ollama](https://ollama.com) with `qwen3.5:9b` (a 6.6 GB download); any
OpenAI-compatible endpoint works. The corpus and the index take about 430 MB
of disk, plus another 205 MB while `devil index` builds a new index next to
the old one.

```sh
pip install git+https://github.com/sicofit/devils-advocate
ollama pull qwen3.5:9b

devil fetch    # BGE corpus from OpenCaseLaw (220 MB), pinned to a commit hash
devil index    # ~6,800 decisions since 2000 -> ~127k passages in SQLite FTS5
devil          # opens the conversation
devil Ein Vertrag braucht nur Konsens.   # opens it with this position
```

Give Ollama a context of at least 8,192 tokens, for example by starting it
with `OLLAMA_CONTEXT_LENGTH=8192 ollama serve`. The largest call in the
evaluation took 4,157 tokens (prompt and reply). Ollama's default context
depends on its version and the GPU memory, and can be 4,096 tokens.

Reply to the devil's objections to continue the argument. In the conversation:

| command | |
| --- | --- |
| `/sources` | full text of the passages behind the last answer, quote in bold |
| `/discarded` | toggle showing the quotes that failed a check or were set aside, and why |
| `/export [file]` | save the conversation as a Markdown memo |
| `/new` | start over |
| `/help` | list the commands |
| `/quit` | leave (also `/exit`, `/q`) |

A follow-up turn sees the claims and citations of the earlier ones and is
answered from passages not cited yet. End a line with `\` to continue it on
the next; a block pasted into the Windows console becomes one turn. Ctrl+C
while the devil is working cancels that turn, not the conversation. Ctrl+C at
the prompt leaves and drops what was typed. Ctrl+D on an empty line (Ctrl+Z,
Enter on Windows) leaves too, but first sends the lines already continued with
`\`. `/export` without a file name writes
`devils-advocate-<date>.md` in the current directory, or `-2`, `-3`, … if that
file exists.

One question without a conversation, for scripts:

```sh
devil argue --json "Eine fristlose Kündigung ist bei jeder Verfehlung zulässig."
```

With `--json`, stdout carries the JSON and nothing else; messages and errors
go to stderr. Per claim, it lists the search queries, the retrieved passages,
the counterarguments (argument, quote, citation, `match` exact or close,
whether the quote starts or ends mid-sentence, whether it reports the lower
court's or a party's view), the counterarguments already quoted under an
earlier claim, and what was discarded or set aside, with the reason.

### Configuration

Flags override `DA_*` environment variables, which override the defaults.
`--model`, `--embed-model`, `--base-url` and `--data-dir` go before the
command or the position: `devil --model qwen3:8b argue …`. An option typed
after the position is refused, not read as part of it.

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
| `DA_EMBED_TIMEOUT` | `60` | seconds to wait for one embedding batch; `devil embed` tries a batch up to five times |
| `DA_REASONING_EFFORT` | `none` | passed through; `none` turns off the thinking phase on Ollama, `omit` sends nothing |

`devil fetch` downloads from Hugging Face. Apart from that, the devil talks
only to `DA_BASE_URL`, so nothing leaves your machine unless that points
elsewhere. `devil status` shows what is configured and installed, and names
the problem if the server refuses the API key.

## How it works

```text
position
  │
  ├─ 1. analysis      model splits it into claims (max. 4), each with search
  │                   queries in German and French            structured JSON
  ├─ 2. retrieval     BM25 for the claim and each query, fused by
  │                   reciprocal rank; top 8 passages, max. 2 per decision
  ├─ 3. objection     model sees the passages as P1…P8, returns up to 3
  │                   (passage, quote, argument, effect)      structured JSON
  └─ 4. verification  is the quote in that passage?           no model
                      yes  -> show the court's wording with its citation
                      no   -> discard; none left -> "Counterarguments were
                              proposed, but none passed verification …"
                      none proposed, or all set aside -> "No BGE passage
                              found that contradicts this claim."
```

- **Passages, not pages.** Each decision is split at its Erwägung numbers
  (`E. 2.1`, `E. 3b/aa`) into 127,388 passages. Page markers are removed from
  the text but recorded, so a quote is cited with the page it is on,
  `BGE 146 III 441 E. 2.5.3 S. 451`, `S. 451 f.` when it runs onto the next
  page, `S. 451 ff.` when it runs further. Long Erwägungen are split at
  sentence breaks; the Dispositiv is left out. Text before the first numbered
  Erwägung is cited by its page alone.
  ([`segment.py`](src/devils_advocate/segment.py))
- **Verification** normalises quotation marks, dashes, whitespace and case and
  keeps an index back into the original, so the quote shown is always the
  court's text. Matches respect word boundaries. An ellipsis splits a quote
  into fragments that must appear in order, and the part of a quoted sentence
  it skips must not contain a negation or a qualifier (*nicht*, *nur*,
  *grundsätzlich*, *in der Regel* …); whole sentences may be left out. A quote that leaves out the denial its sentence starts
  with (*Es trifft nicht zu, dass …*) is rejected. A fragment that differs only
  in inflection or punctuation is accepted as close and marked `≈` instead of
  `✓`. Quotes under five words are rejected. A quote that restates the lower
  court's or a party's view is shown with a warning.
  ([`verify.py`](src/devils_advocate/verify.py))
- **Arguments that agree are set aside.** The model labels each proposal as
  contradicting, limiting or supporting the claim. A supporting one is dropped,
  and so is one whose argument says that the passage supports the claim,
  unless a negation or a limiting word stands next to that phrase. The
  devil also drops an argument that names a decision neither the cited passage
  nor the user named, and keeps one counter per decision and claim.
- **The Regeste** (headnote) is searchable but cited as `(Regeste)`, because it
  summarises the reasoning rather than being part of it.
- **Dense and hybrid retrieval** are implemented (`devil embed`, then
  `devil argue --retriever hybrid`) but not measured; once vectors exist that
  fit the current index, `auto` uses hybrid, and falls back to BM25 with a
  notice if the embedding model fails, for example because the server does not
  have it. See
  [the decision record](docs/DECISIONS.md#2-bm25-is-the-default-retriever-dense-retrieval-is-optional).

The reasoning behind each of these choices, and what would change it, is in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## Evaluation

Both evaluations run from this repository and write their results, with the
dataset commit, the index build and the model, to
[`evals/results/`](evals/results). Each number below is from a single run.

### Retrieval: the court's own citations

When a decision states a proposition and cites `(BGE 137 IV 105 E. 2.3.3)`,
that sentence, with the citation removed, becomes a query and the cited
Erwägung its answer. That gives 1,000 labelled queries without anyone writing
labels. The citing decision is excluded from the results.

Recall@10: is the cited decision among the first ten decisions, and is a
passage of the cited Erwägung among the first ten passages?

| recall@10 | all | same language | cross language |
| --- | --- | --- | --- |
| BM25, decision | 0.446 | **0.615** | 0.115 |
| BM25 + translations, decision | **0.486** | 0.550 | **0.361** |
| BM25, Erwägung | 0.368 | **0.534** | 0.039 |
| BM25 + translations, Erwägung | **0.400** | 0.479 | **0.243** |
| queries (decision / Erwägung) | 1,000 / 905 | 662 / 601 | 338 / 304 |

A third of the citations cross languages, mostly a German decision citing a
French one or the reverse (285 of 338); the rest involve Italian. Lexical search does not match *Kündigung* to
*résiliation*, so BM25 finds the cited decision for 62 % of same-language
queries but for only 12 % of cross-language ones. Searching with a German and
a French translation of each query as well (written by `qwen3.5:9b`) raises
cross-language recall@10 to 36 %. It costs 6.5 points on same-language
queries, because the translated rankings carry the same weight in the fusion
as the original. Overall, recall@10 rises by 4 points, but recall@1 falls
(0.197 to 0.184) and MRR@10 stays level (0.273 and 0.270): the translations
bring the cited decision into the first ten more often, not to the top. The
devil writes its search queries in both languages for the first reason.

BM25 takes 126 ms per query, 270 ms with the two translations. Recall@1/5/20,
MRR@10 and every query with its ranks are in
[`retrieval.json`](evals/results/retrieval.json). Dense retrieval is
implemented but not measured, for the reason given in
[the decision record](docs/DECISIONS.md#2-bm25-is-the-default-retriever-dense-retrieval-is-optional).

### End to end

[`evals/positions.jsonl`](evals/positions.jsonl) holds 22 positions written to
contradict the holding of one specific BGE, and 4 inputs the devil should not
argue with (small talk, foreign law). A position counts as found only if that
specific decision comes back with a verified quote.

| `qwen3.5:9b`, BM25, temperature 0.2 | |
| --- | --- |
| expected decision quoted with a verified quote | **21 / 22** |
| expected decision retrieved | 21 / 22 |
| positions with at least one verified counter-quote | 22 / 22 |
| out-of-scope inputs where the devil abstained | 4 / 4 |
| proposed quotes that verified | 59 / 65 (48 exact, 11 close) |
| seconds per position | median 16.0, max 32.2 |

Not counted as proposed: 25 quotes set aside, 24 because the decision was
already quoted for that claim and 1 because the argument said the passage
supports the claim. Of the 6 quotes discarded, 3 changed a word of the
passage, 2 were not in it and 1 came with an argument naming another
decision. The one position missed is the Italian one (*curatela-perizia*): for
one of its claims the model's reply could not be used, so that claim was
skipped, and the expected decision was not retrieved for the others. An
earlier run of nearly the same code found 19 of 22.

Other decisions may contradict a position just as well, which this count
misses. In the other direction, the positions were written from the Regeste of
the expected decision, so their wording is close to the court's. The model
samples at temperature 0.2, so a second run can come out differently. Read the
table as a regression test of the whole pipeline, not as recall on questions
from practice. Per-case results and the settings of the run:
[`end_to_end.json`](evals/results/end_to_end.json).

```sh
devil bench --translate   # retrieval benchmark
devil eval                # end to end, updates the badge above
```

## Limitations

- Verification proves that the court wrote the quote, not that it supports the
  argument built on it, as the second quote in the run above shows. That
  judgement stays with the reader, who gets the full passage with `/sources`.
- The corpus is in German, French and Italian. A position can be written in
  English; the analysis step writes German and French queries for it. The one
  English position in the evaluation (eviction, BGE 141 III 262) was answered
  with verified quotes, but a single case does not show that English input
  works as well as German or French.
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
of this repository. `devil fetch` downloads it and, when Hugging Face
publishes a SHA-256 for the file, checks the download against it. It records the commit and
the hash. The results in this README were produced from revision `6336cd6`
(SHA-256 `fd363e6c…4931179`, 6,817 decisions, 127,388 passages). To index
exactly that version:

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
