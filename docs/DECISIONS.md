# Design decisions

Short records of the choices that shape the tool, with the reason for each and
what would change it.

## 1. A quote is shown only if it is in the cited passage

The model proposes counterarguments as `(passage label, quote, argument)`. Each
quote is checked against the text of the passage behind that label
([`verify.py`](../src/devils_advocate/verify.py)):

- Both texts are normalised first: typographic quotes and dashes unified,
  whitespace collapsed, case folded, soft hyphens dropped, spaces around
  punctuation evened out. Every character of the normalised text keeps an
  index into the original. What the user sees is therefore always cut from the
  court's text, never the model's copy.
- An ellipsis (`...`, `…`, `[…]`) splits the quote into fragments, and the
  fragments must appear in the passage in that order.
- A fragment that is not found exactly may still be accepted as **close** if at
  least 92 % of its characters line up with one window of the passage. This
  happens when the model changes an inflection or drops a comma. The passage's
  own wording is shown and marked "≈", not "✓".
- Quotes shorter than five words are rejected. They are too easy to find by
  chance and too short to carry an argument.
- A label the model was not shown ("P99") is rejected before any matching.

A claim with no verified quote gets an explicit "no counter-precedent found"
instead of an unsupported argument. The rejected proposals are kept and can be
shown (`--show-discarded`, `/discarded`), so the rejection rate is visible
rather than hidden.

**Why not an LLM judge?** Deterministic string matching is cheap, gives the same
answer every time and needs no trust in a second model. Its blind spot is
deliberate: it checks that the court said it, not that it supports the
argument. That second question is left to the reader, who gets the full passage
with `/sources`.

## 2. BM25 is the default retriever, dense retrieval is optional

The index holds 125,807 passages from 6,819 BGE decisions (2000 onwards).
Retrieval runs on SQLite FTS5 with BM25 ranking. That needs no model, no GPU
and no extra service; the benchmark measures 126 ms per query for the top 100
passages on a laptop.

Dense and hybrid retrieval (reciprocal-rank fusion, k = 60) are implemented for
any OpenAI-compatible embedding endpoint (`devil embed`, `--retriever hybrid`).
They are not the default because on the development machine the local
embedding server could not get through 125k passages reliably: batches hung for
20–30 s and about a quarter failed even with retries. Rather than publish
numbers for a setup that did not run cleanly, the README reports BM25 only.

**What would change it:** a stable embedding endpoint. `devil bench` scores
`bm25`, `dense` and `hybrid` side by side as soon as vectors exist.

## 3. The language gap is closed with queries, not with embeddings

The citation benchmark (decision 5) showed that BM25 recall collapses when the
query and the cited decision are written in different languages. The court
writes in German, French and Italian and cites across all three. The analysis
step therefore writes every search query in both German and French. The
benchmark's `+translate` mode measures what that is worth, by adding a German
and a French translation of each benchmark query written by the same chat model.

Result on 1,000 queries: decision recall@10 across languages rises from 0.119
to 0.311, within one language it falls from 0.605 to 0.541, overall it rises
from 0.438 to 0.462. A one-off check outside `devil bench` searched with the
translation into the other language only, leaving out the same-language
paraphrase: cross-language recall rose further (0.343), same-language recall
fell further (0.521). So the cost does not come from the paraphrase. It comes
from fusing every ranking with equal weight, so passages in the other language
push same-language hits down.

**What would change it:** weighted fusion, or a multilingual dense retriever.
Either should be tuned on queries other than the 1,000 it is then scored on.

## 4. The Regeste is indexed but marked

The Regeste is the headnote that the court publishes with each leading
decision. It is often the most direct statement of the holding, so it is
retrieved like any passage. But it summarises the reasoning and is not the
reasoning itself, so a quote from it is cited as `BGE 146 III 441 (Regeste)`
instead of with an Erwägung. Regesten of French and Italian decisions are in
German in the dataset, which also helps across languages.

## 5. The retrieval benchmark is built from the court's own citations

When a decision states a proposition and cites `(BGE 137 IV 105 E. 2.3.3)`, the
sentence becomes a query and the cited Erwägung becomes its answer. This gives
1,000 labelled queries with nobody writing labels, drawn at random with a fixed
seed, with at most one query per pair of citing and cited decision. The citing
decision is excluded from the results because it contains the sentence word
for word.

Scores are reported at two levels: is the cited decision among the first k
distinct decisions, and is a passage of the cited Erwägung among the first k
passages? When the index holds the Erwägung only at a coarser level (the
citation says `E. 2.1.4`, the index has `E. 2.1`), the closest indexed ancestor
counts. Results are split by whether the citing and the cited decision are in
the same language.

Limitations: the court often cites a decision for a side point, and one
sentence is less context than the devil gets in a conversation. The numbers are
for comparing retrievers and catching regressions, not a claim about the
pipeline's accuracy.

## 6. The end-to-end evaluation is a lower bound

[`evals/positions.jsonl`](../evals/positions.jsonl) holds positions written to
contradict the holding of a specific BGE, taken from its Regeste, plus inputs
the devil should not argue with (small talk, foreign law). A case counts as
found when that specific decision is quoted with a verified quote. Other
decisions may be equally valid counterarguments, so this undercounts. The
out-of-scope inputs check that the devil abstains instead of inventing a
dispute.

## 7. Local first, any OpenAI-compatible endpoint

The default is Ollama with `qwen3.5:9b`, structured output via `json_schema`,
and the model's thinking phase switched off (`reasoning_effort: "none"`). That
cuts a structured call from about 15 s to about 1 s with no visible loss on
this task. Nothing leaves the machine unless `DA_BASE_URL` points elsewhere.
The HTTP client uses the standard library only.

## 8. Data provenance is recorded

The corpus is OpenCaseLaw's `voilaj/swiss-caselaw` dataset on Hugging Face
(CC0). `devil fetch` resolves the branch to a commit hash and stores it, along
with the file's SHA-256 (equal to its Git LFS object id), in `corpus.json`.
Both end up in the index metadata and in every result file under
`evals/results/`.
