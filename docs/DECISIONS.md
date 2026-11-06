# Design decisions

Short records of the choices that shape the tool, with the reason for each and
what would change it.

## 1. A quote is shown only if it is in the cited passage

The model proposes counterarguments as `(passage label, quote, argument,
effect)`. Each quote is checked against the text of the passage behind that
label ([`verify.py`](../src/devils_advocate/verify.py)):

- Both texts are normalised first: quotation marks dropped (`„…“`, `«…»`,
  `‚…‘` and `"…"` say the same), dashes unified, whitespace collapsed, case
  folded, soft hyphens and zero-width characters dropped, spaces around
  punctuation evened out. An apostrophe between two letters stays
  (`l'art.`). Every character of the normalised text keeps an index into the
  original. What the user sees is therefore always cut from the court's text,
  never the model's copy, with the quotation marks the court used.
- A match must start and end at word boundaries: "zulässig" is not found
  inside "unzulässig".
- An ellipsis (`...`, `…`, `[…]`, `(…)`) splits the quote into fragments of
  at least three words each, and the fragments must appear in the passage in
  that order. If the text an ellipsis skips contains a negation (*nicht, kein,
  pas, non, not* …) or a qualifier (*nur, grundsätzlich, in der Regel,
  seulement, en principe* …), the quote is rejected: "ist […] gerechtfertigt"
  is not a quote of "ist nicht gerechtfertigt". Only the parts of sentences the
  quote cuts count; leaving out whole sentences in between changes no quoted
  sentence. If the fragment before an ellipsis was found exactly and occurs
  again closer to the next one, that occurrence is tried before the quote is
  rejected.
- A quote that starts inside a sentence must not leave out the denial the
  sentence starts with: "dass die Frist stillsteht" is not a quote of "Es
  trifft nicht zu, dass die Frist stillsteht", and "que le délai soit
  suspendu" is not one of "On ne saurait admettre que le délai soit suspendu".
- A fragment that is not found exactly may still be accepted as **close** if
  it and one window of the passage line up to at least 92 % (twice the
  matching characters over the length of both) and every word has its
  counterpart: none added or dropped, no number changed, and a changed word
  keeps its stem ("schwerer" for "schweren", not "unzulässig" for
  "zulässig"). An article may change with the case ("dem" for "den"), and the
  last word may be cut short if at least four letters are kept; the passage's
  whole word is then shown. This happens when the model changes an
  inflection or drops a comma. The passage's own wording is shown and marked
  "≈", not "✓".
- Quotes shorter than five words are rejected. They are too easy to find by
  chance and too short to carry an argument.
- The model must name its passage by one of the labels it was shown (the
  JSON schema lists them); a label it was not shown ("P99") is rejected before
  any matching. The labels never reach the reader: the argument names the
  decision instead, and an argument that still names a label the model was
  not shown is rejected. An argument that names a decision that neither the
  cited passage (also in a list such as "ATF 143 I 388; 133 I 259") nor the
  user named is rejected, because it pins the quote on another decision.

A verified quote is shown with what the reader needs to weigh it. A quote that
starts or stops inside a sentence of the court is marked `[…]` at that end. A
quote that restates what the lower court held or a party argued ("Die
Vorinstanz erwog, …") is marked as such, because it is not the Federal Supreme
Court's view; when the attribution opens the quote's own sentence, it is shown
with the quote.

A claim with no verified quote gets an explicit "No BGE passage found that
contradicts this claim." instead of an unsupported argument. When at least one
proposal was discarded and none verified, it gets a note that none passed
verification; when the model's reply for the claim could not be used, a note
that the claim was skipped. The
rejected proposals are kept and can be shown (`--show-discarded`,
`/discarded`), so the rejection rate is visible rather than hidden.

**Why not an LLM judge?** Deterministic string matching is cheap, gives the same
answer every time and needs no trust in a second model. Its blind spot is
deliberate: it checks that the court said it, not that it supports the
argument. That second question is left to the reader, who gets the full passage
with `/sources`.

## 2. BM25 is the default retriever, dense retrieval is optional

The index holds 127,388 passages from 6,817 BGE decisions (2000 onwards).
Retrieval runs on SQLite FTS5 with BM25 ranking. That needs no model, no GPU
and no extra service; the benchmark measures 126 ms per query for the top 100
passages on the development laptop.

Dense and hybrid retrieval (reciprocal-rank fusion, k = 60) are implemented for
any OpenAI-compatible embedding endpoint (`devil embed`, `--retriever hybrid`).
They are not the default because on the development machine the local
embedding server could not get through all passages reliably: batches hung for
20–30 s and about a quarter failed even with retries. Rather than publish
numbers for a setup that did not run cleanly, the README reports BM25 only.

Vectors record the index build they were embedded from. After `devil index`
rebuilds the index, passage ids point at different text, so old vectors are
not used: `auto` falls back to BM25 with a notice, `dense` and `hybrid` refuse
until `devil embed` has run again. In `auto` mode, an embedding call that fails
during a conversation (the server has no such model, for example) also falls
back to BM25 for the rest of the session. A server that is down or refuses the
API key ends the turn instead, because the chat model would fail the same way.

**What would change it:** a stable embedding endpoint. `devil bench` scores
`bm25`, `dense` and `hybrid` side by side as soon as vectors exist.

## 3. The language gap is closed with queries, not with embeddings

The citation benchmark (decision 5) showed that BM25 recall collapses when the
query and the cited decision are written in different languages. The court
writes in German, French and Italian and cites across all three. The analysis
step therefore writes every search query in both German and French. The
benchmark's `+translate` mode measures what that is worth, by adding a German
and a French translation of each benchmark query written by the same chat
model. A query the model could not translate is counted
(`translations_missing`, `translations_partial` in `retrieval.json`); in the
run reported here there were none.

Result on 1,000 queries: decision recall@10 across languages rises from 0.115
to 0.361, within one language it falls from 0.615 to 0.550, overall it rises
from 0.446 to 0.486. Recall@1 falls overall (0.197 to 0.184), and MRR@10 stays
level (0.273 and 0.270). The translations bring the cited decision into the
first ten more often, but do not move it to the top.

A one-off check during development, on an earlier build of the index and
outside `devil bench`, searched with the translation into the other language
only, leaving out the same-language paraphrase. Cross-language recall rose
further and same-language recall fell further. So the cost does not come from
the paraphrase. It comes from fusing every ranking with equal weight, so
passages in the other language push same-language hits down.

**What would change it:** weighted fusion, or a multilingual dense retriever.
Either should be tuned on queries other than the 1,000 it is then scored on.

## 4. The Regeste is indexed but marked

The Regeste is the headnote that the court publishes with each leading
decision. It is often the most direct statement of the holding, so it is
retrieved like any passage. But it summarises the reasoning and is not the
reasoning itself, so a quote from it is cited as `BGE 146 III 441 (Regeste)`
instead of with an Erwägung. Regesten of French and Italian decisions are in
German in the dataset, which also helps across languages; a quote from a
Regeste is therefore checked with the rules of the language it is written in.

Text that the index cannot place under a numbered Erwägung (facts the
reasoning opens with, a section heading) is cited by page alone, as
`BGE 150 IV 114 S. 116`, rather than with an Erwägung it does not belong to.

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
the same language. The report gives recall@1, 5, 10 and 20 and MRR@10.

Limitations: the court often cites a decision for a side point, and one
sentence is less context than the devil gets in a conversation. The numbers are
for comparing retrievers and catching regressions, not a claim about the
pipeline's accuracy.

## 6. The end-to-end evaluation is a regression test, not an accuracy figure

[`evals/positions.jsonl`](../evals/positions.jsonl) holds positions written to
contradict the holding of a specific BGE, taken from its Regeste, plus inputs
the devil should not argue with (small talk, foreign law). A case counts as
found when that specific decision is quoted with a verified quote.

The count errs in both directions. Other decisions may be equally valid
counterarguments, and they do not count. But the positions were written from
the Regeste of the expected decision, so their wording is close to the
court's, which makes that decision easier to find. The out-of-scope inputs
check that the devil abstains instead of inventing a dispute;
`out_of_scope_with_claims` counts those where it abstained only because
nothing verified, after the analysis had found a legal claim.

Each reported number is from one run at temperature 0.2, and a second run can
differ. `end_to_end.json` records the model, the retriever, the temperature,
the reasoning setting, `top_k`, `max_claims`, the package version and a hash
of the prompts, so a change in the numbers can be traced to a change in the
setup. The badge shows the found count, coloured by share: bright green from
80 %, green from 60 %, yellow from 40 %, orange from 20 %, red below, grey when
nothing was measured.

## 7. Arguments that agree with the claim are set aside, not counted

The model is asked to argue against the claim, but sometimes returns a
passage that supports it. Each proposal therefore carries an `effect`
(`contradicts`, `limits`, `supports`). A `supports` item is set aside, and so
is one whose argument says in so many words that the passage supports the
claim ("bestätigt die Behauptung", "supports the claim"), unless a negation or
a limiting word stands next to that phrase ("does not support", "stützt die
Behauptung nur teilweise").

These items, a second verified quote from a decision already quoted for the
same claim, and proposals over the limit of three per claim are listed as set
aside (`–`), not as discarded (`✗`): they did not fail a check, and they are
not counted as proposed quotes. A verified quote that lies within text already
quoted under an earlier claim of the same answer is not shown twice; the later
claim refers to the earlier one. A different quote from the same passage is
shown.

## 8. Local first, any OpenAI-compatible endpoint

The default is Ollama with `qwen3.5:9b`, structured output via `json_schema`,
and the model's thinking phase switched off (`reasoning_effort: "none"`). In
one comparison on the development machine that cut a structured call from
about 15 s to about 1 s, with no loss on this task that we could see. Nothing
leaves the machine unless `DA_BASE_URL` points elsewhere; only `devil fetch`
goes to Hugging Face. The HTTP client uses the standard library only.

The prompts need room: the largest call in the evaluation took 4,157 tokens,
prompt and reply. With Ollama, set `OLLAMA_CONTEXT_LENGTH=8192` or more. If the
model's reply is cut off at the token limit anyway, the devil retries that
claim once with the better-ranked half of its passages, then gives the claim
up.

## 9. Data provenance is recorded

The corpus is OpenCaseLaw's `voilaj/swiss-caselaw` dataset on Hugging Face
(CC0). `devil fetch` resolves the branch to a commit hash and downloads the
file at that commit. When Hugging Face publishes the Git LFS object id of the
file (its SHA-256), the download is compared with it and deleted if it does
not match; when the lookup fails, `devil fetch` says that the download is not
checked. Hash and commit are stored in `corpus.json`. An interrupted download
resumes only for the same commit. Both values end up in the index metadata,
together with the index schema version and an id of the build, and in both
result files, `retrieval.json` and `end_to_end.json`.
