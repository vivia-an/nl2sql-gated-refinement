# BASELINES: implementation notes, fidelity, budgets, smoke results (B1)

Code: `code/nl2sql/systems/baselines/` (all `System` subclasses; `BASELINES = {...}` in `__init__.py`).
Vendored upstream files: `baselines/_vendor/` (byte-identical, hashes below). Smoke tools: `code/tests_b1/`.
Every number in section 5 is copied from the smoke outputs listed there; nothing else in this file is a measurement.

## 0. Registry names

| brief name | EXPERIMENT_SPEC key (alias) | class | calls/q native (smoke) |
|---|---|---|---|
| `zero_shot` | `zs` | `ZeroShot` | 1 |
| `dail_sql` | `dail` | `DailSQL` | 2 |
| `self_debug` | `sdebug` | `SelfDebug` | 1 to 8 (explain mode: 3 calls in round 1, 2 later, `max_rounds`=3) |
| `mac_sql` | `macsql` | `MACSQL` | 1 to 2 + `max_refine` (Selector only for schemas with avg. columns > 6 or total > 30) |
| `ac_sql` / `ac_sql_x` | `acsql_s` / `acsql_x` | `ACSQL` / `ACSQLCross` | 1 to 2 x `max_regen` + 1 |
| `memo_sql` | `memosql` | `MemoSQL` | 42.6 native, 18.1 light |
| `alpha_sql` (EXPENSIVE) | `alphasql` | `AlphaSQL` | 33.7 at `rollouts=2` (section 5) |
| `bon_exec_vote` | `bon_vote` | `BoNExecVote` | n |
| `self_consistency` | `sc_cot` | `SelfConsistency` | n |

Common conventions (all baselines):
* the SQL dialect is SQLite, evidence is given to every system (appended to the question as "Evidence: ..."), the schema is the
  same "rich" DDL (`data.get_schema(item)["ddl"]`: descriptions + sample values) unless a note says otherwise;
* `calls` = number of `LLMClient.chat` requests (matches the real Meter); an `n=k` request counts once, `extra`/`ctx.samples`
  keeps the completions; embedding requests are not counted (spec section 4);
* thinking is off, temperature 0 for headline runs; sampled members use `temperature=0.7` and seed `1000*r + j` (r from the run seed 0/101/202);
* exemplar pools are `bird_train_pool` (POOL57) only; nothing reads `bird_dev`;
* each row has the CONTRACT section 4 fields (`qid, db_id, split, system, model, seed, final_sql, exec_ok, abstained, critic, iterations, actions, calls, prompt_tokens, completion_tokens, latency_s, error_class, extra`); `abstained` is always false.

## 1. Sources and commit hashes

| baseline | source | commit | licence | what is vendored |
|---|---|---|---|---|
| MAC-SQL | https://github.com/wbbeyourself/MAC-SQL | `31a9df5e0d520be4769be57a4b9022e5e34a14f4` (2025-02-27) | NONE in the repository (see risk R1) | `core/const.py` -> `_vendor/macsql_const.py` |
| Alpha-SQL | https://github.com/HKUSTDial/Alpha-SQL | `216e61fc86467591fa1dc5b7031ca536577b4d21` (2026-06-10) | MIT | `alphasql/templates/*.txt` -> `_vendor/alphasql_templates/` (+ LICENSE) |
| DAIL-SQL | https://github.com/BeachWang/DAIL-SQL | `2061f68112222083134a0c9e2877961ff315ff44` (2025-03-07) | Apache-2.0 | nothing (re-expressed, prompts reproduced as strings) |
| Self-Debug | arXiv 2304.05128 (Chen et al., ICLR 2024) | paper only | n/a | prompts quoted from Appendix E.6/E.7 |
| AC-SQL | arXiv 2410.22082 (Zheng et al.) | paper only; no code found | n/a | prompts quoted from Sec. 4.3 + Appendix A |
| Memo-SQL | arXiv 2601.10011 (Yang et al.) | paper only; no code found | n/a | taxonomy E1-E9 (Table 4) only |

sha256 of vendored files (verified byte-identical to the clones in `refs/b1/` with `cmp`):

```
dcf385e3f5cb13c12c418e8fd92b7b13d9cf1b470348414a750113e4d4ebaeae  _vendor/macsql_const.py
8f7a8ca9f5785c5c112a76f7450e1b9778f29ed9c7ec52188ad018a4633505d5  alphasql_templates/identify_column_functions.txt
50e375db53d5e5d04cf6a20f2511aad41dc940c8d742e73c91597794c7cc1afa  alphasql_templates/identify_column_values.txt
57b727d112d4c95bee353e467ed4a3c99bc9cdd71df52dbe0f956ef152be8eae  alphasql_templates/keywords_extraction.txt
4d57f3ef270168cf5a826d7402e3f08fc14fd0f203d6f13ef6ebb63fb05a24ac  alphasql_templates/raphrase_question.txt
a5708afe189e21da48e4372d2d817d8b712a65a9caf7e58f2f2a83d7bf06c5b5  alphasql_templates/schema_selection.txt
95da322c5020af8d7d97d4744c8b9d18597499a3666f9521349270338781f849  alphasql_templates/sql_generation.txt
cf3d018d26999ef0c66fe59c36a53f066ea45c86748fde970b0b614eba778abb  alphasql_templates/sql_revision.txt
f8fa2b070f118001121ce12e1bf5a0384a38246e0a4d58a7af7c63182bcb8239  alphasql_templates/LICENSE.alpha-sql
```

Official-code check for the reimplemented ones (2026-10-24/25): the AC-SQL and Memo-SQL papers print no code link; a GitHub
search for the titles and author names returned no repository. Both are therefore reimplementations from the paper text.

## 2. Fidelity notes

Legend: IDENTICAL = same text/algorithm as the source; ADAPTED = changed on purpose (reason given); OURS = no source exists, written by B1 ("our adaptation").

### 2.1 zero_shot (OURS)
No original. One request: system "You write one SQLite query that answers the question. Return only the SQL in a ```sql code block, with no explanation.", user = schema block + "Question: ... Evidence: ...". `max_tokens` 1024. The prompt is the common base reused by `bon_exec_vote` and `self_consistency`.

### 2.2 dail_sql (DAIL-SQL, Gao et al., VLDB 2024)
IDENTICAL (algorithm, from the commit above): two-stage selection (preliminary query, then re-selection with the predicted-skeleton Jaccard filter, threshold 0.85, padding with nearest), Euclidean distance on masked-question embeddings, target prompt "/* Given the following database schema: */ ... /* Answer the following: Q */", examples under "/* Some SQL examples are provided based on similar problems: */", skeleton post-processing rules and Jaccard.
ADAPTED:
* sentence encoder all-mpnet-base-v2 -> `qwen3.7-text-embedding` (L2-normalised, so Euclidean order equals cosine order);
* CoreNLP + RAT-SQL schema-linking mask -> regex n-gram matcher over table/column names (`<mask>`) and numbers/quoted strings (`<unk>`), no cell-value lookups;
* sqlparse skeleton -> regex tokenizer with the original post-processing;
* completion-style `SELECT ` suffix -> a one-line system message for a chat model;
* k = 5 (spec); the repo's BIRD scripts use 7;
* schema = rich DDL (default `schema_style="rich"`; `"plain"` gives the original bare CREATE TABLE text);
* pool = POOL57 only, same-database examples are excluded.
`organization="dail"` (default; question+SQL pairs, the DAIL_O organisation of the DAIL-SQL system) or `"full"` (FI_O: schema+question+SQL per example).
Discrepancy to flag: EXPERIMENT_SPEC section 4 describes the DAIL organisation as "FI_O full information", whereas the DAIL-SQL system itself uses DAIL_O. Default follows the system; `organization=full` reproduces the spec wording at much higher token cost (decision for the spec owner, R3).
Calls: exactly 2 (preliminary + final), no budget knob (reported natively, spec section 4).

### 2.3 self_debug (Chen et al., ICLR 2024)
IDENTICAL: the chat prompts of Appendix E.6/E.7, quoted verbatim in the module docstring ("You are an expert programming assistant.", "Write the SQL query for the given question.", "Summarize the return type of the SQL execution.", "Infer the return type of the question.", "Does the SQL match the question? If not, generate the fixed SQL."), one conversation across turns, execution result (first rows) fed back, stop when the model keeps the SQL.
ADAPTED / OURS: evidence appended to the question; when the SQL raises an execution error the explanation step is skipped and the error text is fed back (this error prompt is our adaptation: Spider-style tasks in the paper have no runtime errors, only the code-translation tasks do); `max_rounds` = 3 = the iteration budget of our system (spec section 4 "same max-iteration budget"); `feedback` = `explain` (E.7) or `simple` (E.6).
The paper only states "a maximum number of debugging turns"; the native run uses the same cap of 3 unless the spec owner fixes another value.

### 2.4 mac_sql (Wang et al., COLING 2025)
IDENTICAL: `core/const.py` (selector_template, decompose_template_bird, refiner_template) byte-for-byte; Selector rendering of columns with value examples and descriptions, "need prune" rule (no Selector call when the average column count is <= 6 and the total is <= 30), Decomposer (answer = last ```sql block), Refiner triggers for BIRD (empty result, a None among the first rows, or an execution error).
ADAPTED:
* client: openai 0.28 ChatCompletion at T=0.1 -> `LLMClient.chat` at the run temperature, one user message per call as in the original;
* Refiner rounds: `max_refine` = 3 (paper: "a maximum of three rounds"); the public code shares `MAX_ROUND = 3` between agents so effectively allows ONE refinement without re-execution; `max_refine=1` reproduces the repository (R5);
* no ```sql block in a Refiner reply -> previous SQL kept (original emitted an "error:" string);
* `*_tables.json` is read from the BIRD train directory; PRAGMA-synthesised entry if absent;
* per-database value statistics are cached on disk; distinct-value scan capped at 5000 values per column (original unbounded).

### 2.5 ac_sql (Zheng et al., arXiv 2410.22082) - reimplementation
IDENTICAL: Actor, Regenerate and Critic prompts quoted from Sec. 4.3 / Appendix A (module docstring); Execution Critic first (False on any execution error, True otherwise even for an empty result), then the LLM Critic; Actor regenerates until True or `max_regen`; the last regeneration is output without a Critic check; same LLM for Actor and Critic, zero-shot.
ADAPTED: evidence appended to the question; rich DDL; `max_regen` = 3 in the matched setting (paper: 5; native run uses 5); `ac_sql_x` replaces the Critic by `deepseek-v4.1-flash-1017` (spec `acsql_x`); critic output limited to 16 tokens.
No official code was found (section 1).

### 2.6 memo_sql (Yang et al., arXiv 2601.10011) - reimplementation, all prompts OURS
The paper prints no prompts, so every prompt in `memo_sql.py` is "our adaptation" written from the textual description.
IDENTICAL (structure): offline error-correction memory of quintuples <q, s+, s-, E, delta> with the nine-type taxonomy E1-E9 (Table 4); three decomposition strategies (table-wise, hierarchical, atomic operational) with a no-decomposition fallback; ReAct+Reflect for each sub-question (reason, sub-SQL, observe execution, reflect, one revision); final generation from the trace plus 5 retrieved exemplars (retrieved with question + skeleton of the latest sub-SQL) in three styles (CTE, flat JOIN, nested) = up to 9 candidates; memory-augmented refinement (key = question + candidate skeleton, top-40 retrieval, drop entries whose error-type set is subsumed by an earlier kept one, keep 3 to 5, critic then refine, at most 3 rounds); majority vote over execution results.
ADAPTED: backbone Qwen3-Coder-30B-A3B -> the system model; GTE-base -> `qwen3.7-text-embedding`; skeleton = DAIL regex skeleton; schema linking (App. A.2) not reproduced (every baseline gets the full rich schema); exec-error candidates go to the refiner directly (the taxonomy excludes pure syntax errors, so the critic only judges executable SQL).
Memory building (`build_memory`, offline, POOL57 only): the system model samples 3 zero-shot candidates per pool question at T=0.7; an executable candidate whose result set differs from the gold result is the s-; the model assigns E-types and a suggestion. The paper builds its memory from BIRD-train candidates with its own sampler/annotator; ours is a proxy. Built memory used in the smoke: first 400 pool questions -> 108 entries, 1325 calls, 2.80M prompt tokens, 0.09M completion tokens (one-time, not charged to per-question budgets; recorded in `internal note B1`).
Knobs: `strategies` (1-3), `styles` (1-3), `react` (bool), `refine_rounds` (0-3), `mem_keep`, `icl_k`, `memory_path` (None = memory ablation). `preset=native` = full 3 x 3 pipeline, `preset=light` = 2 strategies x 1 style, 1 refine round.

### 2.7 alpha_sql (Li et al., ICML 2025) - EXPENSIVE
IDENTICAL: the seven prompt templates (byte-identical); action space (rephrase, schema selection, identify column values, identify column functions, SQL generation, SQL revision, End) and the node-type transition table; eager expansion of all valid actions; UCT with exploration constant 1.414; random simulation to an End node; reward = self-consistency share (largest execution-result group among the 5 samples); 24 rollouts by default; n = 3 for reasoning actions, 5 samples for SQL generation/revision at T=0.8, top_p 0.8, `SQL_VALIDATION_MAX_TRIES` 15; schema-selection filter with PK/FK closure; DDL rendering with column meaning/description/value description/value examples; final answer = largest execution-result group over all End nodes with ties broken by execution time; value retrieval (keyword LLM call, 3-gram Jaccard >= 0.5, edit similarity >= 0.3, embedding cosine >= 0.6, top-3 values).
ADAPTED: vLLM Qwen2.5-Coder-32B -> `LLMClient.chat` of the system model; sampling requests use the `n` argument of one request (DashScope accepts n <= 4, error text "Range of n should be [1, 4]", so n=5 is sent as 4 + 1; each request counts as one call); every request has its own seed so the response cache cannot replay identical samples; MinHash-LSH (datasketch) -> exact in-memory 3-gram Jaccard index (values per text column capped at 50000); execution-time tie-break uses 3 repeats (original `EXECUTION_TIME_REPEAT = 20`); the original's unbounded retry loops (schema selection, SQL generation) are bounded by `max_attempts` = 3 rounds (an executed but empty/invalid SQL is accepted with reward 0 when no valid one was found); a node whose generation never parses, or whose request fails after the client's own retries, is dropped (`extra.api_fail` counts the latter); `max_tokens` 4096 -> 2048 (a few degenerate n=4 requests exceeded the client's 120 s read timeout at 4096); sqlglot normalisation is used when the package is installed, otherwise whitespace normalisation.
Budget knob: `rollouts` (24 native). The spec threshold (calls/q > 60 -> 500-question dev subset) is very likely to trigger at the native setting (section 5, extrapolation).

### 2.8 bon_exec_vote and self_consistency (OURS, voting rule from the spec)
`bon_exec_vote`: n samples of the zero-shot prompt at T=0.7 (seed `1000*r + j`), drop non-executable, cluster by result set (frozenset of row tuples = BIRD EX equivalence), largest cluster wins, ties -> lowest sample index. `self_consistency`: same selection rule, but the prompt asks for a brief plan before the SQL (our adaptation; last code block extracted, `max_tokens` 1536). Each sample is one request, `calls` = n. `n` is the budget knob; the compute-matching rule sets it to `round(B_ref)`; default n = 5.

## 3. Budget knobs and presets (for the compute-matching rule, EXPERIMENT_SPEC section 4)

| system | knob | native | notes |
|---|---|---|---|
| zero_shot, dail_sql | none | 1 / 2 calls | reported natively |
| self_debug (`sdebug`) | `max_rounds` | paper's own cap | matched: 3 |
| ac_sql (`acsql_s`), ac_sql_x | `max_regen` | 5 | matched: 3 |
| mac_sql (`macsql`) | `max_refine` | 1 (repo) / 3 (paper) | matched: 3 |
| memo_sql (`memosql`) | `preset`, `strategies`, `styles`, `refine_rounds` | native = 3 x 3, 3 rounds | `light` = 2 x 1, 1 round |
| alpha_sql (`alphasql`) | `rollouts` | 24 | not measured at 24; 33.7 calls/q at 2 rollouts |
| bon_exec_vote (`bon_vote`), self_consistency (`sc_cot`) | `n` | n/a | matched n = round(B_ref) |

B_ref (mean calls/q of full_fixed on HR) was not available to B1; the tuning step is therefore NOT done here. Presets above are starting points. The four families that must also be run natively (r0 only) are `sdebug`, `macsql`, `acsql_s`, `memosql`; their native values are in the table.

## 4. How to run

```
cd code && PYTHONPATH=. python -m tests_b1.smoke_b1 --system NAME --split bird_train_HT --n 10 [--cfg key=value ...] --out PATH.jsonl
python -m tests_b1.build_memo_memory --out PATH.json --n 400      # Memo-SQL offline memory
python -m tests_b1.summarize_b1 DIR                               # markdown table over smoke outputs
```
`--cfg` values are JSON (`memory_path="/path"`, `preset="light"`, `rollouts=6`). The key is read by `LLMClient` from `<KEY_FILE>`; no script prints it.

## 5. Smoke results (10 questions of `bird_train_HT`)

Setup: split `bird_train_HT` (real `exp/splits/train_holdout.json`, loaded by X2's `data.load_split`), every 30th question of the 300 (`items[0::30][:10]`), primary model `qwen3.7-plus-2026-05-26`, T = 0 (voting members T = 0.7), seed 0, 4 worker threads, on worker machine port <port>. Raw rows: `code/tests_b1/results/*.jsonl` (copied from `b1_work/out/`). EX uses X2's `db.ex_match`.
These are 10-question plumbing checks; a single question is 10 points of EX, so the differences are not evidence about method quality.

| run | n | EX | exec_ok | calls/q | prompt tok/q | completion tok/q | wall s/q (cache-affected) | errors |
|---|---|---|---|---|---|---|---|---|
| ac_sql | 10 | 0.70 | 1.00 | 3.0 | 5316 | 83 | 5.1 | - |
| ac_sql_native5 | 10 | 0.70 | 1.00 | 3.8 | 7054 | 105 | 1.2 | - |
| ac_sql_x | 10 | 0.70 | 1.00 | 2.5 | 4134 | 66 | 1.3 | - |
| bon_exec_vote | 10 | 0.80 | 1.00 | 5.0 | 8180 | 278 | 5.7 | - |
| dail | 10 | 0.80 | 1.00 | 2.0 | 4164 | 108 | 91.5 | - |
| mac | 10 | 0.70 | 1.00 | 2.0 | 5013 | 873 | 1.0 | - |
| mac_repo_refine1 | 10 | 0.70 | 1.00 | 1.8 | 4771 | 463 | 0.1 | - |
| memo | 10 | 0.80 | 1.00 | 42.6 | 94546 | 3746 | 183.2 | - |
| memo_light | 10 | 0.80 | 1.00 | 18.1 | 35224 | 1736 | 7.4 | - |
| self_consistency | 10 | 0.80 | 1.00 | 5.0 | 8290 | 955 | 10.1 | - |
| self_debug | 10 | 0.70 | 1.00 | 4.2 | 13169 | 416 | 1.0 | - |
| self_debug_simple | 10 | 0.80 | 1.00 | 2.2 | 3892 | 530 | 9.5 | - |
| zero_shot | 10 | 0.80 | 1.00 | 1.0 | 1636 | 55 | 0.3 | - |

Row names are file names under `code/tests_b1/results/`: `ac_sql_native5` = `max_regen=5` (paper setting), `mac_repo_refine1` = `max_refine=1` (repository behaviour), `self_debug_simple` = E.6 simple-feedback prompts, `memo`/`memo_light` = native/light preset, `mac` = default `max_refine=3`.

`alpha_sql` was smoke-tested separately because of its cost (3 questions of `bird_train_HT`, `--offset 3`, `rollouts=2`, all 3 from `video_games`; file `alpha_r2.jsonl`):

| run | n | EX | exec_ok | calls/q | prompt tok/q | completion tok/q | wall s/q |
|---|---|---|---|---|---|---|---|
| alpha_r2 (`rollouts=2`) | 3 | 1.00 | 1.00 | 33.7 (22 / 33 / 46) | 48799 | 48221 | 424 / 624 / 822 |

A 6-rollout run on 3 other questions did not finish (it was stopped after about 55 minutes of wall clock in its last attempt, after earlier attempts exposed and fixed two problems: the `n <= 4` limit and unbounded retries); no numbers are reported for it. Calls and tokens grow roughly in proportion to the number of rollouts, so the native 24 rollouts would be about an order of magnitude above the `rollouts=2` row (extrapolation, NOT measured): several hundred calls and well over one hour of wall clock per question on this API. EXPERIMENT_SPEC section 4's rule (calls/q > 60 -> 500-question dev subset) therefore applies.

Notes on the table: `wall s/q` is the wall-clock time of the question inside that run; runs that were repeated after a code or configuration change were answered mostly from the response cache, so only first-time runs are indicative of latency (e.g. `memo` 183 s/q, `dail` 92 s/q including its first pool embedding). `memo` = native preset, `memo_light` = light preset (same 10 questions; its prompts are a subset of the native prompts, so most were cache hits). Embedding requests (DAIL pool, Memo pool and memory, Alpha value retrieval) are not in `calls`.

Reading: the call counts measured here are what the compute-matching rule needs. With the default configuration of each class: `zero_shot` 1, `dail_sql` 2, `mac_sql` 2.0, `ac_sql` 3.0, `self_debug` 4.2, `bon_exec_vote`/`self_consistency` 5, `memo_sql` 42.6 (18.1 light), `alpha_sql` 33.7 at only `rollouts=2` (row above).

## 6. Known limits

* Only 10 questions per system were run; no statistical claim is made.
* `memo_sql` memory is a proxy built by our own sampler (108 entries from 400 pool questions); a larger build (`--n`) is cheap (the 400-question build above took 537 s wall clock with 8 worker threads) and changes the memory content, so the memory file must be fixed (hash recorded) before the final runs.
* `alpha_sql` is a faithful port but slower than every other system by two orders of magnitude; the `n <= 4` API limit splits its 5-sample steps in two requests, which slightly raises its call count relative to a backend that accepts n=5.
* `mac_sql` has no licence upstream (R1); only `core/const.py` is vendored.
