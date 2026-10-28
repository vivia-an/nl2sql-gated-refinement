# EXPERIMENT_SPEC v1 (X1, 2026-10-25)

Executable specification for E1–E11. Audience: X2 (engine), B1 (baselines), runners R1/R2/R3. Companion: `docs/SAP.md` (statistics), `exp/models.yaml` (models). Where this file and `docs/CONTRACT.md` conflict, CONTRACT wins. Numbers marked **[measure]** are planning assumptions to be replaced by smoke-test measurements; numbers marked **[verify]** are dataset facts the engineer must confirm from the files (write them to `exp/splits/data_manifest.json`; the paper may only cite the manifest, never this file).

---

## 0. Global conventions

| Item | Rule |
|---|---|
| Run id | `<exp>-<system>-<modelkey>-s<seed>-<cfgsha8>`; output `runs/<exp>/<run_id>/preds.jsonl` (schema: CONTRACT §4) |
| LLM cache | key = sha256(model ‖ canonical-JSON(temperature, seed, top_p, max_tokens, extra_body) ‖ messages). Hits count toward `calls` and tokens as recorded in the original response (so cost tables are cache-independent); `latency_s` of a hit = original latency |
| Temperature / seeds | **Headline run r0: T=0, seed=0.** Stability runs r1, r2: T=0.7, seeds 101, 202 (top_p=1.0). BoN/SC sample j of run r uses seed `1000*r + j`. Split seed 20261025; HT/HR partition seed 20261026; annotation sample seed 20261027; E4 permutations 1..5; RL seeds 0..4; bootstrap seed 20261101 |
| Thinking | `extra_body={"enable_thinking": false}` for every model and role (models.yaml). Only E2.T turns it on |
| Output parsing | strip ```sql fences (glm-5.2, kimi-k2.6 and qwen add them); JSON via `json.loads` → fallback regex of the outermost `{...}` → one retry at T=0 with "Return only JSON" (the retry counts as a call). Unparseable after retry → `error_class="parse_fail"`, counts as EX=0 |
| API failure | after 6 retries the item is marked `api_fail`, EX=0, and the run is re-queued once; report the count (must be <0.5 %, else rerun the shard) |
| SQL execution | SQLite read-only copy of the BIRD database; timeout 30 s (BIRD default); one connection per worker; row cap for previews 5 rows × 40 chars/cell |
| Result keys | `results/<exp>.json` flat keys `<exp>.<system>.<modelkey>.<metric>[.mean\|ci_lo\|ci_hi\|sd\|n]`, e.g. `E1.full.qwen37plus.ex.mean` |
| Repro | every run folder stores `config.yaml` (resolved), `models.yaml` sha, `prompts/` sha, git hash |

---

## 1. Datasets and splits

### 1.1 Datasets

| Role | Dataset | Use | Notes |
|---|---|---|---|
| Main test | BIRD dev (**1534** questions, 11 databases [verify]) with evidence | E1–E3, E5–E7, E9, E10, E11 | one-shot after lock |
| Tuning / RL / streams | BIRD train, held-out by db_id (§1.2) | all tuning, E4, E6 training | never touches dev |
| Exemplar/history pool | BIRD train minus held-out dbs = **POOL57** (expect 57 dbs; **[verify]** train = 69 dbs, 9 428 pairs) | few-shot, M2 history log, M6 seed | same pool for every system on BIRD dev and held-out |
| Corrected labels | Jin et al. (arXiv 2601.08778) and BIRD-Platinum (2603.20004) corrected dev gold, **only if publicly released** | E1 re-scoring of stored predictions (0 new LLM calls) | engineer checks availability + licence, stores source URL + sha256 in `data_manifest.json`. If unavailable: report "not available"; **we do not correct labels ourselves** (LLM annotation in E10, Amendment 2, estimates the gold-suspect rate only) |
| Robustness | Spider-Syn, Spider-Realistic, Spider-DK (expected ≈1 034 / 508 / 535 [verify]); Spider dev (sanity) | E8 | exemplar pool = Spider train (7 000 [verify]); config frozen from BIRD, no retuning. Use Spider test-suite evaluation if the distilled suites are available, else plain EX (flag it) |
| Ambiguity | AmbiQT | E7 | sample 500 ambiguous items stratified over the ambiguity types present in the file (seed 20261027) + 500 unambiguous Spider-dev questions (same seed) → label `ambiguous∈{0,1}` |
| Contamination | LiveSQLBench-Base-Lite (PostgreSQL) | E9, appendix | only tasks with public gold; if gold is not public, E9b is dropped and listed as a limitation. Executor needs a `dialect` switch (`EXPLAIN (FORMAT JSON)` for efficiency) |

`thrombosis_prediction` is a BIRD dev database: it is kept in the main analysis, reported separately per database, and excluded in a sensitivity analysis (SAP §8).

### 1.2 BIRD train held-out split (`exp/splits/make_splits.py`, output `exp/splits/train_holdout.json`)

Deterministic algorithm:

1. Load train and dev. Assert `set(train_db) ∩ set(dev_db) = ∅`. Write counts + sha256 of every input file to `data_manifest.json`.
2. `n_q[db]` = number of train questions. Eligible = `{db : n_q ≥ 100}`. Sort eligible by `(n_q, db_id)`; cut into three equal-rank terciles (small/mid/large).
3. `rs = np.random.RandomState(20261025)`; draw 4 dbs per tercile without replacement → **12 held-out dbs** (if a tercile has <4 eligible, take all and fill from the next-larger tercile; assert 12).
4. Partition the 12 into **HT** (tuning sandbox, 4 dbs) and **HR** (RL environment + E4 streams, 8 dbs): `perm = RandomState(20261026).permutation(sorted(held12))`; HT = `perm[:4]`, HR = `perm[4:]`; accept only if HT contains ≥1 db from each of ≥2 terciles and HR contains ≥2 from each tercile; else re-draw with seed+1 (max 100 tries; log the seed used).
5. **POOL57** = all other train dbs. Held-out dbs never appear in the pool, never in M2/M6 stores of any run on HT/HR/dev.
6. **HT-300**: 300 questions sampled from HT, proportional to db size, stratified by SQL complexity tercile (number of tables in gold SQL via sqlglot; fallback token length), seed 20261026. This is the only set used for prompt/hyper-parameter search of any system.
7. **HR per-db roles** (`N_db = min(n_q, 150)` questions drawn with seed 20261026): `seed_set` = 15 %, `probe_set` = 30 %, `stream` = remaining ≈55 % (≈21 / 41 / 75 at N=137). E6 uses **all** HR questions (≈1 100 [measure]), E4 uses the three roles above.
8. **Grouped CV on HR**: 4 folds × 2 dbs; assign dbs sorted by `n_q` desc in snake order (f0,f1,f2,f3,f3,f2,f1,f0). Folds are used for E5 learned critic, E6 controller, baseline selection.
9. Output JSON fields: `ht_dbs, hr_dbs, pool_dbs, terciles, ht300_qids, hr_roles{db:{seed,probe,stream}}, cv_folds, seeds_used, input_sha256`. IDs are `"<split>:<db_id>:<question_id>"`.

Contingency (decided before lock, logged): if the E6 grouped-CV shows RL data starvation (CV validation EX gain over fixed policy has a 95 % CI width > 6 pp), add 4 more train dbs from the eligible set to HR and **remove them from POOL57 for all systems**, rebuilding caches. At most once.

### 1.3 Dev lock procedure (implements CONTRACT §3)

1. Pre-lock phase uses only HT/HR (and Spider/others never). Every tuning trial is appended to `exp/tuning_log.jsonl` `{system, config_sha, split, n_q, metric, timestamp}`. **Equal effort rule:** every system (ours and each baseline) gets ≤10 configuration trials on HT-300 (coarse-to-fine, first 100 questions then 300).
2. Freeze: `exp/configs/frozen/<system>.yaml` for every system + `exp/frozen/controller_<rule>.pt`, `exp/frozen/corpus_burnin_<sha>.sqlite` (M6 burn-in store, §3.M6), `prompts/` directory. `exp/dev_lock.json` entries: `{system, config_path, config_sha256, models_yaml_sha256, prompts_sha256, controller_sha256, corpus_sha256, git_hash, date, datasets_allowed:[bird_dev,spider_*,ambiqt,livesqlbench], seeds:[0,101,202]}`. Main thread commits the lock before any dev run.
3. Runner start-up check: recompute sha256 of the config + referenced artefacts; refuse to run on any non-HT/HR split if there is no matching lock entry or any hash differs.
4. Dev gold is loaded only by `eval/` (evaluator process); pipeline processes receive question, evidence, db_id only. Every evaluator access is logged to `exp/dev_access.log`.
5. Each (system, seed) runs on dev **once**. Re-run only for infrastructure failure with an identical config hash; the log records both attempts and the first complete run is the one analysed.
6. No post-hoc config edits. A bug found after lock → new lock entry `v2`, both versions reported, the paper states which was pre-registered.
7. The SAP (`docs/SAP.md`) is committed before the lock; its commit hash is written in `dev_lock.json`.

---

## 2. Models (from `exp/models.yaml`)

Live `/models` list queried 2026-10-25 (180 entries). Eight models, one 1-call test each for default/`enable_thinking=false`/`thinking:{type:disabled}` (all accepted; reasoning tokens disappear with `enable_thinking=false` for 7/8, kimi-k2.6 never emitted reasoning):

| key | id | vendor | access | snapshot | role |
|---|---|---|---|---|---|
| qwen38max | qwen3.8-max-0902 | Alibaba | proprietary | dated | tier2 / E2 |
| **qwen37plus** | qwen3.7-plus-2026-05-26 | Alibaba | proprietary | dated | **primary actor**, tier1 |
| qwen37flash | qwen3.7-flash-2026-07-15 | Alibaba | proprietary | dated | E2 (small/fast) |
| qwen3527b | qwen3.5-27b | Alibaba | open-weight (unverified) | undated | E2 |
| qwen36a3b | qwen3.6-35b-a3b | Alibaba | open-weight (unverified), MoE | undated | E2 |
| **dsv41flash** | deepseek-v4.1-flash-1017 | DeepSeek | open-weight (unverified) | dated | **cross-vendor critic**, E2 |
| glm52 | glm-5.2 | Zhipu | open-weight (unverified) | undated | E2 |
| kimik26 | kimi-k2.6 | Moonshot | open-weight (unverified) | undated | E2 |

Deviations from PLAN §E2: `qwen3.8-flash` has no dated snapshot (only an alias), so `qwen3.7-flash-2026-07-15` is used; `deepseek-v4-flash` → `deepseek-v4.1-flash-1017` (newest dated); `glm-5.x` → `glm-5.2` (no dated GLM id exists). Undated ids are pinned by recording `created` and by the full request/response cache; the paper must say these four vendors do not expose dated snapshots. Open-weight status and parameter counts must be confirmed from model cards by L1/X2 before the paper states them (the yaml says "unverified"; do not guess).

Measured on 4.3k-token prompts, 112 output tokens, thinking off, no 429 up to 24 concurrent: qwen3.7-plus 1.74 calls/s at c=8 (3.13 at c=24); qwen3.8-max 2.0 (5.19); deepseek-v4.1-flash 3.1 (7.07).

Embedding `qwen3.7-text-embedding` (dim 1024, verified), rerank `qwen3.7-text-rerank` via `/compatible-api/v1/reranks` (verified).

E2.T (appendix): primary actor and qwen3.8-max, zero-shot, `enable_thinking=true`, max_tokens 8192, dev, r0 only. Tests the "thinking vs our loop" question. kimi-k2.6: n/a.

---

## 3. Method components M1–M7

Common prompt rule: all prompts live in `code/prompts/*.txt`, versioned, sha in lock. English instructions. Schema block format = `CREATE TABLE` DDL with FK lines + `-- col: description | samples: a, b, c` comments.

### M1 Intent modelling
- **In:** question, evidence, db_id. **Out:** task graph JSON `{objects:[{name,kind:entity|measure|attribute,mention}], filters:[{text,kind:time|range|equality|text|other}], time_scope, aggregation:[…], group_by:[…], order_limit:{…}, ambiguities:[…]}`.
- **Prompt sketch (P-M1):** "Extract the structured request. Do not write SQL. Fields: …; copy mentions verbatim; list in `ambiguities` anything with more than one plausible reading." One call, T=0, max_tokens 700. Also normalises evidence (strip, split on `;`) deterministically.
- **Params:** none tuned. Ablation −M1: no graph in any downstream prompt/retrieval query.

### M2 Multi-source knowledge fusion (deterministic code + embeddings)
Builds per-db **knowledge store** (indexed once, cached by db hash):

| Source (patent term) | Public-data realisation | Record |
|---|---|---|
| schema/metadata | DDL, BIRD `database_description` CSVs, per text column top-3 distinct values (≤40 chars) | `kind=schema` |
| historical SQL log | POOL57 `(question, evidence, gold SQL)`; skeleton = SQL with literals/identifiers masked (sqlglot) | `kind=history` |
| lineage | FK graph + inferred joins (same column name & type & ≥30 % value overlap on a 1 000-row sample) | `kind=lineage` edges |
| business metadata | POOL57 evidence strings split into `(term → definition)` entries | `kind=term` |

Because BIRD dev/HT/HR databases are disjoint from POOL57, history and term entries are **cross-database** (patterns, not same-db facts); co-occurrence lineage cannot be mined for the target db, hence FK + inferred joins. The paper must state that M2 is a simulated, public-data realisation. The question's own evidence is an input to **every** system (standard BIRD protocol); a no-evidence run (primary model, zero-shot and Full, r0) is reported as a secondary row.

### M3 Domain-routed retrieval (deterministic; "domain agents" are retrieval workers, not LLM agents)
1. Domain split: cluster tables (Louvain on FK graph, edge weight 1, plus 0.5·cosine of table-description embeddings); ≤6 tables → one domain.
2. Router: embed (question + M1 object mentions); pick top-2 domains by max cosine to member tables.
3. Each domain worker retrieves tables/columns: BM25 (k1=1.2, b=0.75 over names+descriptions+samples) ⊕ dense (cosine) → reciprocal-rank fusion (k=60) → top 20 → rerank → keep top `k_t` tables (default 6) + their columns; add tables on the shortest FK path between selected tables (≤2 extra hops).
4. Schema fallback: if full DDL+samples < 6 000 tokens, pass the **full schema** with the top `k_t` tables flagged "likely relevant" (BIRD schemas are small; pruning only below that threshold would hurt). 
5. Exemplars: hybrid (BM25 over masked question + dense) over `history` records of POOL57 → rerank → top `k_e` (default 4). Also top-3 `term` entries.
- **Tuned on HT-300:** `k_t∈{4,6,8}`, `k_e∈{2,4,6}`. Ablation −M3: dense-only top-`k_t` tables and top-`k_e` exemplars, no routing/RRF/rerank/lineage expansion.
- Embedding/rerank calls are logged separately (`embed_calls`, `rerank_calls`) and excluded from `calls`.

### M4 RAG generation
- **In:** question, evidence, M1 graph, M3 schema context, exemplars (+terms). **Out:** one SQLite query.
- **Prompt (P-GEN):** system "You write one SQLite query…"; blocks `# Schema`, `# Lineage hints (join keys)`, `# Examples` (question, evidence, SQL), `# Business terms`, `# Request` (question, evidence, task graph). "Return only the SQL." T per run (0 or 0.7), max_tokens 1024. 1 call.

### M5 Critic and Actor revision
- **Score** = `w1·s_syn + w2·s_sem + w3·s_eff`, w = (0.3, 0.5, 0.2), range 0–10.
- `s_syn` (executor, deterministic): ok→10; ok but 0 rows or single all-NULL cell→8; identifier error ("no such column/table", "ambiguous column")→3; syntax error→2; other runtime error→3; timeout→1.
- `s_eff` (executor, deterministic): exec error→0; else 10, −2 per `SCAN` of a table with >10 000 rows in `EXPLAIN QUERY PLAN` (cap −6), −1 per `USE TEMP B-TREE`, −1 per `CORRELATED SCALAR SUBQUERY`, −2 if elapsed >2 s, −4 if >10 s (non-cumulative: the larger), clip [0,10]. Table row counts come from M2.
- `s_sem` (LLM, **P-CRIT-SEM**): input question, evidence, M1 graph, schema context, SQL, execution preview (columns, row count, first 5 rows). Output JSON `{"semantic": int 0-10, "issues":[{"type":"semantic|efficiency|syntax|other","description":"…"}], "suggestions":["…"]}`. Rubric anchors: 10 fully answers incl. all filters/aggregates; 8–9 minor omission; 5–7 plausible but a filter/aggregate/column/join is wrong or missing; ≤4 does not answer. T=0, max_tokens 512. 1 call.
- **Policy P0 (patent-fixed):** after each Critic call: `score ≥ θ_acc=8.5` → accept; else if `iter < T_max=3` → **revise** (P-REV: question, schema, previous SQL, execution preview, Critic JSON; "Return only the corrected SQL"; if `θ_fb=7.5 ≤ score < θ_acc` the prompt says "optional improvements", below 7.5 "must fix"); after `T_max`, return the highest-scoring candidate. Route-to-human if the final best score `< θ_h` (default 7.5; swept in E7). 
- **Critic variants (E5):** V1 binary (AC-SQL style: "Is the SQL correct? Answer yes/no + one-sentence reason", given the same inputs); V2 pure-LLM weighted (one call returns syntax, semantic, efficiency + issues); **V3 execution-grounded weighted (default)**; V4 = V3 with `dsv41flash` as Critic model; V5 learned (CPU; §E5).
- **Primary-system critic rule (fixed before lock):** self-critic V3 (Actor model = Critic model) unless, on HR (grouped CV, candidates of §E5), AUROC(V4) − AUROC(V3) ≥ 0.02 with paired db-clustered bootstrap CI excluding 0; then the primary system uses V4. Both V3 and V4 systems are run on dev (`full`, `full_x`).
- **Sensitivity (no tuning of the headline):** w over the simplex with step 0.1 (66 points), `θ_fb∈{6.5,7.0,7.5,8.0}`, `θ_acc∈{8.0,8.5,9.0}`, `T_max∈{1,2,3,4}`; evaluated on stored candidate sets (0 extra calls).
- Ablation −M5: single generation, no Critic (also disables M7, which needs the Critic).

### M6 Value-weighted experience corpus
- **Entry** `e = (q, evidence, SQL, db_id, C̄, F̄, n, WS)`. `WS = α·C̄ + β·F̄ + γ·Φ`, α,β,γ = 0.4, 0.5, 0.1 (sum 1, WS∈[0,10]); `Φ = 10·min(1, ln(1+n)/ln(1+20))` (n = times retrieved into a prompt).
- **Online estimate:** EMA `x̄ ← (1−η)x̄ + η·x`, η=0.3 (first observation sets x̄=x). Priors for entries lacking an observation: `C̄=F̄=7.0`, n=0 ⇒ WS₀=6.3 (≥τ, not evicted).
- **Updates:** (i) on creation from a stream item: `C̄` = Critic score of the accepted SQL; `F̄` = `10·EX_self` with probability ϕ=0.5 (simulated user feedback, from gold execution on train data only), else prior. (ii) On every later generation that used e as an exemplar: n+=1, and with probability ϕ the feedback `x = 10·EX_final` of that generation updates `F̄` (shared credit among the k exemplars; noisy by design). Seed-set entries start at the prior (no feedback in a "history log").
- **Eviction:** `WS < τ=6.0` and `n_fb ≥ 1` feedback observation ⇒ remove.
- **Priority recall:** candidates = top-20 by hybrid similarity; rank by `sim_norm + ρ·WS/10` (ρ=0.2); top `k_e` into the prompt.
- **Tuned on HT mini-stream (§E4 tuning):** `η∈{0.2,0.3,0.5}`, `ρ∈{0,0.1,0.2,0.4}`. Sensitivity (reported, not tuned for the headline): `τ∈{5,5.5,6,6.5,7}`; (α,β,γ)∈{(.4,.5,.1),(.5,.4,.1),(.3,.6,.1),(.4,.4,.2),(.6,.3,.1),(.5,0,.5)}; ϕ∈{0.25,0.5,1.0}.
- **Burn-in for dev (the corpus is frozen on dev: dev gold is never available as feedback):** run the primary pipeline over 1 500 POOL57 questions (stratified by db, seed 20261028) with the E4 store logic, simulated feedback ϕ=0.5, then freeze the store (`corpus_burnin_<sha>.sqlite`: WS values, evicted entries removed). Ablation −M6 on dev: raw POOL57 (WS ignored, no eviction, ρ=0).
- E4c (dev, no feedback): online store on dev stream without gold; `F̄` is never updated (β-term replaced by prior); reported as the "unsupervised deployment" scenario.

### M7 RL controller (CPU)
Defined in §E6. At run time the controller replaces policy P0; M1–M6 unchanged.

---

## 4. Systems under test

Primary model for all systems: `qwen37plus` unless noted; evidence given to all systems; same POOL57 exemplars where a system uses exemplars; same executor/parsers.

| key | System | Implementation |
|---|---|---|
| zs | Zero-shot | DDL + sample values + evidence → SQL. 1 call |
| dail | DAIL-style RAG | DAIL_S selection: question-skeleton similarity over POOL57 + preliminary-SQL skeleton filter, k=5, "full information" organisation. 2 calls (preliminary + final) |
| sdebug | Self-Debug (Chen et al., ICLR 2024) | generate → execute → feed result/error + "explain the SQL" → fix, ≤3 rounds, stop when no error |
| macsql | MAC-SQL | official repo prompts/agents (Selector, Decomposer, Refiner ≤3), only the LLM client swapped |
| acsql_s / acsql_x | AC-SQL (2410.22082), reimplemented from the paper | Actor generates; binary Critic (same LLM / `dsv41flash`) with execution result; regenerate until "correct" or ≤3 rounds. B1 documents every deviation |
| memosql | Memo-SQL (2601.10011), reimplemented (check for code first) | memory of successful queries + error→fix pairs built from POOL57 (successes = gold pairs; error-repair pairs from a 1 500-question POOL57 run of the same generator), retrieval-augmented generation per the paper |
| alphasql | Alpha-SQL (ICML 2025, 2502.17248) | official code with our client. If measured calls/q on smoke >60 ⇒ run on a 500-question dev subset (stratified by db, seed 20261027) and compare against all other systems on that subset only |
| bon_vote | Best-of-n + execution voting | n samples (T=0.7), drop non-executable, cluster by result set (BIRD EX equivalence), pick the largest cluster, tie → lowest sample index |
| sc_cot | Self-consistency (CoT) | same voting; prompt asks for a brief plan then the SQL; n samples |
| bon_critic | Best-of-n + Critic rerank | n samples, pick max V3 score (isolates the Critic's value as a selector) |
| full_fixed | M1–M6 + P0 | patent thresholds |
| full_rl | M1–M7 | frozen controller |
| full_x | full_fixed or full_rl with Critic = `dsv41flash` | E1/E5 secondary |
| full | alias of the pre-registered primary system | rule below |

**Primary-system rule (fixed before lock, evaluated on HR grouped CV):** `full := full_rl` if its CV validation EX at calls ≤ calls(full_fixed) is not worse than full_fixed by more than 1.0 pp (paired CI lower bound > −1.0 pp) **and** deferral rate ≤10 %; else `full := full_fixed` and RL is reported as a negative or neutral result.

**Compute-matching rule.** `B_ref` = mean LLM calls/question of `full_fixed` on HR (all LLM calls: M1, generation, Critic, revise; embedding/rerank excluded and reported). Each baseline with a budget knob is configured so that its mean calls/q on HR is within [0.85, 1.15]·`B_ref` (knobs: Self-Debug/AC-SQL/MAC-SQL max rounds and forced verification; Memo-SQL candidate count; BoN/SC `n`; BoN-critic `n`, with each sample costing 2 calls). Also report mean total tokens/q (must be within ±25 % of `full_fixed`, else adjust prompt budget knobs, not content). Baselines without a knob (zs, dail) are reported natively. Four families (sdebug, macsql, acsql_s, memosql) are additionally run at native paper settings (r0 only). **K1 baseline = the compute-matched baseline with the highest HR EX** (ties → fewer calls), chosen before lock.
Training-based systems (SQL-R1, Arctic-Text2SQL-R1, Reasoning-SQL, …): published BIRD-dev numbers only, flagged "different model, trained", never in tests.

---

## 5. Metrics

| Metric | Definition / source |
|---|---|
| **EX** | Official BIRD semantics: execute predicted and gold on the same SQLite database; correct iff `set(pred_rows) == set(gold_rows)` (row tuples, column order matters, duplicates/row order ignored); execution error or 30 s timeout ⇒ 0. Use BIRD's own evaluation scripts verbatim (main repo `llm/src/evaluation.py`; VES: `evaluation_ves.py`; Mini-Dev scripts for Soft-F1 and R-VES — file names from memory, **verify** in the downloaded repo and store its commit hash in the manifest); do not re-implement |
| VES / R-VES | BIRD scripts. Timing is noisy on shared worker machines: run VES in an isolated evaluator job (one process, no other heavy task on the worker machine), `iterate_num=100` for headline systems (full, full_fixed, zs, dail, best baseline), 20 for the rest; gold timings measured once per batch and reused |
| Soft-F1 | BIRD Mini-Dev cell-level soft F1 script |
| Jaccard / overlap | row-set Jaccard `\|P∩G\|/\|P∪G\|` and overlap coefficient `\|P∩G\|/min(\|P\|,\|G\|)`; both empty ⇒ 1; execution failure ⇒ 0 |
| AUROC | of the confidence score for the event EX=1 (db-clustered bootstrap CI). Binary critics give a 1-bit score (AUROC = balanced-accuracy-equivalent; stated) |
| ECE | 15 equal-mass bins on `score/10` after isotonic calibration fitted on HR (cross-fitted by CV folds) and applied to dev; raw (uncalibrated) ECE also reported; Brier score |
| Risk–coverage | sort by confidence desc.; at coverage c, risk = 1 − EX on the accepted; AURC (area, lower better); coverage at 10 % and 20 % risk |
| TrustSQL-style score | `RS(c)=(1/N)Σ s_i`: feasible and answered correctly +1; feasible answered wrong −c; feasible and abstained 0; infeasible and abstained +1; infeasible and answered −c. c∈{0,1,5,10,N}. On BIRD all items are feasible; on the AmbiQT mix ambiguous items are treated as "should abstain" (declared proxy) |
| Auto error classes (E10) | see §6.E10; exclusive classes by decision list |
| Cost | `calls`, prompt/completion tokens, per-question latency = Σ call latencies + execution time (sequential), plus wall-clock of the run; price only if `models.yaml` has verified prices |
| Human-in-loop reporting | `coverage`, `EX_accepted` (on non-abstained), `EX_all` (abstained counted wrong), and `EX_with_oracle_human = cov·EX_accepted + (1−cov)` labelled as upper bound |

---

## 6. Experiments

Unless stated: dev = BIRD dev 1 534, primary model `qwen37plus`, runs r0/r1/r2.

### E1 Main result
- Systems: zs, dail, sdebug, macsql, acsql_s, acsql_x, memosql, bon_vote, sc_cot, bon_critic, full_fixed, full_rl, full_x (r0, r1, r2); alphasql (r0 only; subset rule above); native variants of sdebug, macsql, acsql_s, memosql (r0).
- Outputs: EX, VES, R-VES, Soft-F1, Jaccard, calls, tokens per system with db-clustered CIs; per-database table (11 rows; thrombosis_prediction highlighted); re-scoring on corrected gold when available; seed mean±SD.
- Primary test K1: `full` vs HR-selected best compute-matched baseline (SAP).

### E2 Eight models
- For each of the 8 models: zs, dail, full_fixed (self-critic V3 with that model; the same model also as actor; escalation not used), r0. Primary-model runs are reused from E1 via cache.
- Outputs: EX, auto error-class distribution, calls/tokens/latency; E2.T appendix (thinking on).
- Hypothesis on "smaller models hallucinate less" is only written if the hallucinated-identifier rate (auto class) shows it; otherwise dropped.

### E3 Module ablation (dev, r0, base = `full`)
Configs: full; −M1; −M2 (no history exemplars, terms, lineage hints; DDL+descriptions+samples+evidence kept); −M3; −M5 (no Critic loop; includes −M7); −M6 (raw POOL57); −M7 (= full_fixed; only if full = full_rl). Omnibus Cochran's Q + post hoc (SAP). Error-class analysis: for items fixed by a module (ablated wrong, full right) tabulate the ablated output's auto class.

### E4 Corpus evolution and pollution (HR, no dev)
- **Unit:** per database an independent store; 8 HR dbs. Roles per §1.2: `seed_set` (≈21/db), `stream` (≈75/db), `probe_set` (≈41/db).
- **Initial store:** POOL57 `history` (clean, static, shared) + per-db experience store initialised with the `seed_set` (SQL = gold) with pollution applied.
- **Pollution p∈{0,10,20,40}%:** `round(p·|seed_set|)` seed entries (chosen with seed 20261030+p) get a wrong executable SQL that returns a different result set from gold: 50 % target "natural" wrong SQL (an actual failure of `zs` on that question from a POOL57-style run of the primary model; run `zs` on all HR seed questions first) and 50 % programmatic mutation of gold via sqlglot (comparison operator flip, aggregate swap SUM↔AVG / MAX↔MIN, drop one WHERE conjunct, replace a column by a same-type sibling, change join key); each mutation is verified to execute and to differ from gold in result (≤10 tries, else use a natural failure, else mark the entry skipped and log). Polluted entries start with the neutral prior, i.e. they carry no marker. Natural pollution from the system's own wrong outputs accumulates additionally.
- **Stream:** each `stream` item: retrieve (per policy) → pipeline P0 (M1–M5) → final SQL → store update (policy) with gold-simulated feedback at rate ϕ. 5 permutations (seeds 1–5) of the stream order, independently per db.
- **Policies (5):**
  | key | write rule | eviction | recall ranking |
  |---|---|---|---|
  | static | none | none | sim |
  | append | write every accepted output | none | sim (Memo-SQL-style, only appends) |
  | ws_evict | M6 updates | WS<τ | sim (ρ=0) |
  | ws_full | M6 updates | WS<τ | sim + ρ·WS/10 (**ours**) |
  | memrl | write every accepted output; utility `Q←Q+0.3(r−Q)`, Q₀=0.5, r=EX at feedback rate ϕ | none | z(sim)+λ_m·z(Q), λ_m∈{0.25,0.5,1} tuned on HT (MemRL 2601.03192 reimplemented from the paper) |
- **Checkpoints:** frozen-store probe (no writes) after 0, 40 % and 100 % of the stream: EX on the db's `probe_set` (≈328 items in total). Prequential online EX in windows of 15 stream items. Store diagnostics at each checkpoint: store size, share of polluted entries, polluted-entry retrieval rate, evictions of clean vs polluted entries (precision/recall of eviction), mean WS clean vs polluted.
- **Matrix:** 5 policies × 4 pollution × 5 perms = 100 stream runs. Fallback if throughput <4 calls/s: perms = 3 for p∈{10,20} (pre-registered).
- **Tuning (HT mini-stream):** 4 HT dbs, 40-question streams (seed 20261026), `η, ρ, λ_m` as above, ≤10 trials; `τ=6.0` and α/β/γ stay at defaults for the headline.
- **E4c:** dev stream (5 permutations), `ws_full` and `static` only, no gold feedback; learning curve of prequential EX per 150-item window.
- Outputs: learning curves, pollution–EX curves, diagnostics, GLMM (SAP).

### E5 Critic study
- **Candidates:** HR: root + depth-1 nodes of the E6 tree (≈6 per question ≈6.6k); dev: initial and final candidates of `full_fixed` (≈3.1k). Gold label = EX.
- **Variants:** V1–V4 (§M5), each scored by 1 LLM call per candidate; V5 learned: `HistGradientBoostingClassifier` (sklearn, CPU) on features [V3 sub-scores, V4 sub-scores, weighted scores, exec status one-hot, log rows, #cols, SQL length, #joins, top-1 retrieval sim, mean WS of exemplars, agreement of result with the other candidates of the same question], trained on HR with grouped 4-fold CV, then frozen (trained on all HR) and applied to dev.
- **Outputs:** AUROC, AUPRC, ECE before/after isotonic, Brier, reliability diagrams; selective accuracy; weight simplex heat-map (66 points) and threshold sensitivity on HR+dev with stated "post hoc on dev" label for the dev side; effect on end-to-end EX of `full_fixed` with each Critic variant (dev, r0, 4 runs).
- Weights/thresholds are **not** retuned for the headline; if a tuned set beats the default on HR by ≥1 pp it is reported as `full_tuned` (extra dev run, secondary).

### E6 RL controller
**MDP.** Episode = one question. Horizon H=6 steps. Actions `A = {accept, revise, re_retrieve, escalate, route_human}` (mask `escalate` in the main setting `A4`; `E6-tier` uses `A5`).

| Component | Definition |
|---|---|
| State (27 dims) | Critic `s_syn, s_sem, s_eff, score` /10 (4); Δscore vs previous candidate (1); exec status one-hot {ok_nonempty, ok_empty, err_syntax, err_identifier, timeout} (5); `log1p(rows)/log1p(1000)`, `min(cols,10)/10` (2); result set changed vs previous candidate (1); `k/T_max`, #re-retrievals/2, remaining call budget fraction (3); top-1 and mean top-k retrieval sim, mean WS of exemplars/10 (3); question features: tokens/64 (clipped), #objects in M1, has time filter, #tables/20, #columns/200, evidence present (6); tier one-hot {tier1, tier2} (2); previous action one-hot is **not** included (Markov via the features above) |
| Actions | `accept`: terminate with the current candidate. `revise`: P-REV with the Critic JSON → new candidate → Critic (2 calls). `re_retrieve`: new M3 query = question + Critic issue text + M1 mentions, excluding previous exemplars, `k_t+2` tables → P-GEN from scratch → Critic (2 calls, +embed/rerank). `escalate`: P-REV executed by tier2 (`qwen38max`) → Critic (2 calls). `route_human`: terminate, abstain |
| Reward | accept: +1 if EX=1 else −1; route_human: −c_h; every non-terminal step: −λ·(tokens_step/1000)·m_tier. Forced termination at H or call budget 8: accept the highest-scoring candidate (standard reward). γ=1 (finite horizon) |
| Sweep | c_h∈{0.1,0.25,0.5,0.75}; λ∈{0, 0.01, 0.03} ⇒ 12 cells. Break-even for accept vs route: accept iff P(correct) > (1−c_h)/2 (0.45 / 0.375 / 0.25 / 0.125). `m_tier`: 1 (tier1), `tier_cost_multiplier_fallback` or verified price ratio (tier2) |

**Cached branching environment (HR only; built before lock, ≈82k calls).** Per HR question, a tree: root = M1 + M3 + P-GEN + Critic (tier1, T=0). Depth-1: `revise`×2 (T=0 and T=0.7 seed 7), `re_retrieve`×2 (same), `escalate`×1. Depth-2: from each depth-1 node `revise`×1, `re_retrieve`×1, `escalate`×1. Depth-3: `revise`×1 from each depth-2 node. ≈36 nodes/question, ≈75 calls. Nodes with identical SQL and identical context are merged. Each node stores SQL, Critic JSON and sub-scores, exec status, rows, tokens, latency, gold EX (reward only; never in state). Env transition = look up the child of the chosen action (uniform among stored children); an action with no stored child at the frontier is masked (so the effective horizon is 3 generations). Replay fidelity check: run the trained controller live (not cached) on 100 HR validation questions and compare cached-replay EX/calls with live: |ΔEX| ≤3 pp required, else report the gap.

**Algorithm.** PPO (clipped actor–critic, GAE λ=0.95, γ=1) primary; A2C as ablation. Separate actor and critic MLPs, each 27→64→64 (tanh), Adam lr 1e-3 (annealed to 1e-4), entropy 0.01, clip 0.2, 4 epochs × minibatch 256, 2 048 env steps per update, 200 updates, invalid-action masking. PyTorch CPU (or a numpy fallback decided by X2). Runs on a worker machine (CPU); env tree exported as `npz` to that disk.

**Protocol.** 4-fold database-grouped CV on HR (train 6 dbs / validate 2) × 12 cells × 3 seeds = 144 trainings; model selection per cell by validation return at the last 20 % of updates (no early peeking at dev). Final controllers: train on all 8 HR dbs, 5 seeds per cell (60); the deployed controller of a cell is the seed with the median CV validation return (pre-registered; no ensembling). **Operating points for dev:** (a) headline = cell with the highest CV validation EX_all among cells with mean calls ≤ calls(P0) and deferral ≤10 %; (b) the full Pareto set of all 12 cells for the curve. Dev live runs: 12 RL cells + baselines.

**Baselines.** (1) P0 fixed 7.5/8.5, T_max=3; (2) P0 threshold grid θ_acc∈{7.0,7.5,8.0,8.5,9.0,9.5} (Pareto envelope); (3) fixed-k revisions k∈{0,1,2,3} ignoring scores; (4) logistic-regression stopper on the same 27 features, label "an extra revise improves the candidate to EX=1", threshold swept; (5) greedy oracle on the tree (upper bound); (6) A2C variant; (7) E6-tier: always-tier1, always-tier2 (root regenerated by tier2), vs `A5` controller on the EX–cost plane.
**Outputs:** EX–calls and EX–tokens Pareto, deferral rate, action histograms, return curves, CV table, fidelity gap.

### E7 Abstention / escalation to human (dev, AmbiQT mix)
- Confidence sources: (a) final V3 Critic score; (b) RL controller's routing (operating point + sweep over c_h); (c) consistency: `n=5` T=0.7 samples (reuse `bon_vote` samples), confidence = share of the largest result cluster; (d) verbalised confidence 0–100 (extra field in P-GEN of a separate run: `full` + 1 call); (e) V5 learned.
- Outputs: risk–coverage curves + AURC, coverage at 10/20 % risk, RS(c) for c∈{0,1,5,10,N}; AmbiQT mix: AUROC of "ambiguous" detection from (1−confidence), RS(c) with ambiguous = should-abstain; secondary "interpretation coverage": share of gold valid SQLs whose result matches ≥1 of the 5 samples (exploratory).

### E8 Robustness (Spider-Syn / Realistic / DK + dev sanity)
- Systems: zs, dail, full_fixed, full (frozen BIRD config), r0, primary model. Metrics: EX; **linking recall** = share of gold-SQL tables/columns present in the M3 context (full schema fallback counted as recall 1, so also report the context size in tokens); drop relative to Spider dev. M2 pool = Spider train; M6 burn-in not rerun (corpus = Spider train gold, WS=prior).

### E9 Contamination checks
- **Probe-Q (question completion):** input `db_id` + first 50 % of question words; "Continue the question exactly as in the dataset"; T=0, max_tokens 128. Reproduced iff normalised Levenshtein similarity to the true remainder ≥0.9 (0.7 also reported). 
- **Probe-SQL (schema-free):** `db_id` + question + evidence, **no schema**; exact match of sqlglot-normalised SQL to gold, and table-name overlap.
- Sets: BIRD dev (1 534), 300 POOL57 train questions (seed), control = LiveSQLBench-Base-Lite (post-cut-off by construction, if usable). 8 models × 2 probes.
- **Trend consistency:** zs vs full_fixed and the ablation deltas on LiveSQLBench-Lite vs BIRD dev (sign agreement, CI overlap). Appendix only.

### E10 Error classes
- **Objective auto classes** (`analysis/error_classify.py`, decision list, applied to every failed dev prediction): 
  1. `exec_error` with sub-types {syntax, unknown_identifier (hallucinated table/column), ambiguous_column, other}; 2. `timeout`; 3. `empty_result` (pred empty, gold non-empty); 4. `shape_mismatch` (column count differs); 5. `column_permutation` (equal after reordering columns); 6. `row_superset` / 7. `row_subset` (same columns; pred ⊃ / ⊂ gold: filter too loose / too strict); 8. `value_mismatch` (same shape, different values); 9. `other`.
  Non-exclusive AST tags (sqlglot diff): table set differs, join condition differs, missing/extra predicate, wrong aggregate function, GROUP BY / ORDER BY / LIMIT differs, literal differs.
- **Exploratory "wrong criterion" (口径错):** an executable, plausible SQL with the right shape that implements a different metric/definition than the gold/evidence (human label only; reported as exploratory with weighted prevalence CI).
- **LLM-annotation protocol (300 stratified items; Amendment 2, SAP section 14, replaces the original two-author protocol):**
  - Frame: failed dev items of `zs` and `full` (r0). Strata = 6 collapsed auto classes {exec_error+timeout, empty_result, shape_mismatch(+permutation), row_superset/subset, value_mismatch, other} x 2 systems = 12 strata x 25 items (strata with <25 are taken whole and the shortfall is redistributed proportionally); seed 20261027. Post-stratification weights from the population stratum sizes for prevalence estimates. Unchanged.
  - Annotators: **two independent Claude Sonnet 5.5 runs (A1, A2)**, each blind to the system and to the auto class, each with its own random item order (`annotation_A1.csv`, `annotation_A2.csv`), no access to the other run; instructions verbatim from `docs/ANNOTATION_PROMPT.md`. A third Claude Sonnet 5.5 run adjudicates the disagreements with the two labels in blinded positions. A 30-item pilot (disjoint from the 300) calibrates the guide and the prompt once; both are then frozen; pilot runs are discarded. No human takes part in the labelling.
  - Material per item: question, evidence, schema, gold SQL, predicted SQL, first 10 rows of both results; item order randomised. Unchanged.
  - Labels (primary, one per item): L1 schema linking (wrong table/column), L2 join path, L3 filter/value, L4 aggregation/grouping/ordering, L5 **wrong criterion** (metric definition differs while linking is right), L6 gold suspect (gold SQL or answer is questionable), L7 ambiguous question (>=2 defensible readings), L8 other; secondary label optional; free-text rationale mandatory for L5-L7. Unchanged.
  - Agreement: Cohen's kappa (unweighted) on the primary label and on binary L5-vs-rest, with item-bootstrap 95 % CI (2 000 reps); Gwet's AC1 alongside (kappa paradox); percent agreement; confusion matrix; computed by `code/tools/annotation_kappa.py`. Kappa is reported before adjudication, whatever its value, and is described as agreement between two runs of one model, not as validity.
  - **Optional human spot-check (only reported if actually done):** the authors label about 50 randomly drawn items (seed 20261027, after the LLM labels are frozen, blind to them); agreement with the adjudicated LLM label is reported as a validity check. If it is not done, the paper does not mention a spot-check.
  - Wording: "LLM annotation (two independent Claude Sonnet 5.5 runs)"; never "author annotation", "human annotation", "expert review" or "expert annotation" for the LLM labels. The limitation sentence of `docs/ANNOTATION_GUIDE.md` is quoted in the paper.

### E11 Cost
All runs: calls, prompt/completion tokens, latency distribution (p50/p95), wall-clock; cost per correct answer (tokens per EX-correct); price only where verified in `models.yaml`. Aggregated from `usage.jsonl`; no extra runs.

---

## 7. Run order, ownership, worker machines

| Phase | Work | Where |
|---|---|---|
| P0 | splits, manifest, executor/eval tests (gold vs gold = 100 % EX on HT), BIRD scripts pinned | host → worker machines |
| P1 smoke | 50 HT questions per system (all 14 pipelines), measure calls/q, tokens/q, 429 rate, parse failures; fix budget table | a worker machine |
| P2 tuning | HT-300 sweeps (≤10 trials/system); E4 mini-stream; baseline matching on HR (≈44k calls); candidate sets for E5; E6 tree (R2/<port>) → training on <port> | <port>–<port> + <port> |
| P3 freeze | choose primary rules (critic, full, K1 baseline); write `dev_lock.json`, SAP hash | main thread |
| P4 run | **R1** (worker machines): E1 → E2 → E11 aggregation. **R2** (worker machines): E3 → E5(dev) → E6(dev live) → E7. **R3** (worker machines): E4 → E4c → E8 → E9 → E10 sampling. Max 2 heavy jobs per worker machine; API-bound jobs are light but SQLite VES timing jobs get a worker machine to themselves. Per-model concurrency cap 8 **per process**; at most 4 processes per model (R1+R2+R3+builder) |
| P5 | annotation (E10), analysis (S1), pull results to host `results/` after every shard |

Resumability: runs write `preds.jsonl` append-only with flush per item; restart skips finished `qid`s; worker machine rebuild loses nothing on the shared disk (cache + runs).

---

## 8. Call, token and wall-clock budget (upper bounds without cache hits; replace with **[measure]**)

Assumptions: average prompt 3.0k + completion 0.15k tokens per call; `full_fixed` 5 calls/q; native single-lane throughput 1.74 calls/s at concurrency 8 (qwen3.7-plus, measured). Scenarios: **A** one 8-lane in total (account-wide cap), **C** 3.13 calls/s (measured saturation at 24 concurrent), **B** 4 worker machines × 8 = 6.96 calls/s (nominal).

| Item | Calls | Tokens (B) | A (h) | C (h) | B (h) |
|---|---|---|---|---|---|
| Pre: tuning HT-300 (13 systems × ≤10 trials) | 156 000 | 0.49 | 24.9 | 13.8 | 6.2 |
| Pre: smoke-50 | 6 000 | 0.02 | 1.0 | 0.5 | 0.2 |
| Pre: baseline matching on HR (8 × 1 100 × 5) | 44 000 | 0.14 | 7.0 | 3.9 | 1.8 |
| Pre: E4 M6 parameters (HT mini-stream) | 30 000 | 0.09 | 4.8 | 2.7 | 1.2 |
| Pre: E5 critic variants on HR | 26 400 | 0.08 | 4.2 | 2.3 | 1.1 |
| Pre: E6 tree build (1 100 q × 75) | 82 500 | 0.26 | 13.2 | 7.3 | 3.3 |
| E1 (13 systems × 3 runs + 4 native) | 272 000 | 0.86 | 43.4 | 24.1 | 10.9 |
| E1 Alpha-SQL native (1 run, ~60 calls/q) | 92 040 | 0.29 | 14.7 | 8.2 | 3.7 |
| E2 (7 models × 8 calls × 1 534 + thinking probe) | 87 438 | 0.28 | 14.0 | 7.8 | 3.5 |
| E3 (6 ablations) | 47 861 | 0.15 | 7.6 | 4.2 | 1.9 |
| E4 main (20 configs × 5 600 q-runs × 4.5) | 504 000 | 1.59 | 80.5 | 44.7 | 20.1 |
| E4c dev stream | 38 350 | 0.12 | 6.1 | 3.4 | 1.5 |
| E5 dev | 12 400 | 0.04 | 2.0 | 1.1 | 0.5 |
| E6 dev live (18 runs) | 138 060 | 0.43 | 22.0 | 12.3 | 5.5 |
| E7 | 13 534 | 0.04 | 2.2 | 1.2 | 0.5 |
| E8 (≈3.1k q × 13.5) | 41 850 | 0.13 | 6.7 | 3.7 | 1.7 |
| E9 | 27 544 | 0.09 | 4.4 | 2.4 | 1.1 |
| **Total** | **≈1.62 M** | **≈5.1** | **≈259 h (10.8 d)** | **≈144 h (6.0 d)** | **≈65 h (2.7 d)** |

Cache sharing (M1 outputs, identical first generations, primary-model runs reused across E1/E2/E3) will reduce actual calls; the cache hit rate is a **[measure]**. E6 training: 144 CV + 60 final trainings × ~3 CPU-min ≈ 10 h on one 4-core worker machine, no API. E10 annotation: 330 items × ~3 min ≈ 17 person-hours per annotator. Main risk: token-per-minute rate limits (5.1 B tokens; scenario B ≈ 25 k tokens/s) — verify at smoke with a 10-minute burst at 32 in-flight; if 429 rate >2 %, fall back to scenario C and apply the pre-registered downscales (E4 perms 3 for p∈{10,20}; E6 dev live cells 12→6).

Budget script: `exp/budget.py` recomputes the table from measured calls/q and throughput (engineer writes it).

---

## 9. Pre-registered decisions and rules (summary for freezing)

| Decision | Rule | Evaluated on |
|---|---|---|
| Primary Critic variant | V4 only if AUROC(V4)−AUROC(V3) ≥0.02 with CI excluding 0 | HR candidates |
| `full` = RL or fixed | §4 rule | HR CV |
| K1 baseline | highest-EX compute-matched baseline | HR |
| RL headline cell | max CV EX_all s.t. calls ≤ P0 and deferral ≤10 % | HR CV |
| Weights/thresholds | patent defaults for headline | — |
| Dev runs | exactly once per (system, seed) | — |
| Stopping/exclusion | none; api_fail counted as EX=0 | — |

Open issues for X2/main thread are listed in `internal note X1`.
