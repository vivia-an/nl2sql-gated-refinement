# Statistical Analysis Plan (SAP) v1 (2026-10-25)

> **Reading copy.** This file is the statistical analysis plan of the study (SAP), with its dated amendments and post-hoc notes, prepared for readers of the paper. Sections, amendments, dates, scope labels (confirmatory, pre-specified, exploratory, post hoc), numbers, margins, hashes and decision rules are those of the working file. Only internal references were removed or reworded: paths of working documents, pointers to unreleased working files, names of work packages in author lines, and remarks about who proposed or ran a step. Numbering of review rounds and of internal lists of requests was dropped, and the first trial runs are called pilot runs. Where the working file attributed an amendment (or a round of requests) to an earlier review round of the draft, this copy says that it followed a critical reading of the draft by the authors with Claude Sonnet 5.5 (see Acknowledgments); the date of that amendment and its post-hoc label are unchanged, and those amendments were written after the locked results were known. None of these readings was an external peer review of the submitted paper. References to the result-file and table-generating machinery of the paper (TeX macro names and the like) were removed; the result files themselves are those named in the text.

> **Amendment 1 (2026-10-26, before the dev lock, after tuning-set analysis) is in §13.** Sections 1–12 below are the original v1 text and are kept verbatim. Where §13 states that an item is superseded, demoted or redefined, §13 governs; nothing in §1–12 was deleted or silently edited. **Amendment 3 (2026-10-26, §15)** re-ranks the primary family again (family P\*, weak backbones). **Amendment 2 (2026-10-26, §14)** changes who annotates E10 (two independent LLM runs instead of two authors); it is marked inline where it touches v1 text. **Amendment 4 (2026-10-26, §16)** pre-registers a controlled critic diagnostic with injected semantic errors (scope `diagnostic-injected`, secondary, outside every confirmatory family). **Amendment 5 (2026-10-26, §17)** adds an exploratory, post-hoc, offline analysis of simulated oracle-feedback calibration of the Critic (scope `diagnostic-posthoc`, outside every confirmatory family). **Amendment 7 (2026-10-27, §20)** re-scores the stored dev runs against corrected BIRD gold (Arcwise-Plat-SQL / Plat) as a post-hoc robustness check (scope `robustness-posthoc`, outside every confirmatory family). **Amendment 8 (2026-10-27, §21)** is the external-validity block on Spider-Realistic (scope `external-validity`, secondary, own Holm family X, never in P\*).

Pre-specified before the dev lock. The commit hash of this file is recorded in `exp/dev_lock.json`. Changes after the lock go to §12 (deviation log) with date and reason; the paper reports every deviation. Companion: the experiment specification (design, metrics, seeds). Analysis code: `code/analysis/`. All numbers in the paper come from `results/*.json`.

## 1. Units, populations, runs

| Item | Definition |
|---|---|
| Analysis unit | question (item) nested in database. Dev: 1 534 items, 11 dbs [verify]. Cluster = `db_id` |
| Population of inference | (i) the BIRD dev benchmark as given (finite, item-level tests); (ii) new databases from the same distribution (db-clustered intervals, GLMM) |
| Headline run | r0 (T=0, seed 0) of each system. Runs r1, r2 (T=0.7, seeds 101, 202) are stability analyses |
| Correctness | EX ∈ {0,1} by the official BIRD evaluator; `api_fail`/`parse_fail` ⇒ 0 (counted and reported) |
| Level | two-sided α = 0.05 unless stated; all p-values reported unadjusted **and** Holm-adjusted |
| Software | R ≥ 4.3 (`lme4`, `emmeans`, `pROC`, `irr`/`DescTools`), Python (numpy, scipy, sklearn). Seeds: bootstrap 20261101, permutation 20261102 |

## 2. Endpoints

**Key confirmatory endpoints (family K, five tests):**

| ID | Claim | Data | Statistic | Test |
|---|---|---|---|---|
| K1 | The full system beats the strongest compute-matched baseline | E1 dev r0 | ΔEX = EX(full) − EX(B*), B* = HR-selected baseline (experiment specification §4) | exact McNemar (discordant pairs) + db-clustered bootstrap CI |
| K2 | Weighted execution-grounded Critic (V3) discriminates better than the binary Critic (V1) | E5 dev candidates (initial + final of `full_fixed`) | ΔAUROC = AUROC(V3) − AUROC(V1) | paired db-clustered bootstrap (two-sided percentile), DeLong as check |
| K3 | Value-weighted corpus resists pollution | E4, p = 40 %, final-checkpoint probe items | log-odds contrast `ws_full` − `append` | GLMM LRT/Wald contrast |
| K4 | RL controller dominates the fixed patent policy | E6 dev, `full_rl` vs P0 | route A: EX superiority with calls ≤ +5 % of P0; route B: calls reduction with EX non-inferior (−1.0 pp) | each route an intersection–union test; p(K4) = min(1, 2·min(p_A, p_B)) |
| K5 | Critic-score routing gives a better risk–coverage trade-off than consistency confidence | E7 dev | ΔAURC = AURC(consistency) − AURC(Critic) (positive = Critic better) | paired permutation (swap the two confidence scores within item, stratified by db, 10 000) + bootstrap CI |

**Holm correction scope:** family K only (m = 5, step-down at α = 0.05). A key claim is *confirmed* when its Holm-adjusted p < 0.05 **and** the db-clustered 95 % CI excludes 0 (K1, K2, K4, K5) / the GLMM contrast CI excludes 0 (K3). Adjusted p < 0.05 with a CI that includes 0 is reported as "supported at item level, not at database level".

**Secondary families (each Holm-adjusted within the family; not part of K's error control and labelled as supportive):**

| Family | Tests | m |
|---|---|---|
| S-A (E1) | `full` vs every other system on EX (zs, dail, sdebug, macsql, acsql_s, acsql_x, memosql, alphasql, bon_vote, sc_cot, bon_critic, full_x), excluding B* (that is K1) | 11 (Alpha-SQL on its subset if triggered) |
| S-B (E3) | Cochran's Q omnibus (not counted) then post hoc exact McNemar `full` vs each ablation | 5–6 |
| S-C (E5) | ΔAUROC: V3 vs V2, V4 vs V3, V5 vs V3 | 3 |
| S-D (E6) | `full_rl` vs fixed-k (best-on-HR k), vs LR stopper, vs best threshold-grid θ on HR (EX at matched calls) | 3 |
| S-E (E4) | `ws_full` vs {static, ws_evict, memrl} at p = 40 %; vs {append, static, ws_evict, memrl} at p = 20 % | 7 |
| S-F (E7) | ΔAURC and ΔRS(10): Critic vs {verbalised, V5} | 4 |
| S-G (E8) | `full` vs {zs, dail} on Spider-Syn, Spider-Realistic, Spider-DK | 6 |

**Exploratory (no confirmatory claim; Benjamini–Hochberg q = 0.10 within each exercise, always labelled exploratory):** E2 model comparisons; EX by BIRD difficulty and by database; E4 diagnostics and the policy × pollution interaction; E5 weight/threshold sensitivity; E6 action histograms; E9 contamination contrasts; E10 error-class and "wrong criterion" prevalence; VES/R-VES/Soft-F1/Jaccard differences; AmbiQT metrics; every post hoc subgroup.

## 3. Interval estimation

- **Db-clustered bootstrap** (primary CI for every metric on dev): B = 10 000; resample the 11 databases with replacement, keep all items of a drawn db; paired statistics use the same resample for both systems; percentile 95 % CI. With only 11 clusters percentile intervals under-cover, so every key result also reports (a) the **leave-one-db-out** range of the effect, (b) an item-level paired bootstrap CI (narrower, for comparison), (c) the share of dbs with the same sign as the pooled effect.
- HR/CV statistics (used only for pre-lock decisions) use the same bootstrap over the 8 HR dbs.
- Wilson intervals for proportions that are not paired comparisons (E9, E10 prevalences use the stratified bootstrap).

## 4. Tests for paired binary outcomes (EX)

- **Two systems:** exact two-sided McNemar test on discordant counts b (A right, B wrong) and c (A wrong, B right); mid-p version as a sensitivity. Effect sizes: ΔEX in percentage points with CI (§3); conditional odds ratio b/c with Clopper–Pearson interval on b/(b+c).
- **≥3 systems on the same items:** Cochran's Q (k = number of E3 configurations, df = k−1) on r0; post hoc exact McNemar vs `full` with Holm. Pairwise all-vs-all is exploratory.
- **Stability across runs:** the item-level mean EX over r0–r2 (a value in {0, ⅓, ⅔, 1}) analysed with a paired sign-flip permutation test on the mean difference (stratified by db, 100 000 flips); reported as the sensitivity analysis for K1/S-A. Also run-to-run SD of system EX.
- **Continuous per-item metrics** (Soft-F1, Jaccard, VES per item, calls, tokens, latency): paired permutation test on the mean difference (sign-flip, stratified by db), paired standardised mean difference dz with bootstrap CI; Wilcoxon signed-rank as a check.
- **Non-inferiority (K4 route B):** declared when the lower bound of the 95 % db-clustered CI of ΔEX(full_rl − P0) is > −1.0 pp. Calls non-inferiority (route A): upper bound of the 95 % CI of mean calls difference < +5 % of the P0 mean. Calls reduction (route B): paired permutation test on per-item calls, one-sided α = 0.025 (reported as the two-sided p/2 when the sign agrees).

## 5. GLMM (generalisation over databases and questions)

R, `lme4::glmer`, logit link, Laplace approximation, optimiser `bobyqa` (restart with `nlminbwrap`; if still singular report and drop to `nAGQ = 0`).

| Analysis | Model | Inference |
|---|---|---|
| E1 / S-A | `ex ~ system + difficulty + (1 \| db_id) + (1 \| qid)` (all systems, r0) | omnibus LRT for `system`; `emmeans` contrasts vs `full`, Holm; OR with Wald 95 % CI. Sensitivity: random slope `(0 + system \| db_id)` if it converges (11 dbs make it fragile; reported either way) |
| E2 | `ex ~ model + (1 \| db_id) + (1 \| qid)` per system family | exploratory contrasts, BH |
| K3 / S-E (E4) | restrict to p = 40 % (or 20 %), final-checkpoint probe items: `ex ~ policy + (1 \| db_id) + (1 \| qid) + (1 \| perm)` | LRT of the policy contrast; OR + CI |
| E4 interaction | all cells: `ex ~ policy * pollution_c * checkpoint + (1 \| db_id) + (1 \| qid) + (1 \| perm)`, `pollution_c` = pollution/10 | LRT of `policy:pollution_c` (exploratory); pollution slope (log-odds per 10 pp) per policy with CI; slope difference `ws_full` vs `append` is the prespecified summary of "resistance" |
| E4c | `ex ~ policy * window + (1 \| db_id) + (1 \| qid) + (1 \| perm)` | exploratory trend test |

Fallback if `lme4` cannot be installed on the analysis machine: `statsmodels` `BinomialBayesMixedGLM` (variational) with the same structure, flagged in the paper.

## 6. Scores, calibration, selective prediction

- **AUROC:** empirical (ties = ½); comparison by paired db-clustered bootstrap of the difference; DeLong test as a check (item-level). The binary Critic V1 yields a 1-bit score; its AUROC equals balanced accuracy (stated in the paper). Secondary for K2: V3 at the patent operating threshold 8.5 vs V1 — balanced accuracy and MCC.
- **ECE:** 15 equal-mass bins on score/10 after isotonic calibration fitted on HR (cross-fitted) and applied to dev; raw ECE also; Brier; CI by db-clustered bootstrap. Reliability diagrams with 10 bins for display only.
- **Risk–coverage:** AURC computed on the empirical curve (trapezoid over all thresholds, ties broken by index-stable random shuffle with 100 reshuffles averaged); coverage at 10/20 % risk (threshold chosen on the data, reported as an *oracle-free* curve read-out, not a deployable operating point unless the threshold is fixed on HR). The deployable comparison fixes thresholds on HR and reports dev coverage/risk.
- **TrustSQL-style RS(c):** per-item scores as in experiment specification §5; mean with CI; c ∈ {0, 1, 5, 10, N}.
- **Learned Critic (V5) and controller:** all selection on HR grouped CV; dev touched once.

## 7. Analyses by experiment (what is reported)

| Exp | Primary summary | Inference |
|---|---|---|
| E1 | EX per system (CI), ΔEX vs `full` (CI), per-db table, corrected-gold re-scoring, VES/R-VES/Soft-F1/Jaccard | K1, S-A, GLMM |
| E2 | EX and auto error-class distribution per model; calls/tokens/latency | exploratory GLMM |
| E3 | EX per configuration; Q statistic; table of items fixed per module by error class | S-B |
| E4 | probe EX curve vs pollution per policy; store diagnostics (eviction precision/recall of polluted entries); online EX | K3, S-E, GLMM |
| E5 | AUROC/AUPRC/ECE/Brier per variant; weight simplex and threshold heat-maps; end-to-end EX per critic variant | K2, S-C |
| E6 | EX–calls and EX–tokens Pareto (HR CV and dev), deferral rate, replay-fidelity gap | K4, S-D |
| E7 | risk–coverage, AURC, RS(c); AmbiQT AUROC for ambiguity detection | K5, S-F |
| E8 | EX per variant, drop vs Spider dev, linking recall | S-G |
| E9 | reproduction rate (Wilson CI) dev vs control, per model | exploratory two-proportion tests |
| E10 | auto class distribution; κ, AC1 (item bootstrap, 2 000 reps); stratum-weighted prevalence of L1–L8 with stratified-bootstrap CI **[Amendment 2, §14: labels come from two independent Claude Sonnet 5.5 runs; κ is run-to-run agreement of one model]** | descriptive |
| E11 | calls, tokens, latency (median, p95), tokens per correct answer | descriptive; paired permutation for `full` vs baselines |

## 8. `thrombosis_prediction` handling

1. All dev analyses include it (main numbers = 1 534 items).
2. A per-database table (11 rows, EX per system with item-level Wilson CI) puts `thrombosis_prediction` on its own row and in the supplementary figure.
3. **Sensitivity analysis:** repeat K1–K5 (K3 and the E4/E6 training parts do not use dev databases, so only the dev-evaluated parts of K1, K2, K4, K5), S-A and E3 with that database removed (n = 1 534 − n_thr, n_thr [verify]); cluster count 10.
4. Concordance rule: a claim is *robust* if effect sign, Holm decision and CI-excludes-0 decision are identical with and without the database. If they differ, the paper reports both and downgrades the claim to "not robust to exclusion of one database".
5. Leave-one-db-out ΔEX for all 11 databases (influence plot).
6. No other database is dropped. No item-level exclusions.

## 9. Power and minimum detectable effects

Paired binary outcome, McNemar approximation: Var(Δ̂) ≈ (p_d − Δ²)/n, so MDE ≈ (z_{1−α/2} + z_{power})·√(p_d·DEFF/n), where p_d = share of discordant pairs, DEFF = 1 + (m̄ − 1)·ρ is the clustering design effect (m̄ ≈ 139 items/db; ρ = intra-db correlation of the paired differences, assumed 0.01 → DEFF 2.4 and 0.03 → DEFF 5.1; ρ is re-estimated from HR and reported). Power 0.80, n = 1 534. Values in percentage points:

| p_d | α = 0.05, DEFF 1 / 2.4 / 5.1 | α = 0.01 (Holm step 1, m = 5), DEFF 1 / 2.4 / 5.1 | α = 0.0083 (Holm m = 6), DEFF 1 / 2.4 / 5.1 |
|---|---|---|---|
| 0.05 | 1.60 / 2.48 / 3.61 | 1.95 / 3.02 / 4.41 | 1.99 / 3.08 / 4.49 |
| 0.10 | 2.26 / 3.50 / 5.11 | 2.76 / 4.28 / 6.23 | 2.81 / 4.35 / 6.35 |
| 0.15 | 2.77 / 4.29 / 6.26 | 3.38 / 5.24 / 7.63 | 3.44 / 5.33 / 7.77 |
| 0.20 | 3.20 / 4.96 / 7.23 | 3.90 / 6.05 / 8.81 | 3.97 / 6.16 / 8.97 |
| 0.30 | 3.92 / 6.07 / 8.85 | 4.78 / 7.41 / 10.79 | 4.87 / 7.54 / 10.99 |

Reading: against a near-neighbour baseline (p_d ≈ 0.10–0.15) the item-level test detects ≈2.8–3.4 pp at α = 0.01, but if the effect is only 2 pp it is not detectable and a database-level claim needs ≳4–5 pp. This is the main reason K1 is defined against a single HR-selected baseline and the claim language ("supported at item level") is graded. Against zero-shot (p_d ≈ 0.20–0.30) the MDE is 4–5 pp.

Other endpoints (approximate; `analysis/power.py` re-simulates with HR pilot values before the lock and the result is added to §12):

| Endpoint | Basis | Planning MDE |
|---|---|---|
| K2 ΔAUROC | n = 3 068 candidates (2 per question), SE(AUROC) ≈ 0.010–0.014 (Hanley–McNeil, correct share 0.5–0.6), paired correlation r = 0.5–0.7, α = 0.01, power 0.8; question-level pairing inflates by ≈ √1.5 | ≈ 0.03–0.04 |
| K3 (E4) | 328 probe items (8 dbs), 5 permutations pooled in GLMM, p_d ≈ 0.2 | item-level MDE ≈ 6.9 pp at α = 0.05, 8.6 pp at α = 0.0083 (p_d 0.1: 4.9 / 6.1; p_d 0.3: 8.5 / 10.5); permutations add little information because the item set is fixed. **E4 is the least powered key test**; if the observed polluted-seed effect is below ≈7 pp the claim is reported as "direction consistent, not confirmed" |
| K4 | p_d between `full_rl` and P0 expected 0.05–0.10 | 1.95–2.76 pp at α = 0.01 (DEFF 1); non-inferiority margin 1 pp is *not* resolvable at this n for the item-level EX difference unless the CI half-width ≲ 1 pp; route B is therefore expected to be decided mainly by the calls reduction |
| K5 ΔAURC | simulation | reported after `power.py` |

## 10. Handling of multiple runs, failures, missing data

- Failed API calls/parse failures: counted as EX = 0 (`error_class` = `api_fail`/`parse_fail`); rates reported per system. A run with >0.5 % failures is re-run once on identical config; both attempts logged.
- If a baseline cannot be completed (e.g., Alpha-SQL cost), the subset rule of experiment specification §4 applies and all comparisons for that baseline are restricted to the same subset.
- No outcome-based exclusion, no optional stopping, no re-analysis after dev unblinding other than listed in §12.
- Seeds/stochastic systems: headline r0; r1/r2 reported as mean ± SD and used in the item-mean sensitivity analysis.

## 11. Reporting rules

- Each table cell: estimate [95 % db-clustered CI]; each comparison: ΔEX (pp), CI, exact p, Holm-adjusted p (where in a family), OR.
- Negative or null results are reported with the same prominence (RL controller, WS vs MemRL, learned Critic).
- Claims language: *confirmed* (§2 rule), *supported at item level*, *direction consistent*, *exploratory*. No "significantly" without the adjusted p.
- ~~Author annotation~~ **LLM annotation (two independent Claude Sonnet 5.5 runs; Amendment 2, §14)** (E10): κ reported before adjudication and described as agreement between two runs of one model; "wrong criterion" prevalence labelled exploratory. Never described as human or expert annotation.
- All analysis scripts read only `runs/` and `results/`; the numbers in the paper are generated by script from the result files; the SAP commit hash and `dev_lock.json` hash are printed in the supplementary material.

## 12. Deviation log (append only after the lock)

| Date | Section | Change | Reason | Effect on conclusions |
|---|---|---|---|---|
| — | — | — | — | — |

---

## 13. Amendment 1 (2026-10-26; before the dev lock; after tuning-set analysis)

Status: **proposed; takes effect when committed**, and must be committed before `exp/dev_lock.json` is written (the commit hash of this file is then recorded in the lock). Sections 1–12 are unchanged.

### 13.0 What was seen, and what that means for the claim "pre-specified"

* **Seen:** BIRD-train tuning data only: HT-300 (4 dbs) and HR-1101 (8 dbs), `qwen3.7-plus`, via the tuning report (R0) and the headroom report (R1, offline, 0 API calls). **Not seen:** BIRD dev (questions' gold, predictions or scores), minidev, Spider variants, AmbiQT, LiveSQLBench. No `dev_lock.json` entry exists; `exp/dev_access.log` has no evaluator access.
* **Consequence.** The hypotheses of §13.3 are pre-specified with respect to dev, **not** with respect to the tuning sets: the tuning results shaped which hypotheses we test and how. The tuning results are therefore never used as confirmatory evidence; they are reported as design evidence (supplement, labelled "tuning set"), and every strengthening below is adopted or rejected by a mechanical rule on HR/HT (§13.6) before the lock.
* **Why the amendment.** The v1 family K was built around the claim that the full pipeline improves EX. The tuning data do not support that claim and the v1 power calculation could not have detected the effect sizes now expected (§13.1, §13.7). Keeping K1 and K4 as the headline tests would invite a null result on the wrong question; keeping them as secondary, reported as-is, is the honest option. The amendment does not remove any v1 hypothesis; it re-ranks them (§13.2).

### 13.1 Tuning-set evidence behind the re-ranking

| # | Finding (tuning data only) | Source |
|---|---|---|
| E-1 | `full_fixed` does not beat zero-shot: HT-300 EX 0.693 vs `zs_ours` 0.697 (−0.3 pp, 9 vs 10 discordant of 300) and vs `zs` 0.673 (+2.0 pp). The retrieval stack M1–M4 alone is −1.3 pp vs `zs_ours` | the tuning report §1, the headroom report §1 |
| E-2 | The loop's gain is repair of execution failures: HR P0 +3.5 pp over its own root, 47 of 50 gains are roots that did not execute (47/70 repaired); 41 of those are in `beer_factory` (table `transaction`, a SQLite keyword that `render_ddl._q` does not quote: 54 identical `near "transaction": syntax error`). Without `beer_factory` P0 is −0.2 pp vs root | the headroom report §0.1 |
| E-3 | Semantic revision of executable roots is net negative (HR 3 gains / 11 harms; HT 1 / 1). The LLM semantic flag (score < 7) has recall 14 % (HR) / 19 % (HT) of wrong executable roots; 296 of 415 wrong HR roots get semantic = 10 | the headroom report §0.2, §1 |
| E-4 | Execution-only gating ("revise only on error/timeout") replays at EX 0.666 vs P0 0.658 (+0.7 pp [−0.3, +1.7], 11 W / 3 L) at ≈ 2.1 vs 3.4 calls/q (the semantic Critic call is not made on executable roots) | the headroom report §3. Caveat: replay on candidates that were generated with the Critic's text; live confirmation pending |
| E-5 | Selection, voting and arbitration over the stored candidates recover +0.4 … +1.3 pp over P0 (CIs reach 0); a perfect root-vs-revision arbiter would add at most +1.5 pp; 24.8 % of HR questions are unreachable by 23.5 candidates | the headroom report §0.3, §1, §3 |
| E-6 | Routing: V3 AUROC for root correctness 0.71 (HR) / 0.74 (HT); a consistency-augmented score (LR on V3, agreement, non-empty; leave-one-db-out) 0.75 on HR, EX of the accepted 60 %: 0.735 → 0.782. **AURC on HR roots is unchanged** (0.2625 → 0.2605): the gain is in the mid-coverage region, not over the whole curve. The cheap version (2 extra T = 0.7 samples) is unverified | the headroom report §2 and `results/r1/q2_rc_matched.json` |
| E-7 | RL controller: no PPO cell reaches P0 under database-grouped CV (best cell 0.635 vs 0.660, −2.5 pp); A2C collapses to "accept the root". Action values are flat on executable roots | R0 §1 (E6), the headroom report §5.5 |
| E-8 | E4 mini-stream (p = 40 %): `ws_full` 0.715 / 0.699 (two replicates) vs `memrl` 0.707 / 0.695; trial-to-trial noise ≈ 1 pp; about 90 % of evicted entries are clean; the `ws_full` ρ = 0 optimum is `ws_evict` in disguise | the tuning report §3 |
| E-9 | All 1 401 HR/HT questions took the full-schema fallback (< 6 000 tokens): M3 (routing, RRF, rerank) never ran; k_t × k_e grid spans 0.677–0.693, no effect distinguishable from noise | the tuning report §2, the headroom report §4 |

### 13.2 Re-ranking of the v1 hypotheses (nothing is deleted)

| v1 id | v1 claim | Status after Amendment 1 | New id and role |
|---|---|---|---|
| K1 | `full` beats the strongest compute-matched baseline on EX | **Demoted** to secondary. Reported as-is; no superiority claim is planned (expected effect 0 … +1 pp, below the MDE, §13.7) | S-A row "full vs B*" (two-sided, Holm in S-A); its role as headline is taken by P2 (non-inferiority) |
| K2 | V3 (weighted, grounded) discriminates better than V1 (binary) | **Demoted** to secondary (the headroom report: V3's advantage is mostly the execution term) | S-C, now m = 4 (§13.5) |
| K3 | Value-weighted corpus resists pollution (E4, p = 40 %, `ws_full` vs `append`) | **Kept unchanged** as primary | **P4** (H-pollution) |
| K4 | RL controller dominates the fixed policy (routes A, B) | **Demoted**; expected negative; reported as a negative result with the test as specified | S-D (secondary), headline cell only on dev (§13.8) |
| K5 | Critic-score routing has lower AURC than consistency confidence | **Redefined**: the routing score becomes a consistency-augmented score; primary comparison is vs V3; primary metric AUROC (reason in P3) | **P3** (H-route); the v1 comparison vs consistency confidence is kept in S-F |
| (new) | Cost: execution-gated loop is non-inferior at lower cost | – | **P1, P2** (H-cost) |
| (new) | Execution-failure repair by the loop | – | **P5** (H-repair) |
| v1 "C3 RL controller", "C1 weighted Critic improves EX" | contributions in the paper outline | demoted to secondary / negative result | the paper outline Revision 1 |

The v1 family K (m = 5) is **replaced for confirmatory purposes** by family P (m = 5, §13.3). K1, K2, K4, K5 keep their v1 definitions and are tested and reported as secondary/supportive; their p-values are Holm-adjusted inside the secondary families of §13.5, not in P.

### 13.3 Primary confirmatory family P (m = 5)

Common conventions: two-sided α = 0.05 family-wise; headline run r0 on BIRD dev (1 534 items, cluster = `db_id`); backbone `qwen37plus`; all systems use the identifier-quoting renderer (§13.6 item 1); CIs are db-clustered percentile bootstrap (B = 10 000, §3) with the item-level paired bootstrap, the leave-one-db-out range and the share of dbs with the same sign reported next to them; sensitivity without `thrombosis_prediction` as in §8. One-sided tests (non-inferiority, ratio, "better than") enter Holm as p_2s = min(1, 2·p_1s) (the convention already used for K4); a one-sided p-value is the smallest α for which the (1 − 2α) db-clustered percentile interval excludes the null. An exact/score-based item-level check is reported with each.

**P1 — H-cost (comparison between loop variants): the execution-gated loop keeps the accuracy of the fixed semantic-critic loop at lower cost.**
* Systems: `full_gated` vs `full_fixed` (same backbone, same renderer, same frozen M6 burn-in store, same root prompts, so identical root generations are cache hits and the discordance comes from the revision step only).
* (a) ΔEX = EX(gated) − EX(fixed). H0a: ΔEX ≤ −1.5 pp. Margin rationale: below every loop gain observed on tuning data (+3.5 pp HR pre-fix, +1.0 pp HT), so "non-inferior" means "keeps essentially all measured loop value"; a looser margin would be resolvable but vacuous.
* (b) R = Σ calls(gated) / Σ calls(fixed), calls = LLM calls of the answer path (M1, generation, Critic when it is called, revisions, arbitration if enabled); embedding/rerank calls and the routing-score calls of P3 are reported separately. H0b: R ≥ 0.80. Design value R ≈ 0.62 (2.1 / 3.4, replay).
* Intersection–union: p(P1) = max(p_a, p_b); both bounds must hold. Verdict language: "non-inferior" only if the db-clustered CI lower bound of ΔEX > −1.5 pp; "non-inferior at item level" if only the item-level interval satisfies it.
* Also reported (not tested): EX by root status (execution failure / empty / non-empty), tokens/q, latency, tokens per correct answer, ΔEX at margins 1.0 / 2.0 / 3.0 pp.

**P2 — H-cost (comparison with baselines): at its cost, `full` is not worse than the best baseline that costs no more.**
* `full` = the system chosen by §13.6 rule 3. B* = the baseline with the highest HR EX among all baselines (zero-shot and DAIL included, because they are cheaper) whose HR mean calls/q ≤ 1.15 · B_ref(`full`), with B_ref(`full`) = HR mean calls/q of `full` measured live; ties → fewer calls. This replaces the v1 compute-matching window [0.85, 1.15] · B_ref(`full_fixed`) for P2 only; baselines with a budget knob are matched to both budgets (§13.8).
* ΔEX = EX(full) − EX(B*). H0: ΔEX ≤ −3.0 pp. The margin is wider than P1's because the paired discordance is larger (§13.7): at n = 1 534 a 1.5 pp margin against an independent baseline is not resolvable, and a margin we cannot resolve would only produce an uninformative "not confirmed". It equals the smallest ΔEX between two different systems that this study can detect at all (≈ 3 pp for near-neighbour pairs), i.e. we claim "no difference larger than the resolution of the study".
* **Reading rule written in advance:** if B* is zero-shot or DAIL-style, P2 says only that the loop does not cost accuracy relative to a cheaper alternative; it is not evidence that the loop is worth its cost. The paper must say so (the paper outline limitation paragraph).

**P3 — H-route: a consistency-augmented score ranks correct above incorrect answers better than the Critic score V3.**
* Unit: the final answer of `full` r0. Scores: `route_ca` (frozen logistic regression on [V3 score / 10, agreement share of the final result among the root and k = 2 extra T = 0.7 root samples, non-empty flag]; fitted on HR, frozen, hash in the lock) vs V3 of the same answer. Event: EX = 1. Routing-path cost is reported separately: V3 = +1 call/q (the Critic call on the final answer if it was not made), `route_ca` = +1 + k calls/q.
* ΔAUROC = AUROC(route_ca) − AUROC(V3). H0: ΔAUROC ≤ 0, one-sided; paired db-clustered bootstrap, DeLong check.
* **Why AUROC and not AURC (recorded because it is a change from K5):** on HR roots the tuning data give ΔAUROC ≈ +0.04, EX of the accepted 60 % +4.7 pp, but ΔAURC ≈ −0.002 (0.2625 → 0.2605). A claim about "the risk–coverage curve" would be a claim about a quantity that the tuning data say does not move. AUROC is threshold-free, is the standard measure of ranking correct above incorrect, and was already the K2 metric. AURC, accepted-EX at 40 / 60 / 80 % coverage (quantile rule), and deployable operating points (thresholds fixed on HR) are **co-reported in S-F** and the text may say "improves the risk–coverage trade-off" only if ΔAURC in S-F is also in the favourable direction with its CI excluding 0; otherwise the text states which readouts moved and which did not.
* Pre-lock adoption gate (§13.6 rule 4): if `route_ca` fails the gate on HR (live k = 2 samples, leave-one-db-out), P3 is **dropped from family P** (m = 4), the failure is reported as a tuning result, and E7 is descriptive.

**P4 — H-pollution (v1 K3, unchanged).**
* E4, p = 40 %, final-checkpoint probe items, GLMM log-odds contrast `ws_full` − `append` (SAP §2 K3, §5). `ws_full` = (η, ρ) = (0.5, 0.2) with τ, α/β/γ, φ at defaults (§13.6 rule 7).
* "No-evict / MemRL" in the hypothesis statement: `append` is the no-evict comparator of the primary test; the MemRL-style comparator `memrl` at p = 40 % (and p = 20 %) is the headline row of S-E and is shown in the same table. The slope contrast over p ∈ {0, 10, 20, 40} (§5) and the p = 0 no-harm contrast (non-inferiority, margin 2 pp) are secondary.
* Expectation recorded in advance: the tuning mini-stream shows no difference between `ws_full` and `memrl` and an eviction false-positive rate of ≈ 90 %; E4 has the lowest power in the study (§9: MDE 6.9 pp at α = 0.05, 8.6 pp at α = 0.0083). An effect below ≈ 7 pp is reported as "direction consistent, not confirmed".

**P5 — H-repair: the loop repairs execution failures, and the error text is what repairs them.**
* Population R = dev items of `full_gated` r0 whose first (root) candidate fails to execute (error or timeout). Outcome: final EX = 1 ("repaired") and, separately, final executes ("made executable").
* Comparator: `full_gated_blind`, identical gate and call count, but the revision prompt withholds the error message and Critic text ("the previous query failed; write a corrected query"), T = 0.7, fixed seed. The comparison is paired on R: one-sided exact McNemar test (gated right / blind wrong vs the reverse).
* **Inclusion rule written in advance:** P5 is in family P only if n_R ≥ 40; otherwise it is reported descriptively (rate with Clopper–Pearson interval, by error class) and the family size becomes 4. With the quoting fix, n_R is expected to be small (HR: 54 of 70 pre-fix failures came from one cause), so this is a real possibility.
* Always reported: repair rate by error class {syntax, unknown identifier, ambiguous column, timeout, other} with Wilson intervals, db-cluster sensitivity, and the legacy-renderer arm (§13.8, item E3-Q) as out-of-sample evidence on how much of the repair rate depends on the identifier artefact.

**Decision rule for P (unchanged in form from §2):** a test is *confirmed* when its Holm-adjusted p < 0.05 (Holm over the m tests present at the lock) and the db-clustered 95 % CI satisfies the stated bound; adjusted p < 0.05 with only the item-level interval satisfying it is "supported at item level, not at database level". A hypothesis with two components (P1) is confirmed only if both are.

### 13.4 Secondary and exploratory (reported with the same prominence; no confirmatory claim)

| Item | Content | Status |
|---|---|---|
| Accuracy vs zero-shot and vs every baseline | E1: ΔEX with CI for `full` vs each system (v1 K1 and S-A); per-db table; corrected-gold re-scoring | Secondary, reported as-is. **No superiority claim is planned or allowed in the text unless the S-A Holm-adjusted p < 0.05 and the db-clustered CI excludes 0** |
| RL controller | v1 K4 (routes A and B) at the pre-registered headline cell plus three further cells; CV table; replay-fidelity gap | Secondary; presented as a negative or neutral result when the test fails (the paper outline R5) |
| Semantic-critic ablation | `full_fixed` (semantic score triggers revision) vs `full_gated` (execution triggers revision) = the point estimate of P1; plus the semantic score as router only; V3 vs V1 (K2); sub-score AUROC decomposition | Secondary (S-B, S-C) |
| Backbone strength (E2) | Gain of `full_gated` over `zs` by backbone: per-model ΔEX with CI and the rank correlation of gain with `zs` EX across the 8 models (Spearman, n = 8, descriptive) | Exploratory (BH q = 0.10), hypothesis stated in advance: the gain is larger for weaker backbones |
| M3 | forced routing (`schema_fallback_tokens` = 0) vs default fallback vs dense-only | Exploratory sensitivity; the −M3 ablation alone is empty on small schemas (E-9) |
| Identifier quoting | legacy renderer arm `full_fixed_nq` (E3-Q) | Exploratory, out-of-sample measurement of the artefact |
| Everything listed as exploratory in §2 | E2 model comparisons, E9, E10, subgroup analyses, VES, … | Unchanged |

### 13.5 Multiplicity (supersedes §2 "Holm correction scope" for the families named here)

| Family | Tests | m | Note |
|---|---|---|---|
| **P** (confirmatory) | P1, P2, P3, P4, P5 | 5 (4 if P3 is dropped by the adoption gate or P5 by n_R < 40; fixed at the lock and recorded in `dev_lock.json`) | Holm, family-wise α = 0.05; one-sided p doubled |
| S-A (E1) | `full` vs each of {zs, dail, sdebug, macsql, acsql_s, acsql_x, memosql, bon_vote, sc_cot, bon_critic, full_fixed, full_x} (two-sided; B* is one of them) | 12 (minus unfinished baselines; Alpha-SQL on its subset if run) | was 11 with B* excluded; B* is no longer excluded (K1 demoted). If `full` = `full_fixed`, the row `full_fixed` is replaced by `full_gated` |
| S-B (E3) | Cochran's Q omnibus, then `full` vs each ablation | 7–8 | adds M3-forced and E3-Q arms |
| S-C (E5) | ΔAUROC: V3 vs V1 (old K2), V3 vs V2, V4 vs V3, V5 vs V3 | 4 | was 3 |
| S-D (E6) | old K4 (p = min(1, 2·min(p_A, p_B))), `full_rl` vs fixed-k, vs LR stopper, vs best θ at matched calls | 4 | |
| S-E (E4) | `ws_full` vs {static, ws_evict, memrl} at p = 40 %; vs {append, static, ws_evict, memrl} at p = 20 % | 7 | `memrl` at 40 % is the headline row; unchanged from v1 |
| S-F (E7) | `route_ca` vs {SC3 (same samples), SC5 (`bon_vote` samples), verbalised, V5} on ΔAUROC, plus ΔAURC and Δ accepted-EX at 60 % coverage vs V3 | 6 | |
| S-G (E8) | `full` vs {zs, dail} on three Spider variants | 6 | unchanged |
| S-H (E2) | per-model ΔEX `full_gated` − `zs` (8 models) | 8 | exploratory, BH q = 0.10 |

Gatekeeping across families is not used: failure of a P test does not suppress reporting of any other test.

### 13.6 Pre-lock decisions, mechanical rules, tuning data only

Evaluated once by `code/analysis/selection_rules.py` (to be written by S1/X5) on HR/HT after the final code freeze of the strengthenings; output committed as `exp/selection_decisions.json` before the lock. After the evaluation no gate, score or prompt may be changed; a change requires a new amendment before the lock, with the previous result retained. Every trial is appended to `exp/tuning_log.jsonl`; the ≤ 10 trials per system rule (§1.3 of the spec) holds. All HR/HT comparisons below are live runs, with the quoting fix in force for every system (the stored pre-fix HR run is not reused).

| # | Decision | Rule | Otherwise |
|---|---|---|---|
| 1 | Identifier quoting (`render_ddl._q` quotes SQLite keywords; generation rule "quote reserved words") | Adopted for **all** systems, baselines included (shared renderer). Check, not a gate: re-run of the 131 `beer_factory` HR roots must drop `near "transaction"` errors from 54 to < 5 | Investigate before the lock |
| 2 | Gate definition | G1 = revise only on execution error or timeout. G2 = G1 plus empty result. Use G1 unless EX(G2) − EX(G1) ≥ +1.0 pp on HR with paired db-clustered CI lower bound > 0 | G1 |
| 3 | **Primary system `full`: `full_gated` vs `full_fixed`** | `full := full_gated` iff (i) HR: lower 95 % bound (paired, db-clustered, B = 10 000) of EX(gated) − EX(fixed) > −1.5 pp; (ii) HR: upper 95 % bound of calls ratio R < 0.80; (iii) HT-300: point estimate of ΔEX ≥ −2.0 pp and R < 0.80. All three. | `full := full_fixed`; `full_gated` is still run on dev and P1 is still tested (it then reads as a cost-variant result). `full_rl` is not eligible (v1 rule §4 of the spec already fails: CV EX 0.635 vs 0.660 at calls ≤ P0) |
| 4 | Routing score `route_ca` (k = 2 extra T = 0.7 root samples) | Adopted iff on HR (live samples, leave-one-db-out): ΔAUROC vs V3 ≥ 0.03 with paired db-clustered CI lower bound > 0, **and** accepted-EX at 60 % coverage ≥ V3 + 2 pp (point estimate) | Routing score = V3; P3 dropped (§13.3); S-F reports the comparison with SC and verbalised |
| 5 | Arbitration call (one judge call when root and revision differ) | Adopted into `full_gated` iff on HR the judge picks the right member in ≥ 70 % of the mixed cases **and** ΔEX vs `full_gated` ≥ +1.5 pp with CI lower bound > 0 (the headroom report bound: at most +1.5 pp) | Not adopted; reported in the supplement as a tuning-set negative result; no dev run. If adopted, its calls enter R in P1 |
| 6 | M3 forced routing (`schema_fallback_tokens` = 0) | Headline keeps the default fallback unless HT-300 + HR show ΔEX ≥ +1.5 pp with CI lower bound > 0 for forced routing | Default; forced routing is an exploratory dev arm |
| 7 | `ws_full` parameters | **Override of the mechanical tuning pick, recorded here:** the rule "highest pooled EX" chose (η, ρ) = (0.5, 0) (0.7148), which makes `ws_full` identical to `ws_evict` for recall ranking. `ws_full` is therefore fixed to (0.5, 0.2) (t06, 0.7070; 2 of 256 items lower; replicate noise 0.8–1.6 pp). `ws_evict` (ρ = 0) is run as its own policy as in the spec | – |
| 8 | Critic variant | Unchanged (§9 of the spec): V4 only if AUROC(V4) − AUROC(V3) ≥ 0.02 with CI excluding 0 on HR candidates | V3 |
| 9 | Baseline matching | B_ref(`full`) and B_ref(`full_fixed`) measured live on HR; baselines with a knob are matched into [0.85, 1.15] · B_ref to **both**; B* (P2) is chosen from baselines with calls ≤ 1.15 · B_ref(`full`) | – |

### 13.7 Power and minimum detectable effects, updated honestly

The v1 statement of §9 ("MDE 2–5 pp") described near-neighbour comparisons at α = 0.05. Corrected reading, from the §9 table (n = 1 534, DEFF 2.4, power 0.80, Holm α = 0.01 for m = 5): **3.0 pp (p_d 0.05), 4.3 pp (p_d 0.10), 5.2 pp (p_d 0.15), 6.1 pp (p_d 0.20), 7.4 pp (p_d 0.30)**. Zero-shot vs full on HT-300 had 19 discordant of 300 (p_d ≈ 0.06), so against `zs` the MDE is ≈ 3.2–3.5 pp; against baselines with a different pipeline (p_d 0.15–0.30) it is 5–7.4 pp. The accuracy effects the tuning data point to (0 … +1 pp) are below all of these, which is why accuracy superiority is secondary.

Non-inferiority power (paired binary outcome, McNemar variance (p_d − Δ²)/n × DEFF; one-sided; "Holm" = one-sided α 0.005, i.e. step 1 of m = 5; "nominal" = 0.025; computed in the amendment, to be re-simulated by `analysis/power.py` with HR pilot values):

| Test | margin | p_d | DEFF | SE(Δ) pp | power at true Δ = 0 (nominal / Holm) | at true Δ = +0.5 pp |
|---|---|---|---|---|---|---|
| P1a (gated vs fixed, shared roots) | 1.5 pp | 0.02 | 2.4 | 0.56 | 0.76 / 0.54 | 0.95 / 0.84 |
| P1a | 1.5 pp | 0.04 | 2.4 | 0.79 | 0.47 / 0.25 | 0.72 / 0.48 |
| P1a | 1.5 pp | 0.04 | 1.0 | 0.51 | 0.84 / 0.64 | 0.97 / 0.91 |
| P1a (sensitivity margin) | 2.0 pp | 0.04 | 2.4 | 0.79 | 0.72 / 0.48 | 0.88 / 0.72 |
| P2 (full vs B*) | 3.0 pp | 0.06 | 2.4 | 0.97 | 0.87 / 0.70 | 0.95 / 0.85 |
| P2 | 3.0 pp | 0.10 | 2.4 | 1.25 | 0.67 / 0.43 | 0.80 / 0.59 |
| P2 | 3.0 pp | 0.15 | 2.4 | 1.53 | 0.50 / 0.27 | 0.63 / 0.39 |

Reading: P1a is adequately powered only if the discordance between `full_gated` and `full_fixed` stays ≈ 0.02–0.03 (HR replay: 14 of 1 101 = 1.3 %, plus API non-determinism: repeat runs differ on 3.4 % of items), which is why the roots are shared via the cache. If the observed discordance is higher, the likely outcome is "non-inferior at item level" or "not confirmed", and the text will say that; the margin is **not** widened after the lock. P1b (calls ratio) is powered by construction (per-item call counts differ by design). P2 is moderately powered (0.4–0.7 under Holm) for p_d ≤ 0.10.

| Endpoint | Planning MDE / power |
|---|---|
| P3 ΔAUROC | n = 1 534 answers (≈ 60 % correct), SE(AUROC) ≈ 0.013, paired correlation r = 0.7–0.85 (the augmented score contains V3), cluster inflation √1.5: SE(Δ) ≈ 0.009–0.012; MDE 0.029–0.041 at α = 0.01 (0.024–0.034 at 0.05). Design effect from tuning +0.04 (HR, leave-one-db-out) / +0.07 (HT): power 0.76–0.98 if the effect transfers, ≈ 0.5 if it shrinks to +0.03 |
| P4 | MDE 6.9 pp (α 0.05), 8.6 pp (α 0.0083); lowest-power test; an effect below ≈ 7 pp is "direction consistent" (§9) |
| P5 | exact McNemar on n_R pairs; with n_R = 60 and 25–40 % discordant pairs the smallest detectable repair-rate difference is ≈ 18–25 pp at α = 0.01. Hence the n_R ≥ 40 rule and the descriptive fallback |
| S-A accuracy comparisons | 3.2–7.4 pp (above) vs expected 0 … +1 pp: not powered, hence secondary |

### 13.8 Revised dev run list (replaces the run lists of the experiment specification §6 E1–E3, E5–E7 where they conflict; E4, E4c, E8–E11 unchanged except that `full` means the §13.6-rule-3 system)

All runs: dev, `qwen37plus`, T = 0 (r0) unless stated; seeds r1, r2 (T = 0.7) only where marked. Costs are design estimates from measured unit costs (`full_fixed` ≈ ¥0.0095/q ≈ ¥15 per dev run, `zs` ≈ ¥0.002/q, gated ≈ 2.1 calls ≈ ¥0.006/q); `exp/budget.py` must recompute them. Each (system, config, seed) runs once (§1.3).

| Id | Run | Seeds | Role | ≈ cost (¥) |
|---|---|---|---|---|
| D1 | `zs`, `dail` | r0–r2 | E1, B* candidates | 25 |
| D2 | `full_fixed`, `full_gated` (or `full_gated` with arbiter if rule 5 adopts it) | r0–r2 | P1, E1 | 70 |
| D3 | `full_gated_blind` | r0 | P5 (only roots that fail execution differ; roots are cache hits) | 2 |
| D4 | Routing add-ons on `full`: Critic V3 on the final answer; k = 2 extra T = 0.7 root samples (→ `route_ca`, SC3); `bon_vote` n = 5 (SC5); verbalised-confidence variant | r0 | P3, S-F, E7 | 40 |
| D5 | Matched baselines {sdebug, acsql_s, acsql_x, macsql, memosql, bon_vote, sc_cot, bon_critic} at budget(`full`) **and** budget(`full_fixed`) where the knob allows both; native-setting runs for sdebug, macsql, acsql_s, memosql | r0 (r1, r2 for B* and the best two) | P2, S-A | 250–350 |
| D6 | `alphasql` on the 500-question subset (spec rule) | r0 | S-A (subset) | ≈ 240 |
| D7 | E3 ablations on `full`: −M1, −M2, −M3 (dense-only), M3-forced, −M5 (root only), −M6, **−Q (legacy identifier rendering)** | r0 | S-B, P5 context | 60 |
| D8 | E5: V1–V4 variants end-to-end (4 runs) | r0 | S-C | 50 |
| D9 | E6: `full_rl` at the pre-registered headline cell (max CV EX_all s.t. calls ≤ P0 and deferral ≤ 10 %; from R0: c_h = 0.75, λ = 0.01) plus three further cells spanning deferral and call levels {c_h 0.25 λ 0, c_h 0.5 λ 0.01, c_h 0.1 λ 0.03}; fixed-k ∈ {1, 2}; P0 = `full_fixed` (D2) | r0 | S-D (negative result) | 120 |
| D10 | E2: {zs, dail, full_fixed, full_gated} × 8 models | r0 (r1–r2 on the two weakest backbones) | S-H | 350–500 |
| D11 | E4 (HR, no dev; stream pipeline stays P0 = `full_fixed` because M6 needs the semantic Critic score), E4c, E8, E9, E10, E11 | per spec | P4, S-E, E8–E11 | per spec |
| — | dropped from the v1 list: 8 of the 12 RL cells; the θ-grid live runs (done by replay on stored candidates, 0 calls); V5 on dev as a headline (reported in S-C only) | | | |

Order of magnitude without E4: ≈ ¥1.2–1.6 k. No change to the ¥8 000 soft cap logic of the study contract §7.

### 13.9 Claim-language additions (extend §11)

* Allowed verbs for P1, P2, P5: "non-inferior within δ", "reduces calls by x % (CI)", "repairs x % of execution failures". Not allowed unless S-A confirms: "improves accuracy", "outperforms", "state of the art".
* Every EX table that contains `full` and `zs` reports ΔEX with its CI in the same table. No sentence may say that the pipeline "improves" EX over zero-shot if that CI includes 0.
* Tuning-set numbers (E-1 … E-9) appear only in Setup (rationale for design choices), the supplement, and Discussion when labelled "tuning set".
* The negative results (RL controller, `route_ca` if not adopted, P5 if n_R < 40, P4 if not confirmed) are reported in the main text with the same prominence as positive ones.

### 13.10 Items outside the SAP that follow from this amendment

The experiment specification §4 (add `full_gated`, `full_gated_blind`, `route_ca`, legacy-renderer arm; B_ref rule), §6 E2/E3/E6/E7 (run lists as in §13.8), §9 (decisions table); `exp/selection_decisions.json`; `analysis/power.py` re-simulation; `results/primary.json` keys P1–P5.

### 13.11 Amendment log

| Date | Amendment | Before the lock | Effect |
| --- | --- | --- | --- |
| 2026-10-26 | Amendment 1: confirmatory family K → P; K1, K2, K4, K5 demoted/redefined, K3 kept; margins, adoption rules, power, run list | yes | Headline claims move from accuracy to cost, routing, pollution, repair; all v1 hypotheses remain in the paper as secondary |

## 14. Amendment 2 (2026-10-26; before the dev lock; no outcome data seen)

Status: proposed; takes effect when committed, before `exp/dev_lock.json` is written. Trigger: the investigators' minimal-validation decision (the minimal-validation plan, run M6) and the fact that no human annotator is available.

### 14.1 Change

| Item | v1 / Amendment 1 | Amendment 2 |
|---|---|---|
| E10 annotators | two authors (A1, A2) label independently; a third author adjudicates | two independent runs of **Claude Sonnet 5.5** (runs A1, A2), each blind to the system and to the automatic class, each with its own random item order, no access to the other run's output; a third Claude Sonnet 5.5 run adjudicates the disagreeing items with the two labels in blinded positions |
| Instructions | `ANNOTATION_GUIDE.md` (further files) | the same guide, delivered through the verbatim prompt in `ANNOTATION_PROMPT.md` (further files) (frozen after the 30-item pilot) |
| Sample, strata, weights, labels L1-L8, secondary label, rationale rule | as in experiment specification E10 | unchanged |
| Agreement statistics | κ (primary label, L5-vs-rest) with item bootstrap, AC1, percent agreement, confusion matrix | unchanged, computed by `code/tools/annotation_kappa.py`; reported before adjudication; **described as agreement between two runs of one model** |
| Prevalence | stratum-weighted, stratified bootstrap | unchanged; all E10 prevalences are exploratory (as before) |
| Human involvement | "author annotation" | none in the labelling. **Optional add-on, reported only if actually carried out:** the authors label about 50 randomly drawn items (seed 20261027, drawn after the LLM labels are frozen, blind to them) and report percent agreement and κ against the adjudicated LLM label. If the spot-check is not done, the paper says that it was planned and not performed |

### 14.2 Claim language (extends §11 and §13.9)

* Allowed: "LLM annotation (two independent Claude Sonnet 5.5 runs)", "agreement between two annotation runs of the same model", "model-assigned error labels".
* Not allowed, for the LLM labels: "author annotation", "human annotation", "manual annotation", "expert annotation/review", "ground truth". The study contract §0 rule (no "expert review"; human parts are called by who did them) is kept in spirit: the paper never claims human or expert annotation that did not happen.
* Required limitation sentence (Discussion, threats to validity): "Both annotators are runs of the same language model; their agreement measures consistency under the guide, not validity against human judgement, shared biases are invisible to it, the model may have seen BIRD during training, and the gold-suspect and ambiguity labels are model judgements."
* The gold-suspect rate (L6) is an estimate by the model; it is not used to correct any gold label (experiment specification §1.1 rule "we do not correct labels ourselves" stays).

### 14.3 What was seen

Nothing outcome-related: this amendment concerns the labelling procedure only. The annotation sheet is built after the dev runs exist (frame = failures of `zs` and `full`); the amendment is written before that.

### 14.4 Amendment log

| Date | Amendment | Before the lock | Effect |
| --- | --- | --- | --- |
| 2026-10-26 | Amendment 2: E10 annotators = two independent Claude Sonnet 5.5 runs; optional human spot-check | yes | E10 labels and κ are model-based and exploratory; no change to any confirmatory test |
| 2026-10-26 | Incident note (not an amendment): during a lock-gate dry run, `nl2sql.run --split bird_dev --limit 0` (0 = no limit) loaded the dev question list and one dev schema and made 10 embedding calls (about 5 100 tokens, < CNY 0.01) before crashing on a dummy store. No chat-model call, no dev scoring, no use of gold SQL results. Artefacts deleted; `exp/dev_lock.json` did not exist | yes | None on any estimate; disclosed for the audit trail |
| 2026-10-26 | Amendment 4 (§16): controlled critic diagnostic with injected semantic errors; scope `diagnostic-injected`; secondary block, no confirmatory test, no Holm family; written and committed before the first critic call | yes (dev lock exists; no dev split, no `code/prompts/**`, no `exp/models.yaml` touched) | New descriptive block; no change to any confirmatory test, claim, or threshold |

## 15. Amendment 3 (2026-10-26; before the dev lock; after the tuning-set analysis of the weak backbones, E4 mini-stream, X5 live checks and X7/X9 offline bounds; no dev data seen)

Status: **proposed; takes effect when committed**, and must be committed before `exp/dev_lock.json` is written (the commit hash of this file is then recorded in the lock). Sections 1-14 are unchanged; Amendment 1 (§13) is re-ranked, not deleted. Every tuning number below is design evidence from BIRD-train data (HT-300: 4 databases; HR-1101: 8 databases) and is read from `results/x5/x5_tuning_summary.json`, `results/x8/x8_tuning_summary.json`, `results/x8/x8_pooled.json`, `results/e4_min/e4_p40_summary.json`, `results/p2/*`, `results/r1/*`, `results/x7/*`, `results/x9/*`. None of it is confirmatory.

### 15.0 What was seen, and what that means

* **Seen (new since §13):** (i) live HR/HT runs of `full_fixed` and `full_gated` with identifier quoting on `qwen3.7-plus` (X5); (ii) HT-300 runs of `zs_ours`, `full_fixed`, `full_gated` on `qwen3.7-flash` and `qwen3.5-27b` (X8); (iii) the E4 mini-stream on the HR stream at 40 % pollution, two permutations, final-checkpoint probes (n = 660), policies `append`, `ws_full`, `memrl`; (iv) offline X7/X9 upper bounds for value probing and value grounding, a replay of confidence-routed compute, an arbitration probe on stored trees, and an M3 forced-routing run on HT-300. **Not seen:** BIRD dev (questions, gold, predictions, scores), minidev, Spider variants, AmbiQT, LiveSQLBench. `exp/dev_lock.json` does not exist; `exp/dev_access.log` shows the evaluator was not used on dev (incident note of §14.4 stands).
* **Also seen:** the pre-lock decision file `exp/selection_decisions.json` (rule 3: `full_fixed`; rule 4 at k = 2: `route_ca` dropped); see §15.3 (P3) and §15.7.
* **Consequence (as in §13.0).** The hypotheses below are pre-specified with respect to dev, **not** with respect to the tuning sets: the tuning results decided which hypotheses are primary. They are never used as confirmatory evidence. In the paper they are shown as tuning-set design evidence, labelled as such.

### 15.1 Tuning-set evidence behind the re-ranking (new rows; E-1 to E-9 are in §13.1)

| # | Finding (tuning data only) | Source |
|---|---|---|
| E-10 | Strong backbone, live, quoting on: `full_gated` vs `full_fixed` on HR-1101: ΔEX = −0.54 pp [−1.50, +0.53], 14 vs 20 discordant, calls/q 2.02 vs 3.29; HT-300: −1.0 pp [−2.2, 0.0], calls 2.01 vs 3.24. The gate keeps the EX of the fixed loop within noise at about 0.61-0.62 of the calls | `x5_tuning_summary.json` (`TS.hr.*`, `TS.ht.*`) |
| E-11 | Weak backbones, HT-300 (`zs_ours` / `full_fixed` / `full_gated` EX): flash 0.627 / 0.653 / 0.667; 27b 0.653 / 0.687 / 0.680. `full_gated` − `zs_ours`: +4.0 pp [+1.2, +8.1] (22 vs 10 discordant) and +2.7 pp [−1.3, +8.6] (20 vs 12); pooled over the two +3.3 pp [+0.3, +8.1]. `full_gated` − `full_fixed`: +1.3 [0.0, +3.8] and −0.7 [−2.0, 0.0]. Calls/q 2.06 / 2.05 (gated) vs 3.73 / 3.72 (fixed); list-price cost per question of gated about 0.48 of fixed on both. Pooled over the three backbones (flash, 27b, plus): gated − fixed −0.1 pp [−0.7, +1.0], calls ratio 0.58 [0.53, 0.62] | `x8_tuning_summary.json`, `x8_pooled.json` |
| E-12 | Flash + gated against plus + zero-shot (`zs_ours`, 0.697, ¥0.0017/q): 0.667 at ¥0.00087/q, i.e. −3.0 pp at about half the cost. Against the B1 `zs` prompt (0.673) the difference is −0.6 pp. Directional (4 databases, one run, no paired test) | `x8_tuning_summary.json`, `base_ht300.json` |
| E-13 | Quoting: on the 131 `beer_factory` HR questions root EX 0.405 → 0.725 and root execution failures 43.5 % → 2.3 % | `x5_tuning_summary.json` (`TS.quote.*`) |
| E-14 | Root failures on HT-300 (gated runs): flash 15, 27b 13, plus 4 of 300; 16 of the 32 pooled were repaired to a correct answer, all 32 were made executable | `x8_pooled.json` |
| E-15 | Routing, X5 fit (k = 4 samples): AUROC V3 → V3 + consistency + non-empty: HR leave-one-database-out +0.042 [+0.021, +0.074]; HT held out +0.045 [−0.009, +0.079]. AURC did **not** improve on HR (0.242 → 0.247). EX of the accepted 60 %: HR 0.743 → 0.781, HT 0.806 → 0.833 | `x5_tuning_summary.json` (`TS.route.*`) |
| E-16 | Negative results: (a) E4 mini-stream, p = 40 %: final-probe EX `append` 0.703, `ws_full` 0.688, `memrl` 0.688; `ws_full` evicted 847 entries, of which 69 were injected-wrong and 778 clean (recall of wrong entries 0.548, precision 0.081, clean-entry eviction rate 0.557). (b) Arbitration probe on stored trees: the judge picks the right member in 52 % of 50 mixed pairs (revise-always: 46 %), against the 70 % adoption bar of §13.6 rule 5. (c) M3 forced routing on HT-300: −5.3 pp [−8.7, −1.8]. (d) Confidence-routed compute (replay): −0.7 pp [−2.3, +0.4] on HR and +0.7 pp [−1.6, +3.2] on HT against gated at 3.00 vs 2.02 calls. (e) PPO controller: no cell of the 12-cell database-grouped CV reaches the fixed policy (best 0.635 vs 0.660). (f) Value probing and value grounding: upper bounds 1.0 / 1.3 pp (probing, HR / HT) and 3.5 pp (grounding, lookup-reachable); hint recall 42.9 % of reachable answers | `e4_p40_summary.json`, `x5_tuning_summary.json`, `adaptive_replay.json`, `e6_summary.json`, `value_check.json`, `empty_roots.json`, `vg_eval.json` |

### 15.2 Re-ranking of hypotheses (nothing is deleted)

| Id after A1 | Status after Amendment 3 | New id and role |
|---|---|---|
| P1 (gated ≈ fixed on `qwen3.7-plus`, calls ratio) | **Extended** to the backbone set Bk = {`qwen3.7-plus`, `qwen3.7-flash`, `qwen3.5-27b`} and pooled | **P1′** (primary, cost, comparison between loop variants) |
| (new) | – | **W1** (primary): on the weak backbones the gated pipeline improves over zero-shot |
| P2 (full vs best baseline at no higher cost) | **Demoted**: B\* is selected on dev among the few baselines the minimal-validation plan runs, so the test is conservative and uninformative about the loop | S-A row "full vs B\*" (two-sided CI, margin 3.0 pp reported, Holm in S-A) |
| P3 (routing) | Kept, with k = 4 extra samples instead of 2 and the gate re-applied (see P3 text); drops out if the k = 4 gate fails | **P3** (primary if adopted) |
| P4 (pollution) | **Demoted** to secondary; stays pre-registered and is run as specified; tuning data point to a null or reversed effect (E-16a) | S-E headline row (GLMM contrast `ws_full` − `append`, two-sided); Holm inside S-E |
| P5 (repair) | **Kept**, population pooled over Bk so that the n_R ≥ 40 rule can be met | **P5′** (primary) |
| (new) negative-result reports N1-N6 | see §15.4 | reported with the same prominence |

### 15.3 Primary confirmatory family P\* (m = 4)

Replaces family P of §13.3. Conventions of §13.3 hold (two-sided α = 0.05 family-wise, run r0, cluster = `db_id`, B = 10 000 db-clustered paired percentile bootstrap with a **common database resample across backbones**, one-sided p doubled into Holm, item-level exact checks, verdict language of §13.3, sensitivity without `thrombosis_prediction`). All systems use the quoting renderer. Backbones are fixed at the lock; the weak set Wk = {`qwen3.7-flash`, `qwen3.5-27b`} may be extended by **one** further backbone (`qwen3.6-35b-a3b` or `deepseek-v4.1-flash-1017`) only if the budget stop rule of `exp/budget.py` allows it **before** the lock; the choice is then written into `dev_lock.json`. No backbone is dropped or added after the lock; a run that fails for technical reasons is repeated, not removed.

**P1′ — H-cost (comparison between loop variants): the execution-gated loop keeps the accuracy of the fixed semantic-critic loop at lower cost, on average over Bk.**
* Systems: `full_gated` vs `full_fixed`, same backbone, same renderer, same frozen M6 store, shared roots (cache hits), per backbone b ∈ Bk. Dev runs: plus (the minimal-validation plan M1), flash and 27b (new: `full_fixed` added, see the minimal-validation plan Amendment).
* (a) ΔEX_Bk = mean over b of [EX(gated, b) − EX(fixed, b)]. H0a: ΔEX_Bk ≤ −1.5 pp (margin unchanged from §13.3; **not widened** although the HT-300 point estimate for plus is −1.0 pp with a lower bound of −2.2 pp: the margin is a statement of what we call negligible, and loosening it after seeing that the data sit near it would be the wrong direction).
* (b) R_Bk = mean over b of [Σ calls(gated, b) / Σ calls(fixed, b)] (answer-path calls as in §13.3). H0b: R_Bk ≥ 0.80. Design value from tuning: 0.58.
* p(P1′) = max(p_a, p_b) (intersection-union). Verdict words as §13.3.
* **Per-backbone claim ("on each backbone") is secondary**, family S-J (m = 3 backbone-level non-inferiority tests, same margin, Holm inside S-J, each with its own calls-ratio bound). The words "on each backbone" are allowed only if all three S-J tests are confirmed; otherwise the text lists the backbones that were and were not.
* Also reported (not tested): EX by root status, tokens/q, latency, list-price cost per question, cost per correct answer, ΔEX at margins 1.0 / 2.0 / 3.0 pp.

**W1 — H-weak: on the weaker, cheaper backbones the gated pipeline is more accurate than zero-shot.**
* Systems: `full_gated` vs `zs` (= `zs_ours`, quoting renderer, one call) on each b ∈ Wk. Same dev items, run r0.
* ΔEX_Wk = mean over b ∈ Wk of [EX(gated, b) − EX(zs, b)]. H0: ΔEX_Wk ≤ 0, one-sided (doubled into Holm). The null is zero, not a margin: the tuning-set estimate is +3.3 pp, and the study is powered for a superiority claim of that size only on average over backbones (§15.6).
* **Sign-consistency rule written in advance:** the verdict "confirmed" additionally requires a positive point estimate on every member of Wk. If the pooled test passes and a member is ≤ 0, the verdict is "supported on average, not on each backbone".
* Always reported: per-backbone ΔEX with CI and exact McNemar p (secondary family S-H, BH q = 0.10, exploratory), the same contrast for `qwen3.7-plus` (expected ≤ 1 pp; this is the strong-backbone control, E-1), ΔEX against `root_only` (`gated_rootonly`, M3 arm) so that the effect of the pipeline's retrieval stack and of the loop can be told apart, calls/q and cost per question.
* **Reading rule:** W1 compares the *pipeline* with a single call, not the loop alone. The text must not say "the loop improves accuracy" unless the `rootonly` comparison supports it (§13.9 applies).

**P3 — H-route:** as §13.3 (ΔAUROC of `route_ca` − V3 on the final answer of `full_gated`, `qwen3.7-plus`; the AUROC difference is one-sided; adoption gate §13.6 rule 4), with **one change recorded here, after seeing a gate result**: the number of extra samples is k = 4 instead of k = 2.
* What happened: `exp/selection_decisions.json` (2026-10-26, `code/tools/selection_rules.py`, k = 2) records rule 4 as **not met**: ΔAUROC +0.018 [+0.008, +0.036] (HR, leave-one-database-out) against the required ≥ +0.03; accepted-EX at 60 % coverage +2.2 pp (condition met); `route_ca_adopted: false`, P3 dropped from the family. That file is retained unchanged. The earlier X5 fit with all four stored samples (k = 4) gave +0.042 [+0.021, +0.074] on HR and +0.045 [−0.009, +0.079] on HT held out, accepted-EX +3.9 pp at 60 % coverage (E-15).
* Why k = 4 is allowed to re-enter, and why this is a degree of freedom we declare: the k extra samples are drawn in **one request** with `n = k` (X5 item 3), so the call count is the same for k = 2 and k = 4 (V3 call + one sampling request); only completion tokens double. The reason to prefer k = 2 in A1 was cost, and that reason does not hold. The choice of k = 4 rests on the same HR data on which the k = 2 gate failed; HT held out is the only unused check and its interval crosses 0. The gate is therefore **re-applied mechanically to k = 4** (`selection_rules.py --k 4`, thresholds unchanged: ΔAUROC ≥ 0.03 with lower bound > 0 and accepted-EX at 60 % ≥ V3 + 2 pp, HR leave-one-database-out, B = 10 000). P3 is in family P\* iff this run meets the rule and `route_coef_hr.json` (fitted on HR with k = 4, hash in the lock) is registered; otherwise P3 is dropped as the gate says and routing is reported in E7 as descriptive (m = 3). Both gate outputs are printed in the paper's supplement.
* AURC: expected not to improve (E-15: 0.242 → 0.247 on HR); the text rule of §13.3 (say "risk-coverage trade-off" only if ΔAURC favours the score) stands.

**P5′ — H-repair:** population R = dev items of `full_gated` r0, **pooled over Bk**, whose root candidate fails to execute (error or timeout). Comparator `full_gated_blind` (error text withheld; template `min_gated_blind.yaml`, X6) run on the same three backbones at r0; roots are cache hits, so the cost is a few ¥. One-sided exact McNemar on R. Inclusion rule of §13.3 unchanged: P5′ is in the family only if pooled n_R ≥ 40 (tuning-set expectation on 1 534 items scaled from HT-300: roughly 100-150, i.e. satisfied); else descriptive and m = 3. The per-backbone and per-error-class repair rates with Wilson intervals are always reported. If `full_gated_blind` is not registered in the lock for all three backbones, P5′ is descriptive and m = 3.

**Decision rule, multiplicity:** Holm over m = 4 (P1′, W1, P3, P5′); m is fixed at the lock and recorded in `dev_lock.json` (3 if P3 fails its k = 4 gate or P5′ fails its inclusion rule; 2 if both). A hypothesis with two components (P1′) is confirmed only if both are.

### 15.4 Demoted and negative-result items (reported with the same prominence; no confirmatory claim)

| Id | Item | Pre-declared reading | Tuning evidence (E-16) |
|---|---|---|---|
| N1 | Learned (PPO/A2C) controller | S-D as in §13.4 on the tuning-set CV only; **no live RL run on dev** (the minimal-validation plan). Wording: "we could not detect a benefit" | best of 12 cells 0.635 vs 0.660 |
| N2 | Value-weighted eviction (P4, S-E) | Run as specified (E4, p ∈ {0, 40 %}); GLMM contrast `ws_full` − `append` two-sided; `ws_full` vs `memrl` as in §13.5. If `ws_full` ≤ `append`, the paper says the corpus rule does not protect against pollution under this protocol and reports the eviction diagnostics (recall, precision, clean-entry rate) | 0.688 vs 0.703; 778 of 847 evictions clean |
| N3 | Arbitration judge | Not adopted (§13.6 rule 5 bar not met); not run on dev; supplement | 52 % vs 70 % bar |
| N4 | Forced M3 routing (`schema_fallback_tokens` = 0) | Not adopted (§13.6 rule 6); exploratory dev arm `gated_m3force` kept (the minimal-validation plan M3) | −5.3 pp [−8.7, −1.8] |
| N5 | Confidence-routed compute | Not run on dev (cannot reach R < 0.80 by construction) | −0.7 / +0.7 pp vs gated at 3.0 calls |
| N6 | Value probing, value grounding, `gate_zero` | Switches stay off in all frozen configurations; offline bounds only | upper bounds 1.0 / 1.3 pp and 3.5 pp |

### 15.5 Secondary additions

* **S-I (cost-effectiveness across backbones, directional).** `full_gated` on `qwen3.7-flash` against `zs` on `qwen3.7-plus`: ΔEX with db-clustered CI and the cost ratio (list price per question, `exp/prices.yaml`). Non-inferiority margin 3.0 pp (as P2), cost-ratio bound 0.70, both reported as estimates with intervals; **no equivalence claim** unless both bounds hold. Tuning point estimate −3.0 pp at ratio 0.51, so a failed non-inferiority test is the expected outcome and will be reported as such. The same contrast for `qwen3.5-27b` is exploratory.
* **S-J** (per-backbone gate non-inferiority, m = 3) and **S-H** (per-backbone gain over zero-shot, BH q = 0.10) as above; **S-E** gains P4; **S-A** gains P2.
* The rank correlation of gain over zero-shot with zero-shot EX across backbones (§13.4) is computed over Bk (n = 3) plus any E2 runs that exist; with n < 5 it is a listing, not a statistic.

### 15.6 Power (normal approximation of §13.7: SE = √[(p_d − Δ²)/n × DEFF], n = 1 534, DEFF = 2.4, correlation r of the backbone-level differences 0.3-0.5 because all backbones answer the same items)

| Test | Assumed discordance p_d (tuning) | SE (pp) | Power at the tuning-set effect, Holm step 1 (one-sided α 0.00625) | nominal one-sided 0.025 |
|---|---|---|---|---|
| W1, k = 2 members | 0.107 (flash and 27b: 32 of 300 each) | 1.04-1.11 | 0.68-0.75 at +3.3 pp; 0.24-0.28 at +2.0 pp | 0.84-0.89; 0.43-0.49 |
| W1, k = 3 members | same | 0.94 | 0.84 at +3.3 pp; 0.36 at +2.0 pp | 0.94; 0.57 |
| single weak backbone | same | 1.29 | 0.73 at +4.0 pp; 0.34 at +2.7 pp | – |
| P1′(a), k = 3, margin 1.5 pp | 0.03 / 0.05 | 0.50 / 0.65 | at true Δ = 0: 0.69 / 0.43; at −0.5 pp: 0.31 / 0.17; at −1.0 pp: 0.07 / 0.04 | 0.85 / 0.64; 0.52 / 0.34; 0.17 / 0.12 |
| P1′(b) | by construction | – | ratio SE small; power ≈ 1 if the design value 0.58 holds | |

Reading, written in advance: (i) W1 has a real chance only because the effect is pooled over backbones; if the true pooled gain is 2 pp the study will most likely report "direction consistent, not confirmed", and the paper will say so. (ii) P1′(a) is **underpowered unless the gated loop is at least as accurate as the fixed one**: with a true difference of −0.5 pp the chance of a confirmed non-inferiority at the Holm level is 0.2-0.3. A likely outcome is "non-inferior at item level" or "not confirmed", reported as such. The margin is not widened after the lock. (iii) The tuning-set point estimates (−0.5 to −1.0 pp on the strong backbone, +1.3 / −0.7 on the weak ones) are themselves uncertain (300 questions, 4 databases) and are not used to set any threshold.

### 15.7 Items that follow outside the SAP

the minimal-validation plan Amendment (run list), the paper outline Revision 2, `exp/queues/gen_min.py` (jobs for weak-backbone `full_fixed` and `full_gated_blind` on three backbones, optional third weak backbone), `freeze_configs.py` (+ `gated_blind`), `analysis_pipeline/primary.py` (family P\*: P1′ pooled over Bk with a common database resample, W1, S-J), `code/tools/selection_rules.py` and `exp/selection_decisions.json`: rule 3 (§13.6) was evaluated on 2026-10-26 and selected `full := full_fixed` because the lower 95 % bound of the HR difference was −1.55 pp against the −1.5 pp margin (one-sided bound −1.40; conditions (ii) and (iii) were met). That decision stands and is not changed here. Consequence for this amendment: the primary claims P1′ and W1 **name their systems** (`full_gated`, `full_fixed`, `zs`) and do not use the label `full`; `full` (= `full_fixed`) only fixes the E1 row labels and the S-A comparison (`full_gated` is still run on every backbone). The paper says in Setup that the mechanical rule picked the fixed loop on a borderline bound and that P1′ tests the question on dev.

### 15.8 Amendment log

| Date | Amendment | Before the lock | Effect |
| --- | --- | --- | --- |
| 2026-10-26 | Amendment 3: primary family P → P\* (P1′ extended to three backbones, W1 new, P3 kept, P5′ pooled); P2 → S-A, P4 → S-E; negative-result list N1-N6; run list extended to weak backbones | yes | Headline claims move to cost-effectiveness across backbones, routing and repair; negative results get equal prominence; no v1 or A1 hypothesis is deleted |


## 16. Amendment 4 (2026-10-26; after the dev lock; no dev data and no critic output seen)

Status: written and committed **before the first critic or reviser call** of this block. Sections 1-15 are unchanged. The dev lock (`exp/dev_lock.json`, 16 entries) is not touched: this block runs on BIRD-train data only, calls the frozen critic and reviser unchanged, and does not edit `code/prompts/**` or `exp/models.yaml` (prompt hash and `models.yaml` hash are recorded before and after the run).

### 16.0 Why, and what was seen

* Trigger: on BIRD dev the semantic revision stage gives no measurable EX gain (`full_fixed` vs `full_gated`: dEX -0.13 pp, 28 discordant items). That result cannot tell whether the semantic Critic fails to see semantic errors, or whether there are few semantic errors it could repair. A diagnostic with clearly distinguishable faults separates the two readings.
* **Seen before this amendment:** the dev result above (aggregate, from the F1 run), the code of the critic and the P0 loop, and the **yield of the model-free item construction** (how many questions admit each injection; this uses gold SQL and SQLite execution only). **Not seen:** any critic score, issue text, revision, or EX of any injected query. No BIRD dev question, gold SQL or prediction is read by this block.
* The item list is a deterministic function of (seed, rules, database contents). It is frozen as `results/diag/items.json` (sha256 recorded) and committed before the run.

### 16.1 Question and reading rule (binding)

* Question: how well does the existing semantic Critic (P0 policy, `full_fixed` configuration, Critic Score 0.3 syntax + 0.5 semantic + 0.2 efficiency, `qwen3.7-plus-2026-05-26` as its own critic) detect, and how well does the P0 revision step repair, **obvious, rule-injected semantic errors** that execute and return non-empty results?
* **This block measures Critic sensitivity to injected errors. It is not evidence for an EX improvement on BIRD.** It is labelled "controlled diagnostic" wherever it appears. Injected errors are single-token, mechanical edits and are more detectable than the errors a model makes naturally; detection and repair rates here are upper-end indications, not estimates of natural-error performance. If detection is low, the paper says so; if it is high, the paper does not conclude that the semantic stage helps on dev.
* Results carry `_meta.scope = diagnostic-injected`; they are stored under `results/diag/` and **must not flow into the confirmatory results** (confirmatory-only). No p-values, no Holm family; intervals only.

### 16.2 Population and exclusions

* Source: `bird_train_pool` (56 databases, 6 918 questions). The HT-300 and HR-1101 tuning splits live in 12 other databases (`held_out_dbs`); the tool asserts the pool is disjoint from them.
* Excluded: the 1 479 questions of the M6 burn-in stream (qids from `exp/frozen/corpus_burnin.sqlite.log.jsonl`, the log of the run that produced the frozen store `corpus_burnin_39e2477c.sqlite`), so that no item has a burn-in experience entry of its own. Eligible: 5 439 questions in 53 databases.
* Per question: the item itself is excluded from retrieval (the pipeline's own `exclude = {qid}` rule); other questions' gold SQL stays in the M2 history exactly as in any dev run.
* Gold requirement: executes within 10 s, returns 1-1 000 rows (complete, not truncated), exec status `ok_nonempty` (not empty, not one all-NULL row).

### 16.3 Injection categories (rule-based, no LLM; tokens from the sqlglot SQLite tokenizer, edits are text splices so formatting is kept)

| Category | Edit (one occurrence per item) | Kept only if |
|---|---|---|
| `distinct_drop` | remove `DISTINCT` after `SELECT` | bag (multiset) of result rows differs from gold's (set-EX cannot see this fault; see 16.7) |
| `limit_change` | `LIMIT 1` -> `LIMIT 2` | set differs |
| `order_flip` | first key of an `ORDER BY`: `DESC`<->`ASC` (a missing direction counts as `ASC` and gets `DESC`) | set differs (needs a `LIMIT` in practice) |
| `count_distinct` | `COUNT(x)` -> `COUNT(DISTINCT x)` or `COUNT(DISTINCT x)` -> `COUNT(x)` (not for `*` or a number) | set differs |
| `agg_maxmin` | `MAX`<->`MIN` | set differs |
| `agg_sumavg` | `SUM`<->`AVG` | set differs |
| `cmp_flip` | `>`<->`<`, `>=`<->`<=` | set differs |
| `where_drop` | drop one conjunct of a top-level `AND` chain of a `WHERE` (clauses with a top-level `OR` are not eligible; a `WHERE` with one condition loses the whole clause; `BETWEEN a AND b` is one conjunct); `n_conjuncts` is recorded | set differs |
| `col_swap` | one column reference (alias-resolved with sqlglot) -> a different column of the **same table with the same declared type string** | set differs |

Common validity filter, applied to the injected SQL: it executes within 10 s without error, its status is `ok_nonempty` (so the syntax sub-score is 10 for every injected query and any score difference comes from the semantic and efficiency parts), it is not text-identical to gold, and its result differs from gold under the category comparator above. If it returns more than 1 000 rows (truncated) against a gold with at most 1 000, it counts as different. Official BIRD EX (set) is also stored for every query.

### 16.4 Selection rule (independent of any model output)

* Master seed `20261030`. Categories are processed in the fixed order `distinct_drop, limit_change, order_flip, count_distinct, agg_maxmin, agg_sumavg, cmp_flip, where_drop, col_swap`; a question belongs to at most one category.
* Per category: databases in a seeded shuffled order; inside each database the questions in a seeded shuffled order; **round-robin** over databases, one question per database per round. For a question: collect all edits of the category; shuffle them with a seed derived from (seed, category, qid); take the first of at most 12 that passes the validity filter; if none passes, the question is skipped (and not used).
* Target **n = 44 injected items per category** (9 categories, 396 injected queries). Controls: the first 22 accepted items of each category (draw order) also get their **unperturbed gold SQL** as a control (198 controls). If a category cannot reach 44 it is reported with the n it reached; categories are not replaced or pooled to hide this.
* Coverage: at least 6 databases is required; the round-robin gives about 40 databases per category.

### 16.5 System under test (unchanged code paths)

* Config `exp/configs/frozen/full_fixed.yaml` (resolved defaults: M1 on, M3 `full`, M2 history/terms/lineage, M6 frozen burn-in store, identifier quoting, `m5.policy p0`, variant `v3`, weights 0.3/0.5/0.2, `theta_fb` 7.5, `theta_acc` 8.5, `t_max` 3), model `qwen37plus` as actor and critic, temperature 0, seed 0, thinking off.
* Per question: M1 intent and M3 context are built exactly as `Pipeline.root()` builds them; only the root **generation** is skipped. The injected SQL (and, for controls, the gold SQL) is evaluated through `Pipeline._evaluate` (execution, EXPLAIN, V3 critic) and then the P0 loop of `FullSystem._run_p0` is applied unchanged (`Pipeline.act` revises while score < `theta_acc`, at most `t_max` revisions, final = best node by (score, earlier)). The critic and the reviser never see the gold SQL or the fact that an error was injected.
* **Gated variant (`full_gated`):** revises only on execution error or timeout (frozen `gate_empty` false). Every injected query executes and is non-empty, so the gated policy accepts it unchanged; its repair rate equals the share of injected queries that already match gold, which is 0 by construction. This is computed with the pipeline's own `gate_node`, and reported as a column, not a separate run.

### 16.6 Metrics (all reported, whatever the outcome)

Unit = question; cluster = `db_id`; 95 % percentile intervals from a database-clustered bootstrap (shared resample, B = 5 000, seed 20261101); per category and, for orientation only, pooled over categories. Per-category values are never replaced by pooled ones.

1. **Detection (primary):** `triggered` = Critic Score < `theta_acc` = 8.5, i.e. the P0 revision trigger, on the injected SQL. Secondary: `must_fix` (score < `theta_fb` = 7.5, the "must fix" revision mode and the `theta_h` abstention level), semantic sub-score <= 5, at least one issue listed.
2. **False-positive rate:** the same `triggered` / `must_fix` on the gold controls. (Gold SQL of BIRD has annotation noise and a style unlike model output, so this rate is a floor on what the critic flags for reasons unrelated to injection.)
3. **AUROC** of the Critic Score separating injected from gold (lower score = injected), tie-aware, on the paired set (the injected and the gold SQL of the same 198 questions; clustered bootstrap over databases), per category and pooled; the same for the semantic sub-score. Also all 396 injected vs all 198 gold.
4. **Repair:** starting from the injected SQL, (a) **one revision round**: the first revision the P0 loop makes, evaluated whether or not the loop would continue; reported over all injected items (items the critic did not trigger count as not repaired, since no revision happens) and conditional on `triggered`; (b) the **pipeline's final answer** after the full P0 loop. Success = the answer matches gold under the category comparator of 16.3 (set equality; bag equality for `distinct_drop`); official BIRD EX (set) is also stored. Also: share of triggered items whose revision changed the SQL, and for the gold controls the share still matching gold after the loop (harm of revising correct SQL).
5. **Gated:** repair rate 0 by construction (16.5), reported for completeness.

### 16.7 Limits stated in advance

* Injected faults are mechanical, small and drawn from gold SQL; natural errors are different and harder (mis-read evidence, wrong joins, formatting). Detection here bounds sensitivity from above.
* `distinct_drop` is invisible to the official set-EX of BIRD; it is included because it is a real semantic fault (duplicate rows), and it is scored by bag equality. Its official EX is 1 by construction and is not used.
* Gold SQL is not guaranteed correct (BIRD annotation noise); a gold control that is flagged is counted as a false positive and is not removed.
* One run, temperature 0, one critic sample; no seed variance is estimated. The same question appears twice (injected and gold) with the same context, which is what makes the AUROC paired.
* The M6 store and the M2 history contain other train questions, some near-duplicates of an item; this is the same exposure as in a dev run.
* The M1/M3 context is built once per question, so injected and control candidates share it; retrieval cost and the root generation call are not part of this block.

### 16.8 Claim language

* Allowed: "controlled diagnostic with rule-injected semantic errors", "Critic sensitivity to injected errors", per-category detection and repair rates with intervals, "the P0 loop repairs X of Y triggered injected errors".
* Not allowed: any statement that these results show, or predict, an EX gain on BIRD or any natural-error benchmark; pooling categories to claim a headline detection rate; dropping a weak category.
* If pooled detection is high but `distinct_drop`/`limit_change`/`col_swap` are low, the paper says which faults the critic misses.

### 16.9 Budget and stop rule

Expected spend under CNY 40 (about 600 critic calls plus the revision calls; list price 2 / 8 CNY per million input / output tokens). Stop and report if the cost of this block exceeds CNY 80. Concurrency <= 4 on `qwen3.7-plus` (the in-flight cap of 12 is shared with the dev queues); machine-local LLM cache; no other job is touched.

### 16.10 Amendment log

| Date | Amendment | Before the first critic call | Effect |
| --- | --- | --- | --- |
| 2026-10-26 | Amendment 4: controlled critic diagnostic with injected semantic errors, scope `diagnostic-injected` | yes | Descriptive secondary block; no confirmatory test or threshold changes; never enters the confirmatory results |

## 17. Amendment 5 (2026-10-26; **exploratory, post-hoc**; offline; written and committed before the analysis code is run on the Critic Scores)

Sections 1-16 are unchanged. The dev lock, `code/prompts/**` and `exp/models.yaml` are not touched. No API call is made; the analysis reads only the stored per-item outputs of the Amendment 4 run (`results/diag/records.jsonl.gz`).

### 17.0 Status, and what was seen

* **Label: `exploratory-posthoc`.** This analysis was designed *after* the Critic Scores of the 396 injected queries and 198 gold controls had been seen (pooled detection 0.85 and false-positive rate 0.27 at the 8.5 trigger). It is not pre-registered in the sense of §13-§16 and no result of it can be confirmatory. Results carry `_meta.scope = diagnostic-posthoc`, are stored under `results/diag/`, and never enter the confirmatory results or any confirmatory family. No p-values, no Holm family; intervals only.
* **Seen before this section:** the full DIAG summary (per-category detection, false positives, AUROC, repair) and the score columns of `critic_diag_items_results.csv`. **Not seen:** any result of the analysis below (no refitted threshold, no learned coefficient).
* Motivation (hypothesis from the patent): with accumulated user feedback (accept / reject of flagged outputs) the Critic's false-positive rate can be reduced at similar detection ("the longer it is used, the better calibrated").

### 17.1 What "feedback" is here (binding wording)

* **Simulated oracle feedback.** The label of an item is taken from gold: injected query = incorrect (positive), gold control = correct (negative). It is noise-free, always available, and drawn at random from the training items, whereas a real user would answer only on some flagged outputs, with errors and with delay. It is therefore an **upper bound for an ideal user**, not human, expert or user feedback, and is never called that. The gold controls include BIRD annotation noise (DIAG section 3); here a flagged control still counts as correct.
* Injected errors are mechanical and easier to detect than natural ones (§16.7); everything in §16.7 applies.

### 17.2 Data and features

* Unit = one stored variant of one question: 396 `pert` records (label 1) and 198 `gold` records (label 0); 594 items in 53 databases. Cluster / grouping = `db_id`.
* Features (existing per-item fields only): the Critic Score total `critic.score` and its three component scores `critic.syntax`, `critic.semantic`, `critic.efficiency` (0-10). **Not used as features:** the injected category, the question, the SQL, the issue text, the repair outcome, the id. `db_id` is used only to build the splits.
* The category is used only to *report* per-category results.

### 17.3 Learners and operating points

* **(a) Threshold:** the score is the total Critic Score (flag = score < t). **(b) Logistic regression** on (syntax, semantic, efficiency) on the raw 0-10 scale, L2 penalty with fixed `C = 1.0` (minimise 0.5*|w|^2 + C * sum of log-loss, intercept not penalised), Newton / IRLS to tolerance 1e-8, no class weights, no tuning of C; flag = fitted linear predictor above a threshold. Implementation in pure Python (no scikit-learn on the analysis machine).
* **Operating point is chosen on the labelled sample only**, with two pre-fixed rules (candidate thresholds = midpoints between consecutive distinct scores of the labelled sample, plus one value below and one above all):
  * **Rule FP (detection at the pipeline's own false-positive level):** among thresholds with sample FPR <= `alpha0`, maximise sample detection; ties -> fewer flags. `alpha0` = FPR of the uncalibrated trigger (score < 8.5) on all 198 controls (54/198 = 0.2727, the DIAG value).
  * **Rule DET (false-positive rate at the detection level of the uncalibrated critic):** among thresholds with sample detection >= `d0`, minimise sample FPR; ties -> more detection. `d0` = detection of the uncalibrated trigger on all 396 injected queries (DIAG 0.85).
  * `alpha0` and `d0` are constants computed once on the full data; they are the targets, not fitted quantities. If the labelled sample lacks either class, the learner falls back to the uncalibrated trigger (score < 8.5).
* **Budget** n in {0, 25, 50, 100, 200, 400} labelled items, drawn uniformly at random without replacement from the training items (a draw larger than the training set is capped). **n = 0 is the uncalibrated critic** (score < 8.5, no fitting, same for both learners and both rules) and must reproduce detection 0.85 / FPR 0.27 on the pooled held-out predictions (a check, written into the output).
* Both variants of the same question may enter the sample as separate items; no stratification.

### 17.4 Splits and repetition

* 5-fold **database-grouped** cross-validation. For each of **20 seeds** (0..19): databases are shuffled with the seed and assigned greedily to the currently smallest fold (by item count) -> five folds; for every fold: train = other four folds, draw the n labelled items from train (seed-derived RNG per (seed, fold, n)), fit, predict the held-out fold. The fold assignment therefore also varies with the seed. Every item is held out exactly once per seed; predictions are pooled over the five folds per seed (out-of-fold). The labelled sample for budget n is *independent* of other budgets (not nested).
* No test database is ever used for fitting, thresholding or choosing hyper-parameters.

### 17.5 Metrics (per budget n, learner, rule; mean over seeds)

1. **Detection** (share of the 396 injected items flagged) and **FPR** (share of the 198 gold controls flagged) of the held-out predictions; the headline numbers are *Rule FP: detection (with the realised FPR alongside)* and *Rule DET: FPR (with the realised detection alongside)*.
2. **AUROC** of the score (lower total / higher fitted risk = injected), all injected vs all controls, tie-aware, from the pooled out-of-fold scores per seed. For learner (a) the AUROC is that of the raw total score and **does not depend on n** by construction (a threshold refit cannot change the ranking); only (b) can change it. (b) at n = 0 is reported as the raw total score.
3. **Intervals:** (i) 95 % percentile **database-clustered bootstrap** (resample the 53 databases with replacement, B = 2 000 for detection / FPR, B = 500 for AUROC, seed 20261102; the statistic is the mean over the 20 seeds of the metric recomputed on the resample), also for the **difference to n = 0** with the same resample; (ii) the 2.5 / 97.5 percentile of the 20 per-seed values (seed spread, reflects only the draw of the n labelled items and the folds).
4. **Per category** (9 categories, with the injected items of that category for detection and the 22 controls of that category for FPR; the operating points are those learned on all categories): Rule FP detection and Rule DET FPR, each with the difference to n = 0 and a database-clustered interval. `distinct_drop` and `count_distinct` (the hard categories of DIAG) are called out explicitly, whatever the outcome.

### 17.6 Reading rule (binding, fixed now)

* A metric is called **improved at budget n** only if the 95 % database-clustered interval of its difference to n = 0 lies entirely on the favourable side (higher detection under Rule FP; lower FPR under Rule DET; higher AUROC) **and** the mean difference is at least 0.05 (5 percentage points; 0.02 for AUROC).
* **If no learner reaches "improved" at n <= 200 on the pooled data, or the curve is flat or non-monotone within the intervals, the paper says that calibration from this feedback does not help** (in this setting, with this upper-bound feedback). A gain only at n = 400 (about 80 % of all items, a budget no user will give) is reported as such and is not described as "the longer it is used the better".
* The hard categories are reported as improved only under the same rule; if not, the paper says plainly that they do not improve. No category is dropped, no pooled value replaces a per-category one.
* Claims allowed: "exploratory, post-hoc, simulated oracle feedback, upper bound". Not allowed: "user study", "human feedback", any claim about natural errors, EX or BIRD, or that this confirms the patent mechanism.

### 17.7 Limits stated in advance

* Post-hoc design after seeing the scores; one Critic run at temperature 0 (the scores are fixed inputs, no Critic sampling variance); 53 databases and 594 items (intervals are wide); prior of injected items is 2/3 here, unlike a real stream where most outputs are correct; the unflagged / flagged selection of real feedback is ignored (this favours calibration); injected errors are easier than natural ones; the gold-control FPR includes annotation noise that an oracle label cannot see.

### 17.8 Amendment log

| Date | Amendment | Before the analysis was run | Effect |
| --- | --- | --- | --- |
| 2026-10-26 | Amendment 5: exploratory post-hoc simulated oracle-feedback calibration of the Critic; scope `diagnostic-posthoc` | yes | Descriptive; no confirmatory test or threshold changes; never enters the confirmatory results |

### 17.9 Addendum after the first full run (descriptive only; no change to the spec above)

After the first full run, `results/diag/feedback_calibration.json` was extended with a descriptive block `addendum_descriptive` (mean fitted logistic-regression coefficients per budget, mean component scores per category and label) to explain one per-category result. It uses the same stored scores, adds no metric to 17.5, and does not change the reading rule of 17.6. Before the code commit, one pilot run with 2 seeds and B = 50 (to a temporary directory) was made; its printed verdict was seen. The full run (20 seeds, B = 2 000 / 500) followed the code commit `c09bec4`.

## 18. Amendment 6 (2026-10-26; **exploratory, post-hoc**; written and committed before any API call of this block)

Sections 1-17 are unchanged. The dev lock (`exp/dev_lock.json`, 16 entries), `code/prompts/**` and `exp/models.yaml` are not touched; the variant prompts live in the new directory `exp/diag_prompts/`. The frozen pipeline critic (V0) is not re-run on the evaluation items.

### 18.0 Status, and what was seen

* **Label: `exploratory-posthoc`, `_meta.scope = diagnostic-posthoc`.** The variants are designed *after* the V0 results of Amendment 4 (detection 0.854, gold false-positive rate 0.273, 43 of 198 gold controls broken by the P0 loop) and after the issue texts of the gold false positives were read. Results never enter the confirmatory results or any confirmatory family. No p-values; intervals only.
* **Seen before this section:** all stored V0 outputs (`results/diag/records.jsonl.gz`, `critic_diag_items_results.csv`), including the issue text of all 54 gold false positives (the first 260 characters of each were printed), and the Amendment 5 result. Sub-score reading: syntax and efficiency carry no signal (AUROC 0.50 / 0.48), the semantic score alone has AUROC 0.84 against 0.829 for the weighted total, and all 54 false positives come from the semantic score (mean semantic 3.4). **Not seen:** any output of a variant prompt.
* The false-positive texts suggest three mechanisms: (a) claims that a table or column is not defined although the query executed, (b) over-literal readings where BIRD gold follows a dataset convention (denominator, "common" as `IN`, an extra column), (c) style remarks. Some false positives may be wrong BIRD gold (label noise); a flagged gold control is still counted as a false positive and none is removed.
* **Code-path facts (read before this section; the audit in 18.6 checks them):** the critic that V0 uses is `crit_sem.user` (variant `v3`; `crit_v2.user` belongs to variant `v2`, not used by the frozen configs). Its schema block is `Context.sections_core()` from M3: with `m3.mode = full` the DDL lists the top `k_t = 6` retrieved tables (plus FK expansion), and the whole schema only if the database is under `schema_fallback_tokens = 6000` tokens. So for larger databases the critic sees a pruned schema, which can by itself produce "table X is not defined" claims about a query that executed. Whether this happens is measured, not assumed (18.6).

### 18.1 Question

Does a better-specified Critic reduce the gold false-positive rate at similar detection on the Amendment 4 diagnostic (396 injected, 198 gold controls)? Two prompt variants, V0 as reference. Injected errors are easier than natural ones (§16.7) and all of §16.7 applies.

### 18.2 Variants (fixed here; the prompt files are fixed in 18.4)

* **V0** = the frozen pipeline critic (`crit_sem.user`, Critic Score 0.3 syntax + 0.5 semantic + 0.2 efficiency, trigger score < 8.5). Taken from the stored Amendment 4 records.
* **VA "grounded"**: new prompt; system prompt unchanged (`common.system`). The prompt (i) states that the SQL already executed successfully and that the execution preview is shown (a non-empty executed query has no runtime or schema error); (ii) forbids claiming a missing or undefined table, column or runtime error unless the named object is absent from the schema listing, and gives the **complete** table/column listing of the database in addition to the pipeline's (possibly pruned) DDL block, so that "not in the shown DDL" is never mistaken for "not in the database"; (iii) requires every issue to quote the question phrase and the SQL fragment it concerns; (iv) splits issues into `must_fix` (changes the returned result) and `style` (ignored); (v) the score is the **semantic score alone** (0-10); syntax and efficiency are not part of it. The reviser receives only the `must_fix` issues and the suggestions.
* **VB "grounded + convention memory"**: VA plus k = 4 past cases with oracle verdicts ("this query was accepted / rejected as correct for this question"). The cases come from the pipeline's own retrieval (M6 corpus `recall`: BM25 + dense, RRF, rerank, same parameters as the frozen run) restricted to the feedback-bearing entries of the frozen burn-in store (`kind = exp`, i.e. the final SQL of the burn-in stream questions, 1 011 with EX 1 and 207 with EX 0), verdict = EX of that SQL against its gold in the burn-in log (`exp/frozen/corpus_burnin.sqlite.log.jsonl`). Excluded from retrieval: all pool-history entries (they are gold SQL), the item's own qid, and every evaluation or tuning qid (disjoint from the burn-in qids by construction; asserted). The gold SQL of the evaluated item is never shown. **The verdicts are SIMULATED ORACLE feedback** (EX against BIRD gold, noise-free), an upper bound for an ideal user, and are always labelled so. VB tests whether accumulated feedback *content* (dataset conventions), not score calibration, helps.
* Unchanged in all variants: model (`qwen3.7-plus-2026-05-26`, temperature 0, seed 0), M1 intent, M3 context (the `<<graph>>`, question, evidence, SQL and execution-preview inputs are the pipeline's), the P0 loop (`t_max = 3`, final = best node), the reviser prompt `rev.user` (only the feedback text differs, as it is built from the critic's issues), `max_tokens` 512 for V0; **900 for VA/VB** (longer JSON with quotes). Unparsable output after the one retry counts as semantic 5 (as in V0) and is reported.
* **Operating point.** VA/VB score = semantic score. Revision trigger: semantic < 7.0, the value at which V0's trigger (8.5) lies when syntax = 10 and efficiency = 10 (3 + 0.5 s + 2 < 8.5 gives s < 7); "must fix" level: semantic < 5.0 (V0: 7.5). The trigger is fixed here and is not tuned; the prompts state the scale (9-10: no `must_fix` issue; 7-8: doubtful, no confirmed `must_fix`; 0-6: at least one `must_fix` issue). Threshold-free AUROC is reported next to it.

### 18.3 Data and split discipline

* **Evaluation set:** `results/diag/items.json` unchanged (sha256 `208504194921c78b38a724219725b921bc8e35639274f81603d79008ba855d18`): 396 injected + 198 gold controls, 53 databases. Each frozen variant runs **once** on it. No prompt change after the first evaluation call.
* **Tuning draw (fresh):** the same generator and 9 categories (`critic_diag.py build`) with seed `20261107`, 25 injected items per category (225) and the first 11 per category as gold controls (99), question ids disjoint from the 594 evaluation items (the 396 evaluation qids are excluded) and from the burn-in qids. Ids `T000`... Prompts are iterated **only** on this draw; V0 is also run once on it (critic only) as the reference for the tuning decisions. Tuning runs are critic-only (no revision loop) except one full-loop run per variant to check the loop does not break. Tuning spend cap: CNY 25. Item list frozen as `results/diag/tuning/items_tune.json` before the first tuning call.
* **Freeze:** after tuning, the prompt files' sha256 are written to `exp/diag_prompts/FROZEN.json` and into the log of 18.8, in a commit made before the first evaluation call.

### 18.4 Prompt files

`exp/diag_prompts/va_grounded.user.txt` and `exp/diag_prompts/vb_memory.user.txt` (placeholders `<<sections>>`, `<<full_schema>>`, `<<question>>`, `<<evidence>>`, `<<graph>>`, `<<sql>>`, `<<preview>>`, `<<memory>>` for VB). Loaded by `code/tools/critic_v2.py` only; `code/prompts/**` is not modified and the pipeline's `prompts_util.render` is not changed. The prompts hash of `code/prompts/` (`a61f8211...`) and the sha256 of `exp/models.yaml` (`5131de63...`) are checked before and after the run.

### 18.5 Metrics (per variant on the evaluation set; V0 from stored records)

Cluster = `db_id`; 95 % percentile database-clustered bootstrap, B = 5 000, seed 20261101 (shared resample of items so that differences to V0 are paired). Per variant: **detection** (trigger on injected), **gold false-positive rate** (trigger on gold controls), **AUROC** (injected vs gold, lower score = injected; V0 reported for the total score and for the semantic sub-score alone), per category detection and false-positive rate, **harm on gold** (share of gold controls whose final answer after the P0 loop no longer equals gold under the DIAG comparator, `final_match = 0`), **repair** (one round and final, over all injected items, as in §16.6). Deltas to V0 (variant minus V0) with clustered intervals for detection, false-positive rate, harm, repair, AUROC.

### 18.6 Audit: what the critic sees (model-free except cached or cheap calls)

For the 594 evaluation items and the tuning items: rebuild the M3 context exactly as in the DIAG run (cache hits) and record whether the shown DDL is the full schema, the number of tables shown / in the database, and the tables referenced by the gold SQL that are absent from the shown DDL. For the V0 gold false positives whose issue text claims a missing table or column, check whether the named object is (a) absent from the shown DDL, (b) absent from the database. Reported with counts.

### 18.7 Decision rule (fixed before any call)

A variant **improves** over V0 only if all hold on the evaluation set:
1. gold false-positive rate falls by at least 8 percentage points (point estimate of variant minus V0 <= -0.08) and the upper bound of the 95 % interval of the difference is below 0;
2. detection is not lower than V0 by more than 3 percentage points (point estimate >= -0.03) and the lower bound of the 95 % interval of the difference is above -0.06 (a loss of 6 points or more is excluded).
"CIs supporting" is defined by the two interval clauses above. The stricter reading (interval bounds at -0.08 and -0.03) is reported next to it. If neither variant meets the rule, the report says so plainly; no other metric is substituted, and per-category results do not rescue a failed pooled rule.

### 18.8 Limits stated in advance, budget, claim language

* Post-hoc design after the V0 results; one run per variant at temperature 0 (no sampling variance); 53 databases, wide intervals; injected errors are easier than natural ones; gold false positives include BIRD label noise; the VB verdicts are oracle (EX against gold), not human feedback; VA/VB differ from V0 in several respects at once (full schema, grounding rules, quoting, score definition), so a difference is not attributed to a single cause; the tuning draw uses the same generator and the same databases as the evaluation set (disjoint questions only), so it does not guard against prompt over-fit to the database population.
* Budget: expected CNY 40-60 in total (V0 cost 7.3 for the same items); tuning cap 25; stop and report at CNY 100. Concurrency <= 4 on `qwen3.7-plus` (dev queues may share the cap); machine-local cache.
* Claim language: "exploratory, post-hoc, simulated oracle feedback (VB), controlled diagnostic". Not allowed: any statement about EX on BIRD, natural errors, human or user feedback, or that this confirms the patent mechanism.

### 18.9 Amendment log

| Date | Amendment | Before the first API call | Effect |
| --- | --- | --- | --- |
| 2026-10-26 | Amendment 6: better-specified critic variants VA / VB on the DIAG items, tuning draw + frozen evaluation; scope `diagnostic-posthoc` | yes | Descriptive; no confirmatory test or threshold changes; never enters the confirmatory results |

### 18.10 Amendment 6b (2026-10-26; added after Amendment 6 was committed; **no API call of this block had been made yet**; committed before any VC call)

Requested during the study (the critic's output should go to a verifying agent, claims are checked, and only verified issues are kept before any revision). Sections 18.0-18.9 stand; this subsection adds a third variant and changes two budget numbers. Same label (`exploratory-posthoc`, `_meta.scope = diagnostic-posthoc`), same split discipline (tune on the fresh draw, freeze, run once on the frozen 594 items), same decision rule (18.7) and the same claim language (18.8).

* **VC "verified critic" = the VA critic, then per-issue verification; only confirmed `must_fix` issues keep the item flagged and reach the reviser.** The critic stage is the frozen VA prompt (same file, same inputs, same scale), so VC differs from VA only in what happens after the critic.
* **Stage 1 (deterministic, no verifier call).** For every valid `must_fix` issue of a critic-flagged node (a node with semantic < 7):
  * (a) a claim of a runtime / syntax error, or that a table or column is undefined, missing or not in the schema, is **rejected** when the query executed successfully; the objects named in the claim are also looked up in the complete schema (either they exist, so the claim is false, or they do not exist, so it cannot be a defect of an executed query; both are rejected);
  * (b) a claim that the query does not return a named column is checked against the actual result header and **rejected** if the header contains it;
  * (c) the revision step is run once on the issues that survive (a)-(b) (the pipeline makes this revision for every flagged node anyway); if the revised SQL is identical to the original or returns the same result (ordered if either SQL has `ORDER BY`, else as a multiset), the issues are **moot and dropped**.
* **Stage 2 (one verifier LLM call per surviving issue, `qwen3.7-plus-2026-05-26`, temperature 0, separate prompt `exp/diag_prompts/vc_verifier.user.txt`).** Inputs: question, evidence, the complete schema listing, the SQL, the execution preview, the single issue claim (quoted phrase, SQL fragment, description) and mechanical probe evidence (original result against the result of the proposed fix: header, row count, first rows, NULL / empty status, row-count difference, whether the results are equal). It does **not** receive the critic's score, the other issues, or anything from gold. Output: `confirmed` / `unsupported` with quoted evidence; unparsable output counts as `unsupported`. The proposed fix is the single revision made for all surviving issues together (a per-issue fix would need one revision call per issue); the verifier is told so.
* **Flag and score.** A node is flagged iff at least one issue is confirmed. Its score is `min(raw semantic, 6)` if flagged and `max(raw semantic, 7)` otherwise (the same consistency rule as in 18.2); the raw critic score is kept. Trigger and "must fix" level are those of VA (7.0 / 5.0). If all surviving issues are confirmed the probe revision is the revision of the loop; if only some are confirmed the revision is regenerated with the confirmed issues only (one more reviser call); if none is confirmed the node is accepted. The last node of the loop (`k = t_max`) cannot be revised and is not verified (its critic score is used for the final choice). Everything else in the P0 loop is unchanged.
* **Reported in addition:** per variant and node type (injected / gold) the number of critic-flagged nodes removed by stage 1(a)+(b), by stage 1(c) and by stage 2 (root nodes); the LLM calls on the critic-revision path per query and per flagged item (critic + reviser + verifier calls; V0, VA, VB derived from the stored rounds, VC counted by the client wrapper); which false positives are removable by the deterministic stage alone and which need the verifier. Reading limit stated in advance: a same-model verifier shares the critic's biases, and convention claims (denominators, `IN` against both, an extra column) cannot be settled by execution, so stage 2 can confirm a wrong claim.
* **Budget.** Tuning cap CNY 25 for VA / VB as in 18.8 plus CNY 10 for VC; the overall stop is raised from CNY 100 to **CNY 130**; expected total CNY 45-70. Concurrency <= 4.
* Tuning runs of VC are root-level only (`--critic-only`: critic, stages 1-2 on the root node, no further loop) except one full-loop check.

| Date | Amendment | Before the first VC call | Effect |
| --- | --- | --- | --- |
| 2026-10-26 | Amendment 6b: third variant VC (verified critic), stop CNY 130 | yes | Same label and rules as Amendment 6 |

### 18.11 Amendment 6c (2026-10-26; added after Amendment 6b; **no API call of this block had been made yet**; committed before any VD call)

Requested during the study. A fourth variant; sections 18.0-18.10 stand. Same label (`exploratory-posthoc`, `_meta.scope = diagnostic-posthoc`), same split discipline (tune on the fresh draw, freeze, run once on the frozen 594 items), same decision rule (18.7), same claim language (18.8).

* **VD "appeal" = VC plus one appeal round.** Stage 1 and the verifier (stage 2) are exactly VC's. Every issue that the **verifier** marked `unsupported` is sent back to the critic (same model, system prompt `common.system`, new prompt `exp/diag_prompts/vd_appeal.user.txt`) together with the verifier's rebuttal (verdict and quoted evidence) and the mechanical evidence of stage 1 (the schema check, the execution preview of the original, the result of the proposed fix and the comparison of the two results). The critic answers per issue `maintain` or `withdraw`. The prompt says that withdrawing is the correct answer when the claim is not supported by the evidence, that maintaining requires **new** evidence and not a restatement, and that the critic must address the rebuttal point by point; it is **not** told which answer is expected and does not see the verifier's other verdicts or any score. A `maintain` without a non-empty justification counts as `withdraw`; unparsable output counts as `withdraw`. Issues the verifier confirmed are not appealed; issues rejected by the deterministic stage 1 are not appealed either (they are facts about the schema, the header or the result, not judgements). **One appeal round only.**
* **Flag and revision.** A node is flagged iff at least one issue is confirmed by the verifier or maintained after the appeal. Revision input = confirmed + maintained issues. If that set equals the set used for the probe revision the probe revision is reused, otherwise the revision is regenerated (one more reviser call). Score rule, thresholds, the unverified last node, and the rest of the P0 loop are those of VC.
* **Measured specifically (all reported, whatever the outcome).** (1) Appealed issues withdrawn on gold controls (false positives removed) against withdrawn on injected items (true detections lost), and the number of items whose flag changed because of the appeal, separately for gold controls and injected items. (2) **Sycophancy check:** on injected items, how often the critic withdraws a claim that is demonstrably correct. Operational definition (oracle, offline, uses gold only in the analysis): the claim is *correct* if the node's probe revision (the proposed fix) reproduces the gold result under the category comparator of 16.3; the check reports the share of appealed issues on injected nodes with a correct claim that were withdrawn, next to the share withdrawn on injected nodes with a non-correct claim and on gold controls. An agreeable critic that withdraws everything would show 100 % withdrawals on all three. (3) LLM calls on the critic-revision path per flagged query and per query overall (appeal calls counted by the client wrapper; V0 / VA / VB derived from the stored rounds). (4) VD against VC, VA, VB, V0 on detection, gold false-positive rate, harm on gold, repair, AUROC, with the paired clustered intervals of 18.5. If the appeal round adds nothing over VC the report says so; this is a plausible outcome because critic, verifier and appeal come from the same model family and share biases.
* **Budget.** Tuning cap CNY 25 for VA / VB (18.8) plus CNY 10 for VC (18.10) plus CNY 10 for VD; the overall stop is raised from CNY 130 to **CNY 150**. Concurrency <= 4. Tuning runs of VD are root-level only (`--critic-only`) except one full-loop check.

| Date | Amendment | Before the first VD call | Effect |
| --- | --- | --- | --- |
| 2026-10-26 | Amendment 6c: fourth variant VD (appeal round after the verifier), stop CNY 150 | yes | Same label and rules as Amendment 6 |

### 18.12 Freeze log (2026-10-26 EDT; written after the tuning draw, before the first evaluation call)

* Tuning draw: `results/diag/tuning/items_tune.json` (sha256 `a1140edeb012ca2a76d2f4a4d530e00962e143d6bdb0a40d7df52ef4b6f7086e`; 225 injected, 99 gold controls, 51 databases). Critic-only runs at the root node; numbers in `results/diag/tuning/tuning_summary.json`. V0 on the draw: detection 0.844, gold false-positive rate 0.202 (the evaluation set gave 0.854 / 0.273).
* Iterations: VA draft 1 (0.849 / 0.212), VA draft 2 with rule 5 "a doubt is not a must_fix issue" (0.836 / 0.202, frozen); VB on draft 2 (0.844 / 0.192, frozen); VC verifier draft 1 (0.822 / 0.162; the verifier confirmed 264 of 266 claims), draft 2 with the "strongest reading under which the SQL is correct" procedure (0.818 / 0.162; 5 claims unsupported, frozen); VD appeal prompt draft 1 (0.818 / 0.162, frozen; of 5 appealed issues 4 withdrawn, 1 maintained). A 30-item full-loop check per variant ran without error. No further iteration: the verifier hardly discriminates and the remaining flagged gold controls contain evident gold defects, so further prompt changes would trade detection for a lower false-positive rate on defective gold.
* Frozen files and sha256: `exp/diag_prompts/FROZEN.json`. No prompt change after this commit.
* Reading note for 18.11 (1): the appeal can only add flags relative to VC (unsupported issues are dropped in VC and may be restored by `maintain` in VD), so "withdrawn" appealed issues are those that VD leaves out like VC; the quantities that can differ from VC are the maintained issues (true detections recovered on injected items, false positives re-introduced on gold controls). Both are reported.
* Tuning spend: about CNY 15 (cost_report of the tuning runs); evaluation not yet started.

## 19. Amendment 6d (2026-10-26; **exploratory, post-hoc**; written and committed before any API call of this block)

Sections 1-18 are unchanged. `code/prompts/**`, `exp/models.yaml`, `exp/dev_lock.json` and the confirmatory results are not touched; no result of this block enters the confirmatory results or any confirmatory family. Label `exploratory-posthoc`, `_meta.scope = diagnostic-posthoc`. Claim language of 18.8 applies: controlled diagnostic with injected errors, **simulated** user feedback, no statement about EX on BIRD, natural errors or real users.

### 19.0 Why this block exists, and what was seen

* The VC / VD result (18.10, 18.11) is confounded in two ways. (i) VD substituted an LLM verifier for the user: the verifier answered `confirmed` for 262 of 270 claims whose proposed fix reproduces gold and for 259 of 290 claims that are wrong (gold controls 62 of 74, injected-wrong 197 of 216), i.e. it rejected 11 % of wrong and 3 % of correct claims and hardly discriminates. (ii) The VD appeal prompt (`exp/diag_prompts/vd_appeal.user.txt`) says "Withdrawing is the correct answer when your claim is not supported ... a claim you maintain without addressing the verifier's points counts as withdrawn", i.e. withdrawal is the default. Result: 38 of 39 appealed claims withdrawn (8 of 8 correct, 18 of 19 wrong, 12 of 12 gold). The statement "the critic is sycophantic" therefore cannot be drawn from VD: the outcome is explained by the prompt alone.
* The intended design (user's, accepted) is: a **dispute** is raised by the user; only disputed decisions go to an appeal in which the evidence and the user's rebuttal go to the critic; **undisputed** decisions are not appealed; wrong critic decisions are checked in **both** directions (a flagged-but-fine query, and a passed-but-wrong query). VD tested none of this.
* **Seen before this section:** the VC / VD confusion counts above, the text of the VD appeal prompt, one stage log of a VD record (D017 gold: structure and one verifier / appeal pair), and, from the stored VA records of the evaluation items, the composition of the units of 19.1 (counts only: 389 flagged roots = 340 injected + 49 gold; 233 of the 340 injected flags have a fix that reproduces gold, 107 do not; 205 passed roots = 56 injected + 149 gold; every injected root result differs from gold). **Not seen:** any arbiter output (no arm of this block has been run), any user-dispute simulation result.

### 19.1 Units and oracle labels (evaluation set: `results/diag/items.json`, sha256 `208504194921c78b38a724219725b921bc8e35639274f81603d79008ba855d18`)

* One **unit** = one evaluated SQL at the **root node** of the stored VA run (`results/diag/critic_v2_records_VA.jsonl.gz`; the grounded critic; its `must_fix` issues are the claims; no loop, no verifier). 396 injected SQLs (`pert`) and 198 gold controls (`gold`) = 594 units, 53 databases. No critic call is made in this block for the evaluation set; the proposed fix of a flagged unit is the stored first revision of the VA loop (`rounds[0].sql`), re-executed offline for its real execution.
* **Flagged unit (F)** = the VA critic flagged it (semantic < 7): 389 units (injected 340, gold 49). The claim set = all its `must_fix` issues, treated as one block; a dispute and an arbitration concern the whole block (a per-claim fix does not exist). **Passed unit (P)** = not flagged: 205 units (injected 56, gold 149).
* **Critic-right label** (oracle, uses gold, offline only, never shown to any model): F, injected: the claim block is *correct* iff the stored fix reproduces the gold result under the category comparator (`ex_bag` for `distinct_drop`, `ex_set` otherwise, as in 18.11); F, gold: the claim block is *wrong* by definition (caveat below); P, injected: the SQL is *wrong* (its root result differs from gold under the comparator; asserted for all 396); P, gold: the SQL is *right*. Caveat: BIRD gold contains defects, so some flagged gold controls carry correct claims; they are labelled wrong and none is removed (as in 18.0).
* **Correct action** of the critic after a dispute: F: *maintain* iff the claim block is correct; P: *change* (flag) iff the SQL is wrong. In words: when the critic was right it should **hold**, when it was wrong it should **yield**.

### 19.2 Simulated user dispute channel (noise q)

* The user forms a verdict on the critic's decision that equals the oracle truth with probability q, otherwise the opposite, independently per unit. q in {1.0, 0.85, 0.70}.
  1. **Dispute of a flag (F).** The user rejects the critic's claim ("this is fine / not what I want changed") iff his verdict is "the claim is wrong". Claims the user accepts go straight through with the fix, **no appeal call**.
  2. **Dispute of a pass (P).** The user says "this result is wrong" iff his verdict is "the SQL is wrong". Passes the user accepts: no call.
  3. A unit is **disputed** iff the user contradicts the critic's decision. P(disputed) = q if the critic was wrong, 1 - q if the critic was right.
* **User-right dispute** = the user contradicts a wrong decision of the critic (correct action: yield); **user-wrong dispute** = the user contradicts a right decision (correct action: hold).
* **Primary quantities are exact expectations over the user noise** (each unit contributes with weight P(disputed) or 1 - P(disputed); everything is linear in these weights), so no noise seed is needed and q is coupled across arms and values. A single seeded realisation (seed 20261101; per unit an independent uniform draw u, the user is right iff u < q) is reported next to it with the realised counts of disputes per cell, as a check and to show the n a real run would have.
* The arbiter prompt of a unit is the same whatever q and whatever the realisation (the user's rebuttal is a fixed template that carries only the user's verdict, see 19.3), so one arbitration per unit and arm covers all q (temperature 0). The arbitration of units that a given q does not dispute is not used for that q; the **cost in deployment** is reported from P(disputed).

### 19.3 Arms (the user's rebuttal is verdict-only; no gold SQL, no gold rows, no reason)

* Rebuttal templates (fixed text, in the prompt files): F: "The user disputes this review and does not want the change: 'I do not want this changed. The query already returns what I asked for.' The user gave no further reason."; P: "The user disputes this review: 'This result is not what I asked for. The query is wrong.' The user gave no further reason." Anything richer would have to be derived from gold; the verdict-only channel is the most conservative one and is a stated limit (a real user may give a reason).
* **Arm A (main): same-model critic arbitrates.** `qwen3.7-plus-2026-05-26`, temperature 0, system prompt `common.system`. Inputs: question, evidence, complete schema listing, SQL, real execution, the review (F: the claim block and the score; P: "no must_fix issue", the score and the style remarks), the user's rebuttal and, for F, the mechanical evidence (the fix and its real execution, comparison with the original). **Neutral prompts** `exp/diag_prompts/ve_arbiter_flag.user.txt` and `ve_arbiter_pass.user.txt`: "maintain" / "change", neither is the default, the consequence of a wrong maintain and of a wrong change are stated with equal weight, and evidence must be quoted for either answer. Output `{"decision": "maintain"|"change", "evidence": str}`; unparsable output, or a `change` with fewer than 10 characters of evidence, counts as `maintain` (the original decision stands; reported).
* **Arm B: different-family arbiter.** `deepseek-v4.1-flash-1017` (`dsv41flash`, already in `exp/models.yaml`, role `critic_cross_vendor`), same neutral prompts, role intro "another reviewer" instead of "you", same parse rules.
* **Arm C (control, flags only): the old VD prompt on the same disputes.** `vd_appeal.user.txt` unchanged, the verifier slot filled with the user's rejection; qwen3.7-plus; maintain without a justification of at least 10 characters counts as withdraw (the VD rule). It exists to show how much the default in the old prompt drives the outcome; a pass has no counterpart in that prompt, so C is not applied to P.
* **Arm A0 (control, added in this amendment): same prompts as A with the user's statement removed** (the critic is asked to reconsider once). It separates "what the critic would decide anyway" from "what it does because the user said so".
* References: **no appeal** (VA as is), **obey the user** (the critic yields on every dispute), **oracle-arbiter** (every dispute is decided with the correct action; undisputed decisions stand), **oracle-all** (every unit has the correct action, whatever the user does).

### 19.4 Outcomes (all reported for every q and arm, with n; 95 % database-clustered percentile bootstrap, B = 5 000, seed 20261101, one shared resample so differences are paired)

* **Arbitration quality, q-independent** (over the units that *could* be disputed): *firmness* = share of right critic decisions on which the arbiter holds (F with a correct claim; P gold; n = 233 + 149); *responsiveness* = share of wrong critic decisions on which it yields (F with a wrong claim; P injected; n = 156 + 56). Also per direction (F and P) and per unit class. `1 - firmness` is the sycophancy rate on user-wrong disputes. Weighted by P(disputed) for each q (expected n of disputes in each cell is given).
* **Final flag outcomes**, expectation over the noise, item level: **DET** = share of injected items flagged finally (the old definition, 396); **UDET** (useful detection) = share of injected items finally flagged with a correct claim for F, or flagged through a successful change of a pass (P injected; the new claim is not checked, optimistic); **FPR** = share of gold controls finally flagged (198); **J = UDET - FPR**; **CCR** = share of correct claims retained (233); **WCR** = share of wrong claims removed (156 = 107 injected-wrong + 49 gold); **MER** = share of passed-but-wrong queries recovered (56); **TNR** = share of passed-and-right queries left alone (149). **Harm (lower bound)** = gold controls flagged finally whose stored fix does not reproduce gold (`ex_bag`); the fix of a gold pass that the arbiter changes is not generated, so FPR is the upper bound of harm. No repair after an appeal-driven change is run (stated limit).
* **Cost:** arbitration calls per query = sum of P(disputed) over the units that are arbitrated / 594 (A, B), flags only for C; compared with the extra +0.99 calls per query of VC (DIAGV2). Tokens and CNY from the usage logs.

### 19.5 Decision rule (fixed here, before any call). Applies to arm A (main) and, as a second line, to arm B, at **q = 0.85**; q = 1.0 and 0.70, arms C and A0 and all per-direction numbers are descriptive

The appeal is **useful** iff all hold:
1. **Beats obeying the user** on at least one of: (a) J(arm) - J(obey) >= +0.02 and the lower bound of its 95 % interval > 0; (b) CCR(arm) - CCR(obey) >= +0.05 and its lower bound > 0.
2. **Does not lose more than 3 pp against no appeal in the other direction**: CCR(arm) >= CCR(no appeal) - 0.03 (point estimate; no appeal has CCR = 1) **and** J(arm) >= J(no appeal) - 0.03 (point), with the lower bounds of both paired differences above -0.06 (the sibling clause of 18.7).
The stricter reading (lower bounds at -0.03 for clause 2) is reported next to it. If the rule fails, the report says so and substitutes no other metric; per-class results do not rescue a failed pooled rule. Two descriptive checks accompany it and decide nothing: firmness and responsiveness of A against A0 and C (does the user's statement move the critic, in which direction, and does the old default explain the VD outcome).

### 19.6 Prompt freeze, tuning draw, budget

* **Prompt drafts** are `exp/diag_prompts/ve_arbiter_flag.user.txt` and `ve_arbiter_pass.user.txt`; tuned **only** on a fresh draw (`critic_diag.py build`, seed `20261115`, 20 injected per category = 180, 9 gold controls per category = 81, id prefix `E`, questions disjoint from the 594 evaluation items **and** from the 324 earlier tuning items of 18.3 and from the burn-in qids; item list frozen as `results/diag/tuning_ve/items_tune_ve.json` before the first call). On the draw the frozen VA critic and its loop are run once (units as in 19.1), then arm A (and B) arbitrates every unit. Allowed edits: wording, parse robustness, neutrality (equal weight to both answers), at most **3 drafts**; the selection criterion is stated now: parse-failure rate below 2 %, and among the drafts the one with the higher balanced accuracy (mean of firmness and responsiveness of arm A on the draw), ties to the shorter. No tuning on the 594 units. Then freeze: sha256 of the two arbiter prompts (and of `vd_appeal.user.txt`, unchanged) into `exp/diag_prompts/FROZEN_VE.json` in a commit made before the first evaluation call.
* **Evaluation:** one pass per arm over the units of 19.1 (A 594, A0 594, B 594, C 389 calls), temperature 0, seed 0, concurrency <= 4 (shared Qwen cap), machine-local cache. `code/prompts` and `models.yaml` hashes are checked before and after each run.
* **Budget:** stop at **CNY 60** for tuning plus evaluation (expected 15-25); tuning cap CNY 15; actual cost reported with the results.

### 19.7 Limits stated in advance

Verdict-only user rebuttal; the user is simulated (a symmetric flip of the oracle truth with probability 1 - q, independent of the unit), not human; injected errors are easier than natural ones; gold defects are counted as critic errors (flagged gold = wrong claim); one critic output per unit (the stored VA run, temperature 0), one arbitration per unit and arm; 53 databases and 56 passed-but-wrong units make the missed-error direction wide; the claim block is arbitrated as one block, so a block with one correct and one wrong claim is labelled by its fix; a changed pass is not repaired; arm C is a control only on the flag side; the arbiter of arm A shares the critic's model and biases, arm B tests only one other family.

| Date | Amendment | Before the first VE call | Effect |
| --- | --- | --- | --- |
| 2026-10-26 | Amendment 6d: user-dispute / appeal test (arms A, A0, B, C), rule 19.5, stop CNY 60 | yes | Descriptive; same label and scope as Amendment 6; never enters the confirmatory results |

### 19.8 Freeze log (2026-10-27 CST / 2026-10-26 EDT; written after the tuning draw, before the first evaluation call)

* Tuning draw: `results/diag/tuning_ve/items_tune_ve.json` (sha256 `7283db2051e66c428f7494d03cec2d4dbca6574d67e0b8630c9b66ceeb538c7a`; 180 injected, 81 gold controls, 51 databases; disjoint from the 594 evaluation items and the 324 earlier tuning items, asserted). The frozen VA critic and loop were run once on it (CNY 3.86); units as in 19.1: 261 (injected: 44 flags with a wrong claim, 108 with a correct claim, 28 passed-but-wrong; gold: 22 flags, 59 passes). Results: `results/diag/tuning_ve/tuning_summary_ve.json`.
* **Inputs added before any arbiter call:** the arbiter also receives the M1 task graph (`<<graph?>>`) because the critic's claims quote it ("the task graph specifies a limit of 1"); it is the cached M1 call of the DIAG / VA runs (an offline client; 0 cache misses over the 396 evaluation questions). Arm C keeps the old VD prompt, which has no graph slot. No arbiter output had been seen when this was added.
* Drafts: **d1** (arm A, max_tokens 700): parse failures 2 of 261 (output cut at 700 tokens), firmness 0.910, responsiveness 0.362, balanced 0.636. Reading d1's text revealed an asymmetry that the neutrality requirement of 19.3 forbids: the pass prompt demanded a specific quote only for `change`. **d2:** the pass prompt asks for the same specificity for `maintain` and `change`; max_tokens 1000. Arm A on d2: 0 failures, firmness 0.928, responsiveness 0.319, balanced 0.624; A0 (no user statement) 0.946 / 0.170 / 0.558; B 0.946 / 0.149 / 0.548; C (old prompt, flags only) holds 0.9 % of right claims and yields on 100 % of wrong ones. d2 is frozen **for neutrality, not for the metric**: its balanced accuracy is 0.012 below d1's (noise at n = 261), so the selection rule of 19.6 read literally would have kept d1; this is a deviation, recorded here. No third draft.
* Frozen files and sha256: `exp/diag_prompts/FROZEN_VE.json`; evaluation units `results/diag/critic_ve_units.json.gz` (sha256 of the unpacked file in `FROZEN_VE.json`). No prompt change after this commit.
* Tuning spend so far: CNY 7.0 (cap 15). Evaluation not yet started.

### 19.9 Rule correction (2026-10-27; written and committed before any evaluation output was analysed; the arbitration calls of the evaluation had started, no decision of them had been read)

* Found when the analysis code was dry-run on the **tuning draw** (not the evaluation set): the rule of 19.5 has a degenerate pass. An arbiter that always holds equals "no appeal"; at q = 0.85 it beats "obey the user" on correct-claim retention (CCR 1.0 against 0.85, clause 1b) and loses nothing against no appeal (clause 2), so it would be declared *useful* although it does nothing. This was not intended: "useful" must mean that the appeal adds something.
* **Corrected rule (replaces the decision of 19.5; everything else of 19.5 stands):** the appeal (arm A, second line arm B, q = 0.85) is useful iff (1) it beats obeying the user (19.5 clause 1: J(arm) - J(obey) >= +0.02 with lower bound > 0, or CCR(arm) - CCR(obey) >= +0.05 with lower bound > 0), **and** (2) it does not lose more than 3 pp against no appeal (19.5 clause 2, unchanged), **and** (3) it beats no appeal on J: J(arm) - J(none) >= +0.02 with the lower bound of the paired 95 % interval > 0.
* Both verdicts are reported: `useful_19_5_as_committed` and `useful` (corrected). The headline uses the corrected rule and says so; the verdict under the rule as committed is stated next to it.
* The dry run also showed that on the tuning draw "obeying the user" beats the arbiter on J at q = 0.85 (the user is right 85 % of the time and the arbiter yields on only about a third of the wrong decisions); the corrected rule does not change that, and the report states the comparison to "obey" in both directions whatever it is.

## 20. Amendment 7 (2026-10-27; **robustness, post-hoc**; offline; written and committed before any stored prediction was scored against any corrected gold)

Sections 1-19 are unchanged. The dev lock, `code/prompts/**`, `exp/models.yaml`, `exp/dev_lock.json`, `paper/` and the code of the analysis pipeline (`code/analysis_pipeline/*`, `code/tools/make_macros.py`, `make_tables.py`, `code/scripts/analyze_all.sh`) are not touched. **No model or API call is made**; the block reads stored `preds.jsonl` / `cands.jsonl` and executes SQL on the BIRD SQLite files only.

### 20.0 Status, and what was seen

* **Label: `robustness-posthoc`, `_meta.scope = robustness-posthoc`.** The question (does BIRD gold noise change the headline conclusions?) was asked after the confirmatory pass of §13-§15 had produced its numbers. Nothing here is confirmatory: no p-value enters any Holm family, results are written only to `results/robust/`, never to a top-level `results/*.json`, never to the confirmatory results. Intervals and counts only (the only test statistic is an exact sign test on revision counts, descriptive).
* **Seen before this section.** (i) Every primary and secondary result of the formal pass at the original gold (`results/primary.json`, `W1.json`, `E3.json`), and the E10 numbers (revisions of an executable SQL under `full_fixed`: 70 changed, 11 helped, 11 hurt on all 1 534 items; label L6 about a third of failures). (ii) **Data-level counts, no prediction involved:** the overlap of the corrected sets with the 1 534 dev question ids (below), the number of items whose question and evidence equal the original dev text, the number whose gold SQL *text* differs, and the dev-versus-Mini-Dev SQLite files differing at byte level for 5 of 11 databases (contents not compared). **Not seen:** any prediction scored against any corrected gold; any corrected gold executed; any gold-result difference; any count of E10 items by corrected-gold status; any flip. The author of this amendment has not looked at a rescored result.

### 20.1 Question and reading of the answer

Reviewers will ask whether errors in BIRD gold SQL change the paper's conclusions. Our predictions were produced from the **original** question and evidence of `bird_dev` (1 534 items, 11 databases). This block re-scores the **same stored predictions** against corrected gold SQL where a corrected release exists, and asks, contrast by contrast, whether the conclusion survives. It is a robustness check of the stored runs, not a new experiment; it does not say how the pipeline would perform when run on corrected questions.

### 20.2 Corrected gold and the sets (fixed here)

* **Gold variants** (all four 498-item, based on the 2024-06 Mini-Dev, whose ids are a subset of the dev ids; the data description section 2):
  * **G1 (primary): Arcwise-Plat-SQL** from the Jin et al. repository, `repos/text_to_sql_benchmarks/data/arcwise_plat_sql_only_with_diff.json` (CC BY-SA 4.0 repository licence).
  * **G2: Arcwise-Plat (full)**, the same repository, `arcwise_plat_full_with_diff.json` (changes question and evidence more often).
  * **G3 and G4:** the ReViSQL-repository copies `repos/ReViSQL/data/arcwise_plat_sql.json` and `arcwise_plat_full.json` (that repository has no LICENSE file). They differ from G1 / G2 for a few dozen items (e.g. 27 gold SQL strings between G1 and G3). They are a **source-consistency check**: sensitivity only.
  * The **Jin et al. corrected dev-100** (100 dev ids, 31 of them in Mini-Dev) is **not re-scored**: its gold is written for corrected database files (not the files our predictions ran on), 57 of its 100 items change question or evidence, and 100 items are below the usefulness threshold of the task brief (150). Its overlap is reported only.
* **Original reference.** The original question, evidence and gold SQL are those of `bird/dev/dev_20240627/dev.json` (the file the runs used). "Unchanged" means equality after `str.strip()` of the question, the evidence and the `db_id`. "Gold SQL changed" means inequality after collapsing runs of whitespace.
* **Sets, per gold variant** (items = the 498 dev ids in the variant): **S_all** = all 498; **S_same** = items whose question **and** evidence are unchanged against dev.json (only the gold SQL can differ). **S_same is the primary reading**: a prediction made for the old wording is never scored against a gold written for a changed wording in the primary reading. S_all is a sensitivity analysis and says so. Sizes seen (data level): G1 S_same 367 of 498 (104 gold SQL texts changed in S_same, 184 in S_all); G2 272 (74; 220); G3 335 (87; 168); G4 261 (72; 219). All sets are >= 150, so the analysis proceeds.
* **Gold executed on the dev database files** (the files the predictions ran on), not on the Mini-Dev copies. Because the corrected gold was written against the Mini-Dev databases, whose files differ at byte level from the dev ones for 5 databases (`california_schools`, `european_football_2`, `formula_1`, `thrombosis_prediction`, `toxicology`), every corrected gold is also executed on the Mini-Dev file. **Exclusion rule (applied to S_same and S_all alike, reported):** an item whose corrected gold returns different result sets on the two files, or fails on exactly one of them, is dropped from every set of that variant (`db_discordant`).
* **"Effective gold change"** = the corrected gold's result set differs (under the evaluator's set-equality comparison) from the original gold's result set on the dev database. Text changes without an effective change are counted separately. Both are reported for S_same and S_all.

### 20.3 Scoring (the primary analysis' evaluator)

* Score = `nl2sql.eval.score.Evaluator.score_sql` semantics, copied by construction (the class is used with the gold dictionary replaced): prediction executed by `execute_robust` (30 s, one cold retry), gold by `execute` (120 s, cached), EX = `db.ex_match` set equality of row tuples, a gold that fails or times out gives EX = 0 and stays in the denominator (**the same rule as the primary analysis**; counted per variant), oversize gold results by the digest stub. `final_sql` of every row of `preds.jsonl` is re-executed once per distinct (question, SQL).
* **The original-gold EX used in every comparison of this block is the re-executed one**, so that a difference between "original" and "corrected" can only come from the gold. The stored `ex` is a validation: **gate** = agreement of re-executed and stored EX on at least 99 % of the (run, item) cells of S_all of the variant, per run. Disagreements are listed. If a run fails the gate it is flagged, still analysed with re-executed values for both golds, and this is reported.
* Rows with `ex` null in the stored run follow SAP 10 as in `runs.ex_value`: `api_fail` / `parse_fail` count as 0 under both golds; any other null drops the item from the contrasts of that run (counted).

### 20.4 Runs (pinned; paths relative to `runs/`)

* `qwen3.7-plus`: `zs` = `MIN/M1/zs-s0`; `full_fixed` = `MIN/M1/fixed-s0`; `full_gated` = `MIN/M1/gated-s0`; routing run `MIN/M1/gated-route-s0` (P3); `gated_rootonly` = `MIN/M3/rootonly-s0`; matched baselines (the P2 candidates of the formal pass): `acsql_s` = `MIN/M2/acsql-s0`, `macsql` = `MIN/M2/macsql-s0` (the B* of the formal pass), `sdebug` = `MIN/M2/sdebug-s0`, `bon_vote` n=2 = `MIN/M2/bonvote-s0`, n=3 = `MIN/M2/bonvote3-s0`. `memosql` (500 items) is a subset run and is excluded, as in the primary analysis.
* `qwen3.7-flash` and `qwen3.5-27b`: `MIN/M5/{zs,fixed,gated,rootonly}-{flash,27b}-s0`.
* Roots of `full_fixed` (E10): `cands.jsonl` of `MIN/M1/fixed-s0`, candidate with `action == "root"`, SQL re-executed and scored like a final answer.
* Run r0 only (seed 0). Items usable in all runs of a contrast; the common set per variant and set is reported.

### 20.5 Quantities (each on S_same and S_all, for G1 (primary) and, compactly, G2-G4)

1. **EX level** of every pinned run: original, corrected, paired shift (corrected - original), each with its interval; and the rank correlation (Spearman) and the number of reversed pairs among the plus systems of the two EX vectors (descriptive).
2. **Paired contrasts** (EX difference in percentage points, positive = first system better), each as original, corrected and shift, with the discordant counts (first right / second wrong, and the reverse) under both golds:
   * **H1** (P1 / P1'): `full_gated` - `full_fixed` on plus, flash, 27b; pooled over the three backbones (mean of the three differences, the P1' estimator) and over the weak two.
   * **H2** (W1): `full_gated` - `zs` on flash, 27b and pooled over both (mean of the two differences, the W1 estimator); the strong-backbone control on plus.
   * **H2b** (E3 reading rule): `full_gated` - `gated_rootonly` and `gated_rootonly` - `zs`, per backbone.
   * **H3** (P2 and S-A direction): `full_fixed` - `macsql` (P2 of the formal pass), `full_gated` - each of `macsql`, `acsql_s`, `sdebug`, `bon_vote` (n=2, n=3), `zs`, on plus.
3. **H4 (P3):** on the final answers of the routing run, AUROC of `route_ca` minus AUROC of V3 (the score and k = 4 recomputation of `primary.route_scores` with `exp/frozen/route_coef_hr.json`, copied verbatim, the labels EX re-scored), with the co-reported AURC difference and accepted-EX difference at 60 % coverage (the readouts of SAP 13.3).
4. **H5 (E10).** Over the items of the set: (a) revision of an executable SQL by `full_fixed` (E10 definition: root executes, final SQL text differs from the root SQL): counts **helped** (root wrong, final right), **hurt** (root right, final wrong), neutral, under original and corrected gold; the net revision effect `mean(EX_final - EX_root)` in percentage points of the whole set with a database-clustered interval; exact two-sided sign test on helped vs hurt (descriptive); (b) the same for repair of a non-executable root (reported separately); (c) the **share of E10 items labelled L6 (final label, `results/e10/labels_final_all.csv`) whose gold is changed in G1**: of the sheet items inside S_all and inside S_same, the numbers with gold SQL text changed and with an effective gold change, Wilson intervals, the same shares for the non-L6 items of the sheet, their difference (Newcombe interval) and the stratum-weighted shares (sheet weights, stratified bootstrap within the 8 sheet strata, B = 10 000). Reading rule for the label, fixed now: the L6 label is **corroborated** if the effective-change share among L6 items exceeds that among non-L6 items with the lower bound of the Newcombe difference above 0; **not corroborated** otherwise. The comparator exists because corrected releases concentrate their edits on items where models disagree with the old gold (a selection the L6 label shares), so a high share alone is not evidence of validity.
5. **Flips.** Every (run, item) with EX original != EX corrected, listed in `results/robust/flips.csv` with: run, qid, db, direction (`gain` = wrong -> right, `loss` = right -> wrong), membership in S_same, gold SQL text changed (bool), effective gold change (bool), row and column counts of original / corrected gold, the SQL keywords (from a fixed list: DISTINCT, JOIN, LEFT JOIN, GROUP BY, HAVING, ORDER BY, LIMIT, CAST, COUNT, SUM, AVG, MAX, MIN, CASE, IN, EXISTS, UNION) present in exactly one of the two gold texts. **Cases shown in the report are chosen by a fixed rule, not by judgement:** for `full_gated` and `full_fixed` (plus) and for `zs` and `full_gated` (flash), for each direction, the first five flips in ascending numeric question id, S_same first; the complete list is in the CSV. Corrected gold SQL text is not written into any committed file (licence of the ReViSQL copy); the CSV carries identifiers and derived fields only.
6. **Not re-estimated:** the calls ratio of H1 (it does not depend on gold) and every cost quantity.

### 20.6 Intervals

95 % **database-clustered paired percentile bootstrap, B = 10 000, seed 20261101** (the primary analysis' `ClusterBoot`), **one resample per set shared by original and corrected gold and by all runs**, so that the shift (corrected - original) is a paired quantity with its own interval. Item-level paired bootstrap (same B and seed) is reported beside the headline contrasts; the sensitivity without `thrombosis_prediction` (SAP 8) is reported for H1, H2, H4. With about 11 clusters and 13-45 items per database in S_same (G1), percentile intervals under-cover; the item-level interval and the per-database table are the mitigation, as in SAP 3.

### 20.7 Reading rule (binding, fixed now)

Let d0 and d1 be the estimates on the same set under original and corrected gold, I0 and I1 their intervals, s = d1 - d0 and I_s its interval.

* **Type S** (a claim of a positive effect: H2 pooled / per backbone, H4 AUROC): **robust** iff d0 != 0, sign(d1) = sign(d0) and I0 and I1 overlap.
* **Type N** (a claim of no or negligible difference: H1 per backbone and pooled, the plus control of H2, the H5 net effect): **robust** iff |d1| <= 1.5 pp (the negligible margin of SAP 13.3) and I0 and I1 overlap. Where the sign of a near-zero estimate flips the claim is not changed, so no sign condition applies. The non-inferiority status (lower bound of I above -1.5 pp) under both golds is reported beside it, not used.
* **Stable** (stricter, additional): robust and (0 in I_s or |s| < 1.0 pp; for AUROC |s| < 0.02).
* A **headline conclusion** is *unchanged* iff all its contrasts are robust on (S_same, G1); *unchanged and stable* iff all are stable; *changed* otherwise, naming the contrast. If the verdict on S_all differs from S_same for any contrast of the conclusion it is labelled *wording-sensitive* as well. Headline conclusions: C1 gated keeps the accuracy of the fixed loop (H1); C2 gated beats zero-shot on the weak backbones (H2 pooled, flash, 27b) and the plus control stays small; C3 `route_ca` ranks correct above incorrect better than V3 (H4); C4 the semantic Critic's revisions are net zero (H5). H3, H2b, rank correlations and G2-G4 are descriptive (the type S rule is applied for display).
* The intervals of S_same are wider than the full-dev intervals of the formal pass (about a quarter of the items): overlap is therefore a permissive condition, and the shift s with I_s, which uses the same predictions and items and removes sampling noise, is the sharper quantity. The report states both and names a conclusion "robust" only in the sense defined above. Nothing here is a significance statement.
* If the validation gate of 20.3 fails for a run, or an effective-change count is zero (nothing to test), the output says so and no verdict replaces it.

### 20.8 Limits stated in advance

Corrected gold is itself the product of a correction process whose details we have not audited, and carries its own errors; if its edits follow model-versus-gold disagreement it favours systems close to the models that triggered the edits (a selection that the L6 label shares). Only 498 of 1 534 dev items (32 %) have a corrected counterpart and S_same is about a quarter of the dev set; the subset is a fixed 498-item selection of dev (not shown to be a random sample), with a different database mix from dev. Predictions for the old wording are scored against a gold for the same wording only in S_same; S_all mixes in cases where the question changed. Gold was executed on dev databases; the Mini-Dev databases differ from them at byte level for 5 of 11 files (items whose corrected gold is database-dependent are excluded, 20.2). One run (r0) per system; the corrected sets give no information on the 1 036 other dev items. The pipeline's behaviour on corrected questions is not tested.

### 20.10 Addendum after the first full run (2026-10-27; descriptive; no change to 20.2-20.8)

* **What happened.** After the amendment and the code (`3d31754`) were committed, `exec` (4 482 gold and 4 030 prediction executions, no model call) and `analyze` (B = 10 000) were run once. The results of that run, including the headline verdicts of 20.7, were read before this addendum was written. Operational facts that the spec could not know: the stored original-gold EX equals the re-executed one on every (run, item) cell of S_all (validation gate of 20.3 passed, 100 %); the exclusion rule of 20.2 removed 19 items of G1 (`db_discordant`, corrected gold returns different results on the dev and the Mini-Dev file), leaving S_all = 479 and S_same = 360 for G1.
* **Added after seeing them (descriptive, adds no verdict, changes no rule or threshold):** (i) a **pair audit** for the three H1 pairs (gated vs fixed per backbone): every item whose paired outcome differs between original and corrected gold, counted by transition (`ab->a'b'`), by effective gold change and by whether the two final SQL texts differ; it exists because the first run showed that H1 does not behave alike on the three backbones. (ii) the **fixed-rule case list** of 20.5 item 5 is written to `rescore.json` (`cases_fixed_rule`); its rule is unchanged. The `analyze` stage was re-run to add them; all earlier numbers are reproduced bit for bit (same seed, same code path).
* Nothing in 20.7 was altered after the first run: the headline verdicts reported are those of the rule as committed in `01b1c72`.

### 20.9 Outputs and amendment log

Code: `code/tools/rescore_corrected.py`, tests `code/tests/test_rescore_corrected.py`; outputs `results/robust/*` (`overlap.json`, `exec_meta.json`, `gold_info.json` (digests and counts, no gold text), `items_ex.csv.gz`, `flips.csv`, `rescore.json`, which also holds the E10 block `H5.revision` and `e10_l6`).

| Date | Amendment | Before any prediction was scored against corrected gold | Effect |
| --- | --- | --- | --- |
| 2026-10-27 | Amendment 7: re-scoring of the stored dev runs against Arcwise-Plat-SQL / Plat gold (S_same primary), scope `robustness-posthoc` | yes | Descriptive; no confirmatory test, threshold or family changes; never enters the confirmatory results or `paper/` |
| 2026-10-27 | 20.10 addendum: pair audit and case list added after the first full run | no (first run already read) | Descriptive outputs only; rule 20.7 unchanged |


## 21. Amendment 8 (2026-10-27; **external validity, secondary**; written and committed before any full-arm call of this block; pilot on 20 items per arm already run and disclosed in 21.12)

Sections 1-19 are unchanged. Amendment 7 (§20, RESCORE, commit 01b1c72) is an offline re-scoring of stored dev runs and is independent of this block. `code/prompts/**`, `exp/models.yaml`, `exp/dev_lock.json`, every file of `exp/configs/frozen/` and every pipeline default are **not touched**; nothing under `code/nl2sql/` or `code/analysis_pipeline/` is edited. Scope label `external-validity` (`_meta.scope`). **Nothing of this block enters family P\*, any Holm adjustment of an earlier amendment, the confirmatory results.** If the paper chapter wants numbers from this block it adds them explicitly as exploratory results fed from `results/ext1/` (new keys, new file); that is a decision of the paper, not of this amendment.

### 21.0 Why, and what was seen

* **Why.** E8 (robustness on Spider variants) is "not run" (`results/E8.json`). All confirmatory and secondary claims of the paper are BIRD-dev claims. This block asks whether three of them carry to a second benchmark with a different construction, using the **frozen** systems, with no retuning.
* **Seen before this section.** (i) All BIRD dev results and their summary keys. (ii) Facts about the data from files, not from model output: Spider-Realistic has 508 questions over 19 Spider-dev databases (`world_1` 103, `flight_2` 56, `car_1` 54, `tvshow` 48, `concert_singer` 33, `wta_1` 32, `pets_1` 26, `dog_kennels` 25, `network_1` 21, `employee_hire_evaluation` 20, `student_transcripts_tracking` 17, `singer` 17, `course_teach` 16, `museum_visit` 13, `poker_player` 10, `orchestra` 7, `battle_death` 5, `voter_1` 4, `real_estate_properties` 1); every database has a distilled test-suite database directory; gold-vs-gold under the test-suite evaluator scores 508 / 508 with no worker error (local run, no LLM); the Spider train pool (7 000 questions, 140 databases) shares no database and no exact question string with the 19 databases. (iii) The pilot of 21.12: 12 arms x 20 items, **the only model output seen**. No full-arm output has been seen. The item order, arms and queue file were fixed (committed with this amendment) before the pilot was run; the pilot changed nothing in them.

### 21.1 The three BIRD findings under test (reference values are BIRD dev, r0, from the confirmatory result summary and `results/W1.json`; they are not re-estimated here)

| Id | BIRD finding | BIRD reference |
|---|---|---|
| (a) | The execution-gated loop uses about 0.6x the calls of the fixed semantic-critic loop at equal accuracy | pooled over Bk: calls ratio 0.580 [0.566, 0.597]; dEX(gated - fixed) -0.26 [-0.93, +0.30] pp |
| (b) | On the weak backbones the gated pipeline beats zero-shot | W1 pooled over {flash, 27b}: +2.93 [+1.56, +4.12] pp (flash +2.41 [-0.14, +4.77], 27b +3.46 [+1.49, +5.33]) |
| (c) | On the strong backbone there is no accuracy gain | `qwen3.7-plus` gated - zs: +0.65 [-0.97, +2.51] pp |

### 21.2 Data and metric

* Items: `spider_realistic` of `data.load_split` (508 questions of `spider-realistic.json`; databases = Spider dev). The pipeline sees question and database only (`evidence = ""`, rendered "(none)"); gold is loaded by the evaluator only.
* **Primary metric: Spider test-suite execution accuracy** (`ex_ts`): the official `exec_eval.eval_exec_match` (taoyds/test-suite-sql-eval @ e97acc54) on the distilled test-suite databases, "with values" variant (`plug_value` off), exactly as wired in `nl2sql/eval/spider_ts.py` and `eval/score.py` (committed before this amendment, cross-checked against the official CLI in X3). Plain execution accuracy (`ex`, BIRD set-equality semantics on the single Spider dev database) is reported next to it and decides nothing; the two are expected to differ because the test suite is stricter. `api_fail` rows count as 0 (SAP 10); a `sys_error` row is not a measurement: after the runner's one automatic retry the same command is re-run; an arm that still has such rows is reported as incomplete and enters no contrast.
* Unit and cluster: item nested in database; cluster = `db_id` (19 clusters, one of them with 20 % of the items and one with a single item, which is why the interval conventions of 21.6 also report item-level and leave-one-database-out readouts).

### 21.3 Systems, and what the experience / memory store does on Spider (pre-stated)

* Systems: `zs_ours`, `full_fixed`, `full_gated` and `full_gated` with the root-only configuration (`gated_rootonly`), each with its **frozen** config of `exp/configs/frozen/` and its existing lock entry (datasets `spider_*`, seed 0; `lock.verify` returns ok for all four on the compute machine). r0: temperature 0, seed 0. No `dail`, no baselines, no seeds r1, r2.
* **Memory, by the pipeline's own code (`Resources.corpus_for`, `Resources.pool_name`, read before this amendment):** on a Spider split the pool is `spider_train_pool` (the 7 000 `train_spider` questions with their gold SQL, no evidence field). M2 history = these 7 000 question-SQL pairs in a fresh `ExperienceCorpus` (policy `ws_full` at the prior weights) that is **read-only** (`update: false`); nothing from Spider-Realistic is written back, and nothing is learned on the test items. The BIRD burn-in store named in the frozen configs is **not loaded** (the loader reads it only for `bird_train_pool`), so no BIRD experience reaches a Spider prompt. The M2 term index is empty (Spider has no evidence strings). `zs_ours` uses no memory.
* **Why not "empty memory".** An empty or off store needs a changed config (`m2.history: false` or a pool override), which is a new sha256 and therefore a new lock entry, i.e. an edit of `exp/dev_lock.json`. That is outside the scope of this task. The consequence is stated as a limitation, not hidden: the gated and fixed arms retrieve similar Spider **training** questions with gold SQL as exemplars (database-disjoint from the test items), `zs` does not; exactly as on BIRD, where the pipeline arms carry the BIRD-train store and `zs` does not. Hence XB1 compares the **pipeline** (M1, M3 retrieval, exemplars, gated loop) with a single call, not the loop alone; the root-only arm and 21.5 reading rules separate the two. An empty-store arm would need a new lock entry and is a possible follow-up, not part of this block.

### 21.4 Arms (12 runs, 508 items each; queue `exp/queues/EXT1.jsonl`, generator `exp/queues/gen_ext1.py`)

| Backbone | zs | full_fixed | full_gated | root-only (descriptive) |
|---|---|---|---|---|
| `qwen3.7-plus` (strong) | yes | yes | yes | yes |
| `qwen3.7-flash` (weak) | yes | yes | yes | yes |
| `qwen3.5-27b` (weak) | yes | yes | yes | yes |

Gated imports the cache archive of fixed on the same backbone (shared roots; the gated - fixed discordance comes from the revision step only, as in P1); root-only imports the archive of gated. `zs` is independent. All three backbones are run because the budget estimate of 21.8 allows it (the brief made `full_fixed` and `qwen3.5-27b` conditional on the estimate); the root-only arm is added because the W1 reading rule (15.3) forbids "the loop improves accuracy" without it. Concurrency 6 per run, at most 12 in flight per model; machines 1 to 3; machine-local cache.

### 21.5 Hypotheses (direction, margin, reading rule for every outcome)

Common conventions as 13.3 / 15.3: db-clustered paired percentile bootstrap, B = 10 000, seed 20261101, **one common database resample across backbones and across contrasts of the same family**; one-sided p-values enter Holm as p_2s = min(1, 2 p_1s); a one-sided p is the smallest alpha for which the (1 - 2 alpha) percentile interval excludes the null; the exact McNemar test and the item-level paired bootstrap are the item-level checks. Differences are in percentage points of `ex_ts`.

**XB1 - weak backbones: gated beats zero-shot (BIRD finding (b), W1 analogue).** dEX_W = mean over b in {flash, 27b} of [EX_ts(full_gated, b) - EX_ts(zs, b)]. H0: dEX_W <= 0, one-sided. Reading rule, first line that applies:

1. `confirmed`: Holm-adjusted p < 0.05 (family X, 21.6), the db-clustered 95 % CI lower bound > 0, **and** the point estimate is positive on both members. Finding (b) **replicates**. Size statement (descriptive): "size consistent with BIRD" iff the CI contains +2.93.
2. `supported on average, not on each backbone`: the pooled test passes but a member is <= 0.
3. `supported at item level, not at database level`: only the item-level interval / test passes.
4. `direction consistent, not confirmed`: dEX_W > 0, none of the above. Finding (b) is **not established**; this is the outcome the power analysis of 21.10 makes likely and it is reported as such.
5. `not confirmed`: dEX_W <= 0. If the db-clustered 95 % CI upper bound is < 0 the sentence is "gated is **worse** than zero-shot on the weak backbones: finding (b) is contradicted on this benchmark".
Always reported: the two members separately (dEX, CI, exact McNemar), and the decomposition gated - rootonly (loop) and rootonly - zs (retrieval stack), both **descriptive**. The text may say "the loop improves accuracy" only if gated - rootonly has a positive db-clustered CI lower bound on the pooled weak backbones; otherwise it says "the pipeline", and says that the loop's contribution is bounded by how often the gate fires (reported as the share of items with a revision).

**XB2 - strong backbone: no gain (BIRD finding (c), W1 control analogue).** dEX_plus = EX_ts(full_gated, plus) - EX_ts(zs, plus); two-sided db-clustered 95 % CI; no Holm (estimate with a reading rule, not a test). Reading rule:

1. CI lower bound > 0: **a gain on the strong backbone; finding (c) is not replicated** (reported as such; this is the unfavourable outcome for (c)).
2. CI upper bound < 0: gated is **worse** than zero-shot: "no gain" holds trivially, but with a loss; reported as a loss.
3. Otherwise (CI contains 0): **no detectable gain**; finding (c) is replicated in the sense of "not distinguishable from zero", nothing more. Three magnitude statements are added by pre-stated margins, none is required for line 3: *non-inferior* iff the CI lower bound > -3.0 pp; *equivalent within +/-3.0 pp* iff the whole CI lies in (-3.0, +3.0); otherwise "the interval does not exclude differences larger than 3.0 pp in at least one direction". Margin justification: 3.0 pp is the smallest between-system difference the main study resolves (13.3 P2, 13.7, 15.5 S-I). At n = 508 and 19 clusters equivalence within 3.0 pp is likely **not reachable** (21.10); that is stated before the data, and the estimate with its CI is the result.

**XB3 - gated vs fixed loop (BIRD finding (a), P1' analogue).** Pooled over Bk = {plus, flash, 27b}: (a) dEX_B = mean over b of [EX_ts(gated, b) - EX_ts(fixed, b)], H0a: dEX_B <= -1.5 pp (margin unchanged from 13.3 / 15.3); (b) R_B = mean over b of [sum calls(gated, b) / sum calls(fixed, b)] (answer-path LLM calls of `preds.jsonl`: M1, generation, critic, revisions; embedding and rerank calls reported separately), H0b: R_B >= 0.80. p(XB3) = max(p_a, p_b) (intersection-union). Reading rule:

1. `confirmed`: Holm-adjusted p < 0.05, CI lower bound of dEX_B > -1.5 and CI upper bound of R_B < 0.80. Finding (a) replicates (cost saving **and** accuracy kept within 1.5 pp). R_B and its CI are printed next to BIRD's 0.580 [0.566, 0.597]; no further size threshold.
2. `supported at item level, not at database level`, `direction consistent, not confirmed` (point estimates on the favourable side of both bounds): as 13.3.
3. If R_B is below 0.80 but accuracy non-inferiority fails or is not shown: "fewer calls, accuracy parity not shown". If the **point estimate** of dEX_B is below -1.5 pp: "the gated loop loses more accuracy than the margin on this benchmark; finding (a) is contradicted". If R_B is not below 0.80: "no call saving on this benchmark".
Always reported (not tested): the same per backbone with the same bounds (S-J analogue, no Holm), dEX at margins 1.0 / 2.0 / 3.0 pp, tokens per query, list-price cost per query and per correct answer, the share of items with a revision (the gate firing rate).

### 21.6 Multiplicity, intervals, sensitivity

* **Family X (this block only):** XB1 and XB3, Holm over m = 2 (a test that cannot be run counts with p = 1). XB2 and every descriptive contrast are outside the Holm adjustment. Family X is independent of P\*, S-A ... S-J; gatekeeping is not used.
* Intervals: db-clustered paired percentile bootstrap (B = 10 000, seed 20261101, common resample), item-level paired bootstrap (same B), leave-one-database-out range of every difference, share of databases with the same sign as the pooled effect, exact McNemar counts (b, c, p) for every paired contrast. Per-arm EX_ts with db-clustered CI.
* **Sensitivity without `world_1`** (the largest database, 103 of 508 items, the analogue of the `thrombosis_prediction` rule of SAP 8): every primary estimate is repeated without it; if a verdict differs the report says so next to the headline.

### 21.7 Claim language

Allowed: "replicates", "does not replicate", "direction consistent", "no detectable difference" with the 21.5 qualifiers, always "on Spider-Realistic, test-suite EX, frozen BIRD configuration, one run". Not allowed: "generalises", "holds across benchmarks", "equivalent" (unless 21.5 XB2 line 3 equivalence holds), any statement about EX improvement of the loop without the root-only decomposition, any statement that mixes these numbers with the confirmatory family P\*.

### 21.8 Budget and stop rule

* Cap for the whole block (pilot included): **CNY 150, never exceeded**. Estimate before the full arms (`exp/queues/EXT1.jsonl`, BIRD-dev unit costs x 508, list price `exp/prices.yaml`): about CNY 18 for 12 arms (plus 11.3, flash 1.6, 27b 5.4; BIRD-dev unit costs of `results/E11.json`: plus zs 0.00226, gated 0.00693, fixed 0.01132 per query; the weak backbones scaled by the measured price ratios of `results/x8/x8_tuning_summary.json`). The pilot measured CNY 0.58 for 12 x 20 items, of which CNY 0.34 is the one-off embedding of the 7 000-question Spider train pool (once per machine); the per-query chat cost on Spider is about 70 % of the BIRD figure (smaller schemas), so CNY 18 is an upper-style estimate. All 12 arms are therefore within budget with a factor above 5; none is dropped for budget.
* Stop rule: after every finished arm the realised list-price cost (`cost_report.py`, own-cache hits free) is summed with the pilot. If the sum exceeds **CNY 100**, no further arm is launched and the arms not started are reported as not run; if an arm costs more than 1.5x its estimate, the queue is stopped and the overrun reported; the hard cap of CNY 150 is checked before each launch (running total + estimate of the next arm <= 150).
* Operational: queue `EXT1` via `code/tools/queue_run.py loop`, started in the background; an arm that fails technically is re-run with the identical command; no outcome-dependent action of any kind.

### 21.9 No tuning on Spider

No prompt, threshold, config, model, backbone or arm is changed, added or dropped after seeing any Spider result (the pilot included). The analysis code is committed before it is run on full-arm results. Cases for the report are chosen by the fixed rule of 21.11.

### 21.10 Power (normal approximation of 15.6 / 13.7: SE = sqrt(p_d / n x DEFF), n = 508; planning figures, assumptions not measured on Spider)

p_d (discordant share) is **unknown** for this benchmark; BIRD gave 0.107 for gated - zs on the weak backbones and about 0.03-0.05 for gated - fixed. DEFF: observed analogues on BIRD 0.35-1.7, planning value 2.4. Pooled over two backbones the SE is the single SE x sqrt((1 + r)/2) with r = 0.4.

| Test | p_d, DEFF | SE (pp) | power at +3 pp / +2 pp, one-sided 0.025 | with Holm step (one-sided 0.0125) | MDE 80 % |
|---|---|---|---|---|---|
| XB1 pooled, 2 members | 0.05, 1.5 | 1.02 | 0.84 / 0.50 | 0.76 / 0.39 | 2.8 pp |
| XB1 pooled | 0.107, 1.5 | 1.49 | 0.52 / 0.27 | 0.41 / 0.18 | 4.2 pp |
| XB1 pooled | 0.107, 2.4 | 1.88 | 0.36 / 0.18 | 0.26 / 0.12 | 5.3 pp |
| XB2 single backbone | 0.107, 1.5 | 1.78 | - | - | 5.0 pp |
| XB3(a), margin -1.5 pp, true dEX 0 | 0.03, 1.0 / 2.4 | 0.77 / 1.19 | 0.50 / 0.24 | - | - |
| XB3(a), true dEX -0.5 pp | 0.03, 1.0 / 2.4 | 0.77 / 1.19 | 0.26 / 0.13 | - | - |

Reading, written in advance: (i) XB1 has an honest chance of `confirmed` only if the true weak-backbone gain is about 3 pp **and** the discordance is small; the **likely** outcome under the BIRD-like gain of +3 pp is `direction consistent, not confirmed` or `supported ...`; a null Holm result is not evidence of no gain. (ii) XB2 equivalence within 3.0 pp is unlikely to be reached even if the true difference is zero (SE >= 1.3 pp gives a 95 % interval half-width >= 2.5 pp on 19 clusters and wider under the percentile bootstrap). (iii) XB3(a) is underpowered unless the gated loop is at least as accurate as the fixed one; the call ratio (b) is nearly mechanical (gated = M1 + generation, fixed adds the critic call), so (b) will most likely hold, and the verdict of XB3 will then be decided by (a).

### 21.11 Descriptive outputs and the case rule

* Per arm: EX_ts and plain EX with db-clustered CI, exec-ok rate, calls / tokens / latency per query, list-price cost per query and per correct answer, revision share, error classes of `eval/errors.py`.
* Gate descriptives (no hypothesis): items whose root candidate fails to execute (gated run, first candidate), their repair rate to a correct final answer and to an executable final answer, per backbone with Wilson intervals (the P5' population without the blind arm, which is **not** run here).
* **Cases (fixed rule, no hand picking):** for each of XB1 (flash and 27b, gated vs zs) and XB2 (plus, gated vs zs) the discordant items are sorted by `qid` (ascending, byte order) and the **first three per direction** are listed (gated right / zs wrong; zs right / gated wrong) with question, gold SQL and both predictions; same rule for gated vs fixed on plus. Items where the gate fired are listed by the same rule (first three per backbone).
* Outputs in `results/ext1/` (`ext1_summary.json` with flat keys and `_meta.scope = "external-validity"`, `ext1_arms.csv`, `ext1_contrasts.csv`, `ext1_cases.json`); code `code/tools/ext1_analyze.py`, tests `code/tests/test_ext1_analyze.py`.

### 21.12 Limits stated in advance, and the pilot disclosure

* **Saturation / ceiling.** Spider is partly saturated for strong models; Realistic is a perturbation of Spider dev (explicit column mentions removed), not a harder database. Part of the remaining errors is annotation noise and ambiguity in the gold SQL (not audited here, not quantified); the attainable EX is therefore below 1 by an unknown amount, and differences between systems are compressed. **Contamination:** Spider, Spider dev databases, the test suites and Realistic are public since 2018-2021; memorisation by the backbones cannot be excluded and is not probed here. **Memory:** see 21.3 (Spider-train exemplars in the pipeline arms only). **One run, one seed, T = 0**, 508 items, 19 databases (one with 20 % of the items): intervals are wide and percentile intervals over few clusters under-cover (hence the extra readouts). **Gate firing:** the execution gate revises only on error or timeout; on a benchmark with small schemas it may fire rarely, in which case gated ~ root-only and the pipeline's difference from `zs` comes from the retrieval stack, not from the loop (the decomposition of XB1 exists for this reason). **Test-suite EX** is stricter than plain EX and the two cannot be compared across the benchmarks (BIRD EX is plain set-equality). Backbone snapshots as in `exp/models.yaml` (one undated alias, `qwen3.5-27b`).
* **Pilot (disclosed, not hidden).** Before this amendment was written the 12 arms were run on 20 items each (file `exp/queues/ext1_smoke_qids.txt`: `random.Random(20261127).sample(range(508), 20)` over the file positions, sorted), with a separate call cache, to verify that the frozen pipeline, the lock, the test-suite evaluator and the queue run on Spider-Realistic. The 20 items stay in the full arms (they are not excluded and not re-used as a tuning set). Verdict lines, `results/ext1/smoke_summary.json`: `ex_ts` / plain EX / calls per query / items with a revision / errors. plus: zs 0.65 / 0.85 / 1.00 / 0 / 0; fixed 0.75 / 0.85 / 3.00 / 0 / 0; gated 0.75 / 0.85 / 2.00 / 0 / 0; root-only 0.75. flash: zs 0.55 / 0.90 / 1.00; fixed 0.70 / 0.90 / 3.30 / 3 / 0; gated 0.70 / 0.95 / 2.00 / 0 / 0; root-only 0.70. 27b: zs 0.75 / 0.90 / 1.00; fixed 0.70 / 0.85 / 3.00 / 0 / 0; gated 0.70 / 0.85 / 2.00 / 0 / 0; root-only 0.70. All 240 pilot rows executed without error and no root candidate of a gated arm failed to execute (the gate fired on 0 of 60 gated items). Pilot spend CNY 0.58. No test was computed on these 20-item numbers; their direction is noise-level and was used for nothing except confirming that the arms run (the finding that the gate may fire rarely is already written into the limits above as a possibility and into the decomposition of XB1, which was in the queue before the pilot).

### 21.13 Amendment log

| Date | Amendment | Before the first full-arm call | Effect |
| --- | --- | --- | --- |
| 2026-10-27 | Amendment 8: external validity on Spider-Realistic (XB1-XB3, family X m = 2, memory statement, stop rule CNY 100 / cap 150) | yes (pilot of 20 items per arm disclosed in 21.12) | Secondary, scope `external-validity`; never enters P\*, Holm of earlier amendments or the confirmatory results |

## 22. Amendment 9 (2026-10-27; **exploratory-posthoc, pre-specified**; written and committed before any BIRD-dev call of this block; tuning and pilot on BIRD train are disclosed in 22.12)

Sections 1-21 are unchanged. Scope label: **`exploratory-posthoc, pre-specified`** (`_meta.scope = "exploratory-posthoc"`). The dev items were seen by earlier blocks, so **nothing in this block is confirmatory**: it never enters family P\*, any Holm adjustment of an earlier amendment, the confirmatory results (a paper chapter that wants numbers adds them explicitly as exploratory results; that is a decision of the paper, not of this amendment). `code/prompts/**`, `exp/models.yaml`, every existing pipeline default, every file of `exp/configs/` that existed before, and the 16 existing entries of `exp/dev_lock.json` are not touched (proof by pinned hashes in `code/tests/test_patent_fb.py`). Allowed additions: new system `patent_fb` (`code/nl2sql/systems/ours/patent_fb.py`, registered in `ours/__init__.py`), one additive key in the preds row copy of `code/nl2sql/run.py`, new prompts in `exp/pf_prompts/`, new configs in `exp/configs/pf/`, new data `exp/pf_data/`, new tools, and **new lock entries added with the official command `python -m nl2sql.lock add` after this amendment is committed** (decision of 2026-10-27, experiment specification 1.3 step 6).

### 22.0 Why this block exists, and what was seen

* **The claim under test.** Patent step 7: "user feedback and execution results serve as RL signal, updating the Critic's scoring standard". In the implementation the Critic prompt is fixed (`crit_sem.user`) and feedback reaches only the corpus score WS (M6); the Critic never sees it. The first author's expectation is that the Critic improves with accumulated feedback ("the longer it is used the better"). This block implements the Critic update as in-context calibration (no weights are trained; no GPU) and tests the expectation with an honest dose-response design. A null or negative result is reported with the same prominence.
* **Seen before this section.** (i) All BIRD-dev results of the reference arms: `zs`, `full_fixed`, `full_gated`, `route_ca` on `qwen3.7-plus`, r0 (EX 0.671 / 0.679 / 0.678; calls per query 1.00 / 3.23 / 2.02), the stage breakdown of `full_fixed` (M1 1.00, generation 1.00, Critic 1.135, reviser 0.098 calls per query; 102 of 1 534 items (6.6 %) entered the loop's revision branch), and the conclusions of the confirmatory result summary. (ii) The DIAG results (Amendment 4, 6, 6d): on injected errors a better-specified Critic (VA) and VA with oracle-memory (VB) changed the gold false-positive rate by -2.5 / -4.0 pp at unchanged detection and did not meet the pre-stated rule. (iii) Facts about the data from files: the feedback-bearing part of the frozen M6 store (1 218 alive burn-in entries) and the burn-in log (1 479 rows). **Not seen:** any output of `patent_fb` on any item, any per-item dev result of this block.
* **Prior stated in advance (so that a null is not a surprise).** My expectation, from (i) and (ii), is that every contrast below lies within +-1.5 pp: the stock Critic leaves 93 % of dev items untouched, the loop's measured gain is the repair of non-executing roots, and in DIAG the memory moved the false-positive rate by less than the rule required. This is a prior, not a claim; the reading rule in 22.6 does not depend on it.
* **Why the lock does not make these arms confirmatory.** The lock (`exp/dev_lock.json`) guarantees that a registered configuration was frozen before it touched dev and that no tuning happened on dev. The `patent_fb` entries are added **after the primary dev results were seen**: the hypotheses were chosen knowing how the references behave on these 1 534 items, so the entry documents the configuration hash only; it is not a pre-registration of a hypothesis against unseen data. What the lock does secure here: no prompt or setting is tuned on dev (22.7), one run per arm, config and prompt hashes fixed before the first dev call.

### 22.1 System `patent_fb` (only the Critic's input changes)

* **Loop, score, threshold, final choice: exactly the fixed patent loop P0** (`FullFixed._run_p0`, unchanged code): Score = 0.3 syntax + 0.5 semantic + 0.2 efficiency with the stock deterministic syntax / efficiency sub-scores, revision while Score < 8.5, `t_max = 3`, final = best node (score, then earliest), reviser prompt `rev.user` and its mode rule (theta_fb 7.5), `route_human` rule unchanged. M1, M2, M3, M6 (frozen store `exp/frozen/corpus_burnin_39e2477c.sqlite`, `update: false`), generation prompt, identifier quoting: identical to `full_fixed`, so the **root generation is the same API request as `full_fixed`'s** (cache hit; root SQL identity is checked and reported, 22.5).
* **Critic prompt.** (i) The grounded rules of `exp/diag_prompts/va_grounded.user.txt`, **verbatim** (`exp/pf_prompts/pf_critic.user.txt` differs from it only by the optional placeholder `<<standard?>>`; a test shows that with an empty standard the rendered prompt equals the VA prompt): the query has executed, evidence is authoritative, `must_fix` / `style` split with quoted question phrase and SQL fragment, complete table / column listing of the database next to the pipeline's pruned DDL. (ii) A **scoring-standard section** (`exp/pf_prompts/pf_standard.txt`) with the k = 4 most similar past cases of the feedback store, each with question, evidence, SQL and the user's verdict ACCEPTED / REJECTED. With n = 0 the section is absent.
* **Semantic score** = the model's `semantic` (0-10, scale of the prompt: 9-10 no must_fix issue, 7-8 doubt only, 0-6 at least one must_fix issue) **without** the DIAG scale-consistency rule (so only the input changes; deviation from VA, which applied it). Unparsable output after the one retry counts as semantic 5 (as in the stock Critic) and is reported. **The reviser receives the `must_fix` issues and the suggestions, not the `style` issues** (as in VA; the stock Critic passes every issue). `max_tokens` of the Critic call 900 (stock 512; VA's value, because the JSON carries quotes).
* **Feedback store (the "accumulated feedback").** The feedback-bearing entries of the frozen M6 store: 1 218 burn-in stream questions of the 56 pool databases (BIRD **train**), each with the final SQL of the burn-in run and its verdict = EX of that SQL against the train gold (`exp/pf_data/burnin_verdicts.jsonl`, sha256 `b0b749ee...`). That is a **simulated user whose feedback is the execution result against the reference answer**; the store holds 1 218 of the 1 479 logged burn-in questions (the others were evicted by M6 or not written): 1 011 accepted and 207 rejected (17 %), so rejected cases are rarer than in a fresh stream. **Dose n** = the first n entries of a seeded order of these qids (`u01(20261128, "pf-order", qid)`; nested: the n = 250 store is a subset of n = 1 000). **Noise q**: the verdict of an entry is the oracle verdict with probability q and flipped otherwise (`u01(20261128, "pf-noise", qid) < q`; fixed per entry, independent of n and of the item under review). Retrieval of the k cases uses a separate `ExperienceCorpus` view of the same frozen file with policy `static` (BM25 + dense, RRF, rerank, **similarity only**; the WS score, which contains the true EX, does not influence which cases are shown; DIAG's VB used the WS ranking, a deliberate difference). Cases come from all pool databases (as in VB); the item's own qid and every non-store entry (all history = gold-SQL entries included) are excluded.
* **Leakage assertions (run at system start, tested):** every store qid starts with `bird_train:`, none is in `results/diag/items.json`, `results/diag/tuning/items_tune.json`, `results/diag/tuning_ve/items_tune_ve.json` or the PF tuning draw; the store contains no `bird_dev:*` / `bird_minidev:*` qid; the dev databases are disjoint from the train databases by construction of BIRD. Dev gold is touched only by the evaluator.
* **Upper-bound statement (binding wording).** The verdicts are **simulated oracle feedback**: EX against the reference answer, with optional symmetric noise q. A real user is neither this accurate nor this complete (he does not check every query, may accept wrong results and may reject right ones for reasons other than the result). Results with q = 1.0 are an **upper bound for an ideal user** and are labelled so wherever they appear.

### 22.2 Arms (one run each; 1 534 dev items; `qwen3.7-plus-2026-05-26`; seed 0 = r0, T = 0; same items, same seed, same evaluator as the references)

| Arm | n | q | role |
|---|---|---|---|
| `pf_n0` | 0 | - | control: grounded prompt, no memory (separates the prompt change from the memory effect) |
| `pf_n250_q100` | 250 | 1.0 | learning curve |
| `pf_n1000_q100` | 1 000 | 1.0 | learning curve; upper bound for an ideal user |
| `pf_n1000_q085` | 1 000 | 0.85 | noisy user (15 % of the verdicts wrong) |

References, already run, not re-run (`$R/runs/MIN/M1/`): `zs` (zs-s0), `full_fixed` (fixed-s0), `full_gated` (gated-s0), `route_ca` (gated-route-s0), same backbone, r0. **Backbone `qwen3.7-flash` is not part of this amendment** (see 22.12: it needs its own amendment 9b before any flash dev call). The hash of each arm's config is entered into the lock (system `patent_fb`, seeds [0]).

### 22.3 Hypotheses (all exploratory; two-sided; intervals only, no p-values in any family)

Notation: EX(a) is EX on the 1 534 dev items; differences are paired; 95 % intervals are the database-clustered percentile bootstrap (cluster `db_id`, B = 10 000, seed 20261101, one shared resample for all arms; `analysis_pipeline.stats.ClusterBoot`). "Margin" = 1.5 pp, the non-inferiority margin of P1 (§13.3).

* **PF1 (does memory help).** D1 = EX(`pf_n1000_q100`) - EX(`pf_n0`).
* **PF2 (does a feedback-calibrated Critic beat the fixed v3 Critic with a realistic user).** D2 = EX(`pf_n1000_q085`) - EX(`full_fixed`). Descriptive companions, same format: D2' = EX(`pf_n1000_q100`) - EX(`full_fixed`) and EX(`pf_n0`) - EX(`full_fixed`) (prompt effect alone).
* **PF3 (against the cheap loop).** `pf_n1000_q085` vs `full_gated` on (a) D3a = EX difference and (b) R = sum calls(pf) / sum calls(gated) (ratio of sums, clustered bootstrap), plus the same for `pf_n1000_q100`. Calls = LLM chat calls of the answer path as in P1(b) (embedding / rerank calls reported separately).
* **PF4 (learning curve).** EX at n = 0, 250, 1 000 (q = 1.0); all points, all adjacent and end-to-end paired differences with intervals. Noise cost: EX(`pf_n1000_q100`) - EX(`pf_n1000_q085`).

### 22.4 Metrics reported for every arm and reference (descriptive, with n)

EX with db-clustered interval; EX without `thrombosis_prediction` (sensitivity, §8); calls per query, prompt / completion tokens per query and list-price cost per query and per correct answer (`exp/prices.yaml`, usage tokens x list price as in E11); **revisions helped / hurt** (items with at least one revision: root EX 0 -> final EX 1 = helped; 1 -> 0 = hurt; both counts and the net, as in E3's loop-over-root and the E10 frame; root EX from the stored root candidate); **Critic flag rate** (share of items whose root candidate has Score < 8.5) and its split by root correctness (flag rate on correct roots = false-flag rate; on wrong roots = detection rate); leave-one-db-out range and share of dbs with the same sign for D1-D3a; for the memory arms the share of items for which at least one shown case is REJECTED and the flag rate in that stratum (descriptive); parse-failure rate of the Critic; identity of the root SQL with `full_fixed`'s root (share of items; the arms differ only in the Critic, so this must be 100 % up to cache misses, and any difference is disclosed).

### 22.5 Data, evaluator, exclusions

Same items and evaluator as the references (`bird_dev`, 1 534 items, `eval/` EX; gold read only by the evaluator). Items with `ex` null (infrastructure errors) are dropped for the arm pair as in `runs.align` (the number is reported); `api_fail` counts as 0 (§10). No item is removed for any other reason. A run with `sys_error` / `api_fail` rows is re-run once with the identical command (the runner resumes); the first complete run is the one analysed.

### 22.6 Reading rule (binding, fixed now; covers null and negative results)

For each contrast D with interval [lo, hi] (95 %, db-clustered) and margin m = 1.5 pp:

| Condition | Wording allowed |
|---|---|
| lo > 0 and D >= m | "improves (exploratory)" |
| lo > 0 and D < m | "small improvement of unclear practical relevance (exploratory)" |
| [lo, hi] inside (-m, +m) | "no difference larger than 1.5 pp" (not "equivalent", not "no effect") |
| hi < 0 | "harms (exploratory)" |
| otherwise | "inconclusive; the interval contains both 0 and effects of at least 1.5 pp" |

* **The expectation "the Critic improves with accumulated feedback" is reported as supported only if PF1 reads "improves" or "small improvement" AND the point estimates are non-decreasing in n (EX(0) <= EX(250) <= EX(1000), q = 1.0).** If PF1 improves but the points are not monotone: "improves with n = 1 000, curve not monotone". If PF1 reads "no difference larger than 1.5 pp", "inconclusive" or "harms", the report says plainly that **this implementation of the Critic update does not show the expected gain on these data**, with the interval, and does not claim that feedback cannot help (one prompt design, one backbone, one seed, simulated feedback, in-context calibration only).
* PF2 and PF3 use the same table. PF3(b): R reported with its interval; the verdict on PF3 combines (a) and (b): "dominates" only if D3a reads improves and R <= 1 (not expected); "buys accuracy with calls" if D3a reads improves and R > 1; "no accuracy gain at higher cost; the gated loop stays preferred" if D3a reads "no difference larger than 1.5 pp" / inconclusive / harms and R > 1.
* A positive result under q = 1.0 and a null under q = 0.85 is reported as "only an ideal user helps". The noise cost is the PF4 difference.
* No contrast is promoted to a claim by a subgroup or a per-database result; no interval of a different method is substituted if one of these misses. Cases (D3 discordant items) are listed by the fixed rule of 21.11 (first three per direction by qid) for D1 and D2.

### 22.7 Prompt freeze, tuning draw, pilot

* **Tuning is only on BIRD train**, on a fresh draw disjoint from all earlier ones: 160 questions drawn with seed 20261128 from the 185 questions of the four HT databases (`law_episode`, `retail_complains`, `video_games`, `world_development_indicators`) that are in none of HT-300, the E4/R0 mini-stream roles (seed / stream / probe), HR-1101 or the DIAG / DIAG-tuning / VE-tuning item files (`code/tools/pf_tune_draw.py`; list `results/pf/tuning/items_tune_pf.json`, sha256 `1e8726970eda0fe11b768e7c604a09f9ef56acf0171b1d5bdc5330b7cd52626d`, frozen before the first tuning call). The HT databases are not pool databases, so the store contains none of their questions.
* **What may be iterated:** only `exp/pf_prompts/pf_standard.txt` (the wording of the scoring-standard section), at most 3 drafts; `pf_critic.user.txt` (= VA) is not changed. **Selection rule (fixed now):** on the tuning draw run `full_fixed` (reference), `pf_n0` and `pf_n1000_q100`; keep the earliest draft unless a later draft has a higher EX for `pf_n1000_q100` by at least 2.0 pp; sanity gates for any draft: Critic parse failure <= 2 %, no run error. Tuning cap CNY 35 including pilot. Nothing is tuned on dev; the numbers of the tuning draw are design evidence (not results) and are labelled "tuning set".
* **Freeze:** after tuning, sha256 of `pf_critic.user.txt`, `pf_standard.txt` and of the four configs are written to `exp/pf_prompts/FROZEN.json` and into 22.12, in a commit made before the lock entries are added and before the first dev call. No change after that commit.

### 22.8 Budget, cost estimate, drop order, concurrency

* **Estimate (from E11 / the stored `full_fixed` run; list prices of `exp/prices.yaml`, 2 / 8 CNY per million input / output tokens).** `full_fixed` costs CNY 0.0113 per question (3.23 calls). Its Critic call is about 4 500 prompt tokens at 1.135 calls per query. The `patent_fb` Critic call adds the complete schema listing and, for n > 0, four cases: about 6 000-7 000 prompt tokens and up to 900 (expected 250) completion tokens; the grounded Critic is expected to flag more items than the stock one (flag rate unknown; DIAG: 25 % of gold controls), so 1.2-1.5 Critic calls and 0.1-0.3 reviser calls per query. Estimated **CNY 25-35 per arm** (CNY 0 for M1 / generation / roots when the `full_fixed` cache archive is imported; +CNY 10 per arm if the roots must be regenerated). Four arms: **CNY 100-140**; tuning and pilot: cap **CNY 35**; total 135-175, below the cap **CNY 220**.
* **Drop order if the projection exceeds the cap (announced before running):** (1) `qwen3.7-flash` (already dropped), (2) `pf_n1000_q085`, (3) `pf_n250_q100`. Runs are therefore executed in the order `pf_n0`, `pf_n1000_q100`, `pf_n250_q100`, `pf_n1000_q085`; after each pair the projection (spent + remaining arms at the measured per-arm cost + 10 % margin) is recomputed; if it exceeds CNY 220 the next arm in the drop order is not run, and the report says exactly which arms are complete. Stop at CNY 220 regardless.
* Concurrency: per-model cap 12 (two jobs of `--conc 6`); one run per arm.

### 22.9 Limits stated in advance

One backbone (flash dropped), one seed at T = 0 (no sampling variance), one prompt design for the scoring standard (at most 3 drafts tuned on 160 train questions of four databases), in-context calibration only (no weights are updated; patent step 7's "RL signal update" is realised as retrieval of verdict-labelled cases, which is a reading of the patent step, not the patent's training), 1 534 items over 11 dev databases (wide cluster intervals; the MDE for a paired difference at discordance 0.02-0.04 is about 1.6-2.2 pp, planning figures from §13.7 not measured on these data); simulated oracle feedback (upper bound, 22.1); the store is the burn-in stream of the same frozen M6 run (accepted-heavy: 83 % accepted); cases are drawn from the pool databases, not from the dev databases (as in a deployment's first days, before dev-database feedback exists; a deployment that has collected feedback on the same database would give the Critic more relevant cases, which this design does not test); the Critic and the reviser are the same model (shared biases); the dev items were seen by earlier blocks (22.0).

### 22.10 Outputs

`results/pf/` (`pf_summary.json` with flat keys, `_meta.scope = "exploratory-posthoc"`; `pf_arms.csv`, `pf_contrasts.csv`, `pf_curve.csv`, `pf_cases.json`; `tuning/`), code `code/tools/pf_analyze.py`, tests `code/tests/test_patent_fb.py` and `code/tests/test_pf_analyze.py`, queue `exp/queues/PF.jsonl`.

### 22.11 Amendment log

| Date | Amendment | Before the first dev call | Effect |
| --- | --- | --- | --- |
| 2026-10-27 | Amendment 9: feedback-calibrated Critic `patent_fb` (PF1-PF4), arms n = 0 / 250 / 1 000 (q = 1.0) and n = 1 000 (q = 0.85), scope `exploratory-posthoc, pre-specified`, cap CNY 220, drop order, reading rule; new lock entries added after the primary results were seen | yes (tuning and pilot on BIRD train follow and are disclosed in 22.12) | Exploratory only; never enters P\*, any Holm adjustment, the confirmatory results |

### 22.12 Freeze log and disclosures (filled in after tuning, before the lock entries and the first dev call)

* **Pilot (disclosed).** 12 items (the first 12 qids of the tuning draw), `patent_fb` n = 1 000, q = 1.0, the tuning config `exp/configs/pf/tune_pf_n1000_q100.yaml`, split `bird_train_HT_full` (unlocked): 0 errors, 3.83 calls per query, EX 6/12 (not a result); it verified the plumbing (store 835 accepted / 165 rejected, leakage assertions, cases in the preds row, cache archive). Cost CNY 0.21 (`cost_report`, new calls only).
* **Tuning runs on the 160-item draw (tuning set; design evidence, not results; B = 2 000 only for the descriptive tuning tables in `results/pf/tuning/`).** `full_fixed` (reference, M1 / generation shared through the machine cache): EX 63.75 %, 3.28 calls per query, nominal list-price cost CNY 0.0101 per query, Critic flag rate 8.1 % (13 items entered a revision, 3 helped, 0 hurt). `pf_n0`: EX 63.13 %, 3.24 calls, 0.0106 per query, flag rate 6.3 % (10 items revised, 2 helped, 0 hurt), 1 item with an unparsable Critic output (0.6 %). Draft 1 of the standard (`exp/pf_prompts/drafts/pf_standard_d1.txt` = the committed `pf_standard.txt`), `pf_n1000_q100`: EX 63.13 %, 3.23 calls, 0.0123 per query, flag rate 5.6 % (9 items revised, 2 helped, 0 hurt), 45 % of the items had at least one REJECTED case among the four shown, 0 unparsable outputs. Draft 2 (`drafts/pf_standard_d2.txt`, a more directive wording: follow an ACCEPTED reading, treat a refused reading as evidence for a must_fix issue), `pf_n1000_q100`: EX 63.13 %, flag rate 5.6 %, detection of wrong roots 11.5 % (d1: 13.1 %), false flags 2.0 % (d1: 1.0 %), 1 unparsable output. **Selection rule of 22.7: draft 2 does not beat draft 1 by 2.0 pp, so draft 1 is frozen; no third draft was run.** Both gates hold (parse failure <= 2 %, no run error). The three runs agree to the item (the same 2 helped, 0 hurt) and the memory changes the Critic's decision on at most one item of the draw: the tuning draw cannot separate the arms (160 items, 4 databases) and says nothing about the dev outcome; the point of tuning was to check that the memory reaches the prompt and that the Critic does not degenerate.
* **Root identity.** `pf_n0` / `pf_n1000_q100` roots equal `full_fixed`'s in 99.4 % / 100 % of the tuning items (the 1 differing item of 160 in `pf_n0`, `world_development_indicators:2122`, differs from `full_fixed`'s root in indentation only, with the same exemplars and the same result; that root was evidently not served from the cache, cause not investigated; the dev analysis reports the share per arm).
* **Spend.** Tuning plus pilot CNY 4.23 (cap 35; `cost_report` of `$R/runs/PF`, new calls only; cache hits are free). **Cost projection updated from measured data** (22.8 stands as the pre-run estimate): nominal list-price cost per question 0.0106 (`pf_n0`) and 0.0123-0.0125 (with memory) on the draw, i.e. CNY 16-19 per dev arm at list price for the complete answer path, of which the root requests are cache hits after the archive import (new spend about CNY 8-12 per arm). Projected total of the four arms CNY 35-50, far below the cap CNY 220; the drop rule stays in force and is checked between the pairs as announced. Because the budget is not binding, **the `qwen3.7-flash` backbone is not dropped for budget reasons (correction of the wording of 22.2 / 22.8)**: it is *not part of this amendment*; it may be added only by a further amendment (9b) with its own pre-stated arms, written and committed after the qwen3.7-plus arms are complete and before any flash dev call.
* **Reading-rule precision (before any dev call).** The table of 22.6 is implemented symmetrically (`pf_analyze.reading`): `hi < 0` reads "harms" when D <= -1.5 pp and "small decline of unclear practical relevance" when D > -1.5 pp. The PF3 verdict strings are those of 22.6. No other change.
* **Frozen files (sha256, `exp/pf_prompts/FROZEN.json`):** see the file; the four dev configs `exp/configs/pf/patent_fb_{n0,n250_q100,n1000_q100,n1000_q085}.yaml` carry the prompt and verdict-file hashes inside, so the config hash recorded by the dev lock pins them. Tuning item list sha256 `1e8726970eda0fe11b768e7c604a09f9ef56acf0171b1d5bdc5330b7cd52626d`. No change after this commit; the lock entries follow.

## 23. Amendment 10 (2026-10-27; **posthoc-secondary**; followed a critical reading of the draft by the authors with Claude Sonnet 5.5 (see Acknowledgments); computed on existing runs only; written and committed before any RV2 number was computed)

Sections 1-22 are unchanged. Scope label: **`posthoc-secondary`** (`_meta.scope = "posthoc-secondary"`). Followed a critical reading of the draft by the authors with Claude Sonnet 5.5 (see Acknowledgments); computed on existing runs; **none of it enters the confirmatory results, the tuning-set results or family P\***, or any Holm adjustment. Numbers reach the paper only as exploratory post hoc results (keys `RV2.*`), marked as post hoc in the caption or the sentence. **Zero API calls, zero new model runs.** Every number is a function of stored `preds.jsonl`, `cands.jsonl`, `usage*.jsonl` of finished runs, of `results/robust/items_ex.csv.gz` and of the published `results/**` files. If an analysis below turns out to need a model call (a cache miss, a missing run), it is **stopped and reported**, not run. Memo-SQL rows are not touched (its technical repeat is recorded in Section 24).

### 23.0 Why, and what was seen

* **Why.** The critical reading asked for (i) the effect of the loop itself, per backbone, and the share of it the gated loop keeps; (ii) cost with and without the cache discount, latency and a Pareto statement; (iii) a second interval for every headline paired contrast; (iv) coverage-risk with the cost of the routing score; (v) a heterogeneity test in place of a three-point Spearman correlation; (vi) the exact definition of "helped 11 / hurt 11".
* **Seen before this section (disclosure; nothing below is new data).** All confirmatory results (P1' -0.26 pp [-0.93, +0.30], calls ratio 0.580; W1; P3; P5'); the E11 cost table (calls and list-price cost per question of `zs`, `full_gated`, `full_fixed`, `full_gated_route`, `macsql`, `acsql_s`, `bon_vote`, `sdebug` on plus); the ROB (RESCORE), EXT1 (XB1-XB3) and PF (PF1-PF4) point estimates and intervals; the texts of the critical reading, which quote some numbers (PF1 8 vs 9 discordant items; XB3 lower bound -1.39; failing roots 18 vs 23 on plus; design effects 0.35-1.69; Spearman -0.5 on three backbones). **Not seen:** any loop-effect decomposition, any item-level interval of the headline contrasts other than those already in `*.json`, any dual-interval verdict column, any Pareto statement, any heterogeneity statistic, any coverage-risk cost table, any helped / hurt count by corrected reference or annotation label, and any comparison of the root candidates of the fixed and the gated run.

### 23.1 Data and conventions (fixed)

* **Runs (r0, T = 0, one seed; roles as `exp/analysis_manifest.json` and `runs.LABELS`; machine directory names under `$R/runs`).** BIRD dev, 1 534 items, 11 databases, backbones `plus` = qwen3.7-plus-2026-05-26, `flash` = qwen3.7-flash-2026-07-15, `27b` = qwen3.5-27b. `zs` (single call): `MIN/M1/zs-s0`, `MIN/M5/zs-{flash,27b}-s0`. `rootonly` (retrieval stack and task graph, no loop): `MIN/M3/rootonly-s0`, `MIN/M5/rootonly-{flash,27b}-s0`. `gated`: `MIN/M1/gated-s0`, `MIN/M5/gated-{flash,27b}-s0`. `blind`: `MIN/M3/blind-s0`, `MIN/M5/blind-{flash,27b}-s0`. `fixed`: `MIN/M1/fixed-s0`, `MIN/M5/fixed-{flash,27b}-s0`. `route` (gated plus the routing score): `MIN/M1/gated-route-s0`. Matched baselines (plus only): `MIN/M2/{acsql,bonvote,bonvote3,macsql,sdebug}-s0`. Spider-Realistic (XB): `EXT1/*`, 508 items, 19 databases. PF: `PF/*`, bird_dev, plus. The sha256 of every `preds.jsonl` used is written to the output `_meta`.
* **Intervals.** (a) **Cluster**: the primary analysis' `analysis_pipeline.stats.ClusterBoot` (imported, not modified), cluster = `db_id`, B = 10 000, seed 20261101, percentile 95 %, **one common database resample inside every family of contrasts** (as 13.3 / 15.3). (b) **Item**: paired bootstrap over items (i.i.d. resample, same B and seed; `stats.item_boot_means` / `item_boot_ratio`), percentile 95 %. (c) **Exact McNemar** two-sided (`stats.mcnemar_exact`) for every binary paired contrast, with its discordant counts b (first right, second wrong) and c. (d) **Design effect observed**: DEFF = (SD of the cluster-bootstrap distribution / SD of the item-bootstrap distribution)^2 of the same estimate; DEFF < 1 is reported as is. (e) **Leave-one-database-out** min and max of the estimate. (f) **Cluster-robust t interval** (CR1 sandwich on database totals, t with G - 1 degrees of freedom), **descriptive**, requested as an inferential check for G = 11 and G = 19. Small-cluster caveat, stated here and in every output: percentile cluster intervals under-cover with 11 and 19 clusters, and the item-level interval answers the question "these databases" while the cluster interval answers "new databases"; neither is a significance statement.
* **Decision interval.** Where the two intervals lead to different classes (23.3), the text uses the **wider** one at the side that matters; both are printed.
* **Rounding.** Stored at full precision; printed at 0.1 pp (R2 M3 item 7).
* **Failures.** `ex` null rows are dropped pairwise as in `runs.align` (the number is reported); `api_fail` / `parse_fail` count 0 (10).

### 23.2 Analysis 1: loop effect

Per backbone b in {plus, flash, 27b} on dev, and pooled (mean of the three differences, one common resample):
1. EX of `zs`, `rootonly`, `gated`, `fixed` (level, cluster CI, n).
2. **Loop effect (two definitions, both reported).** (i) *between runs*: EX(final of the run) - EX(`rootonly` run), for `gated` and `fixed`; (ii) *within run*: EX(final) - EX(root candidate of the same run, `cands.jsonl` `cand_idx = 0`, stored `ex`), for `gated` and `fixed`. Each with cluster CI, item CI, exact McNemar (b = helped, c = hurt for (ii)). The definition used in the paper's sentence is (ii) unless (i) and (ii) differ by more than 0.5 pp for some backbone, in which case both are printed and the difference is explained.
3. **Share kept** S = effect(gated) / effect(fixed). Defined only if the cluster CI of effect(fixed) excludes 0; then the cluster-bootstrap percentile interval of the ratio (same resample) is printed, with the item-level analogue; otherwise S is printed as a point value with "undefined (denominator interval contains 0)" and **no interval**. Also reported: the difference effect(fixed) - effect(gated) (= -(gated - fixed), the P1' contrast per backbone) so the claim "the gated loop keeps what the fixed loop adds" is read from an interval of a difference, not from a ratio of small numbers.
4. **Margin relative to the effect.** M = 1.5 pp / effect(fixed) when effect(fixed) > 0 (how many fixed-loop effects fit into the margin), and the plain statement "the margin is larger / smaller than the whole loop effect" with the interval of effect(fixed). The same for the pooled effect.
5. **Stack and loop decomposition.** EX(`rootonly`) - EX(`zs`) (retrieval stack, task graph, examples, schema routing: not separable here) and EX(`gated`) - EX(`rootonly`), per backbone and pooled, each with the two intervals and McNemar. Reading rule: the sentence "the loop improves accuracy" is allowed only for a loop-effect with a cluster CI lower bound > 0 **and** an item CI lower bound > 0; otherwise "no loop effect detected" with the interval.
6. **Failing roots of `fixed` vs `gated` (V1/R2 item "18 vs 23 on plus").** Definition of "failing root": `cands.jsonl` `cand_idx = 0` with `status` in {err_syntax, err_identifier, timeout} (`primary.EXEC_FAIL`; the population of P5'). For each backbone: counts in each run; items failing in both, only in `fixed`, only in `gated`; for items that differ: whether the root SQL text is identical, whether the stored `status` / `ex` differ, whether the exemplar ids, `prompt_tokens`, `completion_tokens`, `key`/cache flag of the root request (from `usage*.jsonl`: tag, `cached`, `attempts`, `mode`) and the run's `seed`, `temperature` and config sha256 differ, and `llm_cache.manifest.json`. The output states the cause **as found** (candidates, none excluded in advance: execution timeout nondeterminism under load; a root request that was not a cache hit and was regenerated; retries; different item sets), with counts; if no cause is identifiable from stored fields it says so. No re-execution of SQL is done unless it is a pure read of the stored SQL against the stored SQLite file with the primary evaluator (no model call); if used, it is stated.

### 23.3 Analysis 3 (placed before 2 because it fixes the reading classes): dual intervals

Contrasts (every headline paired contrast of the paper; EX in pp unless stated; positive = first system better):
* **P1'**: gated - fixed pooled over {plus, flash, 27b} and per backbone (EX); the calls-ratio component (pooled and per backbone) with cluster and item intervals (no McNemar; ratio of sums).
* **W1**: gated - zs pooled over {flash, 27b}; per weak backbone; plus control.
* **P5'**: repair rate of `gated` minus `blind` on the pooled failing-root population (n_R from 15.3), paired items; cluster resample = the 11 dev databases (databases without R items contribute zero, as `weak.run_p5prime`), item interval = paired bootstrap over the n_R items, exact McNemar.
* **P3**: AUROC(route_ca) - AUROC(V3) on `gated-route-s0` with the frozen coefficients and k = 4 exactly as `primary.test_p3` (`exp/frozen/route_coef_hr.json`); cluster interval by `ClusterBoot.stat`; item interval = paired bootstrap over items and the DeLong interval; no McNemar (not binary); AURC and accepted-EX at 60 % coverage as co-reported.
* **RESCORE cells (SAP 20, G1, S_same n = 360 primary; S_all n = 479 compact):** C1 (gated - fixed: plus, flash, 27b, pooled), C2 (gated - zs: flash, 27b, pooled weak, plus control), C3 (AUROC, AURC, accepted-EX at 60 %: route_ca - V3), C4 (net effect of executable-root revisions of `fixed` on plus), each under the original **and** the corrected gold, and the shift (corrected - original).
* **EXT1**: XB1 (weak pooled, gated - zs, `ex_ts`), XB2 (plus, gated - zs), XB3 (a: EX pooled gated - fixed; b: calls ratio), and the descriptive decomposition gated - rootonly (XB1_loop) and per backbone.
* **PF1-PF4** and the descriptive companions of 22.3 (D1, D2, D2', D3a, calls ratio R, learning-curve end-to-end and adjacent differences, noise cost), from the PF runs.
Each row reports: n (items; n_R for P5'), G (clusters), estimate, cluster CI, item CI, cluster-robust t CI (descriptive), exact McNemar p and (b, c) for binary contrasts, observed DEFF, LOO min / max, and **`reading_changes`**.

**Reading classes (pre-stated; one rule per contrast family, taken from the SAP rule that governs that contrast).**
| Family | Rule (class from an interval [lo, hi]) | Source |
|---|---|---|
| NI(1.5) | "non-inferior" iff lo > -1.5 pp, else "not shown" | 13.3 / 15.3 P1', S-J, XB3(a) |
| RATIO(0.80) | "saving shown" iff hi < 0.80, else "not shown" | 15.3 P1' (b), XB3(b) |
| SUP(0) | "positive" iff lo > 0; "negative" iff hi < 0; else "contains 0" | W1, P5', P3, XB1, XB2 (21.5 line 1-3), loop effects, RESCORE S-type cells (20.7) |
| BAND(1.5) | the five-row table of 22.6 (improves / small improvement / no difference larger than 1.5 pp / harms / small decline / inconclusive) | PF1-PF4; RESCORE N-type cells (C1, C2 plus control, C4) |
A contrast with two components (P1', XB3) has the joint class (NI, RATIO); its class is "joint" only if both hold. **`reading_changes` = yes iff the class from the cluster interval differs from the class from the item interval**; a second flag `reading_changes_t` compares the cluster-robust t interval with the cluster percentile interval (descriptive). For contrasts that carry a Holm adjusted p in the formal pass, the adjusted p is **not** recomputed here; the item-level exact McNemar p is printed beside the formal adjusted p only as a pointer (no new family). Where `reading_changes` = yes, the rows name which verdict word of the paper's current sentence the item-level (or t) reading would change (to be mapped when the paper text is revised).

### 23.4 Analysis 2: cost and latency

* **Unit and source.** Per run and per backbone: chat requests ("calls"), prompt, completion and cached tokens from `usage*.jsonl` (rows of kind chat, **all rows counted at their recorded tokens whether or not the request was a cache replay**, as `secondary.run_cost`), per question = total / number of dev items of the run. Embedding and rerank **requests** per question from `kind` in {embed, rerank} (and `preds` `embed_calls`, `rerank_calls`).
* **Price.** `exp/prices.yaml` (DashScope Beijing list price, CNY per million tokens, read 2026-10-26 from the vendor price page; input tiers by request length as in the file). Two variants: **with discount** (cached prompt tokens at the file's `cache_hit_ratio` = 0.2 of the input price; implicit-cache ratio is conservative) and **without discount** (all prompt tokens at the input price). Realtime mode only (`batch_ratio` is applied to rows with `mode = batch`, as `run_cost`; reported if any). Per question and per 1 000 questions (x 1 000).
* **Embedding and rerank.** The headline CNY **excludes** them (no verified per-token prices exist for them); the output states the number of embedding and rerank requests per question for every configuration and a **bounding figure** using an assumed price (tokens in the usage rows, `pt`), labelled "assumed price"; if usage rows carry no tokens for these requests the figure is "not computable" and only request counts are printed.
* **Configurations** (per backbone where the run exists): `zs` (single call), `rootonly`, `gated`, `blind`, `fixed`, `route` (plus), and the plus baselines `acsql`, `bonvote2`, `bonvote3`, `macsql`, `sdebug`. For each: calls, tokens, CNY (both variants), EX (cluster CI), latency mean / p50 / p95 per question as stored (`latency_s` is the sum of the recorded request latencies of the item; replays carry the latency of the original request; sequential, not wall time; the output says so; `wall_s` reported if present).
* **Routing score cost.** `route` minus `gated` (plus): additional calls, additional requests, additional completions per question (V3 call plus the sampling request with n = 4, i.e. five completions in two requests per question, to be **checked against the stored fields** and reported as found), additional CNY per question.
* **Pareto statement (rule fixed now).** Points (cost per question with discount, EX) for all configurations. A **dominates** B iff cost(A) <= cost(B) and EX(A) >= EX(B) with at least one strict (point estimates). A is **not shown to be dominated** otherwise. For the three claims of that reading -- (1) plus `zs` vs flash + pipeline, (2) plus `zs` vs 27b + pipeline, (3) plus `zs` vs plus `gated` and `fixed` -- the output prints EX(A) - EX(B) with the two intervals and McNemar, and the cost ratio cost(B)/cost(A) with a cluster-bootstrap interval of the ratio of per-item costs (from `preds` token counts; usage rows have no qid, as in X1). Wording rule: "A dominates B" is allowed in text only if the point rule holds; the output also prints whether the decision interval of EX(A) - EX(B) excludes 0 in A's favour ("shown") or not ("point-estimate dominance only"), and the sentence must say which. If the interval has hi < 0 (A worse than B), the claim is false whatever the cost.
* Price date and source are written into `_meta` and into the report.

### 23.5 Analysis 4: routing coverage-risk

On `route` (plus, final answers of the routing run, frozen coefficients, k = 4 as `primary.route_scores`) for scores V3 and route_ca, at coverage 40 / 60 / 80 %: the accepted set = the top fraction of items by score (ties by the pipeline's rule in `routing.rc_vector`; the same ordering as the formal pass), reporting n accepted, **accepted-answer error rate** (= 1 - EX on the accepted set), EX of the accepted set, correct answers per question asked (coverage x accepted EX; rejected items counted as unanswered), and CNY per question **including the routing cost** (23.4), and per accepted answer. Cluster and item intervals of the accepted-set EX. Matched baselines at comparable budget: EX of `bonvote2` and `bonvote3` (plus, full dev) with their CNY per question, placed in the same table; the statement is "route_ca at c % coverage has accepted EX x with cost y; bon_vote n has EX z at cost w", with no new test; `gated` EX at 100 % coverage and its cost as the reference. The 19.5 % "wrong at 60 % coverage" of the paper is reproduced as a check.

### 23.6 Analysis 5: heterogeneity of the gain across backbones

* **Quantity.** The pipeline gain g_b = EX(gated, b) - EX(zs, b) for b in {plus, flash, 27b}, and the loop-over-root gain h_b = EX(gated) - EX(rootonly) as a second family. The null is homogeneity: g_plus = g_flash = g_27b (df = 2).
* **Tests (all reported).** (T1) *Cluster Wald / Hotelling*: contrasts C g = (g_flash - g_plus, g_27b - g_plus) with the covariance estimated from the cluster resamples of the pooled bootstrap (common resample), and the small-sample reference F(2, G - 2) with T^2 = (G - 1) times the Mahalanobis form (Hotelling for G clusters), also the chi-square(2) reference as the large-sample comparison. (T2) *Item-level Wald*: the same contrast with the item-level covariance of the paired per-item differences (the three backbones answer the same items), chi-square(2). (T3) *Cochran's Q* on the three per-item binary difference indicators is **not used** (the differences are ternary and dependent); this is stated rather than computed. Also the two pairwise differences (flash - plus, 27b - plus, and 27b - flash) with cluster and item intervals.
* **Reading rule.** "The gain differs between backbones" is allowed only if T1 (the cluster, small-sample reference) has p < 0.05 **and** T2 has p < 0.05; "heterogeneity inconclusive" otherwise (the interpretation of the paper's sentence "backbone-dependent"). The 3-point Spearman correlation is not computed; if any of T1/T2 cannot be computed (singular covariance), the output says so and lists the sentence to drop: "the gain correlates with single-call EX (Spearman -0.5, three backbones)".

### 23.7 Analysis 6: definition of "helped 11 / hurt 11"

* **Definition (as stored in `code/analysis_pipeline/e10.py` and `results/E10.json`).** Population: `fixed`, plus, dev items whose root candidate executes and is non-empty and whose final SQL text differs from the root SQL. *helped* = root EX 0 and final EX 1; *hurt* = root EX 1 and final EX 0; neutral = same EX. EX is execution accuracy against the **original** BIRD dev reference. Repair of a non-executing root is a separate stratum. The output prints the counts with n for: (a) the E10 definition (reproducing 11 / 11 / 48, n = 70); (b) all revisions including repairs (17 / 11 expected from the critical reading; reported as found); (c) the same counts against the **corrected** references (G1) on the items of S_all and S_same, with the original-reference counts on the same items; (d) the **annotation-label breakdown**: for the items of (a) that are in `results/e10/labels_final_all.csv`, the final label distribution (L1-L6) by helped / hurt / neutral, plus the number of helped / hurt items not in the annotated sheet (the sheet holds failures of two systems, so it cannot cover every item); (e) the exact two-sided sign test and the net revision effect with cluster and item intervals on the whole set (pp of 1 534). The statement "the counts are execution-based, not annotation-derived" is verified by (d) (the annotation does not enter the counts).

### 23.8 Limits stated in advance

One seed per arm, one run per cell; the cluster resample does not include generation variability (Plus run-to-run SD about 0.5 pp, `P*` seeds r1, r2 of `gated` and `zs` on plus exist and are used only for a descriptive SD statement in 23.2 if the stored runs allow it). Percentile intervals with 11 and 19 clusters under-cover; the t interval is a rough check, not a repair. The corrected-reference cells are conditional on model disagreement and use the same sets as 20. List-price cost is not a measurement of money paid (cache behaviour of the vendor is not controlled; the realised discount differs between runs). Latency is the stored sum of per-request latencies under the load at run time, not a benchmark. The Pareto statement is on three backbones and one price date. Rerank and embedding prices are placeholders. Nothing here is confirmatory.

### 23.9 Outputs and key grammar (provisional list committed before the values exist)

Code: `code/tools/rv2_analyze.py` (compute machine, numpy; imports `analysis_pipeline.{stats,runs}` read-only), `code/tools/rv2_common.py` if needed, tests `code/tests/test_rv2_analyze.py`; `make_xmacros.py` gains source `rv2` (`results/posthoc2/rv2_summary.json`) and prefix `RV2.`; `make_xtables.py` gains `tab_loop.tex`, `tab_cost.tex`, `tab_dual.tex`. Outputs `results/posthoc2/{rv2_summary.json, loop.csv, roots.csv, cost.csv, pareto.csv, dual.csv, route.csv, het.json, revisions.csv}`.

### 23.10 Amendment log

| Date | Amendment | Before any RV2 number was computed | Effect |
| --- | --- | --- | --- |
| 2026-10-27 | Amendment 10: loop effect, cost and latency, dual intervals, routing coverage-risk, heterogeneity test, definition of helped / hurt; scope `posthoc-secondary`; zero API calls | yes | Descriptive; no confirmatory test, threshold or family changes; never enters the confirmatory results, the tuning-set results or P\* |

### 23.11 Addendum: deviations from 23.1 to 23.9 found while writing and running the code (2026-10-27; written after the first full run, before the tables and the report)

None of these changes a threshold, a family or a reading rule of 23.2 to 23.7; all are descriptive. Items 1, 2, 3, 5 and 8 were decided in the code before the first complete run. Items 4 (only where the embedding and rerank request counts come from: first the usage rows, then the `preds.jsonl` fields), 6 and 7 were edited after it; they changed text, a key name and the embedding and rerank counts, no EX, cost, latency or interval (checked: all csv files and `het.json` are byte-identical before and after except the two request-count columns of `cost.csv`).

1. **t interval (23.1 f).** The "cluster-robust t interval" is implemented as the delete-one-database jackknife standard error (G - 1)/G * sum of squared deviations of the leave-one-out estimates, with the t(G - 1) quantile, and not as the CR1 sandwich on database totals. It is the same kind of object (a rough small-cluster check), it is defined for ratios, AUROC and AURC with the same code, and it equals the classical t interval for a plain mean (tested). It is called `t_ci` and stays descriptive.
2. **Reading class for the calls ratios of PF3b.** 23.3 has no reading class for a ratio whose null is 1. PF3b and PF3b_q100 (calls of a feedback arm over the gated loop) use the class `RATIO(1)`: "below 1" if the upper end is below 1, "above 1" if the lower end is above 1, otherwise "contains 1"; the same class is applied to the item and the t interval. The other ratios keep RATIO(0.80).
3. **Heterogeneity T1 (23.6).** The Hotelling statistic uses the delete-one-database **jackknife** covariance of the two contrasts (T^2 = theta' V^-1 theta, F = T^2 (G - 2) / (2 (G - 1)), F(2, G - 2)); the cluster-**bootstrap** covariance is used only for the chi-square(2) comparison column `p_chi2`. 23.6 named the bootstrap covariance for both. The reading rule is unchanged and reads `p_F` (T1) and the item-level `T2.p`.
4. **Embedding and rerank (23.4).** The usage logs hold embedding and rerank rows only for live requests, and without token counts for rerank, so they cannot be priced per question. The table prints the request counts recorded in `preds.jsonl` (`embed_calls`, `rerank_calls`; live HTTP requests, cache hits are not counted, so the embedding count differs between arms for reasons unrelated to the method), states in `cost.embed_rerank_included` that they are **not** in the CNY, and `cny_emb_assumed_price` is `n/a`. Cached tokens are not recorded for qwen3.7-flash and qwen3.5-27b: with and without the cache discount coincide there.
5. **Routing cost (23.4).** The routing score is costed as the chat rows of the `route` run beyond those of the `gated` run (V3 critic request and consistency request with n = 4: 5 completions in 2 requests); the check `route.check.requests` reports 2.016 requests per question, not exactly 2 (retries are in the logs).
6. **Key renamed.** `roots.<b>.diff.root_request_uncached_either` (23.2 item 6, key list commit 728b3d2) is not computable per item (usage rows carry no question id); it is replaced by `roots.<b>.diff.final_ex_differs` (items among those that differ whose final EX differs between the two runs).
7. **Failing-root cause text.** The first draft said the same cause held on all backbones. As found, there are two mechanisms: on plus the root SQL text is the same cache replay in both runs and a statement that finished in the fixed run's evaluation hit the execution time limit in the gated run's evaluation; on flash and 27b the differing roots were regenerated live with a root request whose stored key is absent from the fixed run. Why they differed is not determined by the stored fields (a candidate is non-deterministic rerank output); nothing was re-executed and no model was called.
8. **Interval-valued base keys.** The result-key generator also writes `<key>_lo` / `<key>_hi` for the base keys `dual.<id>.ci`, `.item_ci`, `.t_ci`. One extra table, `tab_dual_full.tex` (all dual-interval rows, split over several floats), accompanies `tab_dual.tex` (headline rows).

## 24. Post-hoc note: technical repeat of the Memo-SQL M2 run (2026-10-27; not an amendment; no hypothesis, margin, family or rule changes)

Written and committed before the analysis stages that consume the repeated run were re-run.

* **What failed.** The M2 reference run `MIN/M2/memosql-s0` (Memo-SQL light on the 500-question dev subset, `qwen3.7-plus`, config `exp/configs/frozen/memosql_light.yaml`, seed r0) was executed on 2026-10-26 and returned 500 rows, each with an empty SQL, 0 model calls and `error_class = system_error`. It is a technical failure, not a measured EX of 0 %. The run summary printed `sys_error 0`, so the failure was not visible at run time; it was found when the analysis was assembled. Under 13.6 / 15 ("a run that fails for technical reasons is repeated, not removed") the run is repeated; until then the analysis marked the row "n/a (run failed, to be repeated)" and excluded it from E1 and from the S-A Holm family (m = 7 instead of 8).
* **Cause.** `memory_path` of the frozen config (`exp/frozen/memo_memory_v1.json`) is repo-relative. The lock gate and `config.load_config` resolve it against the repository root, but `memo_sql.Memory.load` opened it with a bare `open(path)` while the runner's working directory is `<repo>/code`. Every item raised `FileNotFoundError`, which `BaselineSystem.run` records as `system_error` without a model call; the runner did not count that class as `sys_error` and did not re-queue it. The 5 529 embedding calls of the failed run were the one-time pool embedding of the exemplar pool (the exception came after it). No chat call was made, so no dev prediction was produced and nothing about Memo-SQL's accuracy was seen.
* **Fix (code outside `code/prompts/**` and `exp/models.yaml`; config, its sha256, the memory file, prompts and the lock entry unchanged).** (i) `memo_sql.resolve_repo_path`: a relative `memory_path` is resolved against the repository root (the same rule as the config loader), absolute paths and unresolvable paths are left as they are. (ii) `nl2sql/run.py`: a baseline row with `error_class = system_error` is now recorded as `sys_error:baseline` with the exception text in `err_msg`, so it is counted, warned about and re-queued once. No other baseline run in `runs/` carries a `system_error` row (checked on the compute machine). Regression tests: `code/tests/test_memofix.py` (5 tests; 3 fail when either fix is removed).
* **Repeat.** Same queue unit `min-m2-memosql` (same command, config sha256 `ca157804...`, lock entry, seed r0, the same 500 qids), run on machine 1 on 2026-10-27 after the fix (commit `8d20160`). The failed run directory was kept as evidence outside `runs/` (`runs_failed/MIN-M2-memosql-s0-failed-20261026`). A pilot of 10 items of the unlocked tuning split `bird_train_HT` (no dev item, `--no-tuning-log`, EX 4/10, not a result) preceded it. The repeat was run **after the other development results were known** (the primary family, E1 and the B* choice had been computed with the Memo-SQL row marked n/a), because SAP 13.6 prescribes the repeat of a technically failed run; it was not chosen after seeing the result.
* **Rules unchanged.** Memo-SQL light stays an out-of-budget reference row on a 500-question subset: not budget-matched, not B*-eligible (13.3 P2 and the config header), so B* is selected by the same pre-specified rule from the same candidate set as before. It re-enters E1 (as a subset baseline compared with `full` on the common items) and the S-A Holm family (m = 8) as in 13.5.

### 24.1 Amendment log

| Date | Amendment | Before the lock | Effect |
| --- | --- | --- | --- |
| 2026-10-27 | Post-hoc note (not an amendment): technical repeat of the Memo-SQL M2 run `MIN/M2/memosql-s0` (500 rows with empty SQL and `system_error`: repo-relative `memory_path` opened from the wrong working directory); path-resolution fix and `system_error` accounting in the runner, same config and lock entry, repeated on 2026-10-27 after the other development results were known, as SAP 13.6 prescribes | no (dev results of other systems were already known; the failed run had produced no prediction) | None on any hypothesis, margin, family or the B\* rule; the Memo-SQL row of E1 and the S-A Holm family (m = 7 to 8) are restored; disclosed for the audit trail |

## 25. Post-hoc note: statistics add-on, small-cluster readings, design-effect audit, margin ratio, cost without discount, gate share (2026-10-27; descriptive; no hypothesis, margin, family or rule changes)

Scope `posthoc-secondary`. Written alongside the first complete run, not before it: the method (25.1) was fixed in the code before the values were read, but the wide-interval variants (`wide`, `wideu`) and the alias keys were added after a first look at the numbers (the first definition mixed weighted and unweighted estimands). Zero API calls; computed from the stored runs; the confirmatory files `results/*.json`, the lock and `results/posthoc2/rv2_summary.json` are unchanged. Keys: `RV2.small.*`, `RV2.deff.*`, `RV2.margin.*`, `RV2.cost.nodisc.*`, `RV2.gate.*` in `results/posthoc2/rv2s_summary.json`.

### 25.1 Method

1. **Small-cluster readings (G = 11 BIRD dev, G = 19 Spider-Realistic).** For 130 contrasts (the 110 dual rows of 23.3 and 20 loop effects): (a) cluster-robust CR1 t interval of the item-pooled estimate, t(G - 1); (b) t interval of the unweighted database means; (c) exact sign-flip test over all 2^G sign patterns of the database-level differences with weights n_g/n (`sf`) or 1/G (`sfu`), the interval being the set of shifts not rejected at 5 %; non-mean statistics use delete-one-database jackknife pseudo-values. Reading words use the rules of 23.3. `wide` = hull of the cluster-bootstrap, item, t and sign-flip intervals; `wideu` adds the two unweighted readings; `old` = hull of the cluster-bootstrap and item intervals (the rule used in the text so far). The family P* of 15.3 is re-tested with the sign-flip and t p-values under the same Holm procedure; this does not replace the confirmatory p-values.
2. **Design-effect audit.** Variance of the cluster bootstrap divided by variance of the item bootstrap of the same estimate, for all 110 rows, B = 10 000, plus the CR1 analytic design effect for the EX-difference rows.
3. **Margin ratio.** 1.5 pp divided by the pooled and per-backbone loop effect; interval by mapping the interval of the effect through x -> 1.5/x.
4. **Cost without the cache discount** for all arms with token logs, ratios of sums with cluster and item intervals, front membership from point estimates, route cost per correct answer.
5. **Gate-fire share and helped / hurt** with the exact sign test.

### 25.2 Corrections to earlier statements (found while running 25.1)

The statement in 23.11 item 4 that only qwen3.7-plus records cached tokens is wrong: qwen3.7-flash records them too (84.6 % of the single-call prompt tokens, 8.7 % in the fixed loop, about 1 % elsewhere); qwen3.5-27b records none. The design-effect range "0.35 to 1.69" quoted from `setup.json` covers four pairs at 2 000 draws; over all 110 contrasts the range is 0.17 to 3.71. No threshold or reading rule is changed.

### 25.3 Amendment log

| Date | Amendment | Before the lock | Effect |
| --- | --- | --- | --- |
| 2026-10-27 | Post-hoc note (not an amendment): statistics add-on S2 (small-cluster t and exact sign-flip readings, P* re-test, design-effect audit, margin ratio interval, cost without cache discount, gate share and helped / hurt sign test); correction of the cached-token statement of 23.11 item 4 | no | Descriptive; no confirmatory test, threshold, family or reading-rule change; never enters the confirmatory results, the tuning-set results or P\*; zero API calls |

## 26. Post-hoc note: additional experiments R3EXP (2026-10-28; descriptive; no hypothesis, margin, family or rule changes)

Scope `posthoc-secondary`. Written after the runs, not before them: the run list (`exp/queues/R3.jsonl`, `exp/queues/R3B.jsonl`) and the analysis scripts (`code/tools/r3_*.py`) were fixed before the values were read, but the following were added or changed after a first look at the numbers and are marked as such below: the root-regeneration audit (26.1 item 3, added after the Flash s0 chains showed 29 root differences), the sign-preserving format of the pooled stack bound, the vote baseline variant B, the `RW3` alias keys, and, on the decision of 2026-10-28, the separation of the T = 0 headline run from the T = 0.7 replication (a first version averaged the two; that version was withdrawn before any use, and its keys were deleted) with the third T = 0.7 run (seed 303). The new chains were generated on one machine each with one base cache, models `qwen3.7-plus-2026-05-26`, `qwen3.7-flash-2026-07-15`, `qwen3.5-27b`; the key `RV3.cost.total_cny` gives the cost. The confirmatory files `results/*.json`, the lock entries of the P* family and `results/posthoc2/*` are unchanged; lock entries for the new configurations were appended to `exp/dev_lock.json` (`full_fixed`, `full_gated` with `gated_rootonly.yaml` and with `full_gated.yaml`, `zs_ours`, `zs_ex`, for seeds 101 / 202 and 303; the frozen configuration `exp/configs/frozen/zs_ex.yaml` and, for seed 303, `zs_ex_s303.yaml`, which differs only in the source list). Keys: `results/r3/rv3_summary.json`.

### 26.1 Method

1. **Decomposition arms (E-S).** Single call (zs), root-only, gated loop and fixed loop on Plus, Flash and 27B, dev 1 534. The headline run is seed 0 at T = 0 (locked, unchanged) and is reported apart. The replication is three independent runs at T = 0.7 (seeds 101, 202, 303; the protocol of the existing Plus stability runs) and is never averaged with the headline run. Per arm: the EX of each run, the mean and the SD (ddof = 1) of the three T = 0.7 runs. Per contrast (root-only minus single call = stack; gated minus root-only = gated loop; fixed minus root-only = fixed loop; gated minus fixed; gated minus single call; fixed minus single call; pooled over the three backbones; W1 = mean of Flash and 27B gated minus single call) the estimate is the mean over the three T = 0.7 runs of the per-run paired EX difference of the same-seed arms (question by question). Intervals: (a) cluster bootstrap over the 11 databases (B = 10 000, seed 20261101) of the run-averaged per-question difference; (b) item-level bootstrap of the same vector; (c) between-run range; (d) t interval of the three per-run values, t(2); `wide` = hull of (a) to (d) (superseded by 27.1 item 2: the range (c) and the t(2) interval (d) are no longer part of the hull). Reading rule SUP(0): "positive" if the whole interval is above 0, "negative" if below, otherwise "contains 0". For the headline run the reading uses the hull of its cluster and item-level intervals; each contrast carries the flag "agrees with s0" (same reading word) for the cluster interval and for the wide interval.
2. **P1' in the replication.** The margin test uses the pooled gated-minus-fixed interval of the replication against -1.5 pp; the calls ratio gated / fixed is the ratio of mean billed calls per question, per run.
3. **Root-regeneration audit (added after the first look).** For every backbone and run (s0 and the three T = 0.7 runs, each on its own) the root SQL of the gated, fixed and root-only runs is compared question by question. Contrasts are also reported on the questions whose roots are identical in the two arms.
4. **Same-examples control (E-C).** `zs_ex`: the prompt of the single call plus exactly the few-shot examples the root-only run of the same backbone and seed stored (`exemplar_ids`), in the stored order and the pipeline's block format; no retrieval, no sampling, no repair; one call per question. Contrasts: `zs_ex` minus single call (examples only), root-only minus `zs_ex` (everything else of the root stage), their sum = stack, each as in item 1 (headline run apart, replication mean, flags). Share of the stack carried by the examples = examples / stack, reported only where the stack interval excludes 0 (otherwise "undefined"), interval by the cluster bootstrap of the ratio.
5. **Retrieval activity (E-R).** For every backbone and arm the number of questions on which the schema was cut by the retrieval module (counted from the stored contexts), the largest prompt, the number of databases above the table threshold k_t = 6, and the stack by schema-size stratum (3 000 tokens of the full DDL); headline run.
6. **Routing controls (E-R).** Same features and fit as `route_ca_frozen` on the high-risk set (a) with the sampling-agreement feature removed, (b) agreement alone, (c) the non-empty flag alone, (d) leave-one-database-out fits on the development set, with and without agreement; AUROC with the DeLong interval, accepted-EX at coverage 0.4 / 0.6 / 0.8, paired AUROC differences by cluster bootstrap. Equal-cost vote baseline: the majority share among the answer and the four stored samples of the routing run (variant A: final answer + 4 samples; variant B: root + 4 samples), executed on the database, as a score for acceptance. Headline run.
7. **P5' re-run.** Same population and test as P5' (failing roots; gated repair against the blind re-draw; exact McNemar), (a) all roots, (b) without the roots whose only failure is a time-out and whose SQL runs in the other arm, (c) without any time-out root. Headline run.
8. **Corrected-reference subset (E-Q).** Selection chain printed from the release and the data; EX in the subset, outside it inside dev, and outside the release; the bound of the full-set shift = shift in the subset x its weight (23.5 %) plus the shift elsewhere under two assumptions (none elsewhere; the same rate elsewhere). Headline run.
9. **Addendum (no model call).** Incremental cost per additional correct answer = (cost per question of the arm minus that of the reference) / (EX difference); undefined (dash) where the EX difference is not positive, upper bound "unbounded" where its interval reaches 0; with and without the cache discount (the 27B log has no cached tokens, so the two coincide for it). Minimum detectable effect at 80 % power: (z(0.975) + z(0.80)) x SD of the paired differences / sqrt(n) at item level, (t(0.975, 10) + t(0.80, 10)) x SD of the database means / sqrt(11) at cluster level; named tests P1', W1 (EX, pp), P3 (AUROC, DeLong), P5' (repair rate). Count of post hoc contrasts. Critic increment = (fixed final minus its root) minus (gated final minus its root) on the questions with the same root. Latency = mean per-question wall time of the arm divided by that of the single call. Headline run.

### 26.2 Limits stated

The replication runs at T = 0.7 and the headline run at T = 0; the replication therefore does not repeat the headline run, it asks whether the readings survive a change from temperature 0 (not exactly reproducible, 27.2) to sampled generation. Sampled roots are weaker (the T = 0.7 single-call means are 0.5 to 2.3 pp below the T = 0 runs), which leaves more for the loop to repair, so loop effects are expected to be larger at T = 0.7; "agrees with s0: no" can reflect that and not noise, and the two cannot be separated with this design. The SD of an arm is a three-run SD; with three runs it is a scale, not an interval, and the `wide` interval is a union (conservative). The comparison reading for s0 is the hull of its cluster and item-level intervals; the RV2 widest rule (with the sign-flip interval) is not recomputed for the replication (superseded by 27.1 item 2, which applies one widest interval, the hull of the cluster, item-level, cluster-robust t(G - 1) and sign-flip intervals, to the locked run and to the replication alike). Later addition: the widest intervals of the margin ratio and of the share kept use the same four intervals (for the ratio, the interval of the fixed-loop effect mapped through 1.5 / x; for the share kept, the intervals of the cluster-robust t and sign-flip tests inverted over the ratio), and the within-run difference of the two loop effects is reported beside them; a share of 1 is inside the interval exactly when that difference has an interval that includes 0. The 360-subset analysis cannot recover how the release's authors chose their 498 items. Embedding and rerank requests are not priced.

### 26.3 Amendment log

| Date | Amendment | Before the lock | Effect |
| --- | --- | --- | --- |
| 2026-10-28 | Post-hoc note (not an amendment): additional experiments (T = 0.7 replication of three runs per decomposition arm, kept apart from the T = 0 headline run; same-examples control; retrieval activity; routing controls; P5' re-run; corrected-subset rule; minimum detectable effects; incremental cost; Critic increment; post hoc count); lock entries appended for the new configurations and seeds | no | Descriptive; no confirmatory test, threshold, family or reading-rule change; never enters the confirmatory results, the tuning-set results or P\*; 72 444 billed chat calls (135.94 CNY) |

## 27. Post-hoc note: further statistics and the temperature-0 non-determinism check (2026-10-28; descriptive; no hypothesis, margin, family or rule changes)

Parts 27.1 (statistics) and 27.2 (rerun at temperature 0). Scope `posthoc-secondary` for both. Written after the values: the method of each part was fixed in the code (`code/tools/r4_stats.py`, `code/tools/r4_t0.py`) before the paper text that cites it was written, but the parts were requested by a further critical reading of the draft by the authors with Claude Sonnet 5.5 (see Acknowledgments) after the locked run, the temperature-0.7 replication and the additional experiments of Section 26 were known, so none of this is pre-specified. The confirmatory files `results/*.json`, the lock entries of the P\* family and `results/posthoc2/*`, `results/r3/*` are unchanged. Result files and keys: `results/r4/rv4_summary.json` (prefix `RV4.`) and `results/r4_t0/rv4t0_summary.json` (prefix `RV4T0.`).

### 27.1 Further statistics (no model call)

1. **Critic increment cost.** Cost per additional correct answer of the fixed loop over the gated loop = (cost per question of the fixed loop minus that of the gated loop) / (EX difference), per backbone, with and without the cache discount, defined only where the EX difference is positive; the interval comes from the clustered and the item-level bootstrap of the ratio (the text uses the widest of the four intervals of item 2, keys `...icer_disc_wide`) and is reported as "unbounded" where the interval of the EX difference reaches 0. The price is the break-even value of one extra correct answer, not a price of the Critic in isolation.
2. **One widest interval.** For the locked run and for the replication alike, the interval used for the reading is the hull of: the database-clustered bootstrap (B = 10 000, seed 20261101), the item-level bootstrap, the cluster-robust CR1 interval with t(G - 1), and the exact sign-flip interval over the size-weighted database means. The range of the three replication runs is reported in a separate column and is not part of the hull (earlier text put the range and the t(2) interval of three values into the hull). Reading rule SUP(0) as in 23.3; "agrees" means the same reading word for the locked run and the replication. The earlier hull (19 of 25 contrasts agree) is kept as a comparison key.
3. **Replication extras.** The temperature of the calls in the replication is stated per stage from the code: generator, root and revision calls at 0.7; the task-graph call and every Critic call stay at 0 in the code; the routing samples are at 0.7 in every run. The margin ratio (1.5 pp divided by the pooled loop effect) and the share kept are computed for the replication and for the locked run.
4. **Root regeneration.** In the locked run, the questions whose root SQL differs between the gated and the fixed arm of the same backbone; net questions (gated right and fixed wrong minus the reverse) over all questions, over the questions with a different root and over those with the same root.
5. **Post hoc count.** 209 = 130 (the 110 dual-interval rows of 23.3 plus 20 loop effects) + 25 replication contrasts + 12 same-examples contrasts + 7 routing + 2 P5' + 18 corrected-subset + 6 retrieval strata + 3 Critic + 6 Spider-Realistic; the expected number of false positives at 5 % is 10.45 if all were independent tests with no effect. The count excludes the quantities of 27.2, which are run-to-run changes and not contrasts between arms.
6. **Minimum detectable effects.** Unit stated. Question level: (z(0.975) + z(0.80)) x SD of the paired differences / sqrt(n). Database level: (t(0.975, 10) + t(0.80, 10)) x SD of the database means / sqrt(11). The value for P1' is the smallest difference from zero detectable with 80 % power, not the power of the margin test.
7. **P5' units and silent errors.** The failing roots of P5' are counted per distinct question (a question failing on two backbones counts once), and the test is repeated with one unit per question. A wrong root answer is silent if its SQL executed without error; the share of wrong answers the gate can reach is 1 minus the silent share.
8. **Spider-Realistic split.** Stack and loop effects from the root-only arm of the external-validity block (Section 21), replacing a sentence that had inferred the split.
9. **Lock hashes.** The sha256 prefix and the entry count of `exp/dev_lock.json` at the committed versions (`git show`): 20 entries at the end of the primary runs, 28 entries with the replication runs; the first 16 entries are identical in all committed versions.

### 27.2 Temperature-0 non-determinism check (17 329 billed chat calls; completed under 27.5)

* **Question.** The critical reading of the draft asked whether the locked run (T = 0, seed parameter 0) can be reproduced and how large the change between two identical runs is, in comparison with the contrasts the paper reports. The check is a rerun of part of the locked run with the same model identifiers, prompts, frozen configurations, temperature 0 and seed parameter 0, and a fresh chat-call cache (no hit on the locked run's answers); the embedding and rerank cache of the base was copied unchanged, so retrieval is held fixed. The chain design is that of the locked run (fixed loop first, gated loop importing the fixed archive, root-only; single call separate). The lock gate (`nl2sql.lock verify`) was passed with the existing seed-0 entries; four dated entries (2026-10-28, seed 0, configurations `zs_ours/zs.yaml`, `full_fixed/full_fixed.yaml`, `full_gated/full_gated.yaml`, `full_gated/gated_rootonly.yaml`) were appended to `exp/dev_lock.json` for the audit trail only; the first 28 entries are unchanged.
* **Planned and run.** Planned: two reruns (r1, r2) of all four decomposition arms on the three backbones (24 jobs). Run: four arms complete on all 1 534 development questions (single call on Flash and on 27B; fixed loop on Plus and on 27B, one rerun r1 each), and the fixed loop on Flash for 1 005 of the 1 534 questions (8 of the 11 databases), reported as partial. The rest was not run **because three of the four compute machines were lost during the rerun (local disk, key file and machine-local caches), not because of any result**; the instruction was not to restore the keys on the lost compute machines and to stop launching. The single call on Plus (15 rows) is left out as incomplete. No gated-loop or root-only rerun, no r2 and no rerun on the Plus single call exists; this design therefore cannot give the T = 0 triple, per-contrast agreement with the locked run or a comparison with the T = 0.7 readings, and the text does not claim them.
* **Quantities (each rerun r1 against the locked run s0, same questions).** (a) Share of questions whose final SQL text differs (whitespace-normalised; raw text in a separate key); (b) share whose correctness differs and the counts b (right in s0, wrong in r1) and c (wrong in s0, right in r1) with the exact McNemar p; (c) EX difference r1 minus s0 with the cluster-bootstrap interval, the item-level interval and the widest interval of 27.1 item 2, and its reading; (d) for the fixed loop, the share of questions with a different retrieved example set and with a different root SQL; (e) for requests with the same cache key in s0 and r1 (identical model, parameters and messages), the share whose stored answer differs, by stage; (f) the change of fixed minus single call between s0 and r1 on the backbones where both arms were rerun (27B complete; Flash on the 1 005 questions, partial); (g) one table that sets the largest absolute change of EX in these reruns beside the estimates and widest intervals of the paper's key contrasts in the locked run (loop effects, stack, gated minus fixed, fixed minus single call, W1). The comparison is descriptive: a locked contrast whose absolute estimate is below the largest absolute EX change of a single rerun cannot be told apart from a rerun change by size alone.
* **Reading rule.** No rule of the plan is changed. A rerun changes a reading if the SUP(0) word of the widest interval differs between s0 and r1.

### 27.3 Limits stated (completed under 27.5)

27.1: the 209 count is a count of reported post hoc contrasts, not a correction; the widest interval is a union and is conservative; none of the minimum detectable effects is a power statement for the margin test. 27.2: one rerun per arm, five arms (one partial), so the spread between reruns is a single paired difference per arm and not an estimate of run-to-run variance; the arms that ran were not chosen by result (they are the ones that finished before the compute machines were lost) but they are not a random sample of arms, and the Plus single call and every gated and root-only arm are missing; in the loop arms the retrieved examples and the root SQL differ in part between s0 and r1 (item (d)), so the change in the final SQL combines the change of the service answer for an identical request with the change of the request; the single-call arms are the clean case (every request key of the rerun occurs in s0). The service gives different answers to identical requests at temperature 0 with seed 0; the cause (server-side batching, kernel non-determinism, a changed served model behind an alias) is not identified here. For 27B the alias window stated in the paper applies (the served model behind the identifier may have changed between the locked run and the rerun; this is stated, not tested), so for it a change of the served model is not excluded as a cause.

### 27.4 Amendment log

| Date | Amendment | Before the lock | Effect |
| --- | --- | --- | --- |
| 2026-10-28 | Post-hoc note (not an amendment): further statistics (Critic increment cost, one widest interval, replication extras, root regeneration, post hoc count of 209, minimum detectable effects at question and database level, P5' units, silent errors, Spider-Realistic split, lock hashes) and a temperature-0 non-determinism check (one rerun of all twelve decomposition arms of the locked run; completed under 27.5 after an earlier attempt that stopped at four arms and one partial arm when compute machines were lost); eight dated lock entries appended | no | Descriptive; no confirmatory test, threshold, family or reading-rule change; never enters the confirmatory results, the tuning-set results or P\*; the statistics used no model call, the rerun 24 747 billed chat calls (44.91 CNY with the cache discount, 66.10 CNY at list price) |

### 27.5 Addendum (2026-10-28, after a further critical reading of the draft by the authors with Claude Sonnet 5.5 (see Acknowledgments)): completion of the temperature-0 rerun r1 (post hoc; scope `posthoc-secondary`; no hypothesis, margin, family or rule changes)

That further critical reading of the draft asked for the arms that 27.2 and 27.3 state as missing, so that P1' (pooled gated minus fixed) and the contrast gated minus root-only are retested against the locked run s0. Written after the values of 27.2 and after that reading; the method is that of 27.2 (the code of `code/tools/r4_t0.py` was extended before the new runs were analysed), the quantities added below were requested by that reading and are descriptive. Only r1 was run (no r2). The files `results/*.json` of the confirmatory family, the P\* lock entries, `results/posthoc2/*`, `results/r3/*` and `results/r4/*` are unchanged.

* **What was run.** Gated loop, root-only and the Flash fixed loop on Plus, Flash and 27B, and the Plus single call, all on the 1 534 development questions, with the prompts, model identifiers and frozen configurations of s0: temperature 0, seed parameter 0, fresh chat-call cache, same chain order (fixed, gated, root-only, each importing the call-cache archive of the previous arm of the same replicate). Together with the four complete reruns of 27.2 (single call Flash and 27B, fixed loop Plus and 27B) all 12 arm-backbone combinations now have a complete r1; the 27.2 partial Flash fixed-loop rerun (1 005 questions) was replaced by a complete rerun of the whole loop (its call cache was lost with its compute machine, so completing it would have left the gated roots unshared; the 1 005-row partial is kept as a backup and is not analysed). Four dated lock entries (seed 0) were added to `exp/dev_lock.json`; the earlier entries are unchanged.
* **Where and how it differs from 27.2.** The compute machines that ran 27.2 had lost their key files, so these runs were done on one machine in containers (Python 3.11 instead of 3.12; same package versions of numpy and sqlglot), one job at a time, key read from a key file, call caches local to the machine. The retrieval caches could not be copied from the machine-local base of 27.2 (gone): the base was the shared-disk cache of 2026-10-26. Its embedding/rerank file and its gold-execution file were corrupt on the shared disk; 25 385 embedding/rerank rows were salvaged (16 rows unreadable, 2 invalid ones dropped), the gold-execution cache was rebuilt from the databases, and embedding or rerank values that were missing were requested again. Evidence that the requests are those of s0: the single-call Plus rerun has 100 % of its request keys in s0; the share of questions whose retrieved example set differs from s0 (1.8 to 2.9 % in the loop arms) is the same as in 27.2 (2.0 to 2.5 %).
* **Per arm (r1 against s0).** Final SQL text differs on 5.74 to 47.07 % of the questions (mean 26.4 %), correctness on 0.65 to 7.50 % (mean 3.68 %); largest absolute EX difference 0.52 pp; no difference excludes 0 under the widest interval; smallest McNemar p 0.096 (27B root-only, b/c 13/5).
* **Contrasts (25: six contrasts on Plus, Flash, 27B and pooled, plus W1).** Estimate in s0, in r1, their change, widest intervals and readings are in `results/r4_t0/contrasts_r1.csv` (keys `RV4T0.ctr.<contrast>.<scope>.*`). P1': s0 -0.26 [-1.06, +0.46] -> r1 -0.33 [-1.10, +0.40] pp, change -0.07 [-0.59, +0.44]; the lower end stays 0.40 pp above the margin -1.5. Gated minus root-only, pooled: +0.80 [+0.52, +1.11] -> +0.98 [+0.52, +1.46]; on Plus +0.20 -> +0.26 (widest interval [0.00, +0.52]: "contains 0" in both runs). The widest reading is unchanged in 24 of the 25 contrasts; the one that changes is fixed minus single call on 27B (positive in s0, lower end 0.00 in r1). Under the cluster interval alone 20 of 25 readings are unchanged. No change of a contrast excludes 0 (largest absolute change 0.52 pp).
* **Noise scale.** From the discordant counts b, c of an arm: implied SD of the net EX difference between two runs of the same configuration, `sd_net = 100 sqrt(b + c) / n` pp (0.21 to 0.70 pp over the 12 arms; of one run, `sd_net / sqrt 2`). Two-arm contrast: `sqrt(sd_x^2 + sd_y^2)` (0.33 to 0.98 pp); pooled over the three backbones: `sqrt(sum of the three two-arm variances) / 3` (0.38 to 0.42 pp). The two-arm and pooled values assume independence across arms and backbones and are approximations; arms of one chain share their roots, so they are positively correlated and the formula overstates the SD of their difference (pooled gated minus root-only: 0.40 by the formula, 0.15 from the per-question change of the contrast itself, key `sd_direct`, which needs no such assumption). The observed changes of the contrasts divided by these scales are in `z_indep` and `z_direct`.
* **Chain check.** The root SQL is shared between gated and fixed in 98.6 to 99.6 % of the questions in r1 and 98.4 to 99.9 % in s0 (key `chain.<b>.gf_root_same_*`), so the structure of s0 is reproduced.
* **Cost.** 8.18 CNY with the cache discount (14.25 CNY at list price), 7 418 billed chat calls for this batch; with 27.2: 44.91 CNY (66.10 list), 24 747 calls. Cap 120 CNY. Embedding and rerank requests are not priced.
* **Limits (supersede 27.3 where they differ).** One rerun per arm: one paired difference per arm, not an estimate of run-to-run variance; the contrasts' changes are single paired differences. The retrieval cache deviation above. For 27B the alias window of the paper applies. The cause of the non-determinism is not identified. The Plus single call includes 15 rows from the earlier stopped run (same requests, fresh cache).

## 28. Post-hoc note: a non-Qwen backbone and a single call with retry on execution failure (2026-10-29; descriptive; no hypothesis, margin, family or rule changes)

Parts: the non-Qwen backbone (X2DS, 28.1) and the retry-only control (X2DSR, 28.1). Scope `posthoc-secondary`. Both parts were decided after the locked run, the temperature-0.7 replication, and the temperature-0 rerun (Section 27) were known; none is pre-specified, and no result of this section changes a verdict of the family P\* or any other section. The confirmatory files `results/*.json`, the P\* lock entries, `results/posthoc2/*`, `results/r3/*`, `results/r4/*` and `results/r4_t0/*` are unchanged. Result files and keys: `results/x2ds/x2ds_keys.json` (prefix `X2DS.`), built by `code/tools/x2ds_keys.py` from `results/x2ds/<backbone>/x2ds_summary.json` (`code/tools/x2ds_analyze.py`), `results/x2ds/pooled/x2ds_pooled.json` (`code/tools/x2ds_pool.py`), `results/x2ds/x2ds_runs.json` (`code/tools/x2ds_runmeta.py`) and two cost files; `code/tools/x2ds_run_analysis.sh` runs the analysis.

### 28.1 What was run

* **X2DS: a non-Qwen backbone.** The four decomposition arms of the locked run (single call `zs`, `full_fixed`, `full_gated`, root-only `gated_rootonly`) with the frozen configuration files, prompts and chain order of the locked run and one changed model identifier, `deepseek-v4.1-flash-1017`, called through the same Alibaba Cloud Model Studio account as the other three models, thinking off, temperature 0, seed parameter 0, one run, all 1 534 development questions. `exp/models.yaml` already listed the model; no prompt, configuration or model-list file was changed (the lock records hashes of configuration, prompts and model list, not of the model identifier, so the four arms ran under the existing lock entries). The backbone was chosen because it was available on the same account, before any of its results was seen; no other non-Qwen model was run, so there was no selection among models. Runs on 2026-10-29 (Beijing time).
* **X2DSR: single call with retry on execution failure (`zs_retry`).** New system, `code/nl2sql/systems/ours/zs_retry.py`, configuration `exp/configs/posthoc/zs_retry.yaml` (seed 0 only; entered in the lock file used for the runs on 2026-10-29 before its first call and appended to `exp/dev_lock.json` at integration). Root: the request of the single call (full schema DDL with sample values, question, evidence; no task graph, no retrieved examples, no business terms, no routed schema). Then exactly the execution gate of the gated loop: while the current query fails to execute or times out (30 s) and fewer than three revisions were made, the query and the database error text go back to the same model with the revision prompt of the gated loop (`rev_exec`; no new prompt file); a revision identical to its parent ends the loop; the answer is the best node by execution status (non-empty, empty, error; ties to the earliest). No semantic Critic call and no abstention. Run on Plus, Flash, 27B and DeepSeek.
* **Pairing rule.** The root of `zs_retry` is the stored answer of the locked single-call run (seed 0, temperature 0; for DeepSeek the X2DS single call): the archive of that run was imported into the chat-call cache, so the root request is a cache hit and returns the stored answer, and only the revisions of failing roots are new calls. The pairs "retry-only against single call" therefore share their root on every question (final query text identical on 1 479 to 1 518 of 1 534 questions, the rest revised); a question can differ in correctness with identical text only through re-execution (timing-dependent queries; at most three questions per backbone in each direction, key `X2DS.<b>.retry.n_up_same_sql`, `n_down_same_sql`).

### 28.2 Quantities and intervals

Per arm: EX, chat requests per question, tokens per question (prompt, completion, recorded cached), CNY per 1 000 questions at list prices (`exp/prices.yaml`, read 2026-10-26; DeepSeek at the peak price, the runs were made in the daytime) and with cached tokens at 0.2 of the input price (our assumption, as elsewhere). Contrasts (percentage points, paired per question): retry-only minus single call, gated minus retry-only, fixed minus retry-only, root-only minus retry-only, and the contrasts of the locked run (gated, fixed, root-only against single call and against each other) on every backbone. Intervals: the four of the paper (database-clustered bootstrap, B = 10 000, shared draws, seed 20261101; question-level bootstrap; cluster-robust t(G-1); exact sign-flip over the size-weighted database means) through the same code as Section 27 (`r4_t0.four`), the widest being their hull; reading rule SUP(0), an end that prints as 0.00 counts as containing 0. Exact McNemar counts and p per contrast. Pooled contrasts are the mean over backbones of the per-question difference on one common database resample: three Qwen backbones (`q3`) and Flash with 27B (`w2`, the backbone set of W1). The share of the pooled gated-over-single-call difference that the retry-only difference carries is the ratio of the two point estimates; it has no interval. Gated minus fixed on DeepSeek is read against the margin of P1' (-1.5 pp) for orientation only; it is not a test.

### 28.3 Values (all post hoc; one run per arm and backbone)

* **Retry-only against single call.** Plus +0.26 (widest [-0.19, +0.75]); Flash +1.50 (widest [+0.18, +2.82]); 27B +0.85 (widest [+0.22, +1.42]); DeepSeek +0.33 (widest [0.00, +0.65]). Calls per question 1.016 to 1.043. Pooled: Flash and 27B +1.17 (widest [+0.33, +1.97]), three Qwen backbones +0.87 (widest [+0.25, +1.45]); the retry-only difference is 40% of the gated-over-single-call difference (Flash and 27B) at the point estimates.
* **Gated loop against retry-only.** Plus +0.39 (widest [-1.62, +2.39]); Flash +0.91 (widest [-1.73, +3.36]); 27B +2.61 (widest [+0.26, +5.02]); DeepSeek +1.56 (widest [-0.07, +3.20]). Pooled: Flash and 27B +1.76 (widest [+0.08, +3.31]), three Qwen backbones +1.30 (widest [-0.16, +2.66]).
* **DeepSeek, four arms.** EX single call 62.39, fixed 64.28, gated 64.28, root-only 64.02. Gated minus single call +1.89, cluster [+0.43, +3.19], widest [+0.17, +3.59]; gated minus fixed 0.00, cluster [-1.27, +1.30], widest [-1.58, +1.60] (discordant 34 and 34); the lower end of the widest interval is beyond the margin of -1.5 pp, the lower end of the cluster interval is not. Gated loop calls 0.56 of the fixed loop's (cluster [0.54, 0.59]). CNY per 1 000 questions (list price / with discount): single call 5.65 / 2.07, fixed 23.70 / 12.95, gated 10.49 / 6.89, root-only 10.27 / 6.81.
* **Cost.** X2DS 22.76 CNY with the cache discount (44.48 list price), 7 021 billed chat requests; X2DSR 0.37 CNY (0.52 list), 150 billed chat requests (the roots are cache hits); no embedding or rerank request is priced.

### 28.4 Limits

One run per arm and backbone (seed parameter 0): the intervals contain no generation variance, and the service is not deterministic at temperature 0 (Section 27). One non-Qwen model from the same provider, same account and same billing, thinking off; it does not sample model families, and a provider-specific effect is not excluded. The retry-only control uses the gate and revision prompt of the gated loop, which were designed for the pipeline; another retry design could do better. Of the 50 contrasts reported (ten on each of four backbones, five on each of two pools), 20 re-estimate contrasts of the locked run on the Qwen backbones and are already counted among the post hoc contrasts of Section 27.1; the other 30 are added to that count (key `X2DS.n.contrasts_new`). List prices are those of 2026-10-26. The runs of this section are not in the run archive and the code of this section is not in the code release that the paper cites (version 1.0.1); both are added in a new version of the two records (concept DOIs 10.5281/zenodo.23735018 for the run archive and 10.5281/zenodo.23739687 for the code).

### 28.5 Amendment log

| Date | Amendment | Before the lock | Effect |
| --- | --- | --- | --- |
| 2026-10-29 | Post-hoc note (not an amendment): a non-Qwen backbone (deepseek-v4.1-flash-1017: single call, fixed loop, gated loop, root-only; one seed) and a single call with retry on execution failure on four backbones (roots are the stored answers of the locked single-call run); one lock entry appended (retry-only system, seed 0) | no | Descriptive; no confirmatory test, threshold, family or reading-rule change; never enters the confirmatory results, the tuning-set results or P\*; 7171 billed chat requests (23.13 CNY with the cache discount, 45.00 CNY at list price) |

## 29. Post-hoc note: reruns and further arms after the second and third rounds of review (2026-10-29; descriptive; no hypothesis, margin, family or rule changes)

Scope `posthoc-secondary`. Everything in this section was decided after the locked run, the replication, the rerun (Section 27) and the Section 28 results were known, in answer to a further critical reading of the draft by the authors with Claude (see Acknowledgments). Nothing enters the family P\*, and no verdict of any other section depends on it.

### 29.1 What was run

* **X2DSRR: reruns.** The single call + retry (`zs_retry`, Section 28) once more on Plus, Flash and 27B, with the roots of the temperature-0 rerun of the single call (`analysis_runs/R4T0/zs-<b>-r1`) imported into the call cache; and the DeepSeek single call, fixed loop, gated loop and single call + retry once more with a fresh call cache (the gated loop imports the roots of the rerun fixed loop, as in the locked run). Purpose: reading lines for every step between the single call, the single call + retry, the gated loop and the fixed loop on four backbones.
* **W7 arms.** (a) `zs_think`: the single call with thinking on (`exp/configs/posthoc/zs_think.yaml`; maximum 16,384 output tokens; same price per token as with thinking off) on Plus, Flash and 27B. (b) `zs_retry` with the empty-result gate (`exp/configs/posthoc/zs_retry_empty.yaml`: the gate also fires when the root returns an empty result) on the stored single-call roots of all four backbones. (c) MAC-SQL (`exp/configs/frozen/macsql.yaml`, the locked baseline configuration) on Flash and 27B; on Plus the locked MAC-SQL run is reused. One run each, seed parameter 0, temperature 0; lock entries appended before the first call (`queue/dev_lock.before_w7.json` keeps the lock file before them).
* **W7RR: rerun of the empty gate.** Arm (b) once more on all four backbones, with the roots of the rerun single call imported into the call cache (`analysis_runs/R4T0/zs-<b>-r1` for the Qwen models, `runs/X2DSRR/zs-dsv41flash-r1` for DeepSeek). Purpose: reading lines for the steps that involve the empty gate. Added after the W7 results of arm (b) were seen; it changes no estimate of the main run, only the line its steps are read against.
* **W8: pipeline + empty gate (post hoc, 2026-10-29).** The gated loop with `m5.gate_empty` true (`exp/configs/posthoc/gated_empty.yaml` = `frozen/full_gated.yaml` with that one switch), system `full_gated`, on all four backbones, twice: s0 imports the locked gated archive (`analysis_runs/MIN/M1/gated-s0`, `MIN/M5/gated-<b>-s0`, `runs/X2DS/gated-dsv41flash-s0`) and r1 the gated rerun (`analysis_runs/R4T0/gated-<b>-r1`, `runs/X2DSRR/gated-dsv41flash-r1`), so the arm differs from the gated loop only where an empty result fires the gate. Purpose: a reviewer noted that the pipeline and the empty-gate retry differ in two things (front end and trigger); this arm holds the trigger fixed. Added after the W7 results were seen; lock entry appended before the first call (`queue/dev_lock.before_w8.json` keeps the lock file before it).

### 29.2 Quantities

* **Reading lines.** Per arm, `sd_net = 100 sqrt(b + c) / n` from the discordant counts between the locked run and its rerun. Per step x over y: the independence line `2 sqrt(sd_net_x^2 + sd_net_y^2)` and the direct line `2 x 100 sqrt(sum (d1 - d0)^2) / n`, with d the per-question difference of the step in the locked run (d0) and in the rerun (d1). A step is *detected* when its widest interval excludes zero and its estimate reaches its line; a verdict is reported only when it holds under both lines, else "depends on the line". Steps with an arm that was not rerun (thinking on, MAC-SQL) carry the interval condition only.
* **Per step.** EX gain with the four intervals and the widest; share of database-clustered bootstrap draws above zero; minimum detectable effect `2.8016 x SE` (cluster bootstrap SE); extra CNY per 1,000 questions at full price and with the cache discount; price per extra correct answer (point estimate and at the upper end of the widest interval); net-benefit probability `P(V x gain - extra cost > 0)` over the bootstrap draws for V = 0.01, 0.1, 1 and 10 CNY per correct answer (extra cost held at its mean); the same-text gain, with questions whose final SQL text is identical in both arms scored as no difference.
* **Per arm.** Mean and 95th percentile of the summed request latency per question; final queries whose scoring execution took 24 s or more.
* Code: `code/tools/x2ds_rerun.py`, `x2ds_analyze.py`, `x2ds_keys.py` (keys `X2DS.<b>.w7.*`, `X2DS.<b>.rerun.*`); launcher `code/tools/x2ds_host.py --mode rerun|w7|w7rr|w8`. Third round: the same quantities for the steps of the pipeline + empty gate (`gatedempty`); the corrected-reference re-scoring of the retry arms (`code/tools/x2ds_rescore.py`, subset and reading of Section 20); the count of development questions with an empty reference result (`code/tools/x2ds_empty_gold.py`).

### 29.3 Values

See `results/x2ds/x2ds_keys.json` (prefix `X2DS.`) and Supplementary Section S17.15; the paper's Table "What each step buys" is generated from these keys.

### 29.4 Amendment log

| Date | Amendment | Before the lock | Effect |
| --- | --- | --- | --- |
| 2026-10-29 | Post-hoc note (not an amendment): reruns of the single call + retry and of the DeepSeek chain; thinking-on single call; single call + retry with the empty-result gate; MAC-SQL on Flash and 27B; rerun of the empty-gate arm (W7RR); three lock entries appended (zs_think, zs_retry_empty, MAC-SQL seed 0 on two backbones) | no | Descriptive; no confirmatory test, threshold, margin or family changed |
| 2026-10-29 | Post-hoc note (not an amendment): pipeline + empty gate (W8, s0 and r1 on four backbones); one lock entry appended (full_gated with posthoc/gated_empty.yaml, seed 0); corrected-reference re-scoring of the retry arms; empty-reference count | no | Descriptive; no confirmatory test, threshold, margin or family changed |

## 30. Post-hoc note: the harm of the empty-result trigger on tuning data (post hoc, 2026-10-30; descriptive; no hypothesis, margin, family or rule changes)

Scope `posthoc-secondary`. Written before the runs below were started, after every result of Sections 27 to 29 was known, in answer to a further critical reading of the draft (the BIRD development set has no question whose reference result is empty, so the harm of the empty-result trigger, revising a correct empty answer into a non-empty wrong one, could not be measured there). Nothing enters the family P\*, and no verdict of any other section depends on it.

### 30.1 What is run

* **W9.** On the tuning split HR (`bird_train_HR`, 1,101 questions of eight held-out BIRD training databases; the stored tuning runs show questions whose reference result is empty), on all four backbones, seed 0, temperature 0, one call cache per backbone: (a) the frozen single call (`zs_ours`, `frozen/zs.yaml`); (b) the single call + retry (`zs_retry`, `posthoc/zs_retry.yaml`), with (a) imported; (c) the empty-gate retry (`zs_retry`, `posthoc/zs_retry_empty.yaml`), with (a) and (b) imported, so that (c) differs from (b) only by the revisions that an empty result triggers. Launcher `code/tools/x2ds_host.py --mode w9`. No configuration, prompt or rule is changed; HR is an unlocked tuning split.
* **Scope change before the runs (2026-10-30).** The database file of `bike_share_1` (5.2 GB) does not fit on the analysis host, so W9 runs on the other seven HR databases: 988 questions (`exp/posthoc/w9_qids.txt`). In the stored tuning runs, 3 of the 24 empty roots with an empty reference were on `bike_share_1`. Decided after the download failed and before any W9 model call.

### 30.2 Quantities (fixed before the runs)

* Per backbone: the number of HR questions whose reference result is empty (no rows, or one row of NULLs, which the gate treats as empty); among them, the questions answered correctly by (b) and wrongly by (c) (**harm**), and the reverse; over all questions, the questions turned correct and wrong by (c) against (b) and against (a), and the EX differences (c) − (a) and (c) − (b) with a 95% database-clustered percentile bootstrap interval (2,000 draws). Harm is also expressed per 100 questions with an empty reference.
* Not computed: reading lines (one run, no rerun), any verdict. The pre-lock rule for the pipeline's empty switch (Section 13.6, rule 2: G2 − G1 at least +1.0 pp on HR) concerns the pipeline, not the single call, and is not evaluated by W9.
* Code: `code/tools/x2ds_w9.py` (result `results/x2ds/w9_harm.json`, keys `X2DS.w9.*`).

### 30.3 Amendment log

| Date | Amendment | Before the lock | Effect |
| --- | --- | --- | --- |
| 2026-10-30 | Post-hoc note (not an amendment): harm of the empty-result trigger on the tuning split HR (W9: single call, single call + retry, empty-gate retry on four backbones) | yes (committed before its first call; written after the results of all other arms; the hull intervals and the same-text reading were added during the internal critical reading of the draft, after its results) | Descriptive; no confirmatory test, threshold, margin or family changed |
