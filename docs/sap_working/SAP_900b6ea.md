# Statistical Analysis Plan (SAP) v1 (X1, 2026-10-25)

> **Amendment 1 (2026-10-26, before the dev lock, after tuning-set analysis) is in §13.** Sections 1–12 below are the original v1 text and are kept verbatim. Where §13 states that an item is superseded, demoted or redefined, §13 governs; nothing in §1–12 was deleted or silently edited. **Amendment 3 (2026-10-26, §15)** re-ranks the primary family again (family P\*, weak backbones). **Amendment 2 (2026-10-26, §14)** changes who annotates E10 (two independent LLM runs instead of two authors); it is marked inline where it touches v1 text.

Pre-specified before the dev lock. The commit hash of this file is recorded in `exp/dev_lock.json`. Changes after the lock go to §12 (deviation log) with date and reason; the paper reports every deviation. Companion: `docs/EXPERIMENT_SPEC.md` (design, metrics, seeds). Analysis code: `code/analysis/` (S1). All numbers in the paper come from `results/*.json`.

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
| K1 | The full system beats the strongest compute-matched baseline | E1 dev r0 | ΔEX = EX(full) − EX(B*), B* = HR-selected baseline (SPEC §4) | exact McNemar (discordant pairs) + db-clustered bootstrap CI |
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

Fallback if `lme4` cannot be installed on the host: `statsmodels` `BinomialBayesMixedGLM` (variational) with the same structure, flagged in the paper.

## 6. Scores, calibration, selective prediction

- **AUROC:** empirical (ties = ½); comparison by paired db-clustered bootstrap of the difference; DeLong test as a check (item-level). The binary Critic V1 yields a 1-bit score; its AUROC equals balanced accuracy (stated in the paper). Secondary for K2: V3 at the patent operating threshold 8.5 vs V1 — balanced accuracy and MCC.
- **ECE:** 15 equal-mass bins on score/10 after isotonic calibration fitted on HR (cross-fitted) and applied to dev; raw ECE also; Brier; CI by db-clustered bootstrap. Reliability diagrams with 10 bins for display only.
- **Risk–coverage:** AURC computed on the empirical curve (trapezoid over all thresholds, ties broken by index-stable random shuffle with 100 reshuffles averaged); coverage at 10/20 % risk (threshold chosen on the data, reported as an *oracle-free* curve read-out, not a deployable operating point unless the threshold is fixed on HR). The deployable comparison fixes thresholds on HR and reports dev coverage/risk.
- **TrustSQL-style RS(c):** per-item scores as in SPEC §5; mean with CI; c ∈ {0, 1, 5, 10, N}.
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
- If a baseline cannot be completed (e.g., Alpha-SQL cost), the subset rule of SPEC §4 applies and all comparisons for that baseline are restricted to the same subset.
- No outcome-based exclusion, no optional stopping, no re-analysis after dev unblinding other than listed in §12.
- Seeds/stochastic systems: headline r0; r1/r2 reported as mean ± SD and used in the item-mean sensitivity analysis.

## 11. Reporting rules

- Each table cell: estimate [95 % db-clustered CI]; each comparison: ΔEX (pp), CI, exact p, Holm-adjusted p (where in a family), OR.
- Negative or null results are reported with the same prominence (RL controller, WS vs MemRL, learned Critic).
- Claims language: *confirmed* (§2 rule), *supported at item level*, *direction consistent*, *exploratory*. No "significantly" without the adjusted p.
- ~~Author annotation~~ **LLM annotation (two independent Claude Sonnet 5.5 runs; Amendment 2, §14)** (E10): κ reported before adjudication and described as agreement between two runs of one model; "wrong criterion" prevalence labelled exploratory. Never described as human or expert annotation.
- All analysis scripts read only `runs/` and `results/`; macro file generated by script; the SAP commit hash and `dev_lock.json` hash are printed in the supplementary material.

## 12. Deviation log (append only after the lock)

| Date | Section | Change | Reason | Effect on conclusions |
|---|---|---|---|---|
| — | — | — | — | — |

---

## 13. Amendment 1 (2026-10-26; before the dev lock; after tuning-set analysis)

Author: S2 (planning). Status: **proposed to the main session; takes effect when committed**, and must be committed before `exp/dev_lock.json` is written (the commit hash of this file is then recorded in the lock). Sections 1–12 are unchanged.

### 13.0 What was seen, and what that means for the claim "pre-specified"

* **Seen:** BIRD-train tuning data only: HT-300 (4 dbs) and HR-1101 (8 dbs), `qwen3.7-plus`, via `docs/TUNING.md` (R0) and `docs/HEADROOM.md` (R1, offline, 0 API calls). **Not seen:** BIRD dev (questions' gold, predictions or scores), minidev, Spider variants, AmbiQT, LiveSQLBench. No `dev_lock.json` entry exists; `exp/dev_access.log` has no evaluator access.
* **Consequence.** The hypotheses of §13.3 are pre-specified with respect to dev, **not** with respect to the tuning sets: the tuning results shaped which hypotheses we test and how. The tuning results are therefore never used as confirmatory evidence; they are reported as design evidence (supplement, labelled "tuning set"), and every strengthening below is adopted or rejected by a mechanical rule on HR/HT (§13.6) before the lock.
* **Why the amendment.** The v1 family K was built around the claim that the full pipeline improves EX. The tuning data do not support that claim and the v1 power calculation could not have detected the effect sizes now expected (§13.1, §13.7). Keeping K1 and K4 as the headline tests would invite a null result on the wrong question; keeping them as secondary, reported as-is, is the honest option. The amendment does not remove any v1 hypothesis; it re-ranks them (§13.2).

### 13.1 Tuning-set evidence behind the re-ranking

| # | Finding (tuning data only) | Source |
|---|---|---|
| E-1 | `full_fixed` does not beat zero-shot: HT-300 EX 0.693 vs `zs_ours` 0.697 (−0.3 pp, 9 vs 10 discordant of 300) and vs `zs` 0.673 (+2.0 pp). The retrieval stack M1–M4 alone is −1.3 pp vs `zs_ours` | TUNING §1, HEADROOM §1 |
| E-2 | The loop's gain is repair of execution failures: HR P0 +3.5 pp over its own root, 47 of 50 gains are roots that did not execute (47/70 repaired); 41 of those are in `beer_factory` (table `transaction`, a SQLite keyword that `render_ddl._q` does not quote: 54 identical `near "transaction": syntax error`). Without `beer_factory` P0 is −0.2 pp vs root | HEADROOM §0.1 |
| E-3 | Semantic revision of executable roots is net negative (HR 3 gains / 11 harms; HT 1 / 1). The LLM semantic flag (score < 7) has recall 14 % (HR) / 19 % (HT) of wrong executable roots; 296 of 415 wrong HR roots get semantic = 10 | HEADROOM §0.2, §1 |
| E-4 | Execution-only gating ("revise only on error/timeout") replays at EX 0.666 vs P0 0.658 (+0.7 pp [−0.3, +1.7], 11 W / 3 L) at ≈ 2.1 vs 3.4 calls/q (the semantic Critic call is not made on executable roots) | HEADROOM §3. Caveat: replay on candidates that were generated with the Critic's text; live confirmation pending |
| E-5 | Selection, voting and arbitration over the stored candidates recover +0.4 … +1.3 pp over P0 (CIs reach 0); a perfect root-vs-revision arbiter would add at most +1.5 pp; 24.8 % of HR questions are unreachable by 23.5 candidates | HEADROOM §0.3, §1, §3 |
| E-6 | Routing: V3 AUROC for root correctness 0.71 (HR) / 0.74 (HT); a consistency-augmented score (LR on V3, agreement, non-empty; leave-one-db-out) 0.75 on HR, EX of the accepted 60 %: 0.735 → 0.782. **AURC on HR roots is unchanged** (0.2625 → 0.2605): the gain is in the mid-coverage region, not over the whole curve. The cheap version (2 extra T = 0.7 samples) is unverified | HEADROOM §2 and `results/r1/q2_rc_matched.json` |
| E-7 | RL controller: no PPO cell reaches P0 under database-grouped CV (best cell 0.635 vs 0.660, −2.5 pp); A2C collapses to "accept the root". Action values are flat on executable roots | R0 §1 (E6), HEADROOM §5.5 |
| E-8 | E4 mini-stream (p = 40 %): `ws_full` 0.715 / 0.699 (two replicates) vs `memrl` 0.707 / 0.695; trial-to-trial noise ≈ 1 pp; about 90 % of evicted entries are clean; the `ws_full` ρ = 0 optimum is `ws_evict` in disguise | TUNING §3 |
| E-9 | All 1 401 HR/HT questions took the full-schema fallback (< 6 000 tokens): M3 (routing, RRF, rerank) never ran; k_t × k_e grid spans 0.677–0.693, no effect distinguishable from noise | TUNING §2, HEADROOM §4 |

### 13.2 Re-ranking of the v1 hypotheses (nothing is deleted)

| v1 id | v1 claim | Status after Amendment 1 | New id and role |
|---|---|---|---|
| K1 | `full` beats the strongest compute-matched baseline on EX | **Demoted** to secondary. Reported as-is; no superiority claim is planned (expected effect 0 … +1 pp, below the MDE, §13.7) | S-A row "full vs B*" (two-sided, Holm in S-A); its role as headline is taken by P2 (non-inferiority) |
| K2 | V3 (weighted, grounded) discriminates better than V1 (binary) | **Demoted** to secondary (HEADROOM: V3's advantage is mostly the execution term) | S-C, now m = 4 (§13.5) |
| K3 | Value-weighted corpus resists pollution (E4, p = 40 %, `ws_full` vs `append`) | **Kept unchanged** as primary | **P4** (H-pollution) |
| K4 | RL controller dominates the fixed policy (routes A, B) | **Demoted**; expected negative; reported as a negative result with the test as specified | S-D (secondary), headline cell only on dev (§13.8) |
| K5 | Critic-score routing has lower AURC than consistency confidence | **Redefined**: the routing score becomes a consistency-augmented score; primary comparison is vs V3; primary metric AUROC (reason in P3) | **P3** (H-route); the v1 comparison vs consistency confidence is kept in S-F |
| (new) | Cost: execution-gated loop is non-inferior at lower cost | – | **P1, P2** (H-cost) |
| (new) | Execution-failure repair by the loop | – | **P5** (H-repair) |
| v1 "C3 RL controller", "C1 weighted Critic improves EX" | contributions in `STORYLINE.md` | demoted to secondary / negative result | `STORYLINE.md` Revision 1 |

The v1 family K (m = 5) is **replaced for confirmatory purposes** by family P (m = 5, §13.3). K1, K2, K4, K5 keep their v1 definitions and are tested and reported as secondary/supportive; their p-values are Holm-adjusted inside the secondary families of §13.5, not in P.

### 13.3 Primary confirmatory family P (m = 5)

Common conventions: two-sided α = 0.05 family-wise; headline run r0 on BIRD dev (1 534 items, cluster = `db_id`); backbone `qwen37plus`; all systems use the identifier-quoting renderer (§13.6 item 1); CIs are db-clustered percentile bootstrap (B = 10 000, §3) with the item-level paired bootstrap, the leave-one-db-out range and the share of dbs with the same sign reported next to them; sensitivity without `thrombosis_prediction` as in §8. One-sided tests (non-inferiority, ratio, "better than") enter Holm as p_2s = min(1, 2·p_1s) (the convention already used for K4); a one-sided p-value is the smallest α for which the (1 − 2α) db-clustered percentile interval excludes the null. An exact/score-based item-level check is reported with each.

**P1 — H-cost (internal): the execution-gated loop keeps the accuracy of the fixed semantic-critic loop at lower cost.**
* Systems: `full_gated` vs `full_fixed` (same backbone, same renderer, same frozen M6 burn-in store, same root prompts, so identical root generations are cache hits and the discordance comes from the revision step only).
* (a) ΔEX = EX(gated) − EX(fixed). H0a: ΔEX ≤ −1.5 pp. Margin rationale: below every loop gain observed on tuning data (+3.5 pp HR pre-fix, +1.0 pp HT), so "non-inferior" means "keeps essentially all measured loop value"; a looser margin would be resolvable but vacuous.
* (b) R = Σ calls(gated) / Σ calls(fixed), calls = LLM calls of the answer path (M1, generation, Critic when it is called, revisions, arbitration if enabled); embedding/rerank calls and the routing-score calls of P3 are reported separately. H0b: R ≥ 0.80. Design value R ≈ 0.62 (2.1 / 3.4, replay).
* Intersection–union: p(P1) = max(p_a, p_b); both bounds must hold. Verdict language: "non-inferior" only if the db-clustered CI lower bound of ΔEX > −1.5 pp; "non-inferior at item level" if only the item-level interval satisfies it.
* Also reported (not tested): EX by root status (execution failure / empty / non-empty), tokens/q, latency, tokens per correct answer, ΔEX at margins 1.0 / 2.0 / 3.0 pp.

**P2 — H-cost (external): at its cost, `full` is not worse than the best baseline that costs no more.**
* `full` = the system chosen by §13.6 rule 3. B* = the baseline with the highest HR EX among all baselines (zero-shot and DAIL included, because they are cheaper) whose HR mean calls/q ≤ 1.15 · B_ref(`full`), with B_ref(`full`) = HR mean calls/q of `full` measured live; ties → fewer calls. This replaces the v1 compute-matching window [0.85, 1.15] · B_ref(`full_fixed`) for P2 only; baselines with a budget knob are matched to both budgets (§13.8).
* ΔEX = EX(full) − EX(B*). H0: ΔEX ≤ −3.0 pp. The margin is wider than P1's because the paired discordance is larger (§13.7): at n = 1 534 a 1.5 pp margin against an independent baseline is not resolvable, and a margin we cannot resolve would only produce an uninformative "not confirmed". It equals the smallest ΔEX between two different systems that this study can detect at all (≈ 3 pp for near-neighbour pairs), i.e. we claim "no difference larger than the resolution of the study".
* **Reading rule written in advance:** if B* is zero-shot or DAIL-style, P2 says only that the loop does not cost accuracy relative to a cheaper alternative; it is not evidence that the loop is worth its cost. The paper must say so (`STORYLINE.md` limitation paragraph).

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
| RL controller | v1 K4 (routes A and B) at the pre-registered headline cell plus three further cells; CV table; replay-fidelity gap | Secondary; presented as a negative or neutral result when the test fails (`STORYLINE.md` R5) |
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
| 5 | Arbitration call (one judge call when root and revision differ) | Adopted into `full_gated` iff on HR the judge picks the right member in ≥ 70 % of the mixed cases **and** ΔEX vs `full_gated` ≥ +1.5 pp with CI lower bound > 0 (HEADROOM bound: at most +1.5 pp) | Not adopted; reported in the supplement as a tuning-set negative result; no dev run. If adopted, its calls enter R in P1 |
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

### 13.8 Revised dev run list (replaces the run lists of EXPERIMENT_SPEC §6 E1–E3, E5–E7 where they conflict; E4, E4c, E8–E11 unchanged except that `full` means the §13.6-rule-3 system)

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

Order of magnitude without E4: ≈ ¥1.2–1.6 k. No change to the ¥8 000 soft cap logic of CONTRACT §7.

### 13.9 Claim-language additions (extend §11)

* Allowed verbs for P1, P2, P5: "non-inferior within δ", "reduces calls by x % (CI)", "repairs x % of execution failures". Not allowed unless S-A confirms: "improves accuracy", "outperforms", "state of the art".
* Every EX table that contains `full` and `zs` reports ΔEX with its CI in the same table. No sentence may say that the pipeline "improves" EX over zero-shot if that CI includes 0.
* Tuning-set numbers (E-1 … E-9) appear only in Setup (rationale for design choices), the supplement, and Discussion when labelled "tuning set".
* The negative results (RL controller, `route_ca` if not adopted, P5 if n_R < 40, P4 if not confirmed) are reported in the main text with the same prominence as positive ones.

### 13.10 Items outside the SAP that follow from this amendment (owners in `docs/handoff/S2.md`)

EXPERIMENT_SPEC §4 (add `full_gated`, `full_gated_blind`, `route_ca`, legacy-renderer arm; B_ref rule), §6 E2/E3/E6/E7 (run lists as in §13.8), §9 (decisions table); `exp/selection_decisions.json`; `analysis/power.py` re-simulation; `results/primary.json` keys P1–P5.

### 13.11 Amendment log

| Date | Amendment | Author | Before the lock | Effect |
|---|---|---|---|---|
| 2026-10-26 | Amendment 1: confirmatory family K → P; K1, K2, K4, K5 demoted/redefined, K3 kept; margins, adoption rules, power, run list | S2 | yes | Headline claims move from accuracy to cost, routing, pollution, repair; all v1 hypotheses remain in the paper as secondary |

## 14. Amendment 2 (2026-10-26; before the dev lock; no outcome data seen)

Author: F0 (prep). Status: proposed to the main session; takes effect when committed, before `exp/dev_lock.json` is written. Trigger: the user's minimal-validation decision (`docs/MINIMAL_PLAN.md`, run M6) and the fact that no human annotator is available.

### 14.1 Change

| Item | v1 / Amendment 1 | Amendment 2 |
|---|---|---|
| E10 annotators | two authors (A1, A2) label independently; a third author adjudicates | two independent runs of **Claude Sonnet 5.5** (sub-agent runs A1, A2), each blind to the system and to the automatic class, each with its own random item order, no access to the other run's output; a third Claude Sonnet 5.5 run adjudicates the disagreeing items with the two labels in blinded positions |
| Instructions | `docs/ANNOTATION_GUIDE.md` | the same guide, delivered through the verbatim prompt in `docs/ANNOTATION_PROMPT.md` (frozen after the 30-item pilot) |
| Sample, strata, weights, labels L1-L8, secondary label, rationale rule | as in SPEC E10 | unchanged |
| Agreement statistics | κ (primary label, L5-vs-rest) with item bootstrap, AC1, percent agreement, confusion matrix | unchanged, computed by `code/tools/annotation_kappa.py`; reported before adjudication; **described as agreement between two runs of one model** |
| Prevalence | stratum-weighted, stratified bootstrap | unchanged; all E10 prevalences are exploratory (as before) |
| Human involvement | "author annotation" | none in the labelling. **Optional add-on, reported only if actually carried out:** the authors label about 50 randomly drawn items (seed 20261027, drawn after the LLM labels are frozen, blind to them) and report percent agreement and κ against the adjudicated LLM label. The paper contains a clearly marked placeholder for it; if the spot-check is not done, the placeholder and every mention are deleted |

### 14.2 Claim language (extends §11 and §13.9)

* Allowed: "LLM annotation (two independent Claude Sonnet 5.5 runs)", "agreement between two annotation runs of the same model", "model-assigned error labels".
* Not allowed, for the LLM labels: "author annotation", "human annotation", "manual annotation", "expert annotation/review", "ground truth". The CONTRACT §0 rule (no "expert review"; human parts are called by who did them) is kept in spirit: the paper never claims human or expert annotation that did not happen.
* Required limitation sentence (Discussion, threats to validity): "Both annotators are runs of the same language model; their agreement measures consistency under the guide, not validity against human judgement, shared biases are invisible to it, the model may have seen BIRD during training, and the gold-suspect and ambiguity labels are model judgements."
* The gold-suspect rate (L6) is an estimate by the model; it is not used to correct any gold label (SPEC §1.1 rule "we do not correct labels ourselves" stays).

### 14.3 What was seen

Nothing outcome-related: this amendment concerns the labelling procedure only. The annotation sheet is built after the dev runs exist (frame = failures of `zs` and `full`); the amendment is written before that.

### 14.4 Amendment log

| Date | Amendment | Author | Before the lock | Effect |
|---|---|---|---|---|
| 2026-10-26 | Amendment 2: E10 annotators = two independent Claude Sonnet 5.5 runs; optional human spot-check | F0 | yes | E10 labels and κ are model-based and exploratory; no change to any confirmatory test |
| 2026-10-26 | Incident note (not an amendment): during a lock-gate dry run, `nl2sql.run --split bird_dev --limit 0` (0 = no limit) loaded the dev question list and one dev schema and made 10 embedding calls (about 5 100 tokens, < CNY 0.01) before crashing on a dummy store. No chat-model call, no dev scoring, no use of gold SQL results. Artefacts deleted; `exp/dev_lock.json` did not exist | F0 | yes | None on any estimate; disclosed for the audit trail |

## 15. Amendment 3 (2026-10-26; before the dev lock; after the tuning-set analysis of the weak backbones, E4 mini-stream, X5 live checks and X7/X9 offline bounds; no dev data seen)

Author: N2 (planning, S3). Status: **proposed to the main session; takes effect when committed**, and must be committed before `exp/dev_lock.json` is written (the commit hash of this file is then recorded in the lock). Sections 1-14 are unchanged; Amendment 1 (§13) is re-ranked, not deleted. Every tuning number below is design evidence from BIRD-train data (HT-300: 4 databases; HR-1101: 8 databases) and is read from `results/x5/x5_tuning_summary.json`, `results/x8/x8_tuning_summary.json`, `results/x8/x8_pooled.json`, `results/e4_min/e4_p40_summary.json`, `results/p2/*`, `results/r1/*`, `results/x7/*`, `results/x9/*`. None of it is confirmatory.

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
| P1 (gated ≈ fixed on `qwen3.7-plus`, calls ratio) | **Extended** to the backbone set Bk = {`qwen3.7-plus`, `qwen3.7-flash`, `qwen3.5-27b`} and pooled | **P1′** (primary, cost, internal) |
| (new) | – | **W1** (primary): on the weak backbones the gated pipeline improves over zero-shot |
| P2 (full vs best baseline at no higher cost) | **Demoted**: B\* is selected on dev among the few baselines MINIMAL_PLAN runs, so the test is conservative and uninformative about the loop | S-A row "full vs B\*" (two-sided CI, margin 3.0 pp reported, Holm in S-A) |
| P3 (routing) | Kept, with k = 4 extra samples instead of 2 and the gate re-applied (see P3 text); drops out if the k = 4 gate fails | **P3** (primary if adopted) |
| P4 (pollution) | **Demoted** to secondary; stays pre-registered and is run as specified; tuning data point to a null or reversed effect (E-16a) | S-E headline row (GLMM contrast `ws_full` − `append`, two-sided); Holm inside S-E |
| P5 (repair) | **Kept**, population pooled over Bk so that the n_R ≥ 40 rule can be met | **P5′** (primary) |
| (new) negative-result reports N1-N6 | see §15.4 | reported with the same prominence |

### 15.3 Primary confirmatory family P\* (m = 4)

Replaces family P of §13.3. Conventions of §13.3 hold (two-sided α = 0.05 family-wise, run r0, cluster = `db_id`, B = 10 000 db-clustered paired percentile bootstrap with a **common database resample across backbones**, one-sided p doubled into Holm, item-level exact checks, verdict language of §13.3, sensitivity without `thrombosis_prediction`). All systems use the quoting renderer. Backbones are fixed at the lock; the weak set Wk = {`qwen3.7-flash`, `qwen3.5-27b`} may be extended by **one** further backbone (`qwen3.6-35b-a3b` or `deepseek-v4.1-flash-1017`) only if the budget stop rule of `exp/budget.py` allows it **before** the lock; the choice is then written into `dev_lock.json`. No backbone is dropped or added after the lock; a run that fails for technical reasons is repeated, not removed.

**P1′ — H-cost (internal): the execution-gated loop keeps the accuracy of the fixed semantic-critic loop at lower cost, on average over Bk.**
* Systems: `full_gated` vs `full_fixed`, same backbone, same renderer, same frozen M6 store, shared roots (cache hits), per backbone b ∈ Bk. Dev runs: plus (MINIMAL_PLAN M1), flash and 27b (new: `full_fixed` added, see `MINIMAL_PLAN.md` Amendment).
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
| N1 | Learned (PPO/A2C) controller | S-D as in §13.4 on the tuning-set CV only; **no live RL run on dev** (MINIMAL_PLAN). Wording: "we could not detect a benefit" | best of 12 cells 0.635 vs 0.660 |
| N2 | Value-weighted eviction (P4, S-E) | Run as specified (E4, p ∈ {0, 40 %}); GLMM contrast `ws_full` − `append` two-sided; `ws_full` vs `memrl` as in §13.5. If `ws_full` ≤ `append`, the paper says the corpus rule does not protect against pollution under this protocol and reports the eviction diagnostics (recall, precision, clean-entry rate) | 0.688 vs 0.703; 778 of 847 evictions clean |
| N3 | Arbitration judge | Not adopted (§13.6 rule 5 bar not met); not run on dev; supplement | 52 % vs 70 % bar |
| N4 | Forced M3 routing (`schema_fallback_tokens` = 0) | Not adopted (§13.6 rule 6); exploratory dev arm `gated_m3force` kept (MINIMAL_PLAN M3) | −5.3 pp [−8.7, −1.8] |
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

`docs/MINIMAL_PLAN.md` Amendment (run list), `docs/STORYLINE.md` Revision 2, `exp/queues/gen_min.py` (jobs for weak-backbone `full_fixed` and `full_gated_blind` on three backbones, optional third weak backbone), `freeze_configs.py` (+ `gated_blind`), `analysis_pipeline/primary.py` (family P\*: P1′ pooled over Bk with a common database resample, W1, S-J; keys in `docs/handoff/N2.md`), `code/tools/selection_rules.py` and `exp/selection_decisions.json`: rule 3 (§13.6) was evaluated on 2026-10-26 and selected `full := full_fixed` because the lower 95 % bound of the HR difference was −1.55 pp against the −1.5 pp margin (one-sided bound −1.40; conditions (ii) and (iii) were met). That decision stands and is not changed here. Consequence for this amendment: the primary claims P1′ and W1 **name their systems** (`full_gated`, `full_fixed`, `zs`) and do not use the label `full`; `full` (= `full_fixed`) only fixes the E1 row labels and the S-A comparison (`full_gated` is still run on every backbone). The paper says in Setup that the mechanical rule picked the fixed loop on a borderline bound and that P1′ tests the question on dev.

### 15.8 Amendment log

| Date | Amendment | Author | Before the lock | Effect |
|---|---|---|---|---|
| 2026-10-26 | Amendment 3: primary family P → P\* (P1′ extended to three backbones, W1 new, P3 kept, P5′ pooled); P2 → S-A, P4 → S-E; negative-result list N1-N6; run list extended to weak backbones | N2 | yes | Headline claims move to cost-effectiveness across backbones, routing and repair; negative results get equal prominence; no v1 or A1 hypothesis is deleted |
