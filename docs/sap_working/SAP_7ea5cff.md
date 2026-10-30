# Statistical Analysis Plan (SAP) v1 (X1, 2026-10-25)

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
| E10 | auto class distribution; κ, AC1 (item bootstrap, 2 000 reps); stratum-weighted prevalence of L1–L8 with stratified-bootstrap CI | descriptive |
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
- Author annotation (E10): κ reported before adjudication; "wrong criterion" prevalence labelled exploratory.
- All analysis scripts read only `runs/` and `results/`; macro file generated by script; the SAP commit hash and `dev_lock.json` hash are printed in the supplementary material.

## 12. Deviation log (append only after the lock)

| Date | Section | Change | Reason | Effect on conclusions |
|---|---|---|---|---|
| — | — | — | — | — |
