# Analysis runbook (S1b, 2026-10-26)

From finished runs to `results/*.json`, `paper/results_macros.tex`, `paper/tables/*.tex` and `figs/*`. Nothing here calls an LLM API. Red line (CONTRACT 0): a number reaches the paper only as `\res{KEY}`; a missing run gives a missing key (`not run`), never an estimate.

## 1. Running the analysis

The one-command wrapper of the original work area drove several compute machines and is not part of this release. The stages are run directly:

Single stages: `python -m analysis_pipeline.run_all --runs $R/runs --results OUT --only primary E1 ...` (stages: primary W1 E1 E2 E3 E4 E5 E7 E8 E9 E10 setup tuning E6 X1; E11 is written with E1); `python -m analysis_pipeline.primary --runs ...` for P1-P5 only; `python -m analysis_pipeline.selftest`  for the unit/planted-effect/guard tests (42 checks).
Host Python is 3.6 (stdlib): only `make_macros.py` and `make_tables.py` run on the host.

## 2. Inputs and how runs are found

* `runs/<exp>/<run_id>/preds.jsonl` (CONTRACT 4) + `run_meta.json` + `config.yaml` (+ `cands.jsonl`, `usage*.jsonl`). The tree under `--runs` is scanned recursively. A run is **complete** only with >= 1 534 rows (dev; other splits: the largest run found). Incomplete runs are never used (a note says why).
* Roles are resolved from (system, config stem, model key, seed, split): `full_fixed` = system `full_fixed` (config `full_fixed`/`full_quote`); `full_gated` = system `full_gated`, config `full_gated`; `full_gated_route` = `full_gated` with config `full_gated_cons_lr|full_gated_cons|full_gated_route` (routing fields); `full_gated_blind`; `abl_m1|m3|m3force|m5|m6|q`; baselines by system name. Table: `runs.LABELS`. **Ambiguity is an error** (two complete runs for one role) -> pin in `exp/analysis_manifest.json`: `{"runs": {"full_fixed": "MIN/M1/fixed-s0", "full_gated:qwen37flash:r0": "MIN/M5/gated-flash-s0"}}` (path relative to `--runs`; key = `label`, `label:r<k>` or `label:model:r<k>`).
* `exp/selection_decisions.json` (tools/selection_rules.py): `full` (default `full_gated`), `route_ca_adopted` (false -> P3 dropped, m = 4), `bstar` (pre-registered B*, else chosen on the data, see 4.P2), `m_family`.
* E4: `--e4-roots runs/MIN/E4`; units `<policy>-p<pp>-perm<k>-<db>/{events.jsonl,pollution.json,done}` as written by `nl2sql.stream e4`; units without `done` are skipped. Outcome = `ex_all` (abstained = wrong) of the probe items of the **last** probe event.
* E9 `--e9-dirs` (tools/e9_probes.py `summary.json`), E10 `--annot-dir` (layout in the `e10.py` docstring), routing `--route-coef exp/frozen/route_coef_hr.json --route-k 2` (recompute `route_ca` from frozen coefficients; default: use the in-run `route_score`), `--calib-preds` (HR run for isotonic calibration of E7).

## 3. Scope labels (so that dry runs cannot leak)

Every `results/*.json` carries `_meta.scope`: `confirmatory` (dev split, or E4 HR streams), `tuning-export` (the `T.*` export of `results/r1|p2`), `tuning-set` (any analysis of a non-dev split, e.g. `bird_train_HR/HT` = X5 runs), `synthetic` (runs tree contains a `SYNTHETIC` marker). Enforcement: `Result.write` refuses to write `tuning-set`/`synthetic` into the repo's `results/`; `make_macros.py` and `make_tables.py` refuse such files (exit 4/1); figures need `--synthetic` for synthetic input and then only write below a directory starting with `_`. `primary.json` also carries `P.scope`. Dry runs on X5 therefore go to a scratch `--results` directory and are labelled `tuning-set`.

## 4. The five primary tests as implemented (SAP 13.3, 13.5)

Common: headline run r0, backbone `qwen37plus`; db-clustered percentile bootstrap B = 10 000 (seed 20261101) with item-level paired bootstrap, leave-one-db-out range and share of dbs with the same sign beside it; sensitivity without `thrombosis_prediction`; p_1s = (#bootstrap draws on the wrong side of the null + 1) / (B + 1); doubled p_2s = min(1, 2 p_1s) enters Holm; Holm over m = 5 (4 when P3 is dropped by decision or P5 has n_R < 40; a test whose data are missing stays in the family with p = 1 and is shown as `not run`). `p_adj` = Holm on the db-clustered p; `p_adj_item` = Holm on the item-level check (Tango score test, DeLong, exact McNemar); `verdict` follows SAP 13.3 (`confirmed`, `supported at item level, not at database level`, `direction consistent, not confirmed`, `not confirmed`, `not run`, `descriptive only`, `dropped before the lock`).

| Test | Data | Statistic / test | Key prefix |
|---|---|---|---|
| P1 | `full_gated` vs `full_fixed` r0 | (a) dEX, H0 <= -1.5 pp; (b) calls ratio R = sum calls(gated)/sum calls(fixed), H0 >= 0.80; IUT p = max; also tokens, latency, tokens/correct, sens. margins 1/1.5/2/3 pp | `P1.*` (`P1.delta_ex`, `P1.calls_ratio`, `P1.calls_saved`) |
| P2 | `full` vs B* | dEX, H0 <= -3.0 pp. B* = pre-registered choice, else EX-maximal baseline **among those actually run** with calls <= 1.15 x calls(full) (ties: fewer calls; chosen on dev because MINIMAL_PLAN has no HR baseline runs: conservative for `full`, stated in `P2.bstar_selected_on`); `P2.bstar_is_cheap` + `P2.reading_rule` carry the mandatory reading rule | `P2.*` |
| P3 | `full_gated_route` final answers | dAUROC(route_ca - V3), one-sided; paired db-bootstrap, DeLong check; co-reported AURC, accepted-EX at 40/60/80 %, RS(c); `P3.aurc_favourable` gates the wording "improves the risk-coverage trade-off" | `P3.*` |
| P4 | E4 units, p = 40 %, final probe | GLMM `ex ~ policy + (1|db) + (1|qid) + (1|perm)` (own Laplace implementation in `glmm.py`, scipy), LRT p; comparator `append` (or `no_evict`); p = 0 no-harm contrast (margin 2 pp) | `P4.*`, S-E rows in `E4.*` |
| P5 | items whose root fails to execute (`cands.jsonl`) | rate of final EX = 1 (CP interval), repair by error class (Wilson), vs `full_gated_blind` exact one-sided McNemar on R; in the family only if n_R >= 40 **and** the blind run exists, else `descriptive` | `P5.*` |

E4 definition used for P4/S-E (EXPERIMENT_SPEC E4, `stream.summarize_trial`): unit = (policy, pollution p, permutation, database); probe items are the same per db in every unit; each unit gives one probe event per checkpoint (0/40/100 % of the stream); the pooled probe EX of the last checkpoint is the outcome; eviction diagnostics from the `diag` blocks. MINIMAL_PLAN: policies `ws_full`, `memrl`, `append`; p in {0, 40}; permutations 1, 2.

## 5. Other outputs (all flat keys, `E<k>.json`)

`E1` (+`E11`): per system `E1.<sys>.<model>.{ex,soft_f1,jaccard,silent_error,calls,tokens,cost_per_q,latency_*,...}` with CI, `E1.vs_zs.<sys>.<model>.delta_ex`, `E1.full_vs.<sys>.*` (S-A, Holm `p_adj`), `E1.full_vs_best.*`, `E1.perdb.*`, `E1.perdb_delta.<a>_vs_<b>.<db>.*` (+ LOO), `E1.rootstatus.*`; `E2` per-model gain and Spearman; `E3` ablations, Cochran Q, error classes re-introduced, `E3.sem_revision.*`; `E4` curves, slopes, S-E contrasts (`E4.memrl_vs_ws.p40.*`), eviction diagnostics, no-harm; `E5` stored-candidate AUROC/ECE of V1/V3 (end-to-end V1-V4 runs were dropped by MINIMAL_PLAN); `E7` per score AUROC/AURC/accepted-EX/RS/ECE/Brier, curves, bands, `E7.<a>_vs_<b>.*`; `E8` Spider variants; `E9`; `E10` (auto classes, kappa/AC1, L1-L8 prevalence, `E10.spot.*` only if `spot_check.csv` exists); `setup`; `tuning` (`T.*` from `results/r1|p2`); `E6.status` (live RL runs dropped). Key formats for the text: `make_macros.py` docstring; `K` -> `K.mean`, `K.ci` = `[lo, hi]`, aliases for the STORYLINE R3/R4 short names (`ALIASES`).

Not implemented (reported as missing keys with producer): corrected-gold re-scoring (`E1.corrected.*`), AmbiQT ambiguity detection (`E7.ambiqt.*`), `E10.spot.*` unless the optional human spot-check file exists, `setup.devlock.hash` until `exp/dev_lock.json` exists.

### 5b. Backbone family (SAP Amendment 3, section 15; `weak.py`, added 2026-10-27, see `internal note KEYS`)

`W1.json`: W1 (pooled gated-vs-zs over qwen37flash and qwen3527b, sign-consistency rule), `P1.<backbone>.*` (S-J, secondary), `P1prime.*` (pooled P1', all of Bk), `Pstar.*` (family P* Holm: `W1.p_adj`, `W1.verdict`; P3 enters from primary.json, P5' is counted but its pooled test is not run). `X1.json`: S-I flash gated vs plus zero-shot (descriptive). `E3.<backbone>.{loop_over_root,rootonly_vs_zs}.*`, `E4.<policy>.p<pp>.{evicted_clean,evicted_polluted,evicted_total,recall_polluted}`, `setup.{design_effect,resolution_pp,p4_mde_pp}` (SAP 9 / 13.7 definitions) are written by the existing E3 / E4 / setup stages. primary.json keeps the SAP 13.3 family; the two families are different objects (KEYS.md section 2).

## 6. Tables and figures

`code/tools/make_tables.py` writes `paper/tables/tab_{primary,e1,ablation,backbone,routing,pollution,taxonomy,rootstatus}.tex`; cells are `\res{}` references only, rows appear when their keys exist, otherwise a "not run" row; add `\input{tables/tab_x}` where the text needs one (labels `tab:primary`, `tab:e1`, ...; needs `booktabs`, `tabularx`, `float`). Figures (`analysis_pipeline/figures/make_all.py`): `fig_risk_coverage`, `fig_cost_accuracy`, `fig_pollution`, `fig_perdb_delta` (PDF + PNG, Okabe-Ito palette). Synthetic previews: `figs/_preview/` (watermarked "SYNTHETIC - NOT RESULTS"; delete before submission; `.gitignore` is not set for them, do not add them).

## 7. Checks run (2026-10-26, a worker machine)

`selftest`: 42 checks pass (scipy agreement for McNemar/CP/Wilson/AUROC/chi-square; Holm; GLMM recovery and coverage; planted effects recovered and nulls stay null; not-run and incomplete-run handling; scope guards). Synthetic end-to-end (`analyze_all.sh --selftest`): all 13 stages ok, 25 of the 27 `\res` keys of the current paper skeleton resolve, tables compile under pdflatex with 248 keys resolved.
Dry run on X5 tuning runs (scope `tuning-set`, never merged): see `internal note S1` section 3.

## 8. Known limits

* Bootstrap p-values have resolution 1/(B+1): with B = 10 000 the smallest p_1s is 1e-4 (doubled 2e-4). Do not run the confirmatory pass with B < 10 000.
* Percentile intervals with 11 (dev) or 8 (HR) clusters under-cover; the item-level and LOO columns are the SAP's mitigation, and `verdict` already distinguishes the two levels.
* The GLMM is a self-written Laplace fit (no R/lme4 on the worker machine); `P4.glmm_converged`, `sigma_*` are reported; the SAP's R/lme4 route was not used. Cross-check with lme4 before submission if R becomes available (`glmm.py` docstring has the model).
* Isotonic calibration for E7 needs `--calib-preds`; without it ECE is raw only (`E7.calibration` says which).
