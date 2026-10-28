# HEADROOM (R1, 2026-10-26): where is the headroom of `full_fixed`? Offline analysis, tuning data only

Data: HR-1101 (8 BIRD-train dbs: live run `P2_HR`, 1 101 branching trees `P2_E6/trees_1101.jsonl`) and HT-300 (4 dbs: `P2_base` full_fixed, zs_ours, zs). qwen3.7-plus only. No dev/minidev/Spider/AmbiQT, **no LLM call** (the only new computation is re-executing 7 121 stored SQL strings, gold included, on the read-only sqlite files). CIs = db-clustered bootstrap (HR: 8 clusters, HT: 4, so they are wide; 1 000-2 000 draws). Scripts: `code/tools/analysis_r1/` (run order 01 -> 09; `common.py`, `trees.py`, `execdata.py` are helpers), outputs `results/r1/q*.json` (worker machine copies in `$R/r1/out/`). "Replay" = policy re-run on cached candidates. Tree calls are modelled as root 3 (M1+gen+critic), revision 2 (revise+critic); replay P0 = 3.38 calls/q vs 3.37 live.

## 0. Bottom line

1. **The measured gain of the loop is execution-error repair and nothing else.** P0 replay on HR: +3.5 pp over its own root (0.623 -> 0.658; 50 gains / 11 harms). 47 of the 50 gains are roots that failed to execute (70 roots; 67 % repaired), 41 of them in one database (`beer_factory`: the table is named `transaction`, a SQLite keyword; `render_ddl._q` only quotes names with special characters, so the schema shows `CREATE TABLE transaction` and the model copies it unquoted: 54 identical `near "transaction": syntax error`). Without `beer_factory`: P0 -0.2 pp vs root. On executable roots the semantic revision is net negative (HR 3 gains / 11 harms; HT 1 / 1).
2. **The LLM semantic critic is a weak detector:** on executable roots its "revise" flag (semantic < 7) has recall 14 % (HR) / 19 % (HT) of the wrong answers at precision 73 % / 94 %; 296 of 415 wrong HR roots get semantic = 10. AUROC of V3 for root correctness 0.71 (HR) / 0.74 (HT); within-db 0.69 (0.66 without `beer_factory`).
3. **Selection/voting over the stored candidates cannot recover more than ~+1 pp** (best offline selectors +0.4..+1.3 pp over P0, CIs reach 0) even though oracle-in-pool is +2.6..+4.8 pp. 24.8 % of HR questions (273) have no correct node in 23.5 candidates (incl. qwen3.8-max revisions); 21 % of all questions are "stable wrong" (>= 80 % of nodes agree on one wrong result). That is a ceiling from systematic misreading and/or gold noise, not something a critic can fix.
4. **Defensible claims:** (a) execution-grounded Critic repairs execution failures (67 %), at lower cost if the semantic critic is not the revision trigger; (b) selective prediction / human routing: a consistency-augmented score gives a better risk-coverage curve than V3 alone (AUROC 0.71 -> 0.75 HR, 0.70 -> 0.77 HT; EX of the accepted 60 %: 0.735 -> 0.782 HR, 0.800 -> 0.835 HT); (c) cost: exec-only gating matches P0 EX at ~2.1 instead of ~3.4 calls/q. An accuracy claim "full beats zero-shot by >= 2-4 pp" is not supported by any number here; the expected dev gain over P0 is 0..+1 pp, below the SAP MDE (2-5 pp).

## 1. Oracle headroom, harm, per-db, depth (HR trees)

Root EX 0.623 [0.523, 0.702] (tree root; live-run root 0.617). Live P0 0.655 [0.562, 0.709]; replay 0.658 [0.558, 0.719].

| candidate set (oracle = any member correct) | oracle EX [95 % CI] | gain vs root (pp) | nodes/q | calls/q if all generated |
|---|---|---|---|---|
| root only | 0.623 [0.523, 0.702] | 0 | 1 | 3.0 |
| P0 chain (root + serial T=0 revisions, <= 3) | 0.686 [0.589, 0.738] | +6.3 | 4.0 | 9.1 |
| root + depth-1 revise (T=0, T=0.7) | 0.685 [0.590, 0.736] | +6.2 | 2.1 | 5.2 |
| root + depth-1 re_retrieve | 0.649 [0.552, 0.728] | +2.6 | 2.4 | 5.9 |
| root + depth-1 escalate (qwen3.8-max reviser) | 0.681 [0.588, 0.734] | +5.8 | 2.0 | 5.0 |
| root + all depth-1 | 0.707 [0.620, 0.755] | +8.4 | 4.5 | 10.1 |
| root + depth <= 2 | 0.745 [0.659, 0.793] | +12.2 | 14.0 | 29.3 |
| whole tree (23.5 nodes, 48 calls) | 0.752 [0.670, 0.799] | +12.9 | 23.5 | 48.5 |
| whole tree, tier-1 model only | 0.715 [0.631, 0.762] | +9.2 | 13.5 | 28.3 |

* Ideal gate (accept right roots, take the P0 chain's correct node otherwise): 0.686, i.e. **+2.7 pp max over P0 from a perfect accept/revise decision**.
* Harm: live run root right -> final wrong 11, root wrong -> final right 53 (paired +3.8 pp [-0.7, +11.9]); replay 11 / 50. Of the 11 replay harms 3 are abstentions (route-to-human at score < 7.5), 8 wrong revisions. Forced T=0 revision of a right root breaks it in 3.6 % of cases.
* First correct node among the 415 wrong roots: depth 1: 92 (22 %; revise 68, re_retrieve 18, escalate 6), depth 2: 42, depth 3: 8, never: 273 (66 %). P(depth-1 node correct | wrong root): revise 0.159, **escalate (stronger model) 0.154**, re_retrieve 0.052. When the root is right, depth-1 nodes stay right ~96 %.
* Root-status table (P0 replay):

| root status | n | root right | P0 right | gain / harm | revised |
|---|---|---|---|---|---|
| exec error / timeout | 70 | 0 | 47 | 47 / 0 | 70 |
| executable, empty | 82 | 21 | 19 | 1 / 3 | 21 |
| executable, non-empty | 949 | 665 | 659 | 2 / 8 | 44 |

* When root and forced revise(T=0) give different results on an executable root and exactly one is right, the revision wins in only 17 of 42 cases (40 %; 5 vs 10 when the critic had flagged the root). A perfect root-vs-revision arbiter on executable roots would add the 17 revision wins to the 25 root wins already counted: +17/1101 = +1.5 pp at most. Escalate (3.8-max) vs root on executable roots: 18 vs 25.
* Repeatability: live-run root and tree root (same config, T=0, different time) differ in SQL text for 29 % of questions and in EX for 37/1101 = 3.4 % (net +0.6 pp). Treat paired differences below ~2 pp as noise on 1 100 questions.

Per database (HR):

| db | n | root | P0 replay | oracle chain | oracle depth-1 | oracle tree | gain/harm | root exec-err | root empty | root score >= 9.5 | no correct node |
|---|---|---|---|---|---|---|---|---|---|---|---|
| address | 150 | 0.687 | 0.700 | 0.707 | 0.720 | 0.767 | 2/0 | 0 | 0.05 | 0.87 | 0.23 |
| beer_factory | 131 | 0.427 | 0.740 | 0.756 | 0.748 | 0.802 | 41/0 | 0.43 | 0.02 | 0.55 | 0.20 |
| bike_share_1 | 113 | 0.319 | 0.292 | 0.327 | 0.381 | 0.451 | 0/3 | 0.03 | 0.20 | 0.62 | 0.55 |
| donor | 150 | 0.640 | 0.660 | 0.707 | 0.733 | 0.800 | 6/3 | 0.05 | 0.10 | 0.59 | 0.20 |
| public_review_platform | 150 | 0.700 | 0.700 | 0.727 | 0.727 | 0.747 | 0/0 | 0.01 | 0.04 | 0.85 | 0.25 |
| retails | 150 | 0.747 | 0.747 | 0.760 | 0.787 | 0.827 | 0/0 | 0.01 | 0.04 | 0.80 | 0.17 |
| sales | 107 | 0.710 | 0.710 | 0.729 | 0.757 | 0.794 | 0/0 | 0.01 | 0.08 | 0.78 | 0.21 |
| simpson_episodes | 150 | 0.680 | 0.653 | 0.707 | 0.740 | 0.773 | 1/5 | 0 | 0.08 | 0.87 | 0.23 |

`bike_share_1` (5 GB, 55 % unreachable, 7 gold execution failures/timeouts in HR are mostly here and in `retails`) is a data/IO problem, not a method one.

HT-300 (live only; no trees): root 0.683, final 0.693 (4 gains / 1 harm), zs_ours 0.697, zs 0.673. full - zs_ours = -0.3 pp [-4.0, +5.1] (9 vs 10 discordant); **root - zs_ours = -1.3 pp (9 vs 13)**: the retrieval stack (M1-M4) alone does not help, the loop's +1.0 pp (3 of 4 gains are exec errors) brings it back to par. Only 4/300 HT roots fail to execute.

**Annotation noise / ambiguity (no label exists; proxies only, E10 will give the real figure).** Wrong live finals (380): value_mismatch 66 %, shape_mismatch 14 %, empty_result 7 %, row_subset 6 %, row_superset 4 %, gold_fail 2 %, column_permutation/timeout 1 %. Certain data problems (gold fails/timeouts) 0.8 pp of all questions. Output-specification-type classes (shape/subset/superset/permutation) = 24 % of wrong = 8.4 pp of all. 35 HR gold queries (3.2 %) return empty/NULL. 273 questions (24.8 %) are unreachable by any node; 67 % of those are stable-wrong. Upper bound of "noise or systematic misreading" is therefore ~25 pp of the 34 pp error; BIRD-Platinum (per `RELATED_WORK_NOTES`, not re-verified here) reports corrections in 61 % of sampled train instances, which is consistent with a large noise share but is not a measurement on our data. HT-300: 3 systems (zs, zs_ours, full) together reach only 0.733 oracle (26.7 % wrong for all three).

## 2. Critic quality

AUROC for root correctness (db-clustered 95 % CI); HR = tree roots (n 1 101), HT = live roots (n 300):

| signal | HR | HT |
|---|---|---|
| V3 score | 0.708 [0.638, 0.794] | 0.744 [0.669, 0.823] |
| semantic sub-score | 0.632 [0.557, 0.729] | 0.606 [0.560, 0.650] |
| syntax (execution) | 0.645 [0.563, 0.743] | 0.603 [0.567, 0.625] |
| efficiency (EXPLAIN) | 0.613 [0.530, 0.729] | 0.620 [0.567, 0.671] |
| executes (non-empty) | 0.643 [0.563, 0.737] | 0.603 [0.567, 0.625] |
| retrieval similarity top-1 | 0.523 | 0.480 |
| within-db V3 (n-weighted) | 0.687 (0.658 w/o beer_factory) | 0.728 |
| learned LR/HGB on V3 sub-scores + exec/rows/sim features, leave-one-db-out | 0.714-0.718 [0.66, 0.79] | HR->HT transfer 0.71-0.75 |
| agreement of root result with its depth-1 nodes (5 nodes, ~9 extra calls) | 0.711 [0.647, 0.776] | n/a |
| agreement with whole tree (45 extra calls; reference) | 0.770 [0.720, 0.820] | n/a |
| LR(V3, agreement-d1, non-empty), LODO | **0.750 [0.703, 0.816]** | n/a |
| HT: agreement of 3 systems (zs, zs_ours, full), tie-break V3 | n/a | **0.768 [0.755, 0.793]** (V3 alone on finals 0.704 [0.652, 0.762]) |

* The score is nearly three-valued: V3 >= 9.5 for 75 % of HR roots (53 % of the wrong ones; 22 % of the wrong ones have exactly 10); mean score/10 = 0.90 vs EX 0.62 (ECE 0.28, Brier 0.27). Most of the AUROC comes from the execution error class; a learned V5-style critic on the stored features does not beat V3 (+0.01, CIs overlap).
* **X3's V5**: not trained on HR/HT (`exp/frozen/critic_v5.pkl` does not exist; only a smoke on 334 HT candidates, V3 0.77 vs V5 0.69-0.71, same HT split used for fit and evaluation under db-grouped CV). My LODO/HR->HT numbers above are the answer to "can stored features beat V3": no. Leakage risks if V5 is trained on the trees: forced-revision nodes are off-policy vs P0 (revisions of score-10 roots); `k/tier/action/prev_score` encode the policy; `beer_factory` carries 54 of 70 exec-error positives, so pooled AUROC is dominated by one db (use LODO or within-db); the calibrated score (10 x P) needs re-tuned thresholds on the same HR (nested CV); HT must stay untouched until the single final check.
* **Risk-coverage (accepting the answer when score >= theta), V3, roots:** HR theta 10: coverage 0.36, EX of accepted 0.769; 9.5: 0.75 / 0.732; 8.5: 0.88 / 0.693; none: 1.0 / 0.623. HT theta 10: 0.49 / 0.837; 9.5: 0.85 / 0.780; 8.5: 0.93 / 0.734; none: 1.0 / 0.683. Finals (live): HR 8.5: 0.98 / 0.663; HT 8.5: 0.99 / 0.703. The patent thresholds 7.5/8.5 sit where almost everything is accepted; the only informative cut is 10 vs < 10.
* Matched coverage (EX of the accepted fraction; ties random, CI 95 %): HR roots, 60 % coverage: V3 0.735 [0.656, 0.783]; agreement-d1 0.751 [0.696, 0.782]; LR(V3, agreement, non-empty) 0.782 [0.726, 0.816]; whole-tree agreement 0.803; oracle 1.0. HT finals, 60 %: V3 0.800 [0.732, 0.845]; 3-system agreement + V3 tie-break 0.835 [0.733, 0.889]; 40 %: 0.820 vs 0.874 [0.833, 0.912]. AURC HR 0.263 (V3) vs 0.082 (oracle) vs 0.377 (random); HT 0.200 vs 0.166 (agreement + V3) vs 0.053 vs 0.307.
* Reading: even the best offline routing needs to defer ~40-50 % of questions to reach ~0.8-0.85 accepted EX. Gains over V3 at equal coverage are +3-5 pp with CIs that overlap V3's.

## 3. Equal-compute reference: vote / consistency vs the critic-selected answer (HR trees, replay)

Candidates in a tree are **not independent samples** (a revision is conditioned on its parent SQL and the critic text; T=0.7 siblings share the parent; escalate uses a different model), and no independent sampling run (`bon_vote`, `sc_cot`) exists on HR/HT, so true self-consistency at matched compute cannot be estimated offline. Calls modelled with critic / without critic.

| policy (HR, n 1 101) | EX | vs P0 (pp) [CI] | W/L vs P0 | calls (w/ critic / w/o) |
|---|---|---|---|---|
| root only | 0.623 | -3.5 [-11.8, +1.0] | 11 / 50 | 3.0 / 2.0 |
| **P0 (patent, theta 8.5)** | 0.658 | 0 | - | 3.3 / 2.2 |
| P0 theta_acc 7.5 / 9.0 / 9.5 / 10 | 0.662 / 0.658 / 0.660 / 0.660 | +0.4 / -0.1 / +0.2 / +0.2 | 5/1, 0/1, 8/6, 8/6 | 3.3 / 3.4 / 3.8 / 4.8 |
| ExecGate: revise only on error/timeout | 0.666 | +0.7 [-0.3, +1.7] | 11 / 3 | 3.1 / **2.1** |
| ExecGate: error, timeout or empty | 0.664 | +0.5 [-0.7, +1.8] | 15 / 9 | 3.5 / 2.3 |
| AgreeGate (root + 1 revision; disagree -> add T0.7 sample, vote3) | 0.661 | +0.3 [-0.6, +1.2] | 17 / 14 | 5.3 / 3.1 |
| AgreeGate, disagree -> critic-best | 0.664 | +0.5 [-0.6, +1.6] | 12 / 6 | 5.0 / 3.0 |
| AgreeGate, disagree -> escalate result | 0.657 | -0.2 [-0.9, +0.6] | 11 / 13 | 5.3 / 3.1 (+ ~2.3x cost) |
| vote, pool root + 2 revisions | 0.661 | +0.3 [-0.6, +1.2] | 17 / 14 | 7.0 / 4.0 |
| critic-argmax, same pool | 0.666 | +0.7 [-0.5, +1.9] | 14 / 6 | 7.0 / 4.0 |
| oracle in same pool | 0.685 | +2.6 [+1.3, +4.0] | 29 / 0 | 7.0 / 4.0 |
| vote, pool + 2 re_retrieve | 0.668 | +0.9 [0.0, +1.8] | 14 / 4 | 11 / 6 |
| critic-argmax, same pool | 0.668 | +1.0 [-0.4, +2.4] | 19 / 8 | 11 / 6 |
| oracle in same pool | 0.701 | +4.3 [+2.4, +6.2] | 47 / 0 | 11 / 6 |
| vote, all depth-1 (incl. escalate) | 0.662 | +0.4 [-0.6, +1.3] | 9 / 5 | 13 / 7 (+ ~6x on 1 node) |
| critic-argmax, all depth-1 | 0.671 | +1.3 [0.0, +2.5] | 22 / 8 | 13 / 7 |
| oracle, all depth-1 | 0.707 | +4.8 [+2.8, +6.8] | 53 / 0 | 13 / 7 |
| learned selector (LR / HGB, LODO), pool A | 0.668 / 0.666 | +0.9 / +0.7 | 17/7, 15/7 | 7.0 / 4.0 |
| learned selector (LR / HGB, LODO), pool all depth-1 | 0.666 / 0.667 | +0.7 / +0.8 | 17/9, 17/8 | 13 / 7 |
| oracle, whole tree | 0.752 | +9.4 [+7.0, +12.1] | 103 / 0 | 48 / 25 |

Excluding `beer_factory`, every non-oracle row is within +1.4 pp of P0 and within +1.2 pp of the root. Voting by execution result is not better than choosing by critic score on the same candidates; both recover 10-30 % of the oracle gap. At equal compute (P0 = 3.3 calls) the only policies that fit are the exec-gated ones.

HT-300 cross-prompt voting (same model, different prompts: zs, zs_ours, full_fixed; 5.25 calls): vote3 0.697-0.700 vs best member 0.697 (+0.0..+0.3 pp); oracle of the three 0.733. The three agree on 82 % of the questions (EX 0.773 when they agree, 0.321 otherwise).

## 4. Where gains might be larger

* **Model strength:** only qwen3.7-plus is stored. The single available contrast is the stronger reviser (qwen3.8-max) vs the same model as reviser: fixes 15.4 % vs 15.9 % of wrong roots, at ~5x price per node (escalation was ¥113 of the ¥169 tree cost). No evidence that a stronger model is a better reviser; whether it is a better generator needs E2.
* **Database:** loop gain concentrates in dbs with execution errors (`beer_factory` +31 pp); `bike_share_1` has the largest oracle headroom (0.32 -> 0.45) but half is unreachable/IO-bound. Other six dbs: P0 minus root between -2.7 and +2.0 pp (net about zero).
* **Difficulty:** BIRD train has no difficulty label; proxy = gold-SQL component score (joins + 2 x nested + group/having/case + aggregates + set ops), terciles <= 1 / 2 / >= 3. Root EX 0.73 / 0.58 / 0.49; oracle-tree minus root +10 / +16 / +14 pp; P0 minus root +2.7 / +2.8 / +5.6 pp (the high bin contains `beer_factory`, confounded). Harder questions have more oracle headroom but the critic does not exploit it (same pattern as above). HT: full vs zs_ours by bin: low +3.7, mid -2.6, high -4.4 pp (n 134 / 75 / 91, noise).
* **Schema size:** all 1 401 HR/HT questions took the full-schema fallback (`ctx_full_schema` true; < 6 000 tokens), so M3 (domain routing, rerank, FK expansion) was never exercised; its effect is untestable on this data. Context-size bins are confounded with db (not reported). Evidence is present for 95 % of questions: no signal.
* **Reserved words:** 3 of 69 train dbs have a keyword *table* name (`transaction`, `Match`, `current`; 566 of 9 428 train questions), 8 have keyword column names. Whether dev has the same is unknown (dev schemas not inspected here).
* Detectability: with p_d 0.03-0.05 and DEFF 2, MDE (80 % power, alpha 0.05) is 2.1-2.7 pp on 1 101 HR questions and 1.9-2.5 pp on 1 534 dev questions; the +0.3..+1.3 pp effects above are not resolvable by one run.

## 5. Ranked method strengthenings (<= 5, patent-faithful where stated)

| # | change | patent anchor | estimated headroom (offline) | extra calls | cheap test once API is back |
|---|---|---|---|---|---|
| 1 | **Execution-gated loop + identifier hygiene.** Revision trigger = execution error/timeout (syntax sub-score), not the semantic LLM score; the semantic score stays for routing. Quote SQLite keywords in `render_ddl` (`_q`) and add "quote reserved words" to the gen rule | M5 syntax/field-mismatch scoring, M4 prompt | EX +0.7 pp [-0.3, +1.7] vs P0 (11 W / 3 L) = equal within noise; main effect is cost: **2.1 vs 3.4 calls/q (-38 %)**; quoting removes 54 of 70 HR exec errors at the source (up to +4.9 pp on dbs like `beer_factory`, 0 elsewhere) | **-1.2** | (a) re-run roots for the 131 `beer_factory` questions with quoted DDL (~400 calls): expect exec errors 54 -> ~0, root EX 0.43 -> ~0.7. (b) `ExecGate` on HR+HT: roots are cached, only revisions of ~75 error roots are new (<~150 calls); compare to P0 paired (needs the error text as the only feedback, since the tree revisions were made with the critic's text) |
| 2 | **Consistency term in the Critic / routing score.** Add result agreement among k extra samples as a fourth sub-score used for M8 hand-off (not for revising) | M5 (extra weighted term) + M7 route-to-human (C4) | AUROC 0.71 -> 0.75 (HR LODO), HT 0.70 -> 0.77; EX of accepted 60 %: +4.7 pp HR, +3.5 pp HT over V3 | +2 gen calls (T=0.7 roots; stored evidence used 4 revisions + escalate = ~9 calls, cheaper version unverified) | HT-300 + HR: 2 extra T=0.7 root samples per question (~2 800 calls, < ¥10); LR fitted on HR, evaluated HR-LODO and on HT; success = AUROC gain >= 0.03 with paired db-clustered CI > 0 and AURC lower |
| 3 | **Arbitrate only where candidates disagree.** One LLM judge call that sees both result previews when root and revision differ | M5 critic reused as pairwise judge; M7 | perfect arbiter: +1.5 pp (root vs revision on executable roots), +2.6 (3-candidate pool), +4.8 (all depth-1); every offline selector recovers +0.4..+1.3 | ~+0.25-0.5 (judge on the 24.8 % disagreeing questions) | 273 HR questions whose depth-1 pool disagrees (154 contain both a right and a wrong member): 1-2 judge calls each (~500 calls); success = pool EX >= P0 + 1.5 pp, i.e. judge picks the right member in >= ~70 % of the mixed cases (revise-always wins only 40 % there) |
| 4 | **Make M3 count.** Lower `schema_fallback_tokens` to force domain routing/rerank and test it | M3 | unknown: M3 never ran on HR/HT; root - zs_ours = -1.3 pp says M1-M4 as a whole is neutral-to-negative | 0 | HT-300 (900 calls, ~¥3) with fallback 0 vs 6 000: report as sensitivity; without it E3 "-M3" is empty on small schemas |
| 5 | **Retire (do not spend calls on):** threshold sweeps (all within +0.4 pp of P0), re_retrieve as a default action (5 % of wrong roots fixed, +0.4 pp when added to a pool), escalation as a reviser (15 % = same as the cheap model, ~5x price), RL controller (action values are flat on executable roots, which is why PPO could not beat P0) | - | <= +0.4 pp | - | none |

Not testable offline: whether V1 (binary) loses to V3 (K2): both see the execution error text, and the V3 advantage here is almost entirely the execution-grounded syntax term, so K2 may be small; check V1 on the same HR candidates when the API is back (6.6 k calls per R0).

## 6. What the paper can honestly claim (and the evidence that would carry it)

* Not supportable from these data: "the full pipeline improves EX over zero-shot" (HT -0.3 pp; HR has no zero-shot run, +3.5 pp over its own root is the `transaction` artefact); "semantic critic revision improves accuracy" (net negative on executable roots); "RL beats fixed thresholds" (R0 CV).
* Defensible: (i) execution-grounded repair of execution failures (HR 47/70 repaired, HT 3/4), reported by error class; (ii) cost: exec-gated loop at ~2.1 calls vs 3.4 at equal EX (needs live confirmation, test 1b); (iii) selective prediction: risk-coverage with a consistency-augmented score (tests 2), with the honest statement that ~40-50 % deferral is needed for ~0.8-0.85 accepted EX; (iv) pollution robustness (E4) and WS vs MemRL are independent of this analysis; (v) validity: 25 % of HR questions are unreachable by 23 candidates incl. a stronger model, 21 % stable-wrong, so report EX on a corrected subset and the E10 noise estimate next to every headline number.
* Evidence that would change this reading: a positive judge test (3), a positive M3 sensitivity (4), or dev databases with many keyword/quoting failures (then 1 is worth several pp).

## 7. Caveats

* 8 (HR) and 4 (HT) databases: every CI is wide; `beer_factory` alone drives the HR loop gain. Tree nodes are counterfactual forced revisions (including revisions of score-10 roots); P0 would not have generated most of them. Stored node EX vs my re-execution agrees on 99.3 % of 25 898 nodes (the rest are timeout/IO-state cases, e.g. `donor:3255`); analyses use the stored labels. Gold SQL execution failures (5 HR) count as wrong by design. Voting results are not independent-sample self-consistency. The complexity proxy is mine, derived from gold SQL text, never shown to a system. Complexity/bin numbers for HT are n = 75-134 per bin. Literature figures are quoted from `RELATED_WORK_NOTES.md`, not re-verified. No file in the run directories of other agents was written (outputs in `$R/r1/`).
