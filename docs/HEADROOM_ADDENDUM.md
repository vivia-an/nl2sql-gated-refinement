# HEADROOM addendum (X7, 2026-10-26): value grounding, empty-result roots, confidence-routed compute

Offline only, no LLM call, BIRD-train tuning data (HR = bird_train_HR, 1101 q, 8 dbs; HT = bird_train_HT, 300 q, 4 dbs). Inputs: the X5 runs `1b-{hr,ht}-quote` (P0 = full_fixed + quoted DDL), `1b-{hr,ht}-gated` (full_gated), `2-{hr,ht}-cons` (full_gated + 4 samples). Gold SQL is used for these analyses and never shown to a system. Reproduction: `code/tools/{value_check,empty_roots,adaptive_replay}.py`, raw outputs in `results/x7/*.json`. All gains below are upper bounds unless a paired CI is given.

## A1. Value grounding: how many wrong answers are wrong because of a filter literal? (`tools/value_check.py`, `results/x7/value_check.json`)

Population: gated finals of HR + HT (1401 q) with ex = 0 and auto_class in {value_mismatch, row_subset, row_superset, empty_result}: 399 of the 467 wrong answers (85 %). Literals are taken from the WHERE / HAVING clauses of the gold SQL and compared with the predicted SQL and with question + evidence.

| answer level (gated, HR + HT) | answers | share of the 399 |
|---|---|---|
| every gold literal is in the prediction (or equal after removing LIKE %) | 227 | 56.9 % |
| gold has no WHERE / HAVING literal | 65 | 16.3 % |
| (i) at least one gold literal missing in the prediction | 107 | 26.8 % |
| &nbsp;&nbsp;(ii) all missing literals are verbatim in question / evidence | 27 | 6.8 % |
| &nbsp;&nbsp;(iii) some missing literal only approximate (case 18, stem 8, format 7, fuzzy 3 literals), none absent | 34 | 8.5 % |
| &nbsp;&nbsp;(iv) some missing literal nowhere in question / evidence (needs a DB lookup) | 46 | 11.5 % |

Literal level: 150 gold literals are missing in the predictions: 34 verbatim (23 %), 36 approximate (24 %), 80 absent (53 %). 68 answers are "pure value errors" (prediction and gold filter the same columns with different literals). HR alone: 322 of 372 wrong answers in the classes, levels (in the table's order) 174 / 57 / 23 / 30 / 38; HT alone: 77 of 95, levels 53 / 8 / 4 / 4 / 8. P0 gives the same picture (pooled: 211 / 63 / 25 / 34 / 47).

**Upper bound of value grounding** (answers with a missing literal that cannot be copied from the text = approximate + absent): 80 answers = **5.7 pp of all questions** (HR 6.2, HT 4.0). Of these 49 (**3.5 pp**) have the gold literal stored in its column (reachable by a lookup); for the other 31 the gold literal is not stored verbatim in its column (or the existence check could not run), so showing column values would not give the model the right string. The 27 verbatim-only answers (1.9 pp) are mis-copies of text the model already sees (not a retrieval problem). Absent literals by column cardinality (29 of the 80 measured within the 20 s guard): <= 8 distinct values: 11, <= 20: 12, <= 50: 16, > 50: 13, so at most about a third could simply be enumerated in a prompt.

Reading: a *perfect* value oracle would lift EX by about 3.5 pp (lookup-reachable), a realistic prompt-side mechanism by a fraction of that; the pure-value-error subset that is non-verbatim is 40 answers (2.9 pp). This is a loose bound (a wrong answer can have several causes; fixing a literal does not always fix the answer).

## A2. Empty-result roots (`tools/empty_roots.py`, `results/x7/empty_roots.json`, `empty_roots_near.json`)

Roots that execute but return no rows / one all-NULL row (`ok_empty`):

| run | roots | empty | gold non-empty (= wrong) | gold empty too | root right | wrong empties in pp of all q | gold literal visible in `SELECT DISTINCT ... LIMIT 8` | (complete list) | visible, in pp of all q |
|---|---|---|---|---|---|---|---|---|---|
| P0 HR | 1101 | 78 | 54 | 24 | 21 | 4.9 | 12 | 6 | 1.09 |
| P0 HT | 300 | 15 | 14 | 1 | 1 | 4.7 | 4 | 4 | 1.33 |
| gated HR | 1101 | 77 | 53 | 24 | 21 | 4.8 | 11 | 5 | 1.00 |
| gated HT | 300 | 15 | 14 | 1 | 1 | 4.7 | 4 | 4 | 1.33 |

(Wrong empties are 5 % of all questions, i.e. 14 % of HR's wrong answers; 24 of 78 HR empty roots are CORRECT because the gold is empty too, so gating every empty result has a harm risk: P0's own critic revised 5 of them and broke 2.)

Causes of the 54 wrong HR empties (P0): same column filtered with another literal 26 (14 of them differ only in case / punctuation / spacing: `NOBLE` vs `Noble`, `'Community Post Office '`), same literals but another reason (join, aggregation, extra filter) 15, root has no text literal at all 12, gold filters other columns 1. HT: 6 / 4 / 0 / 3 (+1). In 22 of the 26 HR cases the gold literal exists in the column the root filtered, but a plain `DISTINCT ... LIMIT 8` shows it in only 12 (6 with a complete list): the columns have more than 8 distinct short values. The `probe_near` variant (stored values that contain the longest token of the failed literal listed first) raises this from 12 to 13 (HR) and leaves HT at 4: the failed literal usually is not a spelling variant of the stored one (`Vicky` vs `Hartzler`, `Senate` vs `House of Repsentatives`).

Root -> final repair in P0 (critic-driven revision, no probing): HR 19 of 78 empty roots were revised; of the 54 wrong ones 14 were revised and 6 repaired (11 % of all wrong empties, 43 % of the revised ones), 2 correct roots harmed; of the 12 probe-visible cases only 1 was revised (and repaired). HT: 3 revised, 0 repaired. **Upper bound of the probe block** = visible cases not already repaired = 11 / 1101 = **1.0 pp (HR)** and 4 / 300 = **1.3 pp (HT)**; a revision that fixes half of them gives about +0.5 pp (HR), +0.7 pp (HT), below what HT-300 can resolve (see the handoff). `gated` itself never revises an empty result (`gate_empty: false`), which is why gated HR/HT show 0 revisions here.

Concentration: 21 of the 54 wrong HR empties are in `bike_share_1`, 8 in `donor`, 7 in `address`; HT: 7 of 14 in `world_development_indicators` (`Country.IncomeGroup` alone 4: `High income` vs `High income: OECD`).

**Zero aggregates are the larger degenerate population.** One-cell results equal to 0 (COUNT = 0 / SUM = 0) are not caught by `gate_empty`: HR P0 has 32 such roots, 27 of them wrong with a non-zero gold (2.5 pp of all questions; gated: 33 / 28), HT 2 / 1. `m5.gate_zero` (new, default off) treats them as degenerate for the gate and the ranking; `configs/full_gated_probe_zero.yaml` combines it with probing. It is the only X7 switch whose population is larger than 2 pp.

## B. Confidence-routed compute (`tools/adaptive_replay.py`, `results/x7/adaptive_replay.json`)

Replay of `route.adaptive` on the stored `2-*-cons` runs: the sampled SQL texts were recovered from the runs' call-cache archives (HR 1101 / 1101 and HT 300 / 300 questions matched uniquely, 0 ambiguous), root and samples re-executed read-only and scored; the stored `cons_bits` are reproduced for 1078 / 1082 (HR) and 293 / 293 (HT) executable questions (the 4 HR differences are a timeout / non-deterministic function). Rule: consistency (share of the 4 samples whose result equals the root's) >= theta_fast keeps the root, otherwise the execution-result majority over root + samples (ties -> root). One sampling request is the only extra call: 3.0 calls/q (M1 + generation + 1 request with n = 4) against 2.02 for gated.

| split | reference | EX | calls/q | CNY/q |
|---|---|---|---|---|
| HR | root only (M1 + generation) | 65.49 % | 2.00 | 0.0091 |
| HR | gated (same roots) | 66.30 % | 2.02 | 0.0093 |
| HR | oracle best of root + 4 samples | 68.57 % | 3.0 | - |
| HT | root only | 67.00 % | 2.00 | 0.0081 |
| HT | gated | 68.33 % | 2.02 | 0.0082 |
| HT | oracle best of root + 4 samples | 71.33 % | 3.0 | - |

| split | theta_fast | empty rule | EX | fast path | calls/q | CNY/q | dEX vs gated, pp [db-clustered 95 % CI] | wins / losses |
|---|---|---|---|---|---|---|---|---|
| HR | 1.0 | vote | 65.58 % | 82 % | 3.00 | 0.0172 | -0.73 [-2.27, +0.38] | 6 / 14 |
| HR | 0.75 | vote | 65.58 % | 91 % | 3.00 | 0.0172 | -0.73 [-2.27, +0.38] | 6 / 14 |
| HR | 0.5 | vote | 65.58 % | 94 % | 3.00 | 0.0172 | -0.73 [-2.27, +0.38] | 6 / 14 |
| HR | 1.0 | demote | 65.67 % | 82 % | 3.00 | 0.0172 | -0.64 [-2.23, +0.56] | 8 / 15 |
| HT | 1.0 | vote | 69.00 % | 84 % | 3.00 | 0.0153 | +0.67 [-1.57, +3.24] | 4 / 2 |
| HT | 0.75 | vote | 69.00 % | 90 % | 3.00 | 0.0153 | +0.67 [-1.57, +3.24] | 4 / 2 |
| HT | 0.5 | vote | 69.00 % | 93 % | 3.00 | 0.0153 | +0.67 [-1.57, +3.24] | 4 / 2 |
| HT | 1.0 | demote | 69.67 % | 84 % | 3.00 | 0.0153 | +1.33 [-1.10, +3.54] | 6 / 2 |

(k = 2 samples: HR 65.58 %, HT 68.33 % - 68.67 %, same 3.0 calls because the request count does not change; the full grid incl. `vote_always` is in the JSON. calls/q does not fall below 3.0 for any theta_fast because the sampling request is made before the decision. Paired CIs vs an independent gated run are of the same size.)

Reading: the adaptive path **does not beat gated** (HR -0.7 pp, HT +0.7 pp, both CIs include 0) while costing 1.5x the calls; theta_fast does not matter (82 % of the questions are unanimous, the others rarely change the outcome). Its attraction is only the +2 pp over the root alone on HT ([0.0, +4.2]); the best-of-5 oracle is +2.3 pp (HR) / +3.0 pp (HT) above gated, so a majority vote recovers about nothing of it. It cannot satisfy SAP P1b (R < 0.80) by construction.
