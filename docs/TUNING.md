# TUNING (P2, R0, 2026-10-26) — our method only

Scope: tuning of `full_fixed` (M1–M6, policy P0) on HT-300 and of the M6 parameters on the HT mini-stream, plus the HR artefacts needed later (candidates, E6 trees, controllers). No baseline was tuned here (second pass after B1). Dev / minidev / Spider / AmbiQT / LiveSQL were not touched. No lock entry was created.

All runs: `qwen3.7-plus-2026-05-26` (key `qwen37plus`), T = 0, seed 0, evidence given, critic = self (V3), spec defaults except where stated. Snapshot of the code: git `f08f49058d21021e078dc205327cd0fe89a9de4c`, rsynced to `<NL2SQL_ROOT>/snap_r0/{code,exp}` (host `code/` was not edited). Raw trial records: `results/p2/tuning_log_r0.jsonl` (also `exp/configs_tuned/tuning_log_r0.jsonl`); machine-readable tables: `results/p2/{base_ht300,m3_grid,e4_tuning}.json`.

Re-scoring check: all 11 HT-300 runs below (3 300 predictions) were re-scored with a freshly built gold cache (the shared gold cache turned out to be corrupted later, see `internal note R0`): 0 disagreements. The HR run: 1 099 of 1 101 re-scored, 1 disagreement (see R0.md, timeouts depend on IO state).

## 1. Throughput / cost sanity on HT-300 (default config)

| system | EX (95 % db-clustered CI) | soft-F1 | calls/q | embed / rerank per q | prompt / compl. / cached tokens per q | latency mean / p95 (s) | wall 300 q | list-price cost (non-cached calls) |
|---|---|---|---|---|---|---|---|---|
| `zs` (B1) | 0.673 (0.542–0.782) | 0.704 | 1.0 | – | 1 638 / 57 / 1 174 | 2.2 / 3.6 | 238 s (3 workers) | ¥0.56 |
| `zs_ours` | 0.697 (0.556–0.802) | 0.719 | 1.0 | – | 1 708 / 54 / 1 096 | 2.1 / 3.0 | 195 s | ¥0.50 |
| `full_fixed` | 0.693 (0.586–0.765) | 0.716 | 3.25 | 1.67 / 2.32 | 5 937 / 381 / 3 379 | 10.5 / 17.2 | 450 s (8 workers) | ¥2.37 |

* `full_fixed`: 296 accept, 34 revise, 4 stop_best (coverage 0.987, EX on accepted 0.703). Per stage (calls/q): M1 1.00, gen 1.00, crit 1.14, rev 0.11.
* `full_fixed` does **not** beat zero-shot on HT-300 (−0.3 pp vs `zs_ours`, +2.0 pp vs `zs`; paired difference is a handful of questions). Per db (full_fixed / zs_ours): law_episode 0.741 / 0.704, retail_complains 0.692 / 0.731, video_games 0.779 / 0.821, world_development_indicators 0.548 / 0.493.
* Cost scale: tokens-based estimate for `full_fixed` ≈ ¥0.0095 per question (¥0.0079 as billed, 149 of 976 calls were cache hits) ⇒ a 1 534-question dev run ≈ ¥12–15; zero-shot ≈ ¥0.002 per question. Cost is dominated by prompt tokens, not calls.
* `zs` and `zs_ours` share the call cache (identical prompts for part of the questions), so the cost columns of the two zero-shot rows are not independent.
* API: 0 429, 0 api_fail on all HT-300 runs.

## 2. M3 tuning on HT-300: `k_t ∈ {4,6,8}` × `k_e ∈ {2,4,6}`

The full grid is 9 configurations, i.e. within the ≤10-trial budget, so no coarse-to-fine step was needed: every configuration was run on all 300 questions (`full_fixed`, everything else default). Configs: `exp/configs_tuned/tuning/m3_kt{4,6,8}_ke{2,4,6}.yaml`; (6,4) is the default run of §1 (`full_fixed.yaml`).

| trial | k_t | k_e | EX | correct /300 | calls/q | prompt tok/q | compl. tok/q | wins / losses vs (6,4) |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 2 | 0.6833 | 205 | 3.283 | 5 815 | 390 | 4 / 7 |
| 2 | 4 | 4 | 0.6933 | 208 | 3.277 | 6 034 | 400 | 2 / 2 |
| 3 | 4 | 6 | 0.6767 | 203 | 3.257 | 6 177 | 385 | 2 / 7 |
| 4 | 6 | 2 | 0.6800 | 204 | 3.313 | 5 928 | 403 | 4 / 8 |
| 5 | **6** | **4** | **0.6933** | 208 | **3.253** | 5 937 | 381 | – |
| 6 | 6 | 6 | 0.6800 | 204 | 3.313 | 6 358 | 400 | 2 / 6 |
| 7 | 8 | 2 | 0.6900 | 207 | 3.253 | 5 745 | 382 | 5 / 6 |
| 8 | 8 | 4 | 0.6900 | 207 | 3.253 | 5 942 | 382 | 0 / 1 |
| 9 | 8 | 6 | 0.6900 | 207 | 3.300 | 6 318 | 399 | 4 / 5 |

**Chosen: k_t = 6, k_e = 4** (rule: highest EX, ties → fewer calls: (6,4) and (4,4) tie at 0.6933, (6,4) has 3.253 vs 3.277 calls). This equals the spec default, so `exp/configs_tuned/full_fixed.yaml` only writes it explicitly and all other configs (`full_rl`, `full_x`, `crit_*`, `abl_*`, …) need no change.

Reading: the whole grid spans 0.677–0.693 (203–208 correct of 300). No contrast vs the default exceeds 5 net questions, i.e. nothing is distinguishable from noise (a replicate of the same configuration moves by ~1 pp, §3). M3 is a near no-op on these databases because of the full-schema fallback (< 6 000 est. tokens; k_t then only changes which tables are flagged "likely relevant"). Do not report a k_t / k_e effect.

## 3. E4 tuning on the HT mini-stream (η, ρ for `ws_full`; λ_m for `memrl`)

Not wired in the host code (X2 open issue 4): `results/p2/tools_r0/e4_ht_tune.py` (new file, nothing patched) builds per HT db a mini-stream from `bird_train_HT_full` (seed 20261026, per db: 12 seed / 40 stream / 24 probe questions, 4 dbs ⇒ 48 / 160 / 96), swaps it in for `bird_train_HR` inside the process and calls the unmodified `stream.run_unit`. Setting: pollution p = 40 %, permutation 1, φ = 0.5, τ = 6.0, α/β/γ default, M3 (6,4), pool = POOL56. Natural pollution material: `zs` failures on the 48 seed items (16–17 of 48).
Metric = EX_all (abstain = wrong) pooled over the 160 stream questions (prequential) and the 96 final-checkpoint probe questions (256 scored items). Decision rule = highest pooled EX of the first pass, ties → fewer calls/q.

First pass (v1, run while the shared sqlite cache was still readable) and the clean-cache replicate (v2, private cache, cut short when the DashScope account went into arrears at 06:23 worker machine time; 5 of 13 trials replicated):

| trial | system | params | EX v1 | probe-final v1 | stream v1 | calls/q v1 | evicted (polluted) v1 | EX v2 (replicate) |
|---|---|---|---|---|---|---|---|---|
| t01 | ws_full | η 0.3, ρ 0.2 (spec default) | 0.6914 | 0.6458 | 0.7188 | 3.337 | 73 (5) | 0.6992 |
| t02 | ws_full | η 0.3, ρ 0 | 0.7070 | 0.6667 | 0.7312 | 3.299 | 76 (4) | 0.6953 |
| t03 | ws_full | η 0.3, ρ 0.1 | 0.6875 | 0.6458 | 0.7125 | 3.324 | 78 (5) | – |
| t04 | ws_full | η 0.3, ρ 0.4 | 0.6914 | 0.6458 | 0.7188 | 3.310 | 67 (5) | – |
| t05 | ws_full | η 0.2, ρ 0.2 | 0.6953 | 0.6458 | 0.7250 | 3.330 | 67 (4) | – |
| t06 | ws_full | η 0.5, ρ 0.2 | 0.7070 | 0.6667 | 0.7312 | 3.333 | 76 (8) | – |
| **t07** | **ws_full** | **η 0.5, ρ 0** | **0.7148** | 0.6667 | 0.7438 | 3.301 | 83 (9) | 0.6992 |
| t08 | ws_full | η 0.2, ρ 0 | 0.7070 | 0.6667 | 0.7312 | 3.295 | 74 (3) | – |
| m01 | memrl | λ_m 0.25 | 0.6992 | 0.6562 | 0.7250 | 3.308 | 0 | – |
| m02 | memrl | λ_m 0.5 | 0.7031 | 0.6562 | 0.7312 | 3.330 | 0 | 0.7070 |
| **m03** | **memrl** | **λ_m 1.0** | **0.7070** | 0.6458 | 0.7438 | 3.339 | 0 | 0.6953 |

(trials t09 η 0.5/ρ 0.1 and t10 η 0.5/ρ 0.4, planned as the last two `ws_full` trials, were not completed: arrears. ws_full used 8 of 10 trials, memrl 3.)

**Chosen (first pass, rule applied mechanically):** `ws_full`: η = 0.5, ρ = 0 (t07, EX 0.7148); `memrl`: λ_m = 1.0 (m03, 0.7070). Configs: `exp/configs_tuned/e4_ws_full.yaml`, `e4_memrl.yaml`.

Caveats that matter for the paper:
* **Noise.** The five replicated pairs differ by 0.8–1.6 pp (mean |Δ| 1.0 pp, 2–4 of 256 items); the spread of all 8 `ws_full` trials is 2.7 pp. The ranking is therefore within noise: t07's lead over t06/t02/t08 (0.707) is 2 questions, and t07 fell to 0.6992 in the replicate (mean 0.707, tied with four others). The chosen values are weakly supported; the only reasonably consistent signal is "ρ = 0 is not worse than ρ > 0" (ρ = 0.1/0.2/0.4 at η 0.3: 0.6875–0.6914).
* **ρ = 0 turns `ws_full` into `ws_evict`** for recall ranking (eviction by WS stays). If the author wants a genuinely value-ranked headline, use the best ρ > 0 trial: **η = 0.5, ρ = 0.2** (t06, 0.7070; `e4_ws_full_rho_pos.yaml`), which costs 2 questions on this sample. The decision is the author's; I recommend reporting the sensitivity grid rather than a "tuned" claim.
* **Eviction is very active**: 67–83 evictions over 4 dbs × 40 stream questions, of which 3–9 polluted; about 90 % of the evicted entries are clean (X2 open issue 1 confirmed: one EX = 0 feedback on a shared-credit exemplar suffices at the spec rule). `memrl` never evicts (20 polluted entries stay alive).
* The mini-stream is small (4 dbs, 160 + 96 items, one permutation); the spec's E4 matrix is the real test.

## 4. Chosen settings (summary)

| item | value | file |
|---|---|---|
| M3 | k_t = 6, k_e = 4 (= defaults) | `exp/configs_tuned/full_fixed.yaml` |
| M6 ws_full | η = 0.5, ρ = 0 (alt. ρ > 0: η 0.5, ρ 0.2); τ, α/β/γ, φ default | `e4_ws_full.yaml`, `e4_ws_full_rho_pos.yaml` |
| M6 memrl | λ_m = 1.0 | `e4_memrl.yaml` |
| M7 | not tuned here; E6 results in `internal note R0` §E6 | `full_rl.yaml` (controller path still empty) |

## 5. Cost of this task

Computed with `exp/prices.yaml` list prices, non-cached calls only (`results/p2/tools_r0/ucost.py`; `cost_report.py` does not see `usage_builder.jsonl`): HT-300 base ¥3.4, M3 grid ¥9.5, HR `full_fixed` ¥13.5, E6 trees ¥168.7 (of which qwen3.8-max escalations ¥113), pilot ¥0.5, E4 tuning ¥11.1 (v1) + ¥9.5 (v2) ⇒ **≈ ¥216** (≈ ¥230 with the usage lost when I deleted two partial trial directories). Far below the ¥1 500 stop line.
