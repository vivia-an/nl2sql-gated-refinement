"""Default configuration of the full system. Every E3 ablation is a flag here (configs in exp/configs/*.yaml override)."""
DEFAULTS = {
    "temperature": 0.0,
    "models": {"tier1": "qwen37plus", "tier2": "qwen38max", "critic": "self", "embed": "qwen3.7-text-embedding",
               "rerank": "qwen3.7-text-rerank"},
    "m1": {"enabled": True},
    "m2": {"history": True, "terms": True, "lineage": True,
           # X9 (off = legacy behaviour, same prompts and cache keys): M2 business-metadata source = per-database value index; matching question + evidence n-grams
           # to stored short-text values adds a `# Value hints` block (`- table.column: 'stored literal'`) to the generation / revision prompts. Parameters: values.VG_DEFAULTS
           "value_grounding": False,
           "vg": {}},                 # overrides of values.VG_DEFAULTS (max_hints, max_len, fuzzy_threshold, identical, ...)
    "m3": {"mode": "full", "k_t": 6, "k_e": 4, "k_terms": 3, "schema_fallback_tokens": 6000, "n_domains": 2, "rerank": True,
           "expand_fk": True},
    "m4": {"graph_in_prompt": True},
    "m5": {"enabled": True, "variant": "v3", "weights": [0.3, 0.5, 0.2], "theta_fb": 7.5, "theta_acc": 8.5, "t_max": 3,
           "theta_h": 7.5, "route_human": True, "verbalized_confidence": False,
           # X5 (all off = legacy behaviour, same prompts and cache keys as before):
           "policy": "p0",             # p0 = patent thresholds on the V3 score; gated = revise only on execution error / timeout (+ empty if gate_empty)
           "gate_empty": False,        # gated: an executable-but-empty result (no rows / one all-NULL row) also triggers a revision
           "score_for_routing": False, # gated: one LLM critic call on the FINAL answer only, to get the V3 score for routing (route.score v3 / v3+consistency)
           "revise_feedback": "exec",  # gated: exec = the execution error text goes into the revision prompt; blind = it does NOT (P5 control, system full_gated_blind)
           "arbitrate": False,         # one LLM judge call (root vs final, both result previews) when both execute and the results differ
           # X7 (all off = legacy behaviour; gated + gate_empty only): value probing for revisions of empty / all-NULL results
           "gate_zero": False,         # gated: an executable one-cell result equal to 0 (COUNT = 0, SUM = 0) is also revised (like an empty result)
           "probe_values": False,      # read-only SELECT DISTINCT on the short text columns the failed SQL filters with literals; values go into the revision prompt
           "probe_n": 8,               # distinct values shown per column
           "probe_max_len": 40,        # values longer than this (characters) are not probed (short text columns only)
           "probe_near": False,        # True: values resembling the failed literal (case-insensitive token match) are listed first
           "probe_max_cols": 4,        # at most this many columns per revision
           "probe_timeout_s": 5.0},    # per probe statement
    # X5 routing (M7 route-to-human / selective prediction). score: v3 (legacy) | exec | consistency | v3+consistency | exec+consistency
    "route": {"score": "v3", "theta_h": None,     # theta_h None: legacy m5.theta_h for score v3, no abstention for the other kinds
              "coef_path": None, "coef": None,    # logistic combination {features, coef, intercept}; written by tools/fit_route_score.py
              "weights": None,                    # no coefficients: linear mean of the components on the 0-10 scale, weights default equal
              "consistency": {"k": 0, "temperature": 0.7, "n_per_call": 4},   # k extra T>0 samples of the initial generation (DashScope n <= 4)
              # X7 confidence-routed compute (needs m5.policy gated + consistency.k > 0): consistency >= theta_fast -> root answer ("fast"), otherwise
              # execution-result majority over root + samples ("vote"); no revision call, no critic call (tiebreak v3: one critic call per tied group)
              "adaptive": {"enabled": False, "theta_fast": 1.0, "tiebreak": "root", "empty": "vote"}},
    "ddl": {"quote_keywords": False},  # True: SQLite reserved words are double-quoted in the schema block + one rule line in the generation prompts
    "m6": {"policy": "raw", "path": None, "update": False, "params": {}},
    "m7": {"controller": None, "horizon": 6, "call_budget": 8, "max_gen_depth": 3, "actions": 4},
    "pool": None,                       # auto: bird_train_pool / spider_train_pool by dataset
    "max_tokens": {"m1": 700, "generate": 1024, "critic": 512, "revise": 1024},
    "record_candidates": True,
}
