"""27-dim controller state (EXPERIMENT_SPEC E6). Shared by the live system and the tree builder so train and
deploy use the same function. Gold information is never part of the state."""
import math

import numpy as np

STATUSES = ["ok_nonempty", "ok_empty", "err_syntax", "err_identifier", "timeout"]
ACTIONS = ["accept", "revise", "re_retrieve", "escalate", "route_human"]
STATE_DIM = 27
FEATURE_NAMES = (["s_syn", "s_sem", "s_eff", "score", "d_score"] + ["st_" + s for s in STATUSES] +
                 ["log_rows", "n_cols", "result_changed", "k_over_tmax", "rr_over_2", "budget_left", "sim_top1", "sim_meank", "ws_mean",
                  "q_tokens", "n_objects", "has_time", "n_tables", "n_columns", "has_evidence", "tier1", "tier2"])


def question_features(item, graph, n_tables, n_columns):
    objs = len((graph or {}).get("objects") or []) if graph else 0
    has_time = 0.0
    if graph:
        if graph.get("time_scope"):
            has_time = 1.0
        elif any(isinstance(f, dict) and f.get("kind") == "time" for f in graph.get("filters") or []):
            has_time = 1.0
    return {"q_tokens": min(len(item.question.split()) / 64.0, 1.0), "n_objects": min(objs, 10) / 10.0, "has_time": has_time,
            "n_tables": min(n_tables / 20.0, 1.0), "n_columns": min(n_columns / 200.0, 1.0),
            "has_evidence": 1.0 if (item.evidence or "").strip() else 0.0}


def build_state(node, qf, k, n_rr, budget_left, tier, sims, ws_mean):
    """node.critic: dict(syntax, semantic, efficiency, score, status); node.prev_score, node.n_rows, node.n_cols, node.changed."""
    c = node["critic"]
    v = [c["syntax"] / 10.0, (c["semantic"] if c["semantic"] is not None else 5.0) / 10.0, c["efficiency"] / 10.0, c["score"] / 10.0]
    prev = node.get("prev_score")
    v.append(0.0 if prev is None else (c["score"] - prev) / 10.0)
    v += [1.0 if c["status"] == s else 0.0 for s in STATUSES]
    v.append(math.log1p(node.get("n_rows", 0)) / math.log1p(1000))
    v.append(min(node.get("n_cols", 0), 10) / 10.0)
    v.append(1.0 if node.get("changed") else 0.0)
    v += [k / 3.0, n_rr / 2.0, max(0.0, min(1.0, budget_left))]
    v += [max(sims) if sims else 0.0, (sum(sims) / len(sims)) if sims else 0.0, ws_mean / 10.0]
    v += [qf["q_tokens"], qf["n_objects"], qf["has_time"], qf["n_tables"], qf["n_columns"], qf["has_evidence"]]
    v += [1.0 if tier == 0 else 0.0, 1.0 if tier == 1 else 0.0]
    a = np.asarray(v, dtype=np.float32)
    assert a.shape[0] == STATE_DIM, a.shape
    return a
