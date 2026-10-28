"""Feature extraction for the learned critic V5 (E5). One function, two callers: the offline trainer
(`analysis/critic_v5.py`, from cands.jsonl / trees.jsonl records) and the live critic (`m5.Critic` with
`variant: v5`, from the objects the pipeline holds). Only information available BEFORE the gold answer is known is used;
the label (EX) never enters a feature.

Record schema (dict; every key optional -> NaN):
  sql, syntax, semantic, efficiency, score (= V3 sub-scores/score), status (exec_status), n_rows, n_cols,
  top_sim, mean_sim, mean_ws, ctx_full_schema, and the question features of rl/features.question_features in their
  normalised form: q_tokens, n_objects, has_time, qf_n_tables, qf_n_columns (schema size), has_evidence,
  k, tier, action, prev_score, changed, plan_scans, plan_temp, plan_corr, plan_search,
  v4_syntax, v4_semantic, v4_efficiency, v4_score   (cross-vendor critic, when the candidate was re-scored)
"""
import math
import re

NAN = float("nan")
STATUSES = ["ok_nonempty", "ok_empty", "err_syntax", "err_identifier", "timeout"]
ACTIONS = ["root", "revise", "re_retrieve", "escalate"]

_STR = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|`[^`]*`")
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")
_JOIN = re.compile(r"\bjoin\b")
_FROMJOIN = re.compile(r"\b(?:from|join)\s+([`\"\[]?[a-z_][\w$]*[`\"\]]?)")
_AGG = re.compile(r"\b(?:count|sum|avg|min|max)\s*\(")
_CMP = re.compile(r"<>|!=|<=|>=|=|<|>|\blike\b|\bin\b|\bbetween\b|\bis\b")

SQL_FEATURES = ["sql_len", "sql_tokens", "n_joins", "n_tables_sql", "n_subq", "n_agg", "n_conds", "has_group", "has_having",
                "has_order", "has_limit", "has_distinct", "has_like", "has_case", "has_union", "n_str_lit", "n_num_lit", "n_bool_conn"]
BASE_FEATURES = (["c_syntax", "c_semantic", "c_eff", "c_score"] +
                 ["st_" + s for s in STATUSES] + ["log_rows", "n_cols", "is_empty"] + SQL_FEATURES +
                 ["plan_scans", "plan_temp", "plan_corr", "plan_search"] +
                 ["top_sim", "mean_sim", "mean_ws", "ctx_full_schema", "q_tokens", "n_objects", "has_time", "qf_n_tables", "qf_n_columns",
                  "has_evidence"] +
                 ["k", "tier"] + ["act_" + a for a in ACTIONS] + ["d_score_prev", "changed_prev"])
V4_FEATURES = ["v4_syntax", "v4_semantic", "v4_eff", "v4_score", "v4_minus_v3"]
# set-level features (need the other candidates of the same question; offline analysis only, NOT live-computable at the first candidate)
SET_FEATURES = ["agree_sql", "agree_sig", "agree_res", "score_minus_max", "score_rank_frac", "n_cands"]


def sql_features(sql):
    s = (sql or "")
    low = _STR.sub("''", s).lower()
    f = {"sql_len": math.log1p(len(s)), "sql_tokens": float(len(low.split()))}
    f["n_joins"] = float(len(_JOIN.findall(low)))
    f["n_tables_sql"] = float(len({m.strip('`"[]') for m in _FROMJOIN.findall(low)}))
    f["n_subq"] = float(max(0, low.count("select") - 1))
    f["n_agg"] = float(len(_AGG.findall(low)))
    f["n_conds"] = float(len(_CMP.findall(low)))
    f["has_group"] = float("group by" in low)
    f["has_having"] = float(" having " in low)
    f["has_order"] = float("order by" in low)
    f["has_limit"] = float(re.search(r"\blimit\b", low) is not None)
    f["has_distinct"] = float("distinct" in low)
    f["has_like"] = float(re.search(r"\blike\b", low) is not None)
    f["has_case"] = float(re.search(r"\bcase\b", low) is not None)
    f["has_union"] = float(re.search(r"\b(?:union|intersect|except)\b", low) is not None)
    f["n_str_lit"] = float(len(_STR.findall(s)))
    f["n_num_lit"] = float(len(_NUM.findall(_STR.sub("''", s))))
    f["n_bool_conn"] = float(len(re.findall(r"\b(?:and|or)\b", low)))
    return f


def plan_features(plan):
    """EXPLAIN QUERY PLAN detail lines -> counts."""
    scans = temp = corr = search = 0
    for line in plan or []:
        t = line.strip()
        if t.startswith("SCAN"):
            scans += 1
        if t.startswith("SEARCH"):
            search += 1
        if "USE TEMP B-TREE" in t:
            temp += 1
        if "CORRELATED SCALAR SUBQUERY" in t:
            corr += 1
    return {"plan_scans": float(scans), "plan_temp": float(temp), "plan_corr": float(corr), "plan_search": float(search)}


def _f(v, default=NAN):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def record_features(rec, with_v4=False):
    """-> dict feature_name -> float (NaN = missing). Live set (no set-level features)."""
    r = rec
    x = {}
    x["c_syntax"], x["c_semantic"], x["c_eff"], x["c_score"] = (_f(r.get("syntax")), _f(r.get("semantic")), _f(r.get("efficiency")), _f(r.get("score")))
    st = r.get("status")
    for s in STATUSES:
        x["st_" + s] = (1.0 if st == s else 0.0) if st else NAN
    nr = _f(r.get("n_rows"))
    x["log_rows"] = math.log1p(nr) if nr == nr else NAN
    x["n_cols"] = _f(r.get("n_cols"))
    x["is_empty"] = (1.0 if st == "ok_empty" else 0.0) if st else NAN
    x.update(sql_features(r.get("sql")))
    for k in ("plan_scans", "plan_temp", "plan_corr", "plan_search"):
        x[k] = _f(r.get(k))
    for k in ("top_sim", "mean_sim", "mean_ws", "q_tokens", "n_objects", "has_time", "qf_n_tables", "qf_n_columns", "has_evidence"):
        x[k] = _f(r.get(k))
    x["ctx_full_schema"] = _f(r.get("ctx_full_schema"))
    x["k"] = _f(r.get("k"))
    x["tier"] = _f(r.get("tier"))
    act = r.get("action")
    for a in ACTIONS:
        x["act_" + a] = (1.0 if act == a else 0.0) if act else NAN
    ps = r.get("prev_score")
    x["d_score_prev"] = (_f(r.get("score")) - float(ps)) if ps is not None and r.get("score") is not None else NAN
    ch = r.get("changed")
    x["changed_prev"] = float(bool(ch)) if ch is not None else NAN
    if with_v4:
        x["v4_syntax"], x["v4_semantic"], x["v4_eff"], x["v4_score"] = (_f(r.get("v4_syntax")), _f(r.get("v4_semantic")),
                                                                         _f(r.get("v4_efficiency")), _f(r.get("v4_score")))
        x["v4_minus_v3"] = x["v4_score"] - x["c_score"] if x["v4_score"] == x["v4_score"] and x["c_score"] == x["c_score"] else NAN
    return x


def feature_names(with_v4=False, with_set=False):
    return BASE_FEATURES + (V4_FEATURES if with_v4 else []) + (SET_FEATURES if with_set else [])


def vectorize(feats, names):
    return [feats.get(n, NAN) for n in names]


def qf_fields(qf, item=None):
    """question_features dict -> record fields (qf_n_tables / qf_n_columns renamed to avoid clashing with sql n_tables_sql)."""
    if qf:
        return {"q_tokens": qf.get("q_tokens"), "n_objects": qf.get("n_objects"), "has_time": qf.get("has_time"),
                "qf_n_tables": qf.get("n_tables"), "qf_n_columns": qf.get("n_columns"), "has_evidence": qf.get("has_evidence")}
    if item is not None:
        return {"q_tokens": min(len(item.question.split()) / 64.0, 1.0), "has_evidence": float(bool((item.evidence or "").strip()))}
    return {}


def live_record(item, sql, res, crit, ctx, plan, meta=None):
    """Record for the live critic from pipeline objects (m5.Critic.score).
    meta: dict(prev_score, changed, k, tier, action, qf) with qf = rl/features.question_features(...) of the root."""
    meta = meta or {}
    pf = plan_features(plan)
    sims = list(getattr(ctx, "ex_sims", []) or [])
    ws = [e.WS for e in (getattr(ctx, "examples", []) or [])]
    rec = {"sql": sql, "syntax": crit["syntax"], "semantic": crit["semantic"], "efficiency": crit["efficiency"], "score": crit["score"],
           "status": crit["status"], "n_rows": len(res["rows"]) if res["ok"] else 0, "n_cols": len(res.get("cols") or []),
           "top_sim": max(sims) if sims else 0.0, "mean_sim": (sum(sims) / len(sims)) if sims else 0.0,
           "mean_ws": (sum(ws) / len(ws)) if ws else NAN,
           "ctx_full_schema": float(bool(getattr(ctx, "full_schema", False)))}
    rec.update(qf_fields(meta.get("qf"), item))
    rec.update(pf)
    rec.update({k: meta.get(k) for k in ("prev_score", "changed", "k", "tier", "action")})
    return rec
