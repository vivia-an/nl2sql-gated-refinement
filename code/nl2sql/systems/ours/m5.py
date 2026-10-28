"""M5 Critic. Score = w1*s_syn + w2*s_sem + w3*s_eff (patent default 0.3/0.5/0.2, range 0-10).
s_syn and s_eff are deterministic functions of the execution / EXPLAIN QUERY PLAN; s_sem is an LLM judgement.
Variants (E5): v1 binary yes/no, v2 pure-LLM three axes, v3 execution-grounded weighted (default), v4 = v3 with the
cross-vendor critic model (a model setting, not a code path), v5 learned (features exposed; classifier trained in analysis)."""
import re

from ... import db
from ... import prompts_util as P
from ...util import parse_json, trunc
from .m1 import graph_text


def exec_summary(res, max_preview_rows=5, cell=40):
    """Execution preview text + structured status."""
    if not res["ok"]:
        return "Execution error: %s" % res["error"][:300]
    rows = res["rows"]
    n = len(rows) if not res["truncated"] else ">=%d" % len(rows)
    lines = ["Columns: %s" % ", ".join(res.get("cols") or []), "Row count: %s" % n]
    for r in rows[:max_preview_rows]:
        lines.append(" | ".join(trunc(c, cell) for c in r))
    if not rows:
        lines.append("(no rows)")
    return "\n".join(lines)


def exec_status(res):
    """-> one of ok_nonempty, ok_empty, err_syntax, err_identifier, timeout (other runtime errors -> err_syntax)."""
    if not res["ok"]:
        k = db.error_kind(res["error"])
        return {"timeout": "timeout", "identifier": "err_identifier"}.get(k, "err_syntax")
    rows = res["rows"]
    if not rows or (len(rows) == 1 and all(c is None for c in rows[0])):
        return "ok_empty"
    return "ok_nonempty"


def s_syn(res):
    if res["ok"]:
        rows = res["rows"]
        if not rows or (len(rows) == 1 and len(rows[0]) >= 1 and all(c is None for c in rows[0])):
            return 8.0
        return 10.0
    k = db.error_kind(res["error"])
    return {"identifier": 3.0, "syntax": 2.0, "timeout": 1.0}.get(k, 3.0)


_SCAN = re.compile(r"^SCAN (?:TABLE )?(\S+)")


def s_eff(res, plan, row_counts):
    """10, -2 per full SCAN of a table with >10000 rows (cap -6), -1 per TEMP B-TREE, -1 per CORRELATED SCALAR SUBQUERY,
    -2 if elapsed >2 s, -4 if >10 s (non-cumulative: the larger); clip [0,10]. Exec error -> 0."""
    if not res["ok"]:
        return 0.0
    s = 10.0
    scan_pen = 0.0
    tmp = 0
    corr = 0
    rc = {k.lower(): v for k, v in (row_counts or {}).items()}
    for line in plan:
        m = _SCAN.match(line.strip())
        if m:
            name = m.group(1).strip('"`[]').lower()
            if rc.get(name, 0) > 10000:
                scan_pen += 2.0
        if "USE TEMP B-TREE" in line:
            tmp += 1
        if "CORRELATED SCALAR SUBQUERY" in line:
            corr += 1
    s -= min(scan_pen, 6.0) + tmp + corr
    el = res.get("elapsed_s", 0.0)
    s -= 4.0 if el > 10 else (2.0 if el > 2 else 0.0)
    return float(max(0.0, min(10.0, s)))


class Critic:
    def __init__(self, llm, model, cfg):
        self.llm, self.model, self.cfg = llm, model, cfg
        self.w = cfg["m5"]["weights"]
        self.variant = cfg["m5"]["variant"]
        self.learned = None
        if self.variant == "v5":          # learned critic (E5): calibrated P(EX=1) from candidate features, exported by analysis/critic_v5.py
            from .learned_critic import LearnedCritic
            self.learned = LearnedCritic.load(((cfg["m5"].get("learned") or {}).get("path")) or "exp/frozen/critic_v5.pkl")

    def score(self, item, sql, ctx, graph, res, seed, row_counts, variant=None, with_text=True, meta=None):
        """-> dict(syntax, semantic, efficiency, score, issues, suggestions, status, parse_ok, binary)."""
        variant = variant or self.variant
        cfg = self.cfg
        syn = s_syn(res)
        plan = db.explain(item.db_path, sql) if res["ok"] else []
        eff = s_eff(res, plan, row_counts)
        out = {"syntax": syn, "efficiency": eff, "semantic": None, "issues": [], "suggestions": [], "status": exec_status(res),
               "parse_ok": True, "plan_scans": sum(1 for l in plan if l.strip().startswith("SCAN")),
               "plan_temp": sum(1 for l in plan if "USE TEMP B-TREE" in l), "plan_corr": sum(1 for l in plan if "CORRELATED SCALAR SUBQUERY" in l),
               "plan_search": sum(1 for l in plan if l.strip().startswith("SEARCH"))}
        common = [{"role": "system", "content": P.render("common.system")}]
        gt = graph_text(graph) if (graph and cfg["m4"]["graph_in_prompt"] and cfg["m1"]["enabled"]) else ""
        kw = dict(sections=ctx.sections_core(), question=item.question, evidence=item.evidence or "(none)", graph=gt,
                  sql=sql, preview=exec_summary(res))
        mt = cfg["max_tokens"]["critic"]
        if variant == "v1":
            r = self.llm.chat(self.model, common + [{"role": "user", "content": P.render("crit_v1.user", **kw)}],
                              temperature=0.0, seed=seed, max_tokens=128, tag="crit")
            yes = r["text"].strip().lower().startswith("yes")
            out.update({"binary": int(yes), "semantic": 10.0 if yes else 3.0, "issues": [{"type": "semantic", "description": r["text"].strip()[:300]}] if not yes else []})
            out["score"] = 10.0 if yes else 3.0
            return out
        if variant == "v2":
            msgs = common + [{"role": "user", "content": P.render("crit_v2.user", **kw)}]
            j, ok = self._json(msgs, seed, mt)
            out["parse_ok"] = ok
            if j:
                out["syntax"] = _num(j.get("syntax"), 5.0)
                out["efficiency"] = _num(j.get("efficiency"), 5.0)
                out["semantic"] = _num(j.get("semantic"), 5.0)
                out["issues"] = j.get("issues") or []
                out["suggestions"] = j.get("suggestions") or []
            else:
                out["semantic"] = 5.0
        else:   # v3 / v4 / v5 base scores
            msgs = common + [{"role": "user", "content": P.render("crit_sem.user", **kw)}]
            j, ok = self._json(msgs, seed, mt)
            out["parse_ok"] = ok
            if j:
                out["semantic"] = _num(j.get("semantic"), 5.0)
                out["issues"] = j.get("issues") or []
                out["suggestions"] = j.get("suggestions") or []
            else:
                out["semantic"] = 5.0      # neutral on unparseable critic output (deviation: item is not failed)
        w = self.w
        out["score"] = round(w[0] * out["syntax"] + w[1] * out["semantic"] + w[2] * out["efficiency"], 4)
        if variant == "v5" and self.learned is not None:
            from .critic_features import live_record
            p = self.learned.p_correct(live_record(item, sql, res, out, ctx, plan, meta))
            out["score_v3"] = out["score"]
            out["p_correct"] = round(p, 6)
            out["score"] = round(10.0 * p, 4)
        return out

    def score_exec(self, item, sql, res, row_counts):
        """Execution-grounded sub-scores only, NO LLM call (policy `gated`). `score` is the syntax / field-mismatch sub-score
        (10 non-empty, 8 empty, 3 unknown column / other error, 2 syntax error, 1 timeout), the quantity the gate and the
        `exec` routing score are defined on. semantic is None: the semantic critic was not asked."""
        plan = db.explain(item.db_path, sql) if res["ok"] else []
        syn = s_syn(res)
        eff = s_eff(res, plan, row_counts)
        return {"syntax": syn, "efficiency": eff, "semantic": None, "issues": [], "suggestions": [], "status": exec_status(res),
                "parse_ok": True, "score": syn, "score_kind": "exec",
                "plan_scans": sum(1 for l in plan if l.strip().startswith("SCAN")),
                "plan_temp": sum(1 for l in plan if "USE TEMP B-TREE" in l), "plan_corr": sum(1 for l in plan if "CORRELATED SCALAR SUBQUERY" in l),
                "plan_search": sum(1 for l in plan if l.strip().startswith("SEARCH"))}

    def _json(self, msgs, seed, mt):
        r = self.llm.chat(self.model, msgs, temperature=0.0, seed=seed, max_tokens=mt, tag="crit")
        j = parse_json(r["text"])
        if j is None:
            r = self.llm.chat(self.model, msgs + [{"role": "assistant", "content": r["text"]},
                                                  {"role": "user", "content": P.render("m1.retry")}],
                              temperature=0.0, seed=seed, max_tokens=mt, tag="crit")
            j = parse_json(r["text"])
        return j, j is not None


def _num(v, default):
    try:
        return float(max(0, min(10, float(v))))
    except Exception:
        return default


def feedback_text(crit, res):
    lines = []
    if not res["ok"]:
        lines.append("Execution error: " + res["error"][:300])
    for i in crit.get("issues") or []:
        if isinstance(i, dict):
            lines.append("- [%s] %s" % (i.get("type", "other"), i.get("description", "")))
        else:
            lines.append("- " + str(i))
    for s in crit.get("suggestions") or []:
        lines.append("- suggestion: %s" % s)
    if crit.get("efficiency", 10) < 8:
        lines.append("- [efficiency] the query plan shows %d full table scan(s) or temporary structures; reduce them if the result stays identical" % crit.get("plan_scans", 0))
    return "\n".join(lines) or "- no specific issue reported"


# ----------------------------------------------------------------------------- X5: execution gate
GATE_ERR = ("err_syntax", "err_identifier", "timeout")
STATUS_RANK = {"ok_nonempty": 2, "ok_empty": 1, "err_syntax": 0, "err_identifier": 0, "timeout": 0}


def gate_trigger(status, gate_empty=False):
    """Revision trigger of policy `gated`: execution error / timeout, and (gate_empty) an executable but empty result."""
    return status in GATE_ERR or (gate_empty and status == "ok_empty")


def is_zero_result(res):
    """X7 (m5.gate_zero): the result is exactly one row with one column equal to 0 (COUNT = 0, SUM = 0 ...), a degenerate answer for a question that asks
    for something that exists. Not a separate exec status: `status` stays ok_nonempty."""
    if not res.get("ok") or len(res["rows"]) != 1 or len(res["rows"][0]) != 1:
        return False
    v = res["rows"][0][0]
    return (not isinstance(v, bool)) and isinstance(v, (int, float)) and v == 0


def gate_node(node, gate_empty=False, gate_zero=False):
    """Revision trigger for a node: gate_trigger(status, gate_empty), plus (gate_zero) an executable one-cell zero result."""
    st = node["critic"]["status"]
    return gate_trigger(st, gate_empty) or (gate_zero and st == "ok_nonempty" and is_zero_result(node["res"]))


def node_rank(node, gate_zero=False):
    """Ranking of a node for the final choice of the gated loop: STATUS_RANK; with gate_zero a zero result ranks like an empty one (1), so that a
    non-degenerate revision replaces it and an erroring revision does not."""
    r = STATUS_RANK.get(node["critic"]["status"], 0)
    return 1 if (gate_zero and r == 2 and is_zero_result(node["res"])) else r


BLIND_PREVIEW = "(execution output withheld)"
BLIND_FEEDBACK = "The previous query failed. Write a corrected query."


ZERO_FEEDBACK = ("The query executed but returned a single value 0 (for example a count or sum of zero). Check filter values, join keys and conditions "
                 "against the schema and the sample values; if zero is the correct answer, return the same SQL.")


def exec_feedback_text(res, status, zero=False):
    """Feedback of the gated revision: the actual execution error text; for an empty result the fact that it was empty (zero: a single 0 value, X7)."""
    if not res["ok"]:
        return "Execution error: " + str(res["error"])[:300]
    if zero:
        return ZERO_FEEDBACK
    if status == "ok_empty":
        return ("The query executed but returned no rows (or a single row of NULLs). Check filter values, join keys and conditions "
                "against the schema and the sample values; if an empty result is the correct answer, return the same SQL.")
    return "- no execution problem reported"
