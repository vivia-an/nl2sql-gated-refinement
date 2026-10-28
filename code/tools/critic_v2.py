#!/usr/bin/env python3
"""Better-specified Critic variants VA / VB on the DIAG items (SAP Amendment 6, docs/SAP.md section 18).  scope = diagnostic-posthoc.

Nothing here edits code/prompts/** or exp/models.yaml: the variant prompts live in exp/diag_prompts/ and are rendered by this tool only; the
pipeline (Pipeline, P0 loop, reviser prompt rev.user) is used unchanged, with the critic object swapped for a subclass of m5.Critic.

  audit    what the V0 critic actually sees as schema (pruned or full) and whether its "table/column not defined" claims are about objects that are
           hidden by the pruning; uses M1/M3 exactly as the DIAG run (cache hits) and the stored V0 records.
  run      --variant V0|VA|VB  [--critic-only]  one pass over an items.json (tuning draw or the frozen evaluation items).
  analyze  per variant metrics + paired deltas to V0 (database-clustered bootstrap), cases.

Usage (cwd = code dir, python = worker machine venv):
  python tools/critic_v2.py audit   --items I --v0 ../results/diag/records.jsonl.gz --out OUT
  python tools/critic_v2.py run     --items I --variant VA --out OUT [--conc 4] [--critic-only] [--limit N]
  python tools/critic_v2.py analyze --items ../results/diag/items.json --v0 ../results/diag/records.jsonl.gz --runs VA=OUT_A VB=OUT_B --out ../results/diag
"""
import argparse
import collections
import copy
import gzip
import hashlib
import json
import os
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import critic_diag as D  # noqa: E402

SCOPE = "diagnostic-posthoc"
REPO = os.path.dirname(CODE)
PROMPT_DIR = os.path.join(REPO, "exp", "diag_prompts")
PROMPT_FILES = {"VA": "va_grounded.user.txt", "VB": "vb_memory.user.txt", "VC": "vc_verifier.user.txt", "VD": "vd_appeal.user.txt"}
CRITIC_FILE = {"VA": "VA", "VB": "VB", "VC": "VA", "VD": "VA"}    # VC / VD use the frozen VA critic prompt; their own file is the verifier / appeal prompt
THETA_ACC_V = 7.0          # SAP 18.2: V0's trigger (8.5) at syntax 10 / efficiency 10 is semantic < 7
THETA_FB_V = 5.0           # V0's 7.5 "must fix" level at syntax 10 / efficiency 10
MAX_TOKENS_V = 900
MAX_TOKENS_VER = 500
MAX_TOKENS_APP = 600
TL = threading.local()      # per-thread call counters (by tag)
K_MEMORY = 4
SEED_BOOT = 20261101


# ======================================================================================================== rendering
def render_text(text, **kw):
    """Same placeholder syntax as prompts_util.render (<<name>>, optional <<name?>>), for text read from exp/diag_prompts."""
    def rep(m):
        k = m.group(1)
        if k not in kw:
            if m.group(2):
                return ""
            raise KeyError("missing placeholder %s" % k)
        return str(kw[k])
    return re.sub(r"<<(\w+)(\?)?>>", rep, text)


def prompt_text(variant, prompt_dir=None):
    return open(os.path.join(prompt_dir or PROMPT_DIR, PROMPT_FILES[variant])).read()


def full_schema_text(schema):
    """Complete table / column listing (names and declared types only; no samples, no descriptions)."""
    lines = []
    for t, cols in schema["tables"].items():
        lines.append("%s(%s)" % (t, ", ".join(("%s %s" % (c["name"], c.get("type") or "")).strip() for c in cols)))
    return "\n".join(lines)


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def split_issues(raw_issues):
    """-> (must_fix, style).  A must_fix issue needs a non-empty quoted question phrase AND SQL fragment (SAP 18.2 iii); without them it is demoted to style."""
    must, style = [], []
    for i in raw_issues or []:
        if not isinstance(i, dict):
            style.append({"description": str(i)})
            continue
        sev = _norm(i.get("severity"))
        qp, sf = str(i.get("question_phrase") or "").strip(), str(i.get("sql_fragment") or "").strip()
        if sev == "must_fix" and len(qp) >= 2 and len(sf) >= 2:
            must.append({"type": "semantic", "description": "%s (question: \"%s\"; SQL: \"%s\")" % (str(i.get("description") or "").strip(), qp, sf),
                         "question_phrase": qp, "sql_fragment": sf})
        else:
            d = dict(i)
            d["demoted"] = bool(sev == "must_fix")
            style.append(d)
    return must, style


def consistent_semantic(raw, n_must):
    """Scale consistency (prompt scale 9-10 none / 7-8 doubtful / 0-6 must_fix): a score below the trigger needs at least one valid must_fix issue, and a valid
    must_fix issue caps the score at 6.  The raw model score is kept in the record."""
    s = raw
    if n_must == 0:
        s = max(s, THETA_ACC_V)
    else:
        s = min(s, THETA_ACC_V - 1.0)
    return float(s)


# ======================================================================================================== the variant critic
class VariantCritic(object):
    """Drop-in replacement of Pipeline.critic: same score() signature and return dict (score = semantic only)."""

    def __init__(self, base, variant, prompt_dir=None, memory=None):
        self.base = base                       # stock m5.Critic: provides llm, model, cfg, and the execution-grounded syntax / efficiency fields
        self.variant = variant
        self.template = prompt_text(CRITIC_FILE[variant], prompt_dir)
        self.memory = memory if memory is not None else {}   # qid -> memory text (VB)
        self.learned = None
        self.llm, self.model, self.cfg = base.llm, base.model, base.cfg
        self.schemas = {}
        self.ctx_info = {}

    def _full_schema(self, item):
        from nl2sql import data
        return full_schema_text(data.get_schema(item))

    def score(self, item, sql, ctx, graph, res, seed, row_counts, variant=None, with_text=True, meta=None):
        from nl2sql import db
        from nl2sql import prompts_util as P
        from nl2sql.systems.ours import m5 as M5
        from nl2sql.systems.ours.m1 import graph_text
        from nl2sql.util import parse_json
        syn = M5.s_syn(res)
        plan = db.explain(item.db_path, sql) if res["ok"] else []
        eff = M5.s_eff(res, plan, row_counts)
        out = {"syntax": syn, "efficiency": 10.0, "efficiency_raw": eff, "semantic": None, "issues": [], "style_issues": [], "suggestions": [],
               "status": M5.exec_status(res), "parse_ok": True,
               "plan_scans": sum(1 for l in plan if l.strip().startswith("SCAN")), "plan_temp": sum(1 for l in plan if "USE TEMP B-TREE" in l),
               "plan_corr": sum(1 for l in plan if "CORRELATED SCALAR SUBQUERY" in l), "plan_search": sum(1 for l in plan if l.strip().startswith("SEARCH"))}
        gt = graph_text(graph) if (graph and self.cfg["m4"]["graph_in_prompt"] and self.cfg["m1"]["enabled"]) else ""
        kw = dict(sections=ctx.sections_core(), full_schema=self._full_schema(item), question=item.question, evidence=item.evidence or "(none)", graph=gt,
                  sql=sql, preview=M5.exec_summary(res), memory=self.memory.get(item.qid, "(no past cases)"))
        msgs = [{"role": "system", "content": P.render("common.system")}, {"role": "user", "content": render_text(self.template, **kw)}]
        r = self.llm.chat(self.model, msgs, temperature=0.0, seed=seed, max_tokens=MAX_TOKENS_V, tag="crit")
        j = parse_json(r["text"])
        if j is None:
            r = self.llm.chat(self.model, msgs + [{"role": "assistant", "content": r["text"]}, {"role": "user", "content": P.render("m1.retry")}],
                              temperature=0.0, seed=seed, max_tokens=MAX_TOKENS_V, tag="crit")
            j = parse_json(r["text"])
        out["parse_ok"] = j is not None
        raw = 5.0
        if j:
            raw = M5._num(j.get("semantic"), 5.0)
            must, style = split_issues(j.get("issues"))
            out["issues"], out["style_issues"] = must, style
            out["suggestions"] = (j.get("suggestions") or []) if must else []
        else:
            must = []
        out["semantic_raw"] = raw
        out["semantic"] = consistent_semantic(raw, len(must)) if j else raw
        out["score"] = round(out["semantic"], 4)
        return out


# ======================================================================================================== memory (VB)
class Memory(object):
    """k past cases with ORACLE verdicts from the pipeline's own corpus recall, restricted to the feedback-bearing burn-in entries (SAP 18.2)."""

    def __init__(self, corpus, verdict_by_qid, forbidden_qids, k=K_MEMORY):
        self.corpus, self.k = corpus, k
        self.verdict = verdict_by_qid
        ok = set()
        for eid in corpus.order:
            e = corpus.entries[eid]
            if e.kind == "exp" and e.origin == "burnin" and eid.startswith("bi:") and eid[3:] in verdict_by_qid:
                ok.add(eid)
        assert not any(("bi:" + q) in ok for q in forbidden_qids), "evaluation / tuning qid inside the feedback store"
        self.ok = ok
        self.block = set(eid for eid in corpus.order if eid not in ok)

    def cases(self, res, rerank_model, item):
        qv = res.embed_one(item.question)
        ex = set(self.block) | {item.qid, "bi:" + item.qid}
        entries, _ = self.corpus.recall(rerank_model, item.question, qv, self.k, exclude=ex, rerank=True, hybrid=True)
        out = []
        for e in entries:
            assert e.id in self.ok and e.id[3:] != item.qid
            out.append({"id": e.id, "question": e.question, "evidence": e.evidence, "sql": e.sql, "verdict": "accepted" if self.verdict[e.id[3:]] else "rejected"})
        return out

    @staticmethod
    def render(cases):
        if not cases:
            return "(no past cases)"
        parts = []
        for n, c in enumerate(cases, 1):
            v = "ACCEPTED as correct" if c["verdict"] == "accepted" else "REJECTED as incorrect"
            parts.append("Case %d. Verdict (oracle feedback): %s\nQuestion: %s\nEvidence: %s\nSQL: %s" % (n, v, c["question"], c["evidence"] or "(none)", c["sql"]))
        return "\n\n".join(parts)


def load_verdicts(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            d = json.loads(line)
            out[d["qid"]] = int(d["ex"])
    return out


# ======================================================================================================== VC / VD: verification
RUNTIME_CLAIM = re.compile(r"runtime error|will (?:fail|raise|throw|cause an? (?:error|exception))|syntax error|invalid sql|not executable|execution error|"
                           r"would (?:fail|error)|sql error|fails? to execute", re.I)
SCHEMA_CLAIM = re.compile(r"(?:\b(?:table|column|field)s?\b[^.;]{0,80}?(?:not defined|undefined|does not exist|doesn't exist|do not exist|not exist|"
                          r"not in the (?:provided )?schema|not found|is not present|not present|isn't defined|not available)|no such (?:table|column)|"
                          r"(?:not defined|does not exist|doesn't exist|not found) in the (?:provided )?(?:schema|table|database))", re.I)
NOT_RETURN = re.compile(r"(?:fail(?:s|ed)? to|does not|doesn't|did not|not)\s+(?:return|select|include|output|retrieve)\b", re.I)
QUOTE_TOKEN = re.compile(r"[`'\"]([A-Za-z_][\w .]{0,60}?)[`'\"]")


def stage1_ab(issue, res):
    """Deterministic checks (SAP 18.10 a, b).  -> None if the issue survives, else a reason string."""
    d = "%s %s" % (issue.get("description") or "", "")
    if res.get("ok"):
        if RUNTIME_CLAIM.search(d):
            return "a_runtime_claim_but_query_executed"
        if SCHEMA_CLAIM.search(d):
            return "a_schema_existence_claim_but_query_executed"
    if NOT_RETURN.search(d):
        header = set(str(c).lower() for c in (res.get("cols") or []))
        toks = [t.strip().split(".")[-1].lower() for t in QUOTE_TOKEN.findall(d)]
        toks = [t for t in toks if t]
        if toks and header and all(t in header for t in toks):
            return "b_claims_column_not_returned_but_header_has_it"
    return None


def results_equal(sql_a, res_a, sql_b, res_b):
    """Moot test (SAP 18.10 c): same SQL text, or both executed and return the same rows (ordered if either SQL has ORDER BY, else as a multiset)."""
    if sql_a.strip() == sql_b.strip():
        return True
    if not (res_a.get("ok") and res_b.get("ok")):
        return False
    ra, rb = [tuple(r) for r in res_a["rows"]], [tuple(r) for r in res_b["rows"]]
    if "order by" in sql_a.lower() or "order by" in sql_b.lower():
        return ra == rb
    return D._rows_bag(ra) == D._rows_bag(rb)


def comparison_text(res_a, res_b, same):
    from nl2sql.systems.ours import m5 as M5
    if not res_b.get("ok"):
        return "the proposed fix does not execute (%s)" % str(res_b.get("error"))[:120]
    na, nb = len(res_a["rows"]), len(res_b["rows"])
    return ("%s; original %d row(s), fix %d row(s), difference %+d; status of the original: %s, of the fix: %s"
            % ("the two results are IDENTICAL" if same else "the two results DIFFER", na, nb, nb - na, M5.exec_status(res_a), M5.exec_status(res_b)))


def _chat_json(pipe, user, tag, max_tokens, seed):
    from nl2sql import prompts_util as P
    from nl2sql.util import parse_json
    msgs = [{"role": "system", "content": P.render("common.system")}, {"role": "user", "content": user}]
    r = pipe.llm.chat(pipe.critic_model, msgs, temperature=0.0, seed=seed, max_tokens=max_tokens, tag=tag)
    return parse_json(r["text"])


def verify_issue(pipe, vc, item, node, issue, probe, same, seed, prompt_dir):
    """Stage 2: one verifier call; no critic score, no gold.  -> (verdict, evidence)."""
    from nl2sql.systems.ours import m5 as M5
    tmpl = prompt_text("VC", prompt_dir)
    user = render_text(tmpl, full_schema=vc._full_schema(item), question=item.question, evidence=item.evidence or "(none)", sql=node["sql"],
                       preview=M5.exec_summary(node["res"]), question_phrase=issue["question_phrase"], sql_fragment=issue["sql_fragment"],
                       description=issue["description"].split(" (question:")[0], probe_sql=probe["sql"], probe_preview=M5.exec_summary(probe["res"]),
                       comparison=comparison_text(node["res"], probe["res"], same))
    j = _chat_json(pipe, user, "ver", MAX_TOKENS_VER, seed)
    v = str((j or {}).get("verdict") or "").strip().lower()
    return ("confirmed" if v == "confirmed" else "unsupported"), str((j or {}).get("evidence") or "")[:600], j is not None


def appeal_issue(pipe, vc, item, node, issue, rebuttal, probe, same, seed, prompt_dir):
    """VD appeal: the critic decides maintain / withdraw per issue.  Maintain needs a non-empty justification; unparsable -> withdraw."""
    from nl2sql.systems.ours import m5 as M5
    tmpl = prompt_text("VD", prompt_dir)
    user = render_text(tmpl, full_schema=vc._full_schema(item), question=item.question, evidence=item.evidence or "(none)", sql=node["sql"],
                       preview=M5.exec_summary(node["res"]), question_phrase=issue["question_phrase"], sql_fragment=issue["sql_fragment"],
                       description=issue["description"].split(" (question:")[0], verdict="unsupported", rebuttal=rebuttal or "(no quoted evidence)",
                       probe_sql=probe["sql"], probe_preview=M5.exec_summary(probe["res"]), comparison=comparison_text(node["res"], probe["res"], same))
    j = _chat_json(pipe, user, "app", MAX_TOKENS_APP, seed)
    dec = str((j or {}).get("decision") or "").strip().lower()
    just = str((j or {}).get("justification") or "").strip()
    return ("maintain" if (dec == "maintain" and len(just) >= 10) else "withdraw"), just[:600]


def _with_issues(node, issues, raw):
    nd = dict(node)
    c = dict(node["critic"])
    c["issues"] = list(issues)
    c["semantic"] = min(raw, THETA_ACC_V - 1.0)
    c["score"] = round(c["semantic"], 4)
    nd["critic"] = c
    return nd


def _finalize(node, flagged, raw):
    c = node["critic"]
    c["critic_raw_score"] = raw
    if not flagged:
        c["semantic"] = max(raw, THETA_ACC_V)
        c["score"] = round(c["semantic"], 4)
        c["issues_unverified"] = c.get("issues")
        c["issues"] = []
    else:
        c["semantic"] = min(raw, THETA_ACC_V - 1.0)
        c["score"] = round(c["semantic"], 4)


def verified_loop(pipe, vc, cfg, item, node0, gold_rows, theta_acc, t_max, seed, appeal, critic_only, prompt_dir):
    """The P0 loop with the verification stages in front of every revision (SAP 18.10 / 18.11).  Returns (loop record, root stage log)."""
    nodes, cur, actions, rounds = [node0], node0, [], []
    root_stage, logs = None, []
    while True:
        raw = float(cur["critic"]["score"])
        st = {"k": cur["k"], "critic_flagged": bool(raw < theta_acc), "issues_total": len(cur["critic"].get("issues") or [])}
        if raw >= theta_acc:
            _finalize(cur, False, raw)
            actions.append("accept")
            st.update(after_ab=False, after_c=False, after_final=False)
            logs.append(st)
            root_stage = root_stage or st
            break
        if cur["k"] >= t_max:                 # cannot be revised: not verified, raw critic score is used for the final choice
            st.update(unverified_terminal=True)
            logs.append(st)
            root_stage = root_stage or st
            break
        issues = list(cur["critic"].get("issues") or [])
        surv, rej = [], []
        for i in issues:
            why = stage1_ab(i, cur["res"])
            (rej if why else surv).append((i, why))
        st["n_rej_ab"] = len(rej)
        st["rej_ab_reasons"] = [w for _, w in rej]
        surv = [i for i, _ in surv]
        st["after_ab"] = bool(surv)
        if not surv:
            _finalize(cur, False, raw)
            actions.append("accept")
            st.update(after_c=False, after_final=False, removed_by="ab")
            logs.append(st)
            root_stage = root_stage or st
            break
        probe = pipe.act(item, _with_issues(cur, surv, raw), "revise", T=cfg["temperature"], seed=seed, corpus=None)
        same = results_equal(cur["sql"], cur["res"], probe["sql"], probe["res"])
        st["probe_changed_result"] = not same
        st["probe_score_res"] = D._score_sql(item, probe["sql"], gold_rows)
        st["after_c"] = not same
        if same:
            _finalize(cur, False, raw)
            actions.append("accept")
            st.update(after_final=False, removed_by="c")
            logs.append(st)
            root_stage = root_stage or st
            break
        conf, unsup = [], []
        vlog = []
        for i in surv:
            verdict, evid, okp = verify_issue(pipe, vc, item, cur, i, probe, same, seed, prompt_dir)
            (conf if verdict == "confirmed" else unsup).append((i, evid))
            vlog.append({"claim": i["description"][:300], "verdict": verdict, "evidence": evid, "parse_ok": okp})
        st["verifier"] = vlog
        st["n_confirmed"], st["n_unsupported"] = len(conf), len(unsup)
        keep = [i for i, _ in conf]
        if appeal and unsup:
            alog = []
            maintained = []
            for i, evid in unsup:
                dec, just = appeal_issue(pipe, vc, item, cur, i, evid, probe, same, seed, prompt_dir)
                alog.append({"claim": i["description"][:300], "decision": dec, "justification": just, "rebuttal": evid})
                if dec == "maintain":
                    maintained.append(i)
            st["appeal"] = alog
            st["n_maintained"], st["n_withdrawn"] = len(maintained), len(unsup) - len(maintained)
            st["flagged_by_verifier_alone"] = bool(keep)
            keep = keep + maintained
        st["after_final"] = bool(keep)
        if not keep:
            _finalize(cur, False, raw)
            actions.append("accept")
            st["removed_by"] = "s2" if not appeal else "s2_appeal"
            logs.append(st)
            root_stage = root_stage or st
            break
        logs.append(st)
        root_stage = root_stage or st
        _finalize(cur, True, raw)
        cur["critic"]["issues"] = keep
        if critic_only:
            break
        if len(keep) == len(surv):
            child = probe
        else:
            child = pipe.act(item, _with_issues(cur, keep, raw), "revise", T=cfg["temperature"], seed=seed, corpus=None)
        actions.append("revise")
        rounds.append({"sql": child["sql"], "noop": child["sql"].strip() == cur["sql"].strip(), "critic": brief(child["critic"]),
                       "score_res": D._score_sql(item, child["sql"], gold_rows)})
        nodes.append(child)
        if child["sql"].strip() == cur["sql"].strip():
            actions.append("revise_noop")
            break
        cur = child
    out = {"stage_logs": logs}
    if critic_only:
        return out, root_stage
    best = max(nodes, key=lambda n: (n["critic"]["score"], -n["k"]))
    if actions and actions[-1] != "accept":
        actions.append("stop_best")
    out.update({"actions": actions, "rounds": rounds, "final_sql": best["sql"], "final_k": best["k"], "final_score": best["critic"]["score"],
                "final_res": D._score_sql(item, best["sql"], gold_rows)})
    return out, root_stage


# ======================================================================================================== shared setup
def setup(a, variant):
    from nl2sql import config as C
    from nl2sql import paths
    from nl2sql.llm import LLMClient
    from nl2sql.systems.ours import pipeline as PL
    cfg = PL.resolved_cfg(C.load_config(os.path.join(str(paths.REPO), "exp", "configs", "frozen", "full_fixed.yaml")))
    assert cfg["m5"]["policy"] == "p0" and cfg["m5"]["variant"] == "v3" and list(cfg["m5"]["weights"]) == [0.3, 0.5, 0.2]
    model = C.resolve_model("qwen37plus")
    os.makedirs(a.out, exist_ok=True)
    llm = LLMClient(cache_path=str(paths.CACHE / "llm"), usage_path=os.path.join(a.out, "usage.jsonl"), key_file=paths.KEY_FILE,
                    max_concurrency_per_model=a.conc)
    return cfg, model, llm


def prompts_state():
    from nl2sql import paths
    h = hashlib.sha256()
    for f in sorted(os.listdir(str(paths.PROMPTS_DIR))):
        h.update(f.encode())
        h.update(open(os.path.join(str(paths.PROMPTS_DIR), f), "rb").read())
    return h.hexdigest()


def gold_tables(sql):
    import sqlglot
    from sqlglot import exp
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return set()
    ctes = set(c.alias.lower() for c in tree.find_all(exp.CTE) if c.alias)
    return set(t.name.lower() for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in ctes)


def ctx_info(item, ctx, schema, sql_list):
    """What the critic is shown as schema, and which tables of the given SQLs are not in it."""
    all_t = [t.lower() for t in schema["tables"]]
    shown = [t.lower() for t in ctx.tables]
    shown_cols = sum(len(schema["tables"][t]) for t in ctx.tables if t in schema["tables"])
    all_cols = sum(len(c) for c in schema["tables"].values())
    miss = {}
    for k, s in sql_list.items():
        miss[k] = sorted(t for t in gold_tables(s) if t not in shown and t in all_t)
    return {"n_tables_db": len(all_t), "n_tables_shown": len(shown), "full_schema_shown": bool(ctx.full_schema), "n_cols_db": all_cols,
            "n_cols_shown": shown_cols, "tables_missing_for": miss, "ddl_chars": len(ctx.ddl)}


# ======================================================================================================== audit
ABSENT = re.compile(r"(not (?:defined|exist|found|present|available|part|included|provided|in the (?:provided )?schema)|does not exist|doesn't exist|undefined|"
                    r"no such|is missing|are missing|not (?:shown|listed))", re.I)
QUOTED = re.compile(r"[`'\"]([A-Za-z_][\w ]{0,60}?)[`'\"]|`?\b([A-Za-z_]\w*\.[A-Za-z_]\w*)\b`?")


def absent_claims(text):
    """Identifiers named in a sentence that claims something is undefined / missing / not in the schema."""
    out = []
    for sent in re.split(r"(?<=[.;!?])\s+|\n", text or ""):
        if ABSENT.search(sent):
            for m in QUOTED.finditer(sent):
                tok = (m.group(1) or m.group(2) or "").strip()
                if tok:
                    out.append((tok, sent.strip()[:240]))
    return out


def cmd_audit(a):
    from concurrent.futures import ThreadPoolExecutor
    from nl2sql import data
    from nl2sql.systems.ours import m1 as M1
    from nl2sql.systems.ours import m3 as M3
    from nl2sql.systems.ours import pipeline as PL
    a.variant = "V0"
    cfg, model, llm = setup(a, "V0")
    pipe = PL.Pipeline(llm, model, cfg)
    pool = {i.qid: i for i in D.pool_items()}
    body = json.load(open(a.items))
    items = body["items"][: a.limit] if a.limit else body["items"]
    v0 = {}
    if a.v0:
        for line in gzip.open(a.v0, "rt"):
            r = json.loads(line)
            v0[r["id"]] = r
    out_path = os.path.join(a.out, "audit.jsonl")
    lock = threading.Lock()
    rows = []

    def one(x):
        it = pool[x["qid"]]
        graph, _ = M1.intent(llm, model, it, 0, cfg)
        pipe.res.ensure_terms(it)
        corpus = pipe.res.corpus_for(it)
        ctx = M3.retrieve(pipe.res, it, graph, cfg, corpus, exclude_ids=())
        schema = data.get_schema(it)
        sqls = {"gold": x["gold_sql"], "pert": x["pert_sql"]}
        info = ctx_info(it, ctx, schema, sqls)
        names_shown = set(t.lower() for t in ctx.tables) | set(c["name"].lower() for t in ctx.tables if t in schema["tables"] for c in schema["tables"][t])
        names_db = set(t.lower() for t in schema["tables"]) | set(c["name"].lower() for t in schema["tables"].values() for c in t)
        claims = []
        rec0 = v0.get(x["id"])
        if rec0 and "gold" in rec0["variants"] and rec0["variants"]["gold"]["triggered"]:
            for iss in rec0["variants"]["gold"]["critic"]["issues"]:
                d = iss.get("description", "") if isinstance(iss, dict) else str(iss)
                for tok, sent in absent_claims(d):
                    parts = [p.lower() for p in re.split(r"[.\s]+", tok) if p]
                    where = "shown" if all(p in names_shown for p in parts) else ("hidden_by_pruning" if all(p in names_db for p in parts) else "not_in_db")
                    claims.append({"token": tok, "where": where, "sentence": sent})
        row = dict(info, id=x["id"], cat=x["cat"], control=bool(x["control"]), db_id=x["db_id"], v0_gold_triggered=bool(rec0 and "gold" in rec0["variants"] and rec0["variants"]["gold"]["triggered"]),
                   v0_gold_semantic=(rec0["variants"]["gold"]["critic"]["semantic"] if rec0 and "gold" in rec0["variants"] else None), absent_claims=claims)
        with lock:
            rows.append(row)
        return row

    with ThreadPoolExecutor(max_workers=a.conc) as ex:
        list(ex.map(one, items))
    rows.sort(key=lambda r: r["id"])
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(audit_summary(rows), indent=1, ensure_ascii=False))
    json.dump(audit_summary(rows), open(os.path.join(a.out, "audit_summary.json"), "w"), indent=1, ensure_ascii=False)


def audit_summary(rows):
    n = len(rows)
    pr = [r for r in rows if not r["full_schema_shown"]]
    s = {"_meta": {"scope": SCOPE, "block": "what the V0 critic sees as schema (SAP 18.6)", "n_items": n},
         "items_full_schema_shown": n - len(pr), "items_pruned_schema": len(pr), "share_pruned": len(pr) / max(1, n),
         "mean_tables_shown": sum(r["n_tables_shown"] for r in rows) / max(1, n), "mean_tables_db": sum(r["n_tables_db"] for r in rows) / max(1, n)}
    ctl = [r for r in rows if r["control"]]
    s["gold_sql_uses_table_not_shown"] = {"items": sum(1 for r in rows if r["tables_missing_for"]["gold"]), "n": n,
                                          "controls": sum(1 for r in ctl if r["tables_missing_for"]["gold"]), "n_controls": len(ctl)}
    fp = [r for r in ctl if r["v0_gold_triggered"]]
    ok = [r for r in ctl if not r["v0_gold_triggered"]]
    s["v0_gold_false_positives"] = {"n": len(fp), "gold_uses_hidden_table": sum(1 for r in fp if r["tables_missing_for"]["gold"]),
                                    "pruned_schema": sum(1 for r in fp if not r["full_schema_shown"])}
    s["v0_gold_accepted"] = {"n": len(ok), "gold_uses_hidden_table": sum(1 for r in ok if r["tables_missing_for"]["gold"]),
                             "pruned_schema": sum(1 for r in ok if not r["full_schema_shown"])}
    cnt = collections.Counter()
    ex = []
    for r in fp:
        for c in r["absent_claims"]:
            cnt[c["where"]] += 1
            ex.append({"id": r["id"], "where": c["where"], "token": c["token"], "sentence": c["sentence"]})
    s["v0_fp_absence_claims_by_where"] = dict(cnt)
    s["v0_fp_absence_claims"] = ex
    for name, sub in (("fp_gold_uses_hidden_table", [r for r in fp if r["tables_missing_for"]["gold"]]),):
        s[name + "_ids"] = [r["id"] for r in sub]
    return s


# ======================================================================================================== run
def cmd_run(a):
    from concurrent.futures import ThreadPoolExecutor
    from nl2sql import db, paths
    from nl2sql.systems.ours import m1 as M1
    from nl2sql.systems.ours import m3 as M3
    from nl2sql.systems.ours import pipeline as PL
    from nl2sql.rl import features as F
    var = a.variant
    cfg, model, llm = setup(a, var)
    orig_chat = llm.chat

    def counted(*args, **kw):
        c = getattr(TL, "counts", None)
        if c is not None:
            c[kw.get("tag", "?")] += 1
        return orig_chat(*args, **kw)
    llm.chat = counted
    body = json.load(open(a.items))
    items = body["items"][: a.limit] if a.limit else body["items"]
    pool = {i.qid: i for i in D.pool_items()}
    pr0 = prompts_state()
    models_sha0 = hashlib.sha256(open(str(paths.MODELS_YAML), "rb").read()).hexdigest()
    cfg_v = copy.deepcopy(cfg)
    if var != "V0":
        cfg_v["m5"]["theta_fb"], cfg_v["m5"]["theta_acc"] = THETA_FB_V, THETA_ACC_V
    pipe = PL.Pipeline(llm, model, cfg_v)
    theta_acc, theta_fb, t_max = cfg_v["m5"]["theta_acc"], cfg_v["m5"]["theta_fb"], cfg_v["m5"]["t_max"]
    mem = None
    vcrit = None
    if var != "V0":
        if var == "VB":
            verdicts = load_verdicts(a.burnin_log or os.path.join(str(paths.REPO), "exp", "frozen", "corpus_burnin.sqlite.log.jsonl"))
            corpus0 = pipe.res.corpus_for(pool[items[0]["qid"]])
            mem = Memory(corpus0, verdicts, forbidden_qids=set(x["qid"] for x in items) | set(pool_qids_of(a.also_forbid)))
        vcrit = VariantCritic(pipe.critic, var, a.prompt_dir or None, memory={})
        pipe.critic = vcrit
    rec_path = os.path.join(a.out, "records.jsonl")
    done = set()
    if os.path.exists(rec_path):
        for line in open(rec_path):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    lock = threading.Lock()
    fout = open(rec_path, "a")
    seed = 0

    def one(x):
        if x["id"] in done:
            return None
        it = pool[x["qid"]]
        assert it.gold_sql == x["gold_sql"], x["id"]
        t0 = time.time()
        gres = db.execute(it.db_path, it.gold_sql, timeout_s=30, max_rows=None)
        gold_rows = gres["rows"]
        graph, m1_ok = M1.intent(llm, model, it, seed, cfg) if cfg["m1"]["enabled"] else (None, None)
        pipe.res.ensure_terms(it)
        corpus = pipe.res.corpus_for(it)
        ctx = M3.retrieve(pipe.res, it, graph, cfg, corpus, exclude_ids=())
        store = pipe.res.db_store(it)
        qf0 = F.question_features(it, graph, len(ctx.tables), sum(len(store.schema["tables"][t]) for t in ctx.tables if t in store.schema["tables"]))
        rec = {"id": x["id"], "cat": x["cat"], "qid": x["qid"], "db_id": x["db_id"], "variant": var, "m1_ok": m1_ok, "variants": {}, "calls": {}}
        if var == "VB":
            cs = mem.cases(pipe.res, cfg["models"]["rerank"], it)
            vcrit.memory[it.qid] = Memory.render(cs)
            rec["memory"] = [{"id": c["id"], "verdict": c["verdict"]} for c in cs]
        from nl2sql import data as _data
        rec["ctx"] = ctx_info(it, ctx, _data.get_schema(it), {"gold": x["gold_sql"], "pert": x["pert_sql"]})
        todo = [("pert", x["pert_sql"])] + ([("gold", x["gold_sql"])] if x["control"] else [])
        for kind, sql in todo:
            TL.counts = collections.Counter()
            node = pipe._evaluate(it, sql, ctx, graph, seed, 0, qf=qf0, action="root")
            node.update({"usage": {}, "action": "root", "conf": None, "n_rr": 0, "k": 0, "calls_used": 0, "m1_ok": m1_ok, "qf": qf0})
            crit = node["critic"]
            v = {"sql": sql, "root_res": D._score_sql(it, sql, gold_rows)}
            if var in ("VC", "VD"):
                raw_brief = brief(crit)
                loop, rst = verified_loop(pipe, vcrit, cfg_v, it, node, gold_rows, theta_acc, t_max, seed, var == "VD", a.critic_only, a.prompt_dir or None)
                crit = node["critic"]
                v.update({"critic": brief(crit), "critic_raw": raw_brief, "triggered": bool(rst.get("after_final", False)) if rst else False,
                          "must_fix": bool(crit["score"] < theta_fb), "stage_root": rst})
                v.update(loop)
            else:
                v.update({"critic": brief(crit), "triggered": bool(crit["score"] < theta_acc), "must_fix": bool(crit["score"] < theta_fb)})
                if not a.critic_only:
                    v.update(D.eval_variant(pipe, cfg_v, it, node, gold_rows, theta_acc, t_max, seed))
            rec["variants"][kind] = v
            rec["calls"][kind] = {k: int(n) for k, n in TL.counts.items() if k in ("crit", "rev", "rev_t2", "ver", "app")}
        rec["secs"] = round(time.time() - t0, 1)
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
        return rec["id"]

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=a.conc) as ex:
        for r in ex.map(one, items):
            if r:
                n_done += 1
                if n_done % 10 == 0:
                    print("[v2:%s] %d/%d %.0fs stats=%s" % (var, n_done, len(items) - len(done), time.time() - t0, json.dumps(llm.stats)), flush=True)
    models_sha1 = hashlib.sha256(open(str(paths.MODELS_YAML), "rb").read()).hexdigest()
    meta = {"variant": var, "items": os.path.abspath(a.items), "n_done_now": n_done, "critic_only": bool(a.critic_only), "prompts_sha256_before": pr0,
            "prompts_sha256_after": prompts_state(), "models_yaml_sha256_before": models_sha0, "models_yaml_sha256_after": models_sha1,
            "variant_prompt_sha256": ({PROMPT_FILES[k]: hashlib.sha256(open(os.path.join(a.prompt_dir or PROMPT_DIR, PROMPT_FILES[k]), "rb").read()).hexdigest()
                                    for k in ({"VC": ["VA", "VC"], "VD": ["VA", "VC", "VD"]}.get(var, [var]) if var != "V0" else [])}),
            "theta_acc": theta_acc, "theta_fb": theta_fb, "stats": llm.stats, "secs": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(a.out, "run_meta.json"), "a" if False else "w"), indent=1)
    print("[v2:%s] done %d records in %.0fs stats=%s" % (var, n_done, time.time() - t0, json.dumps(llm.stats)), flush=True)
    assert meta["prompts_sha256_before"] == meta["prompts_sha256_after"] and models_sha0 == models_sha1


def pool_qids_of(path):
    if not path:
        return []
    return [x["qid"] for x in json.load(open(path))["items"]]


def brief(c):
    return {k: c.get(k) for k in ("syntax", "semantic", "semantic_raw", "efficiency", "efficiency_raw", "score", "status", "parse_ok", "issues", "style_issues", "suggestions")}


# ======================================================================================================== analyze
def load_runs(path):
    recs = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            recs[r["id"]] = r
    return recs


def load_v0(path):
    recs = {}
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        recs[r["id"]] = r
    return recs


def _ck(cat):
    return "ex_bag" if cat == "distinct_drop" else "ex_set"


def item_table(recs):
    """-> dict id -> flat per-item numbers for one run (pert and, if present, gold)."""
    t = {}
    for i, r in recs.items():
        p = r["variants"]["pert"]
        g = r["variants"].get("gold")
        ck = _ck(r["cat"])
        d = {"db": r["db_id"], "cat": r["cat"], "p_trig": int(p["triggered"]), "p_score": p["critic"]["score"], "p_sem": p["critic"].get("semantic"),
             "p_r1": int(bool(p.get("rounds")) and bool(p["rounds"][0]["score_res"][ck])) if "rounds" in p else None,
             "p_final": int(p["final_res"][ck]) if "final_res" in p else None}
        if g is not None:
            d.update({"g_trig": int(g["triggered"]), "g_score": g["critic"]["score"], "g_sem": g["critic"].get("semantic"),
                      "g_harm": int(g["final_res"]["ex_bag"] == 0) if "final_res" in g else None})
        t[i] = d
    return t


def auroc_from(pos, neg):
    """P(pos < neg) + 0.5 P(=), tie-aware, lower score = injected."""
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    s = np.sort(neg)
    lo = np.searchsorted(s, pos, side="left")           # neg < pos
    hi = np.searchsorted(s, pos, side="right")          # neg <= pos
    gt = len(neg) - hi                                   # neg > pos  -> pos < neg
    eq = hi - lo
    return float((gt + 0.5 * eq).sum() / (len(pos) * len(neg)))


def metrics_of(tab, ids, score_key="p_score", g_score_key="g_score"):
    ids = list(ids)
    P = [tab[i] for i in ids]
    G = [tab[i] for i in ids if "g_trig" in tab[i]]
    m = {"detection": sum(d["p_trig"] for d in P) / max(1, len(P)), "fpr": (sum(d["g_trig"] for d in G) / len(G)) if G else None}
    if G and G[0].get("g_harm") is not None:
        m["harm"] = sum(d["g_harm"] for d in G) / len(G)
    if P and P[0].get("p_r1") is not None:
        m["repair_round1_all"] = sum(d["p_r1"] for d in P) / len(P)
        m["repair_final_all"] = sum(d["p_final"] for d in P) / len(P)
    if G:
        m["auroc"] = auroc_from([d[score_key] for d in P], [d[g_score_key] for d in G])
    return m


def boot_deltas(tabs, ids, B, seed, keys):
    """tabs: name -> (table, score_key, g_score_key).  Cluster bootstrap over databases (one shared resample); returns for each name the CI of the metric and of
    (name - first name) for the metrics in `keys`."""
    import numpy as np
    dbs = sorted(set(tabs[next(iter(tabs))][0][i]["db"] for i in ids))
    by_db = collections.defaultdict(list)
    for i in ids:
        by_db[tabs[next(iter(tabs))][0][i]["db"]].append(i)
    rng = np.random.RandomState(seed)
    names = list(tabs)
    samples = {n: collections.defaultdict(list) for n in names}
    for _ in range(B):
        pick = rng.randint(0, len(dbs), size=len(dbs))
        sel = [i for j in pick for i in by_db[dbs[j]]]
        for n in names:
            t, sk, gk = tabs[n]
            m = metrics_of(t, sel, sk, gk)
            for k in keys:
                if m.get(k) is not None:
                    samples[n][k].append(m[k])
    out = {}
    base = names[0]
    for n in names:
        out[n] = {}
        for k in keys:
            v = np.asarray(samples[n][k], float)
            if len(v):
                out[n][k] = {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
            if n != base and len(samples[n][k]) == len(samples[base][k]) and len(samples[n][k]):
                dl = np.asarray(samples[n][k], float) - np.asarray(samples[base][k], float)
                out[n][k + "_delta"] = {"lo": float(np.percentile(dl, 2.5)), "hi": float(np.percentile(dl, 97.5))}
    return out


def cmd_analyze(a):
    body = json.load(open(a.items))
    by_id = {x["id"]: x for x in body["items"]}
    v0 = load_v0(a.v0)
    runs = {}
    for kv in a.runs:
        n, p = kv.split("=", 1)
        runs[n] = load_runs(os.path.join(p, "records.jsonl"))
    ids = sorted(set(v0).intersection(*[set(r) for r in runs.values()]))
    t0 = item_table(v0)
    tabs = collections.OrderedDict()
    tabs["V0"] = (t0, "p_score", "g_score")
    tabs["V0_semantic_only"] = (t0, "p_sem", "g_sem")
    for n, r in runs.items():
        tabs[n] = (item_table(r), "p_score", "g_score")
    keys = ["detection", "fpr", "harm", "repair_round1_all", "repair_final_all", "auroc"]
    cats = [c for c in D.CATEGORIES if any(by_id[i]["cat"] == c for i in ids)]
    res = {"_meta": {"scope": SCOPE, "label": "exploratory-posthoc, designed after the V0 results (SAP Amendment 6, section 18)",
                     "n_items": len(ids), "n_injected": len(ids), "n_gold_controls": sum(1 for i in ids if by_id[i]["control"]),
                     "ci": "95%% percentile, database-clustered bootstrap, B=%d, seed %d, one shared resample of databases (paired deltas)" % (a.B, SEED_BOOT),
                     "reading": "controlled diagnostic with injected errors; VB feedback is SIMULATED ORACLE feedback; not evidence of EX on BIRD; never enters results_macros.tex",
                     "operating_point": "VA/VB trigger = semantic < %.1f (V0: score < 8.5); must_fix level %.1f" % (THETA_ACC_V, THETA_FB_V),
                     "decision_rule": "SAP 18.7"},
           "variants": {}, "per_category": {}, "decision": {}}
    # V0_semantic_only shares V0's trigger, so only the AUROC differs
    boots = boot_deltas({k: v for k, v in tabs.items() if k != "V0_semantic_only"}, ids, a.B, SEED_BOOT, keys)
    for n in tabs:
        m = metrics_of(tabs[n][0], ids, tabs[n][1], tabs[n][2])
        res["variants"][n] = {"point": m, "ci": boots.get(n, {})}
        if n == "V0_semantic_only":
            res["variants"][n] = {"point": {"auroc": m["auroc"]}, "note": "V0 semantic sub-score alone, same items"}
        for c in cats:
            cid = [i for i in ids if by_id[i]["cat"] == c]
            res["per_category"].setdefault(c, {})[n] = metrics_of(tabs[n][0], cid, tabs[n][1], tabs[n][2])
    for n in runs:
        pt, ci = res["variants"][n]["point"], res["variants"][n]["ci"]
        v0p = res["variants"]["V0"]["point"]
        dF = pt["fpr"] - v0p["fpr"]
        dD = pt["detection"] - v0p["detection"]
        cF, cD = ci.get("fpr_delta"), ci.get("detection_delta")
        rule = bool(dF <= -0.08 and cF and cF["hi"] < 0 and dD >= -0.03 and cD and cD["lo"] > -0.06)
        strict = bool(dF <= -0.08 and cF and cF["hi"] <= -0.08 and dD >= -0.03 and cD and cD["lo"] >= -0.03)
        res["decision"][n] = {"delta_fpr": dF, "delta_fpr_ci": cF, "delta_detection": dD, "delta_detection_ci": cD, "improves_rule_18_7": rule,
                              "improves_strict_reading": strict}
    res["calls"] = calls_report(v0, runs, ids)
    res["stages"] = stages_report(runs, ids, by_id)
    res["appeal"] = appeal_report(runs, ids, by_id)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "critic_v2_summary.json"), "w"), indent=1, ensure_ascii=False)
    import csv
    with open(os.path.join(a.out, "critic_v2_items.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "cat", "qid", "db_id", "kind", "variant", "score", "semantic", "triggered", "final_match"])
        for n, (t, sk, gk) in tabs.items():
            if n == "V0_semantic_only":
                continue
            src = v0 if n == "V0" else runs[n]
            for i in ids:
                r = src[i]
                for kind, v in r["variants"].items():
                    ck = _ck(r["cat"])
                    w.writerow([i, r["cat"], r["qid"], r["db_id"], kind, n, v["critic"]["score"], v["critic"].get("semantic"), int(v["triggered"]),
                                (v["final_res"][ck] if "final_res" in v else "")])
    cases = build_cases(ids, by_id, v0, runs)
    json.dump(cases, open(os.path.join(a.out, "critic_v2_cases.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"variants": {n: res["variants"][n]["point"] for n in res["variants"]}, "decision": res["decision"]}, indent=1))


def calls_report(v0, runs, ids):
    """LLM calls on the critic-revision path (critic + reviser + verifier + appeal; M1 / retrieval excluded) per evaluated SQL.  V0 is derived from the stored rounds
    (1 critic + 2 per revision round); the new runs are counted by the client wrapper (cache hits included)."""
    out = {}

    def per(src, derive):
        tot, n, fl_tot, fl_n, by = 0, 0, 0, 0, collections.Counter()
        for i in ids:
            r = src[i]
            for kind, v in r["variants"].items():
                if derive:
                    c = {"crit": 1 + len(v.get("rounds", [])), "rev": len(v.get("rounds", []))}
                else:
                    c = r.get("calls", {}).get(kind)
                    if c is None:
                        continue
                k = sum(c.values())
                tot += k
                n += 1
                for t, m in c.items():
                    by[t] += m
                st = v.get("stage_root")
                flagged = bool(st["critic_flagged"]) if st else bool(v["triggered"])
                if flagged:
                    fl_tot += k
                    fl_n += 1
        return {"n_queries": n, "calls_per_query": tot / max(1, n), "n_flagged_roots": fl_n, "calls_per_flagged_root": fl_tot / max(1, fl_n),
                "calls_by_tag_total": dict(by)}
    out["V0"] = per(v0, True)
    for n, r in runs.items():
        out[n] = per(r, False)
    base = out.get("VA")
    if base:
        for n in out:
            out[n]["extra_calls_per_query_vs_VA"] = out[n]["calls_per_query"] - base["calls_per_query"]
    for n in out:
        out[n]["extra_calls_per_query_vs_V0"] = out[n]["calls_per_query"] - out["V0"]["calls_per_query"]
    return out


def stages_report(runs, ids, by_id):
    out = {}
    for n in ("VC", "VD"):
        if n not in runs:
            continue
        d = {}
        for kind in ("pert", "gold"):
            rows = [(i, runs[n][i]["variants"][kind]["stage_root"]) for i in ids if kind in runs[n][i]["variants"] and runs[n][i]["variants"][kind].get("stage_root")]
            c = collections.Counter()
            rem = collections.defaultdict(list)
            for i, st in rows:
                c["n"] += 1
                c["critic_flagged"] += int(st["critic_flagged"])
                c["after_ab"] += int(bool(st.get("after_ab")))
                c["after_c"] += int(bool(st.get("after_c")))
                c["after_final"] += int(bool(st.get("after_final")))
                if st.get("removed_by"):
                    c["removed_by_" + st["removed_by"]] += 1
                    rem[st["removed_by"]].append(i)
            d[kind] = dict(c)
            d[kind]["removed_ids"] = dict(rem)
            n_ = max(1, c["n"])
            d[kind]["root_flag_rates"] = {k: c[k] / n_ for k in ("critic_flagged", "after_ab", "after_c", "after_final")}   # after_c = deterministic stage 1 only (no verifier call), root node
        out[n] = d
    return out


def appeal_report(runs, ids, by_id):
    """VD: appealed issues (all nodes of the loop), withdrawn / maintained, split by node kind; sycophancy proxy = the claim is CORRECT if the node's probe fix reproduces gold
    under the category comparator (oracle, offline).  Plus the root-level effect of the appeal on the flag."""
    if "VD" not in runs:
        return None
    out = {"definition": "appealed issue = issue the verifier marked unsupported; correct claim (injected nodes) = the probe revision of that node equals gold under the category comparator"}
    for kind in ("pert", "gold"):
        c = collections.Counter()
        root = collections.Counter()
        for i in ids:
            v = runs["VD"][i]["variants"].get(kind)
            if not v:
                continue
            ck = _ck(by_id[i]["cat"])
            for st in v.get("stage_logs", []):
                if "appeal" not in st:
                    continue
                correct = bool(st["probe_score_res"].get(ck))
                for a in st["appeal"]:
                    key = "correct_claim" if correct else "other_claim"
                    c["appealed"] += 1
                    c["appealed_" + key] += 1
                    c[a["decision"] + "_" + key] += 1
                    c[a["decision"]] += 1
            st = v.get("stage_root")
            if st and st.get("appeal"):
                root["roots_with_appeal"] += 1
                root["flag_added_by_appeal"] += int((not st.get("flagged_by_verifier_alone")) and bool(st.get("after_final")))
                root["flag_kept_without_appeal"] += int(bool(st.get("flagged_by_verifier_alone")))
        out[kind] = {"issues": dict(c), "roots": dict(root)}
    return out


def build_cases(ids, by_id, v0, runs):
    """Six illustrative cases by a FIXED rule (lowest id per class, different categories where possible; not by effect size):
    FP_fixed_by_VA x2, FP_fixed_by_VB x2 (V0 flagged the gold control, the variant did not), FP_still_failing x1 (V0, VA and VB all flagged it),
    detection_lost x1 (V0 flagged the injected query; VB did not, else VA)."""
    def g(r, n):
        return r["variants"]["gold"] if "gold" in r["variants"] else None

    def pick(pred, n, distinct_cat=True, avoid=()):
        out, seen = [], set()
        for i in ids:
            if len(out) >= n:
                break
            if i in avoid:
                continue
            cat = by_id[i]["cat"]
            if distinct_cat and cat in seen:
                continue
            if pred(i):
                out.append(i)
                seen.add(cat)
        return out

    has = lambda n: n in runs
    sel = []
    if has("VA"):
        sel += [("FP_fixed_by_VA", i) for i in pick(lambda i: by_id[i]["control"] and g(v0[i], "V0")["triggered"] and not g(runs["VA"][i], "VA")["triggered"], 2)]
    if has("VB"):
        used = set(i for c, i in sel)
        sel += [("FP_fixed_by_VB", i) for i in pick(lambda i: by_id[i]["control"] and g(v0[i], "V0")["triggered"] and not g(runs["VB"][i], "VB")["triggered"], 2, avoid=used)]
    if has("VA") and has("VB"):
        sel += [("FP_still_failing_all", i) for i in pick(lambda i: by_id[i]["control"] and g(v0[i], "V0")["triggered"] and g(runs["VA"][i], "VA")["triggered"] and g(runs["VB"][i], "VB")["triggered"], 1)]
        lost = pick(lambda i: v0[i]["variants"]["pert"]["triggered"] and not runs["VB"][i]["variants"]["pert"]["triggered"], 1)
        who = "VB"
        if not lost:
            lost, who = pick(lambda i: v0[i]["variants"]["pert"]["triggered"] and not runs["VA"][i]["variants"]["pert"]["triggered"], 1), "VA"
        sel += [("detection_lost_" + who, i) for i in lost]
    out = {"_meta": {"scope": SCOPE, "selection": "fixed rule: lowest id of the class, different categories where possible; illustrative, not representative; VB memory verdicts are SIMULATED ORACLE feedback"},
           "cases": []}
    for cls, i in sel:
        x = by_id[i]
        kind = "gold" if cls.startswith("FP") else "pert"
        c = {"class": cls, "id": i, "category": x["cat"], "source_qid": x["qid"], "db_id": x["db_id"], "question": x["question"], "evidence": x["evidence"],
             "evaluated_kind": kind, "gold_sql": x["gold_sql"], "injected_sql": x["pert_sql"], "edit": x["edit"], "variants": {}}
        for n, src in [("V0", v0)] + list(runs.items()):
            v = src[i]["variants"].get(kind)
            if v is None:
                continue
            ck = _ck(x["cat"])
            e = {"score": v["critic"]["score"], "semantic": v["critic"].get("semantic"), "triggered": v["triggered"],
                 "issues": [(q.get("description") if isinstance(q, dict) else q) for q in v["critic"].get("issues") or []],
                 "style_issues": [(q.get("description") if isinstance(q, dict) else q) for q in v["critic"].get("style_issues") or []],
                 "final_matches_gold": (v["final_res"][ck] if "final_res" in v else None), "final_changed": (v["final_sql"].strip() != v["sql"].strip()) if "final_sql" in v else None}
            if n == "VB":
                e["memory_verdicts"] = [m["verdict"] for m in src[i].get("memory", [])]
            c["variants"][n] = e
        out["cases"].append(c)
    return out


# ======================================================================================================== main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    r.add_argument("--items", required=True)
    r.add_argument("--variant", required=True, choices=["V0", "VA", "VB", "VC", "VD"])
    r.add_argument("--out", required=True)
    r.add_argument("--conc", type=int, default=4)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--critic-only", action="store_true")
    r.add_argument("--prompt-dir", default="")
    r.add_argument("--burnin-log", default="")
    r.add_argument("--also-forbid", default="", help="items.json whose qids must not be in the VB feedback store (e.g. the evaluation items)")
    u = sub.add_parser("audit")
    u.add_argument("--items", required=True)
    u.add_argument("--v0", default="")
    u.add_argument("--out", required=True)
    u.add_argument("--conc", type=int, default=4)
    u.add_argument("--limit", type=int, default=0)
    z = sub.add_parser("analyze")
    z.add_argument("--items", required=True)
    z.add_argument("--v0", required=True)
    z.add_argument("--runs", nargs="+", required=True, help="NAME=dir (dir holds records.jsonl), e.g. VA=... VB=...")
    z.add_argument("--out", required=True)
    z.add_argument("--B", type=int, default=5000)
    a = ap.parse_args()
    {"run": cmd_run, "audit": cmd_audit, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
