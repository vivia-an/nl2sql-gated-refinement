#!/usr/bin/env python3
"""User-dispute / appeal test on the DIAG items (SAP Amendment 6d, docs/SAP.md section 19).  scope = diagnostic-posthoc.

Question: when a (simulated) user disputes a critic decision, does an arbiter that is given the evidence and the user's rebuttal hold a right decision and
yield on a wrong one?  Decisions are checked in BOTH directions (a flagged-but-fine query = F unit with a wrong claim; a passed-but-wrong query = P unit).
Undisputed decisions are never appealed.  The arbiter prompts are neutral (no default); the old VD prompt (default = withdraw) is a control.

Nothing here edits code/prompts/** or exp/models.yaml.  Units come from the stored VA root nodes (no critic call).

  units       VA records + items -> units.json (oracle labels, claims, fix, real executions)       [needs the worker machine's data: db files]
  arbitrate   --arm A|A0|B|C  one LLM call per unit -> decisions_<arm>.jsonl (resumable)
  rates       all-unit firmness / responsiveness per arm (used on the tuning draw)
  analyze     user-noise expectation + seeded realisation + clustered bootstrap + decision rule 19.5 -> critic_ve_summary.json

Usage (cwd = code dir, python = worker machine venv):
  python tools/critic_ve.py units     --records R.jsonl[.gz] --items I --out U.json
  python tools/critic_ve.py arbitrate --units U.json --arm A --out DIR [--conc 4] [--prompt-dir D]
  python tools/critic_ve.py rates     --units U.json --dec A=DIR/decisions_A.jsonl ...
  python tools/critic_ve.py analyze   --units U.json --dec A=... A0=... B=... C=... --out ../results/diag
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import re
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
for _p in (CODE, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCOPE = "diagnostic-posthoc"
REPO = os.path.dirname(CODE)
PROMPT_DIR = os.path.join(REPO, "exp", "diag_prompts")
PROMPT_FILES = {"flag": "ve_arbiter_flag.user.txt", "pass": "ve_arbiter_pass.user.txt", "old": "vd_appeal.user.txt"}
SEED_BOOT = 20261101
SEED_REAL = 20261101
QS = (1.0, 0.85, 0.70)
Q_RULE = 0.85
MAX_TOKENS_ARB = 1000
MIN_EVIDENCE = 10
ARMS = ("A", "A0", "B", "C")
ARM_MODEL = {"A": "qwen37plus", "A0": "qwen37plus", "B": "dsv41flash", "C": "qwen37plus"}

USER_FLAG = ("The user disputes this review and does not want the change: \"I do not want this changed. The query already returns what I asked for.\" "
             "The user gave no further reason.")
USER_PASS = ("The user disputes this review: \"This result is not what I asked for. The query is wrong.\" The user gave no further reason.")
ROLE = {"A": "You reviewed the SQLite query below for the question below and wrote the review shown under \"The review\".",
        "A0": "You reviewed the SQLite query below for the question below and wrote the review shown under \"The review\".",
        "B": "Another reviewer reviewed the SQLite query below for the question below and wrote the review shown under \"The review\"."}
CLAUSE_USER = "The user has disputed the review (see \"The user's statement\"). The user may be right or wrong."
CLAUSE_NONE = "You are asked to look at the review once more."


def _ck(cat):
    return "ex_bag" if cat == "distinct_drop" else "ex_set"


def render_text(text, **kw):
    def rep(m):
        k = m.group(1)
        if k not in kw:
            if m.group(2):
                return ""
            raise KeyError("missing placeholder %s" % k)
        return str(kw[k])
    return re.sub(r"<<(\w+)(\?)?>>", rep, text)


def prompt_text(kind, prompt_dir=None):
    return open(os.path.join(prompt_dir or PROMPT_DIR, PROMPT_FILES[kind])).read()


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_records(path):
    recs = {}
    for line in _open(path):
        if line.strip():
            r = json.loads(line)
            recs[r["id"]] = r
    return recs


# ======================================================================================================== units
def _clean_desc(d):
    return str(d or "").split(" (question:")[0].strip()


def build_units(recs, items_by_id):
    """Units of SAP 19.1 from stored VA records.  Pure: no execution, no model.  Each unit carries the oracle labels (never shown to a model)."""
    units = []
    for rid in sorted(recs):
        r = recs[rid]
        x = items_by_id[rid]
        ck = _ck(r["cat"])
        for side in ("pert", "gold"):
            v = r["variants"].get(side)
            if v is None:
                continue
            inj = side == "pert"
            flagged = bool(v["triggered"])
            u = {"uid": "%s:%s" % (rid, side), "id": rid, "side": side, "db": r["db_id"], "cat": r["cat"], "qid": r["qid"], "inj": inj,
                 "kind": "F" if flagged else "P", "sql": v["sql"], "question": x["question"], "evidence": x.get("evidence") or "",
                 "semantic": v["critic"]["score"], "semantic_raw": v["critic"].get("semantic_raw")}
            if flagged:
                assert v.get("rounds"), "flagged root without a stored revision: %s" % u["uid"]
                rd = v["rounds"][0]
                u["claims"] = [{"question_phrase": i.get("question_phrase", ""), "sql_fragment": i.get("sql_fragment", ""),
                                "description": _clean_desc(i.get("description"))} for i in (v["critic"].get("issues") or [])]
                u["fix_sql"] = rd["sql"]
                u["fix_noop"] = bool(rd.get("noop"))
                u["fix_matches_gold"] = bool(rd["score_res"][ck])
                u["fix_matches_gold_bag"] = bool(rd["score_res"]["ex_bag"])
                u["critic_right"] = bool(rd["score_res"][ck]) if inj else False
            else:
                u["remarks"] = [_clean_desc((s or {}).get("description") if isinstance(s, dict) else s)[:220] for s in (v["critic"].get("style_issues") or [])][:4]
                sql_wrong = bool(inj and v["root_res"][ck] == 0)
                if inj:
                    assert sql_wrong, "injected root equals gold under the comparator: %s" % u["uid"]
                u["sql_wrong"] = sql_wrong
                u["critic_right"] = not sql_wrong
            units.append(u)
    return units


def attach_graph(units, pool):
    """The M1 task graph that the critic saw (same cached call as the DIAG / VA runs; an OFFLINE client, so a cache miss raises and nothing is charged).  A miss
    leaves the graph empty and is counted."""
    from nl2sql import config as C
    from nl2sql import paths
    from nl2sql.llm import LLMClient
    from nl2sql.systems.ours import m1 as M1
    from nl2sql.systems.ours import pipeline as PL
    cfg = PL.resolved_cfg(C.load_config(os.path.join(str(paths.REPO), "exp", "configs", "frozen", "full_fixed.yaml")))
    model = C.resolve_model("qwen37plus")
    llm = LLMClient(cache_path=str(paths.CACHE / "llm"), usage_path=os.devnull, key_file=paths.KEY_FILE, max_concurrency_per_model=1, offline=True)
    use = bool(cfg["m4"]["graph_in_prompt"] and cfg["m1"]["enabled"])
    cache, miss = {}, 0
    for u in units:
        if u["qid"] not in cache:
            txt = ""
            if use:
                try:
                    g, _ = M1.intent(llm, model, pool[u["qid"]], 0, cfg)
                    txt = M1.graph_text(g)
                except Exception:
                    miss += 1
            cache[u["qid"]] = txt
        u["graph"] = cache[u["qid"]]
    return {"graph_in_prompt": use, "qids": len(cache), "cache_misses": miss, "empty": sum(1 for v in cache.values() if not v)}


def attach_exec(units, pool):
    """Real executions of the original SQL and the fix (offline; worker machine data).  Adds preview / fix_preview / comparison strings."""
    from nl2sql import db
    from nl2sql.systems.ours import m5 as M5
    import critic_v2 as V
    for u in units:
        it = pool[u["qid"]]
        res = db.execute(it.db_path, u["sql"], timeout_s=30, max_rows=10000)
        u["preview"] = M5.exec_summary(res)
        if u["kind"] == "F":
            fres = db.execute(it.db_path, u["fix_sql"], timeout_s=30, max_rows=10000)
            same = V.results_equal(u["sql"], res, u["fix_sql"], fres)
            u["fix_preview"] = M5.exec_summary(fres)
            u["comparison"] = V.comparison_text(res, fres, same)
            u["fix_same_result"] = bool(same)


def claims_text(u):
    if not u["claims"]:
        return "(the review gave a low score and stated no must_fix claim)"
    return "\n".join("%d. Question phrase: \"%s\"; SQL fragment: \"%s\"; claim: %s" % (n, c["question_phrase"], c["sql_fragment"], c["description"])
                     for n, c in enumerate(u["claims"], 1))


def review_block(u):
    if u["kind"] == "F":
        return "Semantic score you gave: %s (0-10; below 7 means at least one must_fix issue).\nMust_fix claims:\n%s" % (u["semantic"], claims_text(u))
    rem = u.get("remarks") or []
    return ("You found no must_fix issue. Semantic score you gave: %s (0-10; 7 or more means no must_fix issue).\nOther remarks (style or doubts, not must_fix): %s"
            % (u["semantic"], "; ".join(rem) if rem else "(none)"))


def user_block(u, with_user):
    if not with_user:
        return ""
    return "\n# The user's statement\n%s\n\n" % (USER_FLAG if u["kind"] == "F" else USER_PASS)


def render_arbiter(u, arm, full_schema, prompt_dir=None):
    """The prompt of arms A / A0 / B (neutral)."""
    with_user = arm != "A0"
    kw = dict(full_schema=full_schema, role_intro=ROLE[arm], user_clause=CLAUSE_USER if with_user else CLAUSE_NONE, question=u["question"],
              evidence=u["evidence"] or "(none)", graph=u.get("graph") or "", sql=u["sql"], preview=u["preview"],
              review_block=review_block(u), user_block=user_block(u, with_user))
    if u["kind"] == "F":
        kw.update(probe_sql=u["fix_sql"], probe_preview=u["fix_preview"], comparison=u["comparison"])
    return render_text(prompt_text("flag" if u["kind"] == "F" else "pass", prompt_dir), **kw)


def render_old(u, full_schema, prompt_dir=None):
    """Arm C: the old VD appeal prompt, unchanged; the 'verifier's response' slot carries the user's rejection."""
    assert u["kind"] == "F"
    cl = u["claims"] or [{"question_phrase": "", "sql_fragment": "", "description": "(no must_fix claim stated)"}]
    return render_text(prompt_text("old", prompt_dir), full_schema=full_schema, question=u["question"], evidence=u["evidence"] or "(none)", sql=u["sql"],
                       preview=u["preview"], question_phrase=" | ".join(c["question_phrase"] for c in cl), sql_fragment=" | ".join(c["sql_fragment"] for c in cl),
                       description=" || ".join(c["description"] for c in cl), verdict="rejected by the user", rebuttal=USER_FLAG,
                       probe_sql=u["fix_sql"], probe_preview=u["fix_preview"], comparison=u["comparison"])


def parse_arbiter(j, arm):
    """-> (hold, decision, evidence, valid).  hold = the original decision stands.  Arms A / A0 / B: 'maintain' holds; 'change' needs >= MIN_EVIDENCE characters
    of evidence, else it is invalid and the original decision stands.  Arm C (old VD rule): 'maintain' needs a justification of >= MIN_EVIDENCE characters,
    otherwise it counts as withdraw."""
    j = j if isinstance(j, dict) else {}
    if arm == "C":
        dec = str(j.get("decision") or "").strip().lower()
        just = str(j.get("justification") or "").strip()
        if dec not in ("maintain", "withdraw"):
            return False, dec or None, just, False                # unparsable counts as withdraw (the VD rule)
        return bool(dec == "maintain" and len(just) >= MIN_EVIDENCE), dec, just, True
    dec = str(j.get("decision") or "").strip().lower()
    ev = str(j.get("evidence") or "").strip()
    if dec == "change" and len(ev) >= MIN_EVIDENCE:
        return False, dec, ev, True
    if dec in ("maintain", "change"):
        return True, dec, ev, dec == "maintain"
    return True, dec or None, ev, False


# ======================================================================================================== arbitration run
def cmd_units(a):
    import critic_diag as D
    items = json.load(open(a.items))["items"]
    by_id = {x["id"]: x for x in items}
    recs = load_records(a.records)
    recs = {k: v for k, v in recs.items() if k in by_id}
    units = build_units(recs, by_id)
    pool = {i.qid: i for i in D.pool_items()}
    attach_exec(units, pool)
    gstat = attach_graph(units, pool)
    body = {"_meta": {"scope": SCOPE, "graph": gstat, "records_sha256": hashlib.sha256(open(a.records, "rb").read()).hexdigest(),
                      "items_sha256": hashlib.sha256(open(a.items, "rb").read()).hexdigest(), "n_units": len(units),
                      "counts": dict(collections.Counter("%s-%s%s" % (u["kind"], "inj" if u["inj"] else "gold", "" if u["kind"] == "P" else ("-right" if u["critic_right"] else "-wrong")) for u in units))},
            "units": units}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(body, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(body["_meta"], indent=1))


def cmd_arbitrate(a):
    from concurrent.futures import ThreadPoolExecutor
    from nl2sql import config as C
    from nl2sql import data, paths
    from nl2sql import prompts_util as P
    from nl2sql.llm import LLMClient
    from nl2sql.util import parse_json
    import critic_diag as D
    import critic_v2 as V
    arm = a.arm
    body = json.load(open(a.units))
    units = body["units"][: a.limit] if a.limit else body["units"]
    if arm == "C":
        units = [u for u in units if u["kind"] == "F"]
    model = C.resolve_model(ARM_MODEL[arm])
    os.makedirs(a.out, exist_ok=True)
    pr0 = _prompts_state()
    models_sha0 = hashlib.sha256(open(str(paths.MODELS_YAML), "rb").read()).hexdigest()
    llm = LLMClient(cache_path=str(paths.CACHE / "llm"), usage_path=os.path.join(a.out, "usage_%s.jsonl" % arm), key_file=paths.KEY_FILE,
                    max_concurrency_per_model=a.conc)
    pool = {i.qid: i for i in D.pool_items()}
    path = os.path.join(a.out, "decisions_%s.jsonl" % arm)
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    lock = threading.Lock()
    fout = open(path, "a")
    sysmsg = P.render("common.system")
    schemas = {}

    def schema_of(u):
        if u["db"] not in schemas:
            schemas[u["db"]] = V.full_schema_text(data.get_schema(pool[u["qid"]]))
        return schemas[u["db"]]

    def one(u):
        if u["uid"] in done:
            return None
        fs = schema_of(u)
        user = render_old(u, fs, a.prompt_dir or None) if arm == "C" else render_arbiter(u, arm, fs, a.prompt_dir or None)
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": user}]
        r = llm.chat(model, msgs, temperature=0.0, seed=0, max_tokens=MAX_TOKENS_ARB, tag="arb_" + arm)
        j = parse_json(r["text"])
        hold, dec, ev, valid = parse_arbiter(j, arm)
        rec = {"uid": u["uid"], "arm": arm, "model": model, "kind": u["kind"], "hold": bool(hold), "decision": dec, "evidence": ev[:700], "valid": bool(valid),
               "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": r.get("completion_tokens"), "cached": bool(r.get("cached"))}
        with lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
        return u["uid"]

    t0 = time.time()
    n = 0
    with ThreadPoolExecutor(max_workers=a.conc) as ex:
        for r in ex.map(one, units):
            if r:
                n += 1
                if n % 25 == 0:
                    print("[ve:%s] %d/%d %.0fs stats=%s" % (arm, n, len(units) - len(done), time.time() - t0, json.dumps(llm.stats)), flush=True)
    models_sha1 = hashlib.sha256(open(str(paths.MODELS_YAML), "rb").read()).hexdigest()
    meta = {"arm": arm, "model": model, "units": os.path.abspath(a.units), "n_done_now": n, "prompts_sha256_before": pr0, "prompts_sha256_after": _prompts_state(),
            "models_yaml_sha256_before": models_sha0, "models_yaml_sha256_after": models_sha1,
            "arbiter_prompt_sha256": {PROMPT_FILES[k]: hashlib.sha256(open(os.path.join(a.prompt_dir or PROMPT_DIR, PROMPT_FILES[k]), "rb").read()).hexdigest()
                                      for k in PROMPT_FILES}, "stats": llm.stats, "secs": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(a.out, "run_meta_%s.json" % arm), "w"), indent=1)
    print("[ve:%s] done %d in %.0fs stats=%s" % (arm, n, time.time() - t0, json.dumps(llm.stats)), flush=True)
    assert meta["prompts_sha256_before"] == meta["prompts_sha256_after"] and models_sha0 == models_sha1


def _prompts_state():
    from nl2sql import paths
    h = hashlib.sha256()
    for f in sorted(os.listdir(str(paths.PROMPTS_DIR))):
        h.update(f.encode())
        h.update(open(os.path.join(str(paths.PROMPTS_DIR), f), "rb").read())
    return h.hexdigest()


def load_decisions(path):
    d = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            d[r["uid"]] = r
    return d


# ======================================================================================================== analysis (pure python + numpy)
def p_disputed(critic_right, q):
    """P(the user contradicts the critic): the user's verdict is right with probability q."""
    return (1.0 - q) if critic_right else q


def policy_flags(units, policy, q, dec=None):
    """Expected final flag per unit (0..1) under `policy` and user quality q.  policy: none | obey | oracle_arbiter | oracle_all | arm (dec = uid -> hold)."""
    out = []
    for u in units:
        s = 1.0 if u["kind"] == "F" else 0.0               # original flag state
        pd = p_disputed(u["critic_right"], q)
        yielded = 1.0 - s
        if policy == "none":
            f = s
        elif policy == "obey":
            f = pd * yielded + (1 - pd) * s
        elif policy == "oracle_arbiter":
            ys = 0.0 if u["critic_right"] else 1.0           # correct action: yield iff the critic was wrong
            f = pd * (yielded if ys else s) + (1 - pd) * s
        elif policy == "oracle_all":
            f = s if u["critic_right"] else yielded
        else:
            d = dec.get(u["uid"])
            if d is None:                                    # arm not defined for this unit (C on passes): no appeal
                f = s
            else:
                f = pd * (s if d else yielded) + (1 - pd) * s
        out.append(f)
    return out


def group_masks(units):
    m = collections.OrderedDict()
    m["inj"] = [u["inj"] for u in units]
    m["gold"] = [not u["inj"] for u in units]
    m["F_inj_right"] = [u["kind"] == "F" and u["inj"] and u["critic_right"] for u in units]
    m["F_wrong"] = [u["kind"] == "F" and not u["critic_right"] for u in units]
    m["P_inj"] = [u["kind"] == "P" and u["inj"] for u in units]
    m["P_gold"] = [u["kind"] == "P" and not u["inj"] for u in units]
    m["F_gold"] = [u["kind"] == "F" and not u["inj"] for u in units]
    m["F_gold_fix_wrong"] = [u["kind"] == "F" and not u["inj"] and not u["fix_matches_gold_bag"] for u in units]
    m["F_inj"] = [u["kind"] == "F" and u["inj"] for u in units]
    return m


METRICS = ("DET", "UDET", "FPR", "J", "CCR", "WCR", "MER", "TNR", "HARM_LB")


def metric_terms(f, masks):
    """Numerators / denominators per base metric from per-unit expected flags f.  -> dict name -> (num array, den array); composite metrics are built from these."""
    import numpy as np
    f = np.asarray(f, float)
    M = {k: np.asarray(v, float) for k, v in masks.items()}
    one = np.ones_like(f)
    t = {}
    t["DET"] = (f * M["inj"], M["inj"])
    t["UDET"] = (f * (M["F_inj_right"] + M["P_inj"]), M["inj"])
    t["FPR"] = (f * M["gold"], M["gold"])
    t["CCR"] = (f * M["F_inj_right"], M["F_inj_right"])
    t["WCR"] = (M["F_wrong"] - f * M["F_wrong"], M["F_wrong"])
    t["MER"] = (f * M["P_inj"], M["P_inj"])
    t["TNR"] = (M["P_gold"] - f * M["P_gold"], M["P_gold"])
    t["HARM_LB"] = (f * M["F_gold_fix_wrong"], M["gold"])
    return t


def metric_values(f, masks, idx=None):
    """Point values of METRICS over the units `idx` (None = all), or per-database matrices via metric_matrix."""
    t = metric_terms(f, masks)
    sl = slice(None) if idx is None else idx
    v = {}
    for k, (n, d) in t.items():
        dd = d[sl].sum()
        v[k] = float(n[sl].sum() / dd) if dd else None
    if v["UDET"] is not None and v["FPR"] is not None:
        v["J"] = v["UDET"] - v["FPR"]
    else:
        v["J"] = None
    return v


def db_matrix(units, f, masks):
    """-> (dbs, num[db, metric-term], den[db, metric-term]) for the bootstrap; metric terms in METRIC_BASE order."""
    import numpy as np
    dbs = sorted(set(u["db"] for u in units))
    di = {d: k for k, d in enumerate(dbs)}
    t = metric_terms(f, masks)
    base = ["DET", "UDET", "FPR", "CCR", "WCR", "MER", "TNR", "HARM_LB"]
    N = np.zeros((len(dbs), len(base)))
    Dn = np.zeros((len(dbs), len(base)))
    for j, k in enumerate(base):
        n, d = t[k]
        for ui, u in enumerate(units):
            N[di[u["db"]], j] += n[ui]
            Dn[di[u["db"]], j] += d[ui]
    return dbs, N, Dn, base


def boot_metric_samples(units, flags_by_name, masks, B, seed):
    """flags_by_name: name -> per-unit expected flags.  One shared resample of databases.  -> name -> metric -> array(B)."""
    import numpy as np
    dbs = sorted(set(u["db"] for u in units))
    rng = np.random.RandomState(seed)
    W = np.zeros((B, len(dbs)))
    for b in range(B):
        W[b] = np.bincount(rng.randint(0, len(dbs), size=len(dbs)), minlength=len(dbs))
    out = {}
    for name, f in flags_by_name.items():
        _, N, Dn, base = db_matrix(units, f, masks)
        num, den = W @ N, W @ Dn
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(den > 0, num / den, np.nan)
        s = {k: r[:, j] for j, k in enumerate(base)}
        s["J"] = s["UDET"] - s["FPR"]
        out[name] = s
    return out


def ci_of(x):
    import numpy as np
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return None
    return {"lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5))}


def hold_yield_rates(units, dec):
    """q-independent arbitration quality over the units that could be disputed.  firmness = hold rate on right critic decisions, responsiveness = yield rate on wrong
    decisions, split by direction (F, P) and class.  Units without a decision (C on passes) are skipped."""
    cells = collections.OrderedDict([("firmness_all", lambda u: u["critic_right"]), ("responsiveness_all", lambda u: not u["critic_right"]),
                                     ("firmness_F_claim_correct", lambda u: u["kind"] == "F" and u["critic_right"]),
                                     ("responsiveness_F_claim_wrong", lambda u: u["kind"] == "F" and not u["critic_right"]),
                                     ("responsiveness_F_claim_wrong_injected", lambda u: u["kind"] == "F" and u["inj"] and not u["critic_right"]),
                                     ("responsiveness_F_gold_flag", lambda u: u["kind"] == "F" and not u["inj"]),
                                     ("firmness_P_gold", lambda u: u["kind"] == "P" and not u["inj"]),
                                     ("responsiveness_P_injected", lambda u: u["kind"] == "P" and u["inj"])])
    out = {}
    for name, pred in cells.items():
        sel = [u for u in units if pred(u) and u["uid"] in dec]
        if not sel:
            out[name] = {"n": 0, "rate": None}
            continue
        holds = sum(1 for u in sel if dec[u["uid"]])
        rate = holds / len(sel) if name.startswith("firmness") else 1 - holds / len(sel)
        out[name] = {"n": len(sel), "rate": rate}
    fa, ra = out["firmness_all"]["rate"], out["responsiveness_all"]["rate"]
    out["balanced_accuracy"] = (fa + ra) / 2 if fa is not None and ra is not None else None
    return out


def boot_rates(units, dec, B, seed):
    """Clustered CI of the q-independent rates."""
    import numpy as np
    dbs = sorted(set(u["db"] for u in units))
    by = collections.defaultdict(list)
    for u in units:
        by[u["db"]].append(u)
    rng = np.random.RandomState(seed)
    acc = collections.defaultdict(list)
    for _ in range(B):
        sel = [u for j in rng.randint(0, len(dbs), size=len(dbs)) for u in by[dbs[j]]]
        r = hold_yield_rates(sel, dec)
        for k, v in r.items():
            if k == "balanced_accuracy":
                if v is not None:
                    acc[k].append(v)
            elif v["rate"] is not None:
                acc[k].append(v["rate"])
    return {k: ci_of(v) for k, v in acc.items()}


def realised(units, dec, q, seed):
    """One seeded realisation of the user noise: per unit an independent uniform u (the user is right iff u < q).  -> counts of disputes per cell and the
    realised metrics for obey and the arm."""
    rng = random.Random(seed)
    cells = collections.Counter()
    fl_arm, fl_obey = [], []
    for u in units:
        right = rng.random() < q
        # the user contradicts the critic iff the user's verdict says the critic is wrong
        user_says_critic_wrong = (not u["critic_right"]) if right else u["critic_right"]
        s = 1.0 if u["kind"] == "F" else 0.0
        if not user_says_critic_wrong:
            fl_arm.append(s)
            fl_obey.append(s)
            continue
        cells["user_right" if right else "user_wrong", u["kind"]] += 1
        fl_obey.append(1 - s)
        d = dec.get(u["uid"]) if dec else None
        fl_arm.append(s if (d is None or d) else 1 - s)
        if d is not None:
            cells["arm_holds" if d else "arm_yields", "user_right" if right else "user_wrong", u["kind"]] += 1
    return {"disputes": {"|".join(k): v for k, v in sorted(cells.items())}, "flags_arm": fl_arm, "flags_obey": fl_obey}


def decide_rule(point, ci, name):
    """SAP 19.5 as committed (useful_19_5) and with the correction 19.9 (useful): the corrected rule also requires that the appeal adds value over NO appeal
    (an arbiter that always holds equals no appeal and would pass 19.5).  point[policy][metric], ci[(a, b)][metric] = CI of the paired difference a - b."""
    arm, obey, none = point[name], point["obey"], point["none"]
    d_obey_J, d_obey_C = arm["J"] - obey["J"], arm["CCR"] - obey["CCR"]
    c_obey_J, c_obey_C = ci[(name, "obey")]["J"], ci[(name, "obey")]["CCR"]
    beat_a = bool(d_obey_J >= 0.02 and c_obey_J and c_obey_J["lo"] > 0)
    beat_b = bool(d_obey_C >= 0.05 and c_obey_C and c_obey_C["lo"] > 0)
    d_none_C, d_none_J = arm["CCR"] - none["CCR"], arm["J"] - none["J"]
    c_none_C, c_none_J = ci[(name, "none")]["CCR"], ci[(name, "none")]["J"]
    keep = bool(d_none_C >= -0.03 and d_none_J >= -0.03 and c_none_C and c_none_J and c_none_C["lo"] > -0.06 and c_none_J["lo"] > -0.06)
    keep_strict = bool(d_none_C >= -0.03 and d_none_J >= -0.03 and c_none_C and c_none_J and c_none_C["lo"] >= -0.03 and c_none_J["lo"] >= -0.03)
    adds = bool(d_none_J >= 0.02 and c_none_J and c_none_J["lo"] > 0)                    # 19.9 clause 3: beats no appeal on J
    return {"q": Q_RULE, "arm": name, "clause1a_J_beats_obey": beat_a, "clause1b_CCR_beats_obey": beat_b,
            "delta_J_vs_obey": d_obey_J, "delta_CCR_vs_obey": d_obey_C, "delta_CCR_vs_none": d_none_C, "delta_J_vs_none": d_none_J,
            "clause2_no_loss_vs_none": keep, "clause2_strict": keep_strict, "clause3_J_beats_none_19_9": adds,
            "useful_19_5_as_committed": bool((beat_a or beat_b) and keep), "useful_19_5_strict_reading": bool((beat_a or beat_b) and keep_strict),
            "useful": bool((beat_a or beat_b) and keep and adds), "useful_strict_reading": bool((beat_a or beat_b) and keep_strict and adds)}


def calls_per_query(units, dec, q, flags_only=False):
    n = len(units)
    tot = 0.0
    for u in units:
        if flags_only and u["kind"] != "F":
            continue
        if dec is not None and u["uid"] not in dec:
            continue
        tot += p_disputed(u["critic_right"], q)
    return tot / n


def build_cases(units, decs_raw):
    """Illustrative cases by a FIXED rule (lowest uid of each class; not by effect size).  Classes are defined on arm A's decision and the oracle labels."""
    by = {u["uid"]: u for u in units}
    A = decs_raw["A"]

    def pick(pred, n=1):
        return [u["uid"] for u in units if u["uid"] in A and pred(u, A[u["uid"]])][:n]
    classes = [
        ("A_holds_correct_claim_user_wrong", lambda u, d: u["kind"] == "F" and u["critic_right"] and d["hold"]),
        ("A_yields_on_correct_claim_user_wrong", lambda u, d: u["kind"] == "F" and u["critic_right"] and not d["hold"]),
        ("A_drops_wrong_claim_gold_flag", lambda u, d: u["kind"] == "F" and not u["inj"] and not d["hold"]),
        ("A_drops_wrong_claim_injected", lambda u, d: u["kind"] == "F" and u["inj"] and not u["critic_right"] and not d["hold"]),
        ("A_keeps_wrong_claim_injected", lambda u, d: u["kind"] == "F" and u["inj"] and not u["critic_right"] and d["hold"]),
        ("A_recovers_missed_error", lambda u, d: u["kind"] == "P" and u["inj"] and not d["hold"]),
        ("A_flags_fine_query_user_wrong", lambda u, d: u["kind"] == "P" and not u["inj"] and not d["hold"]),
    ]
    out = {"_meta": {"scope": SCOPE, "selection": "fixed rule: lowest uid of the class under arm A; illustrative, not representative; the user statement is a SIMULATED verdict-only template"},
           "cases": []}
    for cls, pred in classes:
        for uid in pick(pred, 2):
            u = by[uid]
            c = {"class": cls, "uid": uid, "kind": u["kind"], "category": u["cat"], "db_id": u["db"], "injected": u["inj"], "critic_right": u["critic_right"],
                 "question": u["question"], "evidence": u["evidence"], "sql": u["sql"], "claims": u.get("claims"), "remarks": u.get("remarks"), "fix_sql": u.get("fix_sql"),
                 "fix_matches_gold": u.get("fix_matches_gold"), "arms": {}}
            for n in decs_raw:
                if uid in decs_raw[n]:
                    d = decs_raw[n][uid]
                    c["arms"][n] = {"hold": d["hold"], "decision": d["decision"], "evidence": d["evidence"]}
            out["cases"].append(c)
    return out


def cmd_rates(a):
    units = json.load(open(a.units))["units"]
    for kv in a.dec:
        n, p = kv.split("=", 1)
        dec = {k: v["hold"] for k, v in load_decisions(p).items()}
        recs = load_decisions(p)
        r = hold_yield_rates(units, dec)
        bad = sum(1 for v in recs.values() if not v["valid"])
        print(n, json.dumps({"n": len(recs), "invalid": bad, "rates": r}, indent=1))


def cmd_analyze(a):
    units = json.load(open(a.units))["units"]
    meta_u = json.load(open(a.units))["_meta"]
    decs_raw, decs = {}, {}
    for kv in a.dec:
        n, p = kv.split("=", 1)
        decs_raw[n] = load_decisions(p)
        decs[n] = {k: v["hold"] for k, v in decs_raw[n].items()}
    masks = group_masks(units)
    res = {"_meta": {"scope": SCOPE, "label": "exploratory-posthoc (SAP Amendment 6d, section 19); SIMULATED user dispute channel; controlled diagnostic with injected errors; "
                                              "not evidence of EX on BIRD, natural errors or real users; never enters results_macros.tex",
                     "ci": "95%% percentile, database-clustered bootstrap, B=%d, seed %d, one shared resample (paired differences)" % (a.B, SEED_BOOT),
                     "q": list(QS), "q_rule": Q_RULE, "units": meta_u, "decision_rule": "SAP 19.5",
                     "primary": "exact expectation over the user noise; one seeded realisation (seed %d) as a check" % SEED_REAL,
                     "arms": {"A": "qwen3.7-plus-2026-05-26 neutral prompt, user statement", "A0": "same without the user statement (control)",
                              "B": "deepseek-v4.1-flash-1017 neutral prompt, user statement", "C": "old VD prompt, flags only (control)"}},
           "units_count": dict(meta_u.get("counts", {})), "validity": {}, "arbitration_quality": {}, "by_q": {}}
    for n, raw in decs_raw.items():
        res["validity"][n] = {"n": len(raw), "invalid_or_unparsable": sum(1 for v in raw.values() if not v["valid"]),
                              "hold_rate_all": sum(1 for v in raw.values() if v["hold"]) / max(1, len(raw))}
        res["arbitration_quality"][n] = {"point": hold_yield_rates(units, decs[n]), "ci": boot_rates(units, decs[n], a.B, SEED_BOOT)}
    for q in QS:
        flags = collections.OrderedDict()
        for p in ("none", "obey", "oracle_arbiter", "oracle_all"):
            flags[p] = policy_flags(units, p, q)
        for n in decs:
            flags[n] = policy_flags(units, n, q, decs[n])
        samples = boot_metric_samples(units, flags, masks, a.B, SEED_BOOT)
        point = {p: metric_values(f, masks) for p, f in flags.items()}
        ci = {}
        pairs = [(n, "obey") for n in list(decs)] + [(n, "none") for n in list(decs)] + [("obey", "none"), ("oracle_arbiter", "obey")] + \
                [(n, "A0") for n in decs if n not in ("A0",) and "A0" in decs]
        for x, y in pairs:
            ci[(x, y)] = {m: ci_of(samples[x][m] - samples[y][m]) for m in METRICS}
        cell = {"point": point, "ci": {p: {m: ci_of(samples[p][m]) for m in METRICS} for p in flags},
                "delta_ci": {"%s - %s" % k: v for k, v in ci.items()}, "delta_point": {"%s - %s" % (x, y): {m: (point[x][m] - point[y][m]) if point[x][m] is not None and point[y][m] is not None else None for m in METRICS} for x, y in pairs},
                "expected_disputes": {}, "calls_per_query": {}}
        for n in decs:
            cell["calls_per_query"][n] = calls_per_query(units, decs[n] if n == "C" else None, q, flags_only=(n == "C"))
            cell["expected_disputes"][n] = cell["calls_per_query"][n] * len(units)
        cell["expected_disputes_by_class"] = {"user_right_F": sum(p_disputed(u["critic_right"], q) for u in units if u["kind"] == "F" and not u["critic_right"]),
                                              "user_wrong_F": sum(p_disputed(u["critic_right"], q) for u in units if u["kind"] == "F" and u["critic_right"]),
                                              "user_right_P": sum(p_disputed(u["critic_right"], q) for u in units if u["kind"] == "P" and not u["critic_right"]),
                                              "user_wrong_P": sum(p_disputed(u["critic_right"], q) for u in units if u["kind"] == "P" and u["critic_right"])}
        cell["realisation"] = {}
        for n in decs:
            rz = realised(units, decs[n], q, SEED_REAL)
            cell["realisation"][n] = {"seed": SEED_REAL, "disputes": rz["disputes"], "metrics": metric_values(rz["flags_arm"], masks),
                                      "metrics_obey": metric_values(rz["flags_obey"], masks)}
        if abs(q - Q_RULE) < 1e-9:
            cell["decision_rule_19_5"] = {n: decide_rule(point, ci, n) for n in ("A", "B") if n in decs}
        res["by_q"]["%.2f" % q] = cell
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, a.name), "w"), indent=1, ensure_ascii=False)
    json.dump(build_cases(units, decs_raw), open(os.path.join(a.out, a.name.replace("summary", "cases")), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"validity": res["validity"], "rule": res["by_q"].get("0.85", {}).get("decision_rule_19_5")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    u = sub.add_parser("units")
    u.add_argument("--records", required=True)
    u.add_argument("--items", required=True)
    u.add_argument("--out", required=True)
    r = sub.add_parser("arbitrate")
    r.add_argument("--units", required=True)
    r.add_argument("--arm", required=True, choices=list(ARMS))
    r.add_argument("--out", required=True)
    r.add_argument("--conc", type=int, default=4)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--prompt-dir", default="")
    t = sub.add_parser("rates")
    t.add_argument("--units", required=True)
    t.add_argument("--dec", nargs="+", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("--units", required=True)
    z.add_argument("--dec", nargs="+", required=True, help="NAME=decisions_NAME.jsonl")
    z.add_argument("--out", required=True)
    z.add_argument("--name", default="critic_ve_summary.json")
    z.add_argument("--B", type=int, default=5000)
    a = ap.parse_args()
    {"units": cmd_units, "arbitrate": cmd_arbitrate, "rates": cmd_rates, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
