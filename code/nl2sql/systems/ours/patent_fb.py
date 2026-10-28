"""patent_fb (SAP Amendment 9, docs/SAP.md section 22): the fixed patent loop P0 with a FEEDBACK-CALIBRATED Critic.

Patent step 7: "user feedback and execution results update the Critic's scoring standard". In `full_fixed` the Critic prompt is fixed and feedback only
reaches the corpus score WS (M6). Here the Critic prompt carries (i) the grounded rules of exp/diag_prompts/va_grounded.user.txt (verbatim) and (ii) a
"scoring standard" section: the k most similar past cases of a FEEDBACK STORE with the user's verdict (accepted / rejected).
Only the Critic input changes. Score = 0.3 syntax + 0.5 semantic + 0.2 efficiency (stock M5 sub-scores), trigger 8.5, t_max 3, final = best node,
reviser prompt rev.user: all unchanged (FullFixed._run_p0).

Feedback store = the feedback-bearing burn-in entries of the frozen M6 store (BIRD TRAIN stream questions; kind exp, origin burnin), verdict = EX of
the stored SQL against the train gold (exp/pf_data/burnin_verdicts.jsonl) -- a SIMULATED user, an upper bound for an ideal one. Dose n = the first n
entries of a seeded order (nested subsets); noise q = each entry's verdict is kept with probability q and flipped otherwise (fixed per entry, independent
of the item under review). Retrieval is a separate ExperienceCorpus view of the same frozen sqlite with policy `static` (similarity only, no WS), so the true
EX never influences which cases are shown. Dev, DIAG and tuning qids are asserted absent from the store.

Everything here is behind the system name `patent_fb` (config key `pf`); no existing system, config or prompt is touched."""
import hashlib
import json
import os
import re
import threading

from ... import paths
from ... import prompts_util as P
from ...util import parse_json
from . import m5 as M5
from .corpus import ExperienceCorpus, u01
from .m1 import graph_text
from .pipeline import FullFixed

PF_DEFAULTS = {
    "k": 4,                                   # past cases shown
    "n": 0,                                   # feedback entries in the store (0 = no standard section: the grounded prompt only)
    "q": 1.0,                                 # probability that an entry's verdict equals the oracle (EX against train gold)
    "seed": 20261128,
    "max_tokens": 900,                        # VA: longer JSON with quotes
    "prompt_dir": "exp/pf_prompts",
    "template": "pf_critic.user.txt",
    "standard": "pf_standard.txt",
    "verdicts": "exp/pf_data/burnin_verdicts.jsonl",
    "verdicts_sha256": None,
    "prompt_sha256": None,                    # {file: sha256} of template / standard; checked at start (the config hash then pins the prompts in the dev lock)
    "forbid_files": [],                       # json files holding qids that must not be in the store (DIAG / tuning items)
}
FORBID_BIRD_SPLITS = ("bird_dev:", "bird_minidev:")


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _abs(p):
    return p if os.path.isabs(p) else str(paths.REPO / p)


def render_text(text, **kw):
    """Same placeholder syntax as prompts_util.render (<<name>>, optional <<name?>>), for text read from exp/pf_prompts."""
    def rep(m):
        k = m.group(1)
        if k not in kw:
            if m.group(2):
                return ""
            raise KeyError("missing placeholder %s" % k)
        return str(kw[k])
    return re.sub(r"<<(\w+)(\?)?>>", rep, text)


def full_schema_text(schema):
    lines = []
    for t, cols in schema["tables"].items():
        lines.append("%s(%s)" % (t, ", ".join(("%s %s" % (c["name"], c.get("type") or "")).strip() for c in cols)))
    return "\n".join(lines)


def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def split_issues(raw_issues):
    """-> (must_fix, style). A must_fix issue needs a non-empty quoted question phrase AND SQL fragment (as in critic_v2.split_issues, VA)."""
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


def load_prompts(pf):
    """-> (template, standard, {file: sha256}); the shas must equal exp/pf_prompts/FROZEN.json when that file lists them."""
    d = _abs(pf["prompt_dir"])
    out, shas = {}, {}
    for key in ("template", "standard"):
        p = os.path.join(d, pf[key])
        out[key] = open(p).read()
        shas[pf[key]] = sha256_file(p)
    for f, h in (pf.get("prompt_sha256") or {}).items():
        if shas.get(f) != h:
            raise ValueError("pf prompt %s does not match pf.prompt_sha256 of the config" % f)
    fz = os.path.join(d, "FROZEN.json")
    if os.path.exists(fz):
        want = (json.load(open(fz)).get("prompts") or {})
        for f, s in shas.items():
            if f in want and want[f] != s:
                raise ValueError("pf prompt %s differs from exp/pf_prompts/FROZEN.json" % f)
    return out["template"], out["standard"], shas


# ======================================================================================================== feedback store
class FeedbackStore(object):
    """n feedback entries (nested seeded subsets) of the frozen burn-in store with simulated user verdicts (noise q)."""

    def __init__(self, entries_by_qid, oracle, n, q, seed):
        self.oracle = oracle                              # qid -> EX (0/1) of the stored SQL against the train gold
        self.seed, self.q = seed, float(q)
        order = sorted(entries_by_qid, key=lambda x: (u01(seed, "pf-order", x), x))
        if n > len(order):
            raise ValueError("pf.n=%d but the store has only %d feedback-bearing entries" % (n, len(order)))
        self.qids = order[:n]
        self.verdict = {x: self._noisy(x) for x in self.qids}          # 1 = ACCEPTED, 0 = REJECTED (as the simulated user said)
        self.ids = set("bi:" + x for x in self.qids)

    def _noisy(self, qid):
        v = int(self.oracle[qid])
        return v if u01(self.seed, "pf-noise", qid) < self.q else 1 - v

    def stats(self):
        n = len(self.qids)
        return {"n": n, "q": self.q, "accepted": sum(self.verdict.values()), "rejected": n - sum(self.verdict.values()),
                "flipped": sum(1 for x in self.qids if self.verdict[x] != int(self.oracle[x])),
                "oracle_accepted": sum(int(self.oracle[x]) for x in self.qids)}


def load_verdicts(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            d = json.loads(line)
            out[d["qid"]] = int(d["ex"])
    return out


def collect_qids(obj):
    """all strings under a key `qid` (or a list under `qids`) anywhere in a json document."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "qid" and isinstance(v, str):
                out.add(v)
            elif k == "qids" and isinstance(v, list):
                out.update(x for x in v if isinstance(x, str))
            else:
                out |= collect_qids(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= collect_qids(v)
    return out


def forbidden_qids(files):
    out = set()
    for f in files or []:
        p = _abs(f)
        if p.endswith(".txt"):
            out |= set(l.strip() for l in open(p) if l.strip())
        else:
            out |= collect_qids(json.load(open(p)))
    return out


def build_store(corpus, oracle, pf):
    """Feedback-bearing entries of the (static-policy view of the) frozen store -> FeedbackStore of size pf.n. Asserts the leakage conditions."""
    ok = {}
    for eid in corpus.order:
        e = corpus.entries[eid]
        if e.kind == "exp" and e.origin == "burnin" and eid.startswith("bi:") and eid[3:] in oracle:
            ok[eid[3:]] = e
    forb = forbidden_qids(pf.get("forbid_files"))
    bad = [x for x in ok if x in forb or x.startswith(FORBID_BIRD_SPLITS) or not x.startswith("bird_train:")]
    assert not bad, "dev / DIAG / tuning qid inside the feedback store: %s" % bad[:3]
    return FeedbackStore(ok, oracle, int(pf["n"]), pf["q"], pf["seed"]), len(ok)


def render_cases(cases):
    parts = []
    for n, c in enumerate(cases, 1):
        v = "ACCEPTED" if c["verdict"] == 1 else "REJECTED"
        parts.append("Case %d. User verdict: %s\nQuestion: %s\nEvidence: %s\nSQL: %s" % (n, v, c["question"], c["evidence"] or "(none)", c["sql"]))
    return "\n\n".join(parts)


# ======================================================================================================== the critic
class FBCritic(object):
    """Drop-in replacement of Pipeline.critic (same score() signature and return dict); score = stock weighted sum with the stock sub-scores."""

    def __init__(self, base, pipe, pf, store, mem_corpus, template, standard):
        self.base, self.pipe, self.pf = base, pipe, pf
        self.llm, self.model, self.cfg = base.llm, base.model, base.cfg
        self.w = base.w
        self.variant = "pf"
        self.learned = None
        self.store, self.mem_corpus = store, mem_corpus
        self.template, self.standard = template, standard
        self._mem = {}
        self._pfail = {}              # qid -> number of Critic calls whose output stayed unparsable after the one retry
        self._lock = threading.Lock()
        self.k = int(pf["k"])

    # ---- memory
    def cases(self, item):
        """-> list of {id, qid, question, evidence, sql, verdict, oracle}: the k most similar feedback entries (same item -> same cases)."""
        if self.store is None or not self.store.ids:
            return []
        with self._lock:
            c = self._mem.get(item.qid)
        if c is not None:
            return c
        res = self.pipe.res
        qv = res.embed_one(item.question)
        block = set(self.mem_corpus.order) - self.store.ids
        block |= {item.qid, "bi:" + item.qid}
        entries, _ = self.mem_corpus.recall(self.cfg["models"]["rerank"], item.question, qv, self.k, exclude=block, rerank=True, hybrid=True)
        out = []
        for e in entries:
            assert e.id in self.store.ids and e.id[3:] != item.qid
            q = e.id[3:]
            out.append({"id": e.id, "qid": q, "question": e.question, "evidence": e.evidence, "sql": e.sql, "verdict": self.store.verdict[q],
                        "oracle": int(self.store.oracle[q])})
        with self._lock:
            self._mem[item.qid] = out
        return out

    def prompt(self, item, sql, ctx, graph, res, cases):
        gt = graph_text(graph) if (graph and self.cfg["m4"]["graph_in_prompt"] and self.cfg["m1"]["enabled"]) else ""
        from ... import data
        std = render_text(self.standard, cases=render_cases(cases)) if cases else ""
        return render_text(self.template, sections=ctx.sections_core(), full_schema=full_schema_text(data.get_schema(item)), standard=std,
                           question=item.question, evidence=item.evidence or "(none)", graph=gt, sql=sql, preview=M5.exec_summary(res))

    def score(self, item, sql, ctx, graph, res, seed, row_counts, variant=None, with_text=True, meta=None):
        from ... import db
        syn = M5.s_syn(res)
        plan = db.explain(item.db_path, sql) if res["ok"] else []
        eff = M5.s_eff(res, plan, row_counts)
        out = {"syntax": syn, "efficiency": eff, "semantic": None, "issues": [], "style_issues": [], "suggestions": [], "status": M5.exec_status(res),
               "parse_ok": True, "plan_scans": sum(1 for l in plan if l.strip().startswith("SCAN")),
               "plan_temp": sum(1 for l in plan if "USE TEMP B-TREE" in l), "plan_corr": sum(1 for l in plan if "CORRELATED SCALAR SUBQUERY" in l),
               "plan_search": sum(1 for l in plan if l.strip().startswith("SEARCH"))}
        cases = self.cases(item)
        msgs = [{"role": "system", "content": P.render("common.system")},
                {"role": "user", "content": self.prompt(item, sql, ctx, graph, res, cases)}]
        mt = int(self.pf["max_tokens"])
        r = self.llm.chat(self.model, msgs, temperature=0.0, seed=seed, max_tokens=mt, tag="crit")
        j = parse_json(r["text"])
        if j is None:
            r = self.llm.chat(self.model, msgs + [{"role": "assistant", "content": r["text"]}, {"role": "user", "content": P.render("m1.retry")}],
                              temperature=0.0, seed=seed, max_tokens=mt, tag="crit")
            j = parse_json(r["text"])
        out["parse_ok"] = j is not None
        if j:
            out["semantic"] = M5._num(j.get("semantic"), 5.0)
            out["issues"], out["style_issues"] = split_issues(j.get("issues"))
            out["suggestions"] = j.get("suggestions") or []
        else:
            out["semantic"] = 5.0           # neutral on unparseable output, as in the stock critic
            with self._lock:
                self._pfail[item.qid] = self._pfail.get(item.qid, 0) + 1
        w = self.w
        out["score"] = round(w[0] * out["syntax"] + w[1] * out["semantic"] + w[2] * out["efficiency"], 4)
        out["n_cases"] = len(cases)
        return out

    def score_exec(self, *a, **kw):
        return self.base.score_exec(*a, **kw)


# ======================================================================================================== the system
class PatentFB(FullFixed):
    """FullFixed (forced P0) with the feedback-calibrated Critic. Config: `pf` (see PF_DEFAULTS)."""
    name = "patent_fb"

    def __init__(self, llm, model, cfg):
        super().__init__(llm, model, cfg)
        pf = dict(PF_DEFAULTS)
        pf.update((cfg or {}).get("pf") or {})
        self.pf = pf
        if self.rcfg["m5"].get("policy", "p0") != "p0":
            raise ValueError("patent_fb needs m5.policy p0 (the fixed loop)")
        if pf["verdicts_sha256"] and sha256_file(_abs(pf["verdicts"])) != pf["verdicts_sha256"]:
            raise ValueError("pf.verdicts does not match pf.verdicts_sha256")
        template, standard, self.prompt_shas = load_prompts(pf)
        store, mem_corpus, self.store_size = None, None, 0
        if int(pf["n"]) > 0:
            m6 = self.rcfg["m6"]
            if not m6.get("path"):
                raise ValueError("patent_fb with pf.n > 0 needs m6.path (the frozen burn-in store)")
            mem_corpus = ExperienceCorpus.load(_abs(m6["path"]), llm, self.rcfg["models"]["embed"], "static", None)
            store, self.store_size = build_store(mem_corpus, load_verdicts(_abs(pf["verdicts"])), pf)
        self.store = store
        self.pipe.critic = FBCritic(self.pipe.critic, self.pipe, pf, store, mem_corpus, template, standard)
        print("[lineage] patent_fb: n=%s q=%s k=%s store=%s prompts=%s" % (pf["n"], pf["q"], pf["k"], store.stats() if store else None,
                                                                          {k: v[:8] for k, v in self.prompt_shas.items()}), flush=True)

    def run(self, item, seed=0):
        out = super().run(item, seed=seed)
        cases = self.pipe.critic._mem.get(item.qid) or []
        out["pf"] = {"n": int(self.pf["n"]), "q": self.pf["q"], "parse_fail": self.pipe.critic._pfail.get(item.qid, 0), "cases": [{"id": c["id"], "verdict": c["verdict"], "oracle": c["oracle"]} for c in cases]}
        return out
