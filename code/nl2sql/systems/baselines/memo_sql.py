"""Memo-SQL (Yang et al., arXiv 2601.10011, Jan 2026): structured decomposition + experience-driven self-correction.

REIMPLEMENTATION.  No official code is public (paper page, arXiv listing, GitHub search checked 2026-10-24) and the paper
prints no prompts; every prompt below is therefore "our adaptation" written from the paper's textual description.
Identical to the paper (structure, Sections 3.1-3.5):
  * offline error-correction memory of quintuples <q, s+, s-, E, delta> with the nine-type taxonomy E1-E9 (Table 4),
    built from the BIRD training set (here: the shared train pool POOL57 only), entry key = question + skeleton(s-);
  * online: three decomposition strategies (table-wise, hierarchical, atomic operational) with a "no decomposition" fallback;
    each sub-question solved by ReAct+Reflect (reason -> sub-SQL -> observe execution -> reflect, one revision);
  * final generation from the reasoning trace + 5 retrieved (q, SQL) exemplars (retrieved with question + skeleton of the
    latest sub-SQL) in three styles (CTE / flat JOIN / nested) -> up to 9 candidates;
  * memory-augmented refinement: key = question + candidate skeleton, top-40 retrieval, deduplication that drops an entry whose
    error-type set is fully subsumed by an already kept one (descending similarity), 3-5 kept; critic -> refine loop until the
    critic accepts or 3 rounds; final = majority vote over execution results of the refined candidates.
Adapted (our adaptation):
  * backbone Qwen3-Coder-30B-A3B -> the system's model; embedder GTE-base -> `qwen3.7-text-embedding`;
  * all prompts (decomposer, ReAct step, reflection, final generator, critic, refiner, memory annotator);
  * SQL skeleton = the DAIL regex skeleton (`dail_sql.sql_skeleton`); schema preprocessing/linking (Appendix A.2) is not
    reproduced - the full "rich" schema (descriptions + value examples) is given to every baseline instead;
  * memory is built by `build_memory()` from POOL57 with the system's own model (the paper uses its training-set candidates; the
    candidate sampler and annotator are ours); a plain execution mismatch against the gold SQL defines a "semantic" failure;
  * dialect: SQLite (BIRD), same as the paper;
  * budget knobs for compute matching: `strategies`, `styles`, `react`, `refine_rounds`, `mem_keep`; `preset="native"` is the paper's
    full 3x3 pipeline with <=3 refine rounds, `preset="light"` is a reduced one (see docs/BASELINES.md).
"""
from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter

import numpy as np

from ._common import (ApiFail, BaselineSystem, DEFAULT_EMBED_MODEL, datamod, embed_matrix, embed_one, extract_sql, fmt_rows,
                      one_line, question_block, result_key, run_index, schema_text)
from .dail_sql import sql_skeleton

ERROR_TYPES = {
    "E1": "Join Logic Error", "E2": "Filter Condition Error", "E3": "Aggregation and Grouping Error",
    "E4": "Select Output Error", "E5": "Ordering and Limit Error", "E6": "Subquery Logical Error",
    "E7": "Null Handling Error", "E8": "Temporal Semantics Error", "E9": "Quantifier Intent Error",
}
_TAXO = "\n".join("%s: %s" % kv for kv in ERROR_TYPES.items())

# ----------------------------------------------------------------------------- prompts (our adaptation; none are printed in the paper)
SYS = "You are an expert SQLite engineer. Follow the output format exactly."

STRATEGY_TEXT = {
    "table": ("Table-wise decomposition: identify every table/entity mentioned or implied by the question and write one "
              "sub-question per table (with that table's filters and projections); the tables are joined through foreign keys afterwards."),
    "hier": ("Hierarchical decomposition: split the question from the innermost to the outermost semantics (for example the "
             "aggregate or existence condition first, then the outer query that uses it)."),
    "atomic": ("Atomic operational decomposition: break the question into a sequence of atomic relational operations "
               "(selection, projection, join, grouping, aggregation, ordering), one sub-question per step."),
}

DECOMPOSE = """{schema}

{question}

Decompose the question with this strategy.
{strategy}

Return a JSON list of at most {max_sub} short sub-questions in order, inside a ```json block.
If the question is too simple to decompose or this strategy does not fit, return an empty list [].
"""

REACT_STEP = """{schema}

{question}

Strategy: {sname}
Earlier steps:
{history}

Current sub-question: {sub}
First reason briefly about what it needs from the schema (one short paragraph), then write the SQLite query for it in a ```sql block."""

REFLECT = """{schema}

{question}

Sub-question: {sub}
SQL: {sql}
Execution result (first rows): {obs}

Check the sub-SQL for data validity problems (empty or NULL results, wrong values, wrong columns) and semantic misalignment with the sub-question.
If it is fine, answer exactly OK. Otherwise answer with one corrected query in a ```sql block."""

STYLE_TEXT = {
    "cte": "Write the query with Common Table Expressions (WITH ... AS) that materialise the intermediate results.",
    "flat": "Write the query as one flat statement with explicit JOIN clauses and filters in WHERE/ON, no subqueries.",
    "nested": "Write the query with nested subqueries (IN / EXISTS / scalar subqueries) for the auxiliary logic.",
}

FINAL = """{schema}

{examples}{question}

Reasoning trace ({sname} strategy):
{trace}

{style}
Return only the final SQLite query in a ```sql block."""

CRITIC = """{schema}

{question}

Candidate SQL:
{sql}
Execution result (first rows): {obs}

Past failures of similar queries (each: question, wrong SQL, error types, correction advice, corrected SQL):
{cases}

Error taxonomy:
{taxo}

Decide whether the candidate has a semantic error. Answer with a JSON object in a ```json block:
{{"correct": true or false, "error_types": ["E1", ...], "suggestion": "one concrete revision instruction"}}"""

REFINE = """{schema}

{question}

Candidate SQL:
{sql}
{feedback}

Write the corrected SQLite query in a ```sql block."""

ANNOTATE = """{schema}

Question: {q}
Correct SQL: {good}
Wrong SQL: {bad}

Error taxonomy:
{taxo}

Assign one or more error types to the wrong SQL and write one actionable correction suggestion.
Answer with a JSON object in a ```json block: {{"error_types": ["E1"], "suggestion": "..."}}"""


# ----------------------------------------------------------------------------- helpers
def jblock(text):
    m = re.search(r"```json\s*(.*?)```", text or "", re.DOTALL)
    s = m.group(1) if m else (text or "")
    try:
        return json.loads(s.strip())
    except Exception:
        m2 = re.search(r"[\[{].*[\]}]", s, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(0))
            except Exception:
                return None
    return None


def obs_text(res, rows=5):
    if not res["ok"]:
        return "ERROR: %s" % (res.get("error") or "")[:300]
    if not res["rows"]:
        return "(no rows)"
    return fmt_rows(res["rows"], max_rows=rows)


# ----------------------------------------------------------------------------- memory
class Memory:
    def __init__(self, entries, emb):
        self.entries, self.emb = entries, emb

    @staticmethod
    def load(path, llm, model, cache_dir=None):
        entries = json.load(open(path, encoding="utf-8"))
        texts = [e["key"] for e in entries]
        emb = embed_matrix(llm, model, texts, cache_dir=cache_dir, name="memo_mem") if texts else np.zeros((0, 1), np.float32)
        return Memory(entries, emb)

    def retrieve(self, qvec, top=40, keep=5):
        if not self.entries:
            return []
        sims = self.emb @ qvec
        order = np.argsort(-sims)[:top]
        kept, kept_sets = [], []
        for i in order:
            es = frozenset(self.entries[i]["error_types"])
            if any(es <= k for k in kept_sets):      # fully subsumed by an earlier retained entry
                continue
            kept.append(self.entries[i])
            kept_sets.append(es)
            if len(kept) >= keep:
                break
        return kept


_MEM = {}
_POOLX = {}
_LOCK = threading.Lock()


def resolve_repo_path(path):
    """Config paths such as `exp/frozen/memo_memory_v1.json` are repo-relative (the lock gate and `config.load_config` resolve them
    against the repo root).  The runner's cwd is `<repo>/code`, so a bare `open(path)` failed with FileNotFoundError on every item of
    MIN/M2/memosql-s0 (MEMOFIX).  Absolute paths are kept; otherwise the repo-relative file wins when it exists; else unchanged."""
    p = str(path)
    if os.path.isabs(p):
        return p
    try:
        from nl2sql import paths as _paths
        cand = os.path.join(str(_paths.REPO), p)
        if os.path.exists(cand):
            return cand
    except Exception:      # shim environment without the core package: keep the cwd-relative path
        pass
    return p


def get_memory(llm, model, path, cache_dir):
    path = resolve_repo_path(path)
    with _LOCK:
        key = (path, model)
        if key not in _MEM:
            _MEM[key] = Memory.load(path, llm, model, cache_dir)
        return _MEM[key]


def get_pool_emb(llm, cfg, model):
    """question + skeleton(gold SQL) embeddings of the exemplar pool (POOL57)."""
    split = cfg.get("pool_split", "bird_train_pool")
    key = (split, model, cfg.get("pool_limit"))
    with _LOCK:
        if key in _POOLX:
            return _POOLX[key]
    items = datamod.load_split(split)
    if cfg.get("pool_limit"):
        items = items[: int(cfg["pool_limit"])]
    texts = [(it.question + " " + (it.evidence or "")).strip() + " || " + sql_skeleton(it.gold_sql) for it in items]
    emb = embed_matrix(llm, model, texts, cache_dir=cfg.get("emb_cache_dir"), name="memo_pool_" + split)
    with _LOCK:
        _POOLX[key] = (items, emb)
    return _POOLX[key]


def build_memory(llm, model, out_path, *, n_questions=400, n_cand=3, temperature=0.7, split="bird_train_pool", workers=8,
                 seed_base=900000, emb_model=DEFAULT_EMBED_MODEL):
    """Offline phase.  For each pool question sample `n_cand` zero-shot candidates, keep executable-but-wrong ones
    (result set differs from the gold), take one as s-, let the model assign E1-E9 and a correction suggestion.
    Uses the pool split only (never HT/HR/dev).  Writes a JSON list; returns (n_entries, usage dict)."""
    from concurrent.futures import ThreadPoolExecutor
    from . import _common as C
    items = datamod.load_split(split)[:n_questions]
    usage = Counter()
    lock = threading.Lock()

    def one(it):
        schema = schema_text(it)
        gold = C.dbmod.execute(it.db_path, it.gold_sql, timeout_s=30)
        if not gold["ok"]:
            return None
        prompt = "%s\n\n%s\n\nWrite one SQLite query that answers the question. Return only the SQL in a ```sql block." % (schema, question_block(it))
        wrong = []
        for j in range(n_cand):
            r = llm.chat(model, [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}], temperature=temperature,
                         seed=seed_base + j, max_tokens=1024, tag="memo.build")
            with lock:
                usage["calls"] += 1
                usage["pt"] += r["prompt_tokens"]
                usage["ct"] += r["completion_tokens"]
            sql = extract_sql(r["text"])
            if not sql:
                continue
            res = C.dbmod.execute(it.db_path, sql, timeout_s=30)
            if res["ok"] and result_key(res) != result_key(gold):
                wrong.append(sql)
        if not wrong:
            return None
        bad = wrong[0]
        a = llm.chat(model, [{"role": "system", "content": SYS}, {"role": "user", "content": ANNOTATE.format(
            schema=schema, q=(it.question + " " + (it.evidence or "")).strip(), good=it.gold_sql, bad=bad, taxo=_TAXO)}],
            temperature=0.0, seed=seed_base, max_tokens=512, tag="memo.build")
        with lock:
            usage["calls"] += 1
            usage["pt"] += a["prompt_tokens"]
            usage["ct"] += a["completion_tokens"]
        j = jblock(a["text"])
        if not isinstance(j, dict):
            return None
        et = [e for e in (j.get("error_types") or []) if e in ERROR_TYPES]
        if not et:
            return None
        q = (it.question + " " + (it.evidence or "")).strip()
        return {"qid": it.qid, "db_id": it.db_id, "question": q, "good_sql": it.gold_sql, "bad_sql": bad, "error_types": et,
                "suggestion": str(j.get("suggestion", ""))[:400], "key": q + " || " + sql_skeleton(bad)}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        entries = [e for e in ex.map(one, items) if e]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(entries, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    return len(entries), dict(usage)


# ----------------------------------------------------------------------------- the system
class MemoSQL(BaselineSystem):
    name = "memo_sql"
    default_cfg = {
        "preset": "native",            # native | light (overridden by explicit keys)
        "strategies": ["table", "hier", "atomic"],
        "styles": ["cte", "flat", "nested"],
        "react": True,
        "refine_rounds": 3,
        "mem_top": 40, "mem_keep": 5, "icl_k": 5, "max_sub": 5,
        "memory_path": None,           # None -> run without memory (ablation, documented)
        "embedding_model": DEFAULT_EMBED_MODEL,
        "max_tokens": 1536, "parallel": 9,
    }
    PRESETS = {
        "native": {},
        "light": {"strategies": ["table", "hier"], "styles": ["flat"], "refine_rounds": 1},
    }

    def __init__(self, llm, model, cfg=None):
        cfg = dict(cfg or {})
        preset = cfg.get("preset", self.default_cfg["preset"])
        merged = dict(self.PRESETS.get(preset, {}))
        merged.update(cfg)
        super().__init__(llm, model, merged)

    # -- one decomposition path
    def _trace(self, ctx, item, schema, qb, strat, sidx):
        cfg = self.cfg
        raw = ctx.user(DECOMPOSE.format(schema=schema, question=qb, strategy=STRATEGY_TEXT[strat], max_sub=cfg["max_sub"]),
                       system=SYS, seed=ctx.seed + sidx, max_tokens=512, tag="memo.decompose")
        subs = jblock(raw)
        subs = [str(s) for s in subs][: int(cfg["max_sub"])] if isinstance(subs, list) else []
        if not subs:
            return {"strategy": strat, "subs": [], "trace": "(no decomposition: answer the original question directly)", "last_sql": None}
        hist, last_sql = [], None
        for k, sub in enumerate(subs):
            if not cfg["react"]:
                hist.append("%d. %s" % (k + 1, sub))
                continue
            r = ctx.user(REACT_STEP.format(schema=schema, question=qb, sname=strat, history="\n".join(hist) or "(none)", sub=sub),
                         system=SYS, seed=ctx.seed + sidx * 10 + k, max_tokens=int(cfg["max_tokens"]), tag="memo.react")
            sql = extract_sql(r)
            if not sql:
                hist.append("%d. %s -> (no SQL)" % (k + 1, sub))
                continue
            res = ctx.execute(sql)
            ref = ctx.user(REFLECT.format(schema=schema, question=qb, sub=sub, sql=sql, obs=obs_text(res)), system=SYS,
                           seed=ctx.seed + sidx * 10 + k, max_tokens=int(cfg["max_tokens"]), tag="memo.reflect")
            if not re.match(r"\s*OK\b", ref or "") and extract_sql(ref):
                sql2 = extract_sql(ref)
                res = ctx.execute(sql2)
                sql = sql2
            last_sql = sql
            hist.append("%d. %s\n   SQL: %s\n   Result: %s" % (k + 1, sub, one_line(sql), obs_text(res, 3)))
        return {"strategy": strat, "subs": subs, "trace": "\n".join(hist), "last_sql": last_sql}

    def _exemplars(self, ctx, item, tr, pool):
        items, emb = pool
        q = (item.question + " " + (item.evidence or "")).strip()
        key = q + " || " + (sql_skeleton(tr["last_sql"]) if tr["last_sql"] else "")
        v = embed_one(ctx.llm, self.cfg["embedding_model"], key)
        sims = emb @ v
        out = []
        for i in np.argsort(-sims):
            it = items[i]
            if it.db_id == item.db_id or it.question == item.question:
                continue
            out.append(it)
            if len(out) >= int(self.cfg["icl_k"]):
                break
        return "".join("Example question: %s\nExample SQL: %s\n\n" % ((e.question + " " + (e.evidence or "")).strip(), one_line(e.gold_sql))
                       for e in out)

    # -- generation + refinement of one candidate
    def _candidate(self, ctx, item, schema, qb, tr, style, j, pool, mem):
        cfg = self.cfg
        ex = self._exemplars(ctx, item, tr, pool) if int(cfg["icl_k"]) > 0 else ""
        r = ctx.user(FINAL.format(schema=schema, examples=ex, question=qb, sname=tr["strategy"], trace=tr["trace"], style=STYLE_TEXT[style]),
                     system=SYS, seed=ctx.seed + 100 + j, max_tokens=int(cfg["max_tokens"]), tag="memo.final")
        sql = extract_sql(r)
        rounds = 0
        verdicts = []
        if not sql:
            return None, rounds, verdicts
        while rounds < int(cfg["refine_rounds"]):
            res = ctx.execute(sql)
            if res["ok"]:
                cases = ""
                if mem is not None:
                    v = embed_one(ctx.llm, cfg["embedding_model"], (item.question + " " + (item.evidence or "")).strip() + " || " + sql_skeleton(sql))
                    ents = mem.retrieve(v, int(cfg["mem_top"]), int(cfg["mem_keep"]))
                    cases = "\n".join("- Q: %s\n  Wrong: %s\n  Types: %s\n  Advice: %s\n  Fixed: %s" % (
                        e["question"], one_line(e["bad_sql"]), ",".join(e["error_types"]), e["suggestion"], one_line(e["good_sql"])) for e in ents)
                cr = ctx.user(CRITIC.format(schema=schema, question=qb, sql=sql, obs=obs_text(res), cases=cases or "(none)", taxo=_TAXO),
                              system=SYS, seed=ctx.seed + 200 + j * 10 + rounds, max_tokens=512, tag="memo.critic")
                jv = jblock(cr)
                verdicts.append(bool(isinstance(jv, dict) and jv.get("correct")))
                if isinstance(jv, dict) and jv.get("correct"):
                    break
                et = ",".join(e for e in (jv.get("error_types") or []) if e in ERROR_TYPES) if isinstance(jv, dict) else ""
                fb = "Critic: error types %s. %s" % (et or "unknown", (jv.get("suggestion", "") if isinstance(jv, dict) else cr[:300]))
            else:
                fb = "Execution error: %s" % (res.get("error") or "")[:300]
                verdicts.append(False)
            rr = ctx.user(REFINE.format(schema=schema, question=qb, sql=sql, feedback=fb), system=SYS,
                          seed=ctx.seed + 300 + j * 10 + rounds, max_tokens=int(cfg["max_tokens"]), tag="memo.refine")
            new = extract_sql(rr)
            rounds += 1
            if not new or new.strip() == sql.strip():
                break
            sql = new
        return sql, rounds, verdicts

    def solve(self, item, seed, ctx):
        cfg = self.cfg
        r = run_index(seed)
        ctx.seed = 1000 * r
        schema = schema_text(item, quote_keywords=self.qk)
        qb = question_block(item)
        pool = get_pool_emb(ctx.llm, cfg, cfg["embedding_model"]) if int(cfg["icl_k"]) > 0 else None
        mem = get_memory(ctx.llm, cfg["embedding_model"], cfg["memory_path"], cfg.get("emb_cache_dir")) if cfg.get("memory_path") else None
        strategies = list(cfg["strategies"])
        traces = ctx.pmap(lambda s, i: self._trace(ctx, item, schema, qb, s, i), [(s, i) for i, s in enumerate(strategies)])
        jobs = []
        for tr in traces:
            for st in cfg["styles"]:
                jobs.append((tr, st, len(jobs)))
        outs = ctx.pmap(lambda tr, st, j: self._candidate(ctx, item, schema, qb, tr, st, j, pool, mem), jobs)
        cands = [o[0] for o in outs if o[0]]
        if not cands:
            return "", {"iterations": 0, "actions": ["no_candidate"], "error_class": "no_sql", "extra": {}}
        groups = {}
        for s in cands:
            res = ctx.execute(s)
            if res["ok"]:
                groups.setdefault(result_key(res), []).append(s)
        if groups:
            best = max(groups.values(), key=len)       # ties -> first inserted (dicts keep order)
            sql = best[0]
        else:
            sql = cands[0]
        return sql, {"iterations": sum(o[1] for o in outs), "actions": ["decompose x%d" % len(strategies), "candidates x%d" % len(cands), "vote"],
                     "error_class": None,
                     "extra": {"n_candidates": len(cands), "n_groups": len(groups), "refine_rounds_total": sum(o[1] for o in outs),
                               "fallback_traces": sum(1 for t in traces if not t["subs"]), "memory": bool(mem), "samples": ctx.samples}}
