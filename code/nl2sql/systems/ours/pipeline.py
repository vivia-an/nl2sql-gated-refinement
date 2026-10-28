"""The full system: M1 intent -> M3 retrieval (over M2/M6 stores) -> M4 generation -> M5 critic / actor revision ->
policy P0 (patent thresholds) or the learned controller (M7). One code path serves the live system, the E4 stream driver
and the E6 cached-tree builder: `Pipeline.root()` and `Pipeline.act()` create candidate nodes; policies decide which."""
import dataclasses
import hashlib
import json
import re
import threading
import time

import numpy as np

from ... import config as C
from ... import data, db
from ... import meter as _meter
from ... import prompts_util as P
from ...data import Item
from ...llm import LLMClient, LLMError
from ...rl import features as F
from ...util import est_tokens, parse_json, strip_sql
from ..base import System
from . import m1 as M1
from . import m3 as M3
from . import m5 as M5
from . import probe as PR
from . import values as VG
from .corpus import ExperienceCorpus
from .defaults import DEFAULTS
from .knowledge import DbStore, TermIndex
from .route import RouteScorer, adaptive_decide


def resolved_cfg(cfg):
    return C.deep_update(DEFAULTS, cfg or {})


class Resources:
    """Process-wide shared state: db stores, pool corpora, term indexes (thread-safe lazy construction)."""

    def __init__(self, llm, cfg):
        self.llm, self.cfg = llm, cfg
        self.embed_model = cfg["models"]["embed"]
        self.rerank_model = cfg["models"]["rerank"]
        self._stores = {}
        self._pools = {}
        self._lock = threading.RLock()
        self._qv = {}
        self.terms = None
        self.external_corpus = None      # set by the stream driver (per-db corpora)

    def embed_one(self, text):
        with self._lock:
            v = self._qv.get(text)
        if v is None:
            v = self.llm.embed_np([text], self.embed_model)[0]
            v = v / max(float(np.linalg.norm(v)), 1e-9)
            with self._lock:
                if len(self._qv) > 20000:
                    self._qv.clear()
                self._qv[text] = v
        return v

    def db_store(self, item):
        key = item.db_path
        with self._lock:
            s = self._stores.get(key)
            if s is None:
                s = self._stores[key] = DbStore(self.llm, self.embed_model, self.rerank_model, item, self.cfg)
            return s

    def pool_name(self, item):
        if self.cfg.get("pool"):
            return self.cfg["pool"]
        ds = item.qid.split(":")[0]
        return "spider_train_pool" if ds.startswith("spider") or ds == "ambiqt" else "bird_train_pool"

    def ensure_terms(self, item):
        if self.cfg["m2"]["terms"] and self.terms is None:
            name = self.pool_name(item)
            with self._lock:
                if self.terms is None:
                    items = data.load_split(name, with_gold=True)
                    self.terms = TermIndex(self.llm, self.embed_model, items, name)

    def corpus_for(self, item):
        name = self.pool_name(item)
        if self.external_corpus is not None:
            self.ensure_terms(item)
            return self.external_corpus.get(item.db_id) if isinstance(self.external_corpus, dict) else self.external_corpus
        with self._lock:
            if name not in self._pools:
                m6 = self.cfg["m6"]
                items = data.load_split(name, with_gold=True)
                if m6.get("path") and name == "bird_train_pool":
                    from ... import paths
                    corp = ExperienceCorpus.load(str(paths.REPO / m6["path"]) if not str(m6["path"]).startswith("/") else m6["path"],
                                                 self.llm, self.embed_model, m6["policy"] if m6["policy"] != "raw" else "ws_full",
                                                 m6.get("params"))
                else:
                    corp = ExperienceCorpus(self.llm, self.embed_model, m6["policy"], m6.get("params"))
                    if self.cfg["m2"]["history"]:
                        corp.add_history(items, vecs=_pool_vecs(self.llm, self.embed_model, items, name))
                self._pools[name] = (corp, items)
                if self.cfg["m2"]["terms"] and self.terms is None:
                    self.terms = TermIndex(self.llm, self.embed_model, items, name)
            return self._pools[name][0]


def _pool_vecs(llm, model, items, name):
    from .knowledge import _embed_cached
    return _embed_cached(llm, model, [i.question for i in items], "hist_" + name)


class Pipeline:
    def __init__(self, llm, model, cfg, resources=None, critic_model=None, tier2_model=None):
        self.llm = llm
        self.cfg = resolved_cfg(cfg)
        self.models = [model]                       # tier index 0 = tier1 (the run's model)
        t2 = tier2_model or self.cfg["models"].get("tier2")
        self.tier2 = C.resolve_model(t2) if t2 else None
        cm = self.cfg["models"]["critic"]
        self.critic_model = model if cm in ("self", None) else C.resolve_model(cm)
        # X5 switches (all default to the legacy behaviour); validated before the critic is built so config errors fail fast
        m5 = self.cfg["m5"]
        if m5.get("policy", "p0") not in ("p0", "gated"):
            raise ValueError("m5.policy must be p0 or gated, got %r" % m5.get("policy"))
        self.gated = m5.get("policy", "p0") == "gated"
        if m5.get("revise_feedback", "exec") not in ("exec", "blind"):
            raise ValueError("m5.revise_feedback must be exec or blind, got %r" % m5.get("revise_feedback"))
        self.blind = m5.get("revise_feedback", "exec") == "blind"
        if self.blind and not self.gated:
            raise ValueError("m5.revise_feedback blind needs m5.policy gated")
        self.qk = bool((self.cfg.get("ddl") or {}).get("quote_keywords"))
        self.vg_on = bool(self.cfg["m2"].get("value_grounding"))        # X9: stored-literal hints in the prompts (validated here so a bad m2.vg fails fast)
        self.vg_params = VG.params(self.cfg) if self.vg_on else None
        self.router = RouteScorer(self.cfg.get("route"))
        if self.gated and m5["variant"] == "v5":
            raise ValueError("m5.policy gated does not combine with the learned critic v5 (its features need the LLM critic)")
        if self.gated and "v3" in self.router.needs and not m5.get("score_for_routing"):
            raise ValueError("route.score %r needs the V3 score, but policy gated skips the semantic critic: set m5.score_for_routing: true" % self.router.kind)
        # X7 switches (all default off)
        self.probe_on = bool(m5.get("probe_values"))
        self.gate_zero = bool(m5.get("gate_zero"))
        if (self.probe_on or self.gate_zero) and not self.gated:
            raise ValueError("m5.probe_values / m5.gate_zero need m5.policy gated")
        if self.probe_on and not (m5.get("gate_empty") or self.gate_zero):
            raise ValueError("m5.probe_values needs m5.gate_empty (or m5.gate_zero) true: it augments the revision of an empty / zero result")
        ad = (self.cfg["route"].get("adaptive") or {})
        self.adaptive = bool(ad.get("enabled"))
        self.ad_cfg = {"theta_fast": float(ad.get("theta_fast", 1.0)), "tiebreak": ad.get("tiebreak", "root"), "empty": ad.get("empty", "vote")}
        if self.adaptive:
            if not self.gated:
                raise ValueError("route.adaptive needs m5.policy gated (execution-only node scores)")
            if int(((self.cfg["route"].get("consistency") or {}).get("k")) or 0) <= 0:
                raise ValueError("route.adaptive needs route.consistency.k > 0")
            if self.probe_on or m5.get("arbitrate"):
                raise ValueError("route.adaptive does not combine with m5.probe_values / m5.arbitrate (it makes no revision call)")
            if not 0.0 <= self.ad_cfg["theta_fast"] <= 1.0:
                raise ValueError("route.adaptive.theta_fast must be in [0, 1], got %r" % self.ad_cfg["theta_fast"])
            adaptive_decide([None, None], None, self.ad_cfg["theta_fast"], self.ad_cfg["tiebreak"], self.ad_cfg["empty"])    # validates the enums
        self.res = resources or Resources(llm, self.cfg)
        self.critic = M5.Critic(llm, self.critic_model, self.cfg)

    # ------------------------------------------------------------- helpers
    def _msgs(self, user):
        return [{"role": "system", "content": P.render("common.system")}, {"role": "user", "content": user}]

    def _exec(self, item, sql):
        return db.execute(item.db_path, sql, timeout_s=30, max_rows=10000)

    def _evaluate(self, item, sql, ctx, graph, seed, tier, prev=None, usage=None, qf=None, action=None):
        res = self._exec(item, sql)
        rc = self.res.db_store(item).schema["row_counts"]
        key = db.rows_key(res["rows"]) if res["ok"] else None
        meta = None
        if self.critic.learned is not None:       # only the learned critic (V5) uses candidate context features
            meta = {"prev_score": prev["critic"].get("score_v3", prev["critic"]["score"]) if prev else None,
                    "changed": (key != prev.get("rkey")) if prev else None,
                    "k": (prev["k"] + 1) if prev else 0, "tier": tier, "action": action, "qf": qf}
        if self.gated:        # policy gated: execution-grounded sub-scores only, the semantic critic is not called in the loop
            crit = self.critic.score_exec(item, sql, res, rc)
        else:
            crit = self.critic.score(item, sql, ctx, graph, res, seed, rc, meta=meta)
        node = {"sql": sql, "res": res, "critic": crit, "tier": tier, "n_rows": len(res["rows"]) if res["ok"] else 0,
                "n_cols": len(res.get("cols") or []), "prev_score": prev["critic"]["score"] if prev else None,
                "changed": (key != prev.get("rkey")) if prev else False, "rkey": key, "ctx": ctx, "graph": graph}
        if self.probe_on:
            node["probe"] = {"n": 0, "cols": []}            # X7: what the revision prompt of THIS node carried (set by _revise)
        return node

    def _qr(self):
        """Optional prompt rule line (config ddl.quote_keywords); "" keeps every legacy prompt byte-identical."""
        return P.QUOTE_RULE if self.qk else ""

    def _gen_user(self, item, ctx, graph):
        gt = M1.graph_text(graph) if (graph and self.cfg["m4"]["graph_in_prompt"] and self.cfg["m1"]["enabled"]) else ""
        return P.render("gen.user", sections=ctx.sections(), question=item.question, evidence=item.evidence or "(none)", graph=gt,
                        quote_rule=self._qr(), value_hints=ctx.vh_text)

    def _gen_msgs(self, item, ctx, graph):
        user = self._gen_user(item, ctx, graph)
        if self.cfg["m5"].get("verbalized_confidence"):
            user += "\n" + P.render("verbal.suffix")
        return self._msgs(user)

    def _generate(self, item, ctx, graph, model, T, seed, tag="gen"):
        r = self.llm.chat(model, self._gen_msgs(item, ctx, graph), temperature=T, seed=seed, max_tokens=self.cfg["max_tokens"]["generate"], tag=tag)
        return self._parse_gen(r["text"])

    def _parse_gen(self, text):
        conf = None
        if self.cfg["m5"].get("verbalized_confidence") and "CONFIDENCE:" in text:
            text, c = text.rsplit("CONFIDENCE:", 1)
            try:
                conf = float(c.strip().split()[0].strip("%"))
            except Exception:
                conf = None
        return strip_sql(text), conf

    def _revise(self, item, node, model, T, seed, tag="rev"):
        crit = node["critic"]
        if self.gated:        # feedback = the actual execution error text (or the fact that the result was empty), no critic text
            gt = M1.graph_text(node["graph"]) if (node["graph"] and self.cfg["m4"]["graph_in_prompt"] and self.cfg["m1"]["enabled"]) else ""
            zero = False
            if self.blind:    # X6 (P5 control): same prompt, same gate, but neither the error text (preview) nor the feedback carries it
                preview, feedback = M5.BLIND_PREVIEW, M5.BLIND_FEEDBACK
            else:
                zero = self.gate_zero and crit["status"] == "ok_nonempty" and M5.is_zero_result(node["res"])
                preview, feedback = M5.exec_summary(node["res"]), M5.exec_feedback_text(node["res"], crit["status"], zero=zero)
            probes = self._probe(item, node) if (self.probe_on and not self.blind and (crit["status"] == "ok_empty" or zero)) else []
            if probes:        # X7: observed values of the filtered columns; own template, so rev_exec.user stays byte-identical
                node["probe"] = {"n": len(probes), "cols": PR.probe_cols(probes)}
                print("[lineage] probe %s: %d cols %s" % (item.qid, len(probes), node["probe"]["cols"]), flush=True)
                user = P.render("rev_exec_probe.user", sections=node["ctx"].sections(), question=item.question, evidence=item.evidence or "(none)",
                                graph=gt, sql=node["sql"], preview=preview, quote_rule=self._qr(), feedback=feedback,
                                probe=PR.probe_text(probes, self.cfg["m5"].get("probe_n", 8)), value_hints=node["ctx"].vh_text)
            else:
                user = P.render("rev_exec.user", sections=node["ctx"].sections(), question=item.question, evidence=item.evidence or "(none)",
                                graph=gt, sql=node["sql"], preview=preview, quote_rule=self._qr(), feedback=feedback, value_hints=node["ctx"].vh_text)
            r = self.llm.chat(model, self._msgs(user), temperature=T, seed=seed, max_tokens=self.cfg["max_tokens"]["revise"], tag=tag)
            return strip_sql(r["text"])
        mode = "optional improvements" if crit["score"] >= self.cfg["m5"]["theta_fb"] else "must fix"
        gt = M1.graph_text(node["graph"]) if (node["graph"] and self.cfg["m4"]["graph_in_prompt"] and self.cfg["m1"]["enabled"]) else ""
        user = P.render("rev.user", sections=node["ctx"].sections(), question=item.question, evidence=item.evidence or "(none)", graph=gt,
                        sql=node["sql"], preview=M5.exec_summary(node["res"]), score="%.1f" % crit["score"], mode=mode,
                        feedback=M5.feedback_text(crit, node["res"]), quote_rule=self._qr(), value_hints=node["ctx"].vh_text)
        r = self.llm.chat(model, self._msgs(user), temperature=T, seed=seed, max_tokens=self.cfg["max_tokens"]["revise"], tag=tag)
        return strip_sql(r["text"])

    def _probe(self, item, node):
        """X7: observed values of the short text columns the (empty-result) SQL filters with literals; [] when none applies. Read-only, time-guarded."""
        m5 = self.cfg["m5"]
        try:
            schema = self.res.db_store(item).schema
            return PR.probe_failed_sql(item.db_path, node["sql"], schema, n=int(m5.get("probe_n", 8)), max_len=int(m5.get("probe_max_len", 40)),
                                       max_cols=int(m5.get("probe_max_cols", 4)), timeout_s=float(m5.get("probe_timeout_s", 5.0)),
                                       near=bool(m5.get("probe_near", False)))
        except Exception as e:          # probing is an aid: never fail the question because of it
            print("[lineage] probe failed %s: %s" % (item.qid, type(e).__name__), flush=True)
            return []

    def _actor(self, tier):
        return self.tier2 if (tier == 1 and self.tier2) else self.models[0]

    # ------------------------------------------------------------- nodes
    def root(self, item, seed=0, T=None, corpus=None, exclude_ids=()):
        T = self.cfg["temperature"] if T is None else T
        out = {"m1_ok": None}
        with _meter.Meter() as m:
            graph = None
            if self.cfg["m1"]["enabled"]:
                graph, ok = M1.intent(self.llm, self.models[0], item, seed, self.cfg)
                out["m1_ok"] = ok
            corpus = corpus if corpus is not None else self.res.corpus_for(item)
            self.res.ensure_terms(item)
            ctx = M3.retrieve(self.res, item, graph, self.cfg, corpus, exclude_ids=exclude_ids)
            sql, conf = self._generate(item, ctx, graph, self.models[0], T, seed)
            qf0 = F.question_features(item, graph, len(ctx.tables), sum(len(self.res.db_store(item).schema["tables"][t]) for t in ctx.tables if t in self.res.db_store(item).schema["tables"]))
            node = self._evaluate(item, sql, ctx, graph, seed, 0, qf=qf0, action="root")
        node.update({"usage": m.summary(), "action": "root", "conf": conf, "n_rr": 0, "k": 0, "calls_used": 0, "m1_ok": out["m1_ok"],
                     "qf": qf0})
        return node

    def act(self, item, node, action, T=0.0, seed=0, corpus=None):
        """Create the child of `node` for a generating action. Returns child node with its own usage."""
        with _meter.Meter() as m:
            if action == "revise":
                sql = self._revise(item, node, self._actor(node["tier"]), T, seed, tag="rev_t2" if node["tier"] else "rev")
                ctx, graph, tier = node["ctx"], node["graph"], node["tier"]
            elif action == "escalate":
                if not self.tier2:
                    raise ValueError("escalate without tier2 model")
                sql = self._revise(item, node, self.tier2, T, seed, tag="rev_t2")
                ctx, graph, tier = node["ctx"], node["graph"], 1
            elif action == "re_retrieve":
                issues = " ".join(i.get("description", "") if isinstance(i, dict) else str(i) for i in node["critic"].get("issues") or [])
                corpus = corpus if corpus is not None else self.res.corpus_for(item)
                ctx = M3.retrieve(self.res, item, node["graph"], self.cfg, corpus, extra_query=issues[:400],
                                  exclude_ids=set(node["ctx"].exemplar_ids), extra_tables=2)
                graph, tier = node["graph"], node["tier"]
                sql, _ = self._generate(item, ctx, graph, self._actor(tier), T, seed, tag="gen_rr_t2" if tier else "gen_rr")
            else:
                raise ValueError(action)
            child = self._evaluate(item, sql, ctx, graph, seed, tier, prev=node, qf=node["qf"], action=action)
        child.update({"usage": m.summary(), "action": action, "n_rr": node["n_rr"] + (1 if action == "re_retrieve" else 0), "k": node["k"] + 1,
                      "calls_used": node["calls_used"] + m.calls, "qf": node["qf"], "m1_ok": node.get("m1_ok"), "parent_sql": node["sql"]})
        return child

    # ------------------------------------------------------------- X5: routing signals, arbitration
    def score_final_v3(self, item, node, seed):
        """policy gated + m5.score_for_routing: ONE semantic critic call on the final answer (not in the loop). The node keeps
        its execution sub-scores; `critic` becomes the full V3 dict (+ score_exec)."""
        rc = self.res.db_store(item).schema["row_counts"]
        crit = self.critic.score(item, node["sql"], node["ctx"], node["graph"], node["res"], seed, rc)
        crit["score_exec"] = node["critic"]["score"]
        node["critic"] = crit
        return crit

    def draw_samples(self, item, root, seed):
        """The k extra T = route.consistency.temperature samples of the INITIAL generation prompt (same M1 graph and M3 context as the root;
        n <= n_per_call samples per API request, seeds seed+1+chunk) -> list of SQL strings."""
        cc = self.cfg["route"]["consistency"]
        k, T, per = int(cc.get("k", 0)), float(cc.get("temperature", 0.7)), max(1, int(cc.get("n_per_call", 4)))
        if k <= 0:
            return []
        msgs = self._gen_msgs(item, root["ctx"], root["graph"])
        texts, left, i = [], k, 0
        while left > 0:
            n = min(per, left)
            r = self.llm.chat(self.models[0], msgs, temperature=T, seed=seed + 1 + i, max_tokens=self.cfg["max_tokens"]["generate"], n=n, tag="cons")
            t = r["text"]
            texts.extend(t if isinstance(t, list) else [t])
            left -= n
            i += 1
        return [self._parse_gen(t)[0] for t in texts[:k]]

    def sample_results(self, item, sqls):
        """sql -> execution result dict (each distinct SQL executed once; an empty SQL fails)."""
        out = {}
        for sql in sqls:
            if sql not in out:
                out[sql] = self._exec(item, sql) if sql.strip() else {"ok": False, "rows": [], "error": "empty sql"}
        return out

    @staticmethod
    def agreement(fkey, sample_keys):
        """-> (fraction of samples whose key equals `fkey`, n_samples, n_agree, per-sample 0/1 list); a failed (None) key never agrees."""
        bits = [int(fkey is not None and k is not None and k == fkey) for k in sample_keys]
        n, agree = len(bits), sum(bits)
        return (agree / float(n) if n else None), n, agree, bits

    def consistency(self, item, root, final, seed):
        """k extra samples (T = route.consistency.temperature, default 0.7) of the INITIAL generation prompt (same M1 graph and M3
        context as the root; n <= n_per_call samples per API request, seeds seed+1+chunk). Returns (fraction of samples whose execution
        result equals the final result, n_samples, n_agree, per-sample 0/1 list). A failed final or sample execution never agrees (two errors are not agreement)."""
        if int(self.cfg["route"]["consistency"].get("k", 0)) <= 0:
            return None, 0, 0, []
        sqls = self.draw_samples(item, root, seed)
        res = self.sample_results(item, sqls)
        keys = [db.rows_key(res[q]["rows"]) if res[q]["ok"] else None for q in sqls]
        return self.agreement(final["rkey"] if final["res"]["ok"] else None, keys)

    def adopt_sample(self, item, root, sql, seed, action="sample"):
        """Node for an extra sample (not a revision): evaluated like a root child, parent = root; no LLM call under policy gated."""
        with _meter.Meter() as m:
            node = self._evaluate(item, sql, root["ctx"], root["graph"], seed, 0, prev=root, qf=root["qf"], action=action)
        node.update({"usage": m.summary(), "action": action, "n_rr": 0, "k": 1, "calls_used": root["calls_used"] + m.calls, "qf": root["qf"],
                     "m1_ok": root.get("m1_ok"), "parent_sql": root["sql"], "conf": None})
        return node

    def arbitrate(self, item, root, final, seed=0):
        """One LLM judge call (critic model) that sees both candidates with their result previews, only when root and final are
        different nodes, both execute, and their results differ. Order A/B is flipped by a hash of the qid (position-bias control).
        Returns (chosen node, info dict). An unparsable answer keeps `final`."""
        if final is root or not (root["res"]["ok"] and final["res"]["ok"]) or root["rkey"] == final["rkey"]:
            return final, {"arbitrated": 0}
        flip = int(hashlib.sha256(item.qid.encode()).hexdigest(), 16) % 2
        a, b = (final, root) if flip else (root, final)
        gt = M1.graph_text(root["graph"]) if (root["graph"] and self.cfg["m4"]["graph_in_prompt"] and self.cfg["m1"]["enabled"]) else ""
        user = P.render("arbitrate.user", sections=root["ctx"].sections_core(), question=item.question, evidence=item.evidence or "(none)",
                        graph=gt, sql_a=a["sql"], preview_a=M5.exec_summary(a["res"]), sql_b=b["sql"], preview_b=M5.exec_summary(b["res"]))
        r = self.llm.chat(self.critic_model, self._msgs(user), temperature=0.0, seed=seed,
                          max_tokens=self.cfg["max_tokens"].get("arbitrate", 256), tag="arb")
        ch = parse_arbitration(r["text"])
        info = {"arbitrated": 1, "arb_parse_ok": ch is not None, "arb_flip": flip}
        if ch is None:
            info["arb_choice"] = "final"
            return final, info
        picked = a if ch == "A" else b
        info["arb_choice"] = "root" if picked is root else "final"
        return picked, info

    def state(self, node, budget=8):
        ctx = node["ctx"]
        ws = [e.WS for e in ctx.examples] or [6.3]
        return F.build_state(node, node["qf"], node["k"], node["n_rr"], 1.0 - node["calls_used"] / float(budget), node["tier"],
                             ctx.ex_sims, float(np.mean(ws)))


def parse_arbitration(text):
    """-> "A" | "B" | None from the judge's answer (JSON {"choice": ...}; falls back to a regex for truncated JSON)."""
    j = parse_json(text)
    if isinstance(j, dict):
        c = str(j.get("choice", "")).strip().upper()[:1]
        if c in ("A", "B"):
            return c
    m = re.search(r'"?choice"?\s*[:=]\s*"?([AaBb])\b', text or "")
    return m.group(1).upper() if m else None


def cand_record(node):
    c = node["critic"]
    rec = {"sql": node["sql"], "action": node["action"], "tier": node["tier"], "syntax": c["syntax"], "semantic": c["semantic"],
            "efficiency": c["efficiency"], "score": c["score"], "status": c["status"], "n_rows": node["n_rows"], "n_cols": node["n_cols"],
            "k": node["k"], "exemplar_ids": list(node["ctx"].exemplar_ids), "prompt_tokens": node["usage"]["prompt_tokens"],
            "completion_tokens": node["usage"]["completion_tokens"], "top_sim": (node["ctx"].ex_sims[0] if node["ctx"].ex_sims else None),
            "mean_ws": float(np.mean([e.WS for e in node["ctx"].examples])) if node["ctx"].examples else None,
            # additive (V5 learned-critic features); older readers ignore them
            "mean_sim": float(np.mean(node["ctx"].ex_sims)) if node["ctx"].ex_sims else None, "qf": node.get("qf"),
            "prev_score": node.get("prev_score"), "changed": bool(node.get("changed")), "full_schema": bool(node["ctx"].full_schema),
            "plan_scans": c.get("plan_scans"), "plan_temp": c.get("plan_temp"), "plan_corr": c.get("plan_corr"), "plan_search": c.get("plan_search"),
            "score_kind": c.get("score_kind", "v3")}
    if "probe" in node:          # X7 (only with m5.probe_values): the value probe that went into the prompt that REVISED this candidate
        rec["probed"] = node["probe"]["n"]
        rec["probe_cols"] = node["probe"]["cols"]
    return rec


class FullSystem(System):
    name = "full"
    cfg_override = {}
    force_p0 = False

    def __init__(self, llm, model, cfg):
        super().__init__(llm, model, cfg)
        c = C.deep_update(DEFAULTS, self.cfg_override)
        c = C.deep_update(c, cfg or {})
        self.rcfg = c
        self.pipe = Pipeline(llm, model, c)
        self.controller = None
        ctl = (c["m7"].get("controller") or {})
        if ctl.get("path") and not self.force_p0:
            from .controller import LearnedController
            self.controller = LearnedController.load(ctl["path"], greedy=ctl.get("greedy", True))

    # public hook for the E4 stream driver: run an item against a given corpus; returns result + best node
    def run(self, item, seed=0, corpus=None):
        cfg = self.rcfg
        if cfg.get("no_evidence"):
            item = dataclasses.replace(item, evidence="")
        m5 = cfg["m5"]
        root = self.pipe.root(item, seed, corpus=corpus)
        nodes = [root]
        actions = []
        if not m5["enabled"]:
            final, abstain = root, False
        elif self.controller is not None:
            final, abstain, actions = self._run_rl(item, root, nodes, seed, corpus)
        elif self.pipe.adaptive:
            final, abstain, actions = self._run_adaptive(item, root, nodes, seed)
        elif self.pipe.gated:
            final, abstain, actions = self._run_gated(item, root, nodes, seed, corpus)
        else:
            final, abstain, actions = self._run_p0(item, root, nodes, seed, corpus)
        final, abstain, extra = self._post(item, root, nodes, final, abstain, actions, seed)
        out = {"final_sql": final["sql"], "abstained": abstain, "critic": {k: final["critic"][k] for k in ("syntax", "semantic", "efficiency", "score")},
               "iterations": len(nodes) - 1, "actions": actions, "m1_ok": root.get("m1_ok"), "final_status": final["critic"].get("status"),
               "exemplar_ids": list(final["ctx"].exemplar_ids), "confidence": final["critic"]["score"],
               "verbalized_confidence": root.get("conf"),
               # E8 linking recall inputs (root context = the M3 output the first generation saw)
               "ctx_tables": list(root["ctx"].tables), "ctx_full_schema": bool(root["ctx"].full_schema),
               "ctx_tokens": est_tokens(root["ctx"].ddl)}
        out.update(extra)
        if self.pipe.probe_on:           # X7 value probing: totals over the revision chain
            cols = []
            for n in nodes:
                cols.extend(c for c in n["probe"]["cols"] if c not in cols)
            out["probed"] = sum(n["probe"]["n"] for n in nodes)
            out["probe_cols"] = cols
        if self.pipe.vg_on:              # X9 value grounding: what the final answer's prompt carried and whether the stored literals reached the SQL
            out["value_hints"] = VG.log_record(final["ctx"], root["sql"], final["sql"])
        ad = root.get("adaptive")
        if ad:                           # X7 adaptive path (route.adaptive)
            out.update(ad["public"])
        if cfg["record_candidates"]:
            out["candidates"] = [cand_record(n) for n in nodes] + ([cand_record(n) for n in ad["sample_nodes"]] if ad else [])
        out["_nodes"] = nodes
        out["_final"] = final
        return out

    def _run_p0(self, item, root, nodes, seed, corpus):
        m5 = self.rcfg["m5"]
        cur = root
        actions = []
        while True:
            s = cur["critic"]["score"]
            if s >= m5["theta_acc"]:
                actions.append("accept")
                break
            if cur["k"] >= m5["t_max"]:
                break
            child = self.pipe.act(item, cur, "revise", T=self.rcfg["temperature"], seed=seed, corpus=corpus)
            actions.append("revise")
            if child["sql"].strip() == cur["sql"].strip():
                actions.append("revise_noop")
                nodes.append(child)
                break
            nodes.append(child)
            cur = child
        best = max(nodes, key=lambda n: (n["critic"]["score"], -n["k"]))
        abstain = bool(m5["route_human"] and best["critic"]["score"] < m5["theta_h"])
        if actions and actions[-1] != "accept":
            actions.append("stop_best")
        return best, abstain, actions

    def _run_gated(self, item, root, nodes, seed, corpus):
        """Policy `gated` (HEADROOM row 1): revise only while the current SQL fails to execute / times out (and, m5.gate_empty, returns an
        empty result); no semantic critic call, feedback = the execution error text. Final = the best node by execution status
        (non-empty > empty > error), ties to the earliest, so an erroring revision never replaces an executable root."""
        m5 = self.rcfg["m5"]
        cur = root
        actions = []
        while True:
            if not M5.gate_node(cur, m5.get("gate_empty", False), m5.get("gate_zero", False)):
                actions.append("accept")
                break
            if cur["k"] >= m5["t_max"]:
                break
            child = self.pipe.act(item, cur, "revise", T=self.rcfg["temperature"], seed=seed, corpus=corpus)
            actions.append("revise")
            if child["sql"].strip() == cur["sql"].strip():
                actions.append("revise_noop")
                nodes.append(child)
                break
            nodes.append(child)
            cur = child
        best = max(nodes, key=lambda n: (M5.node_rank(n, m5.get("gate_zero", False)), -n["k"]))
        if actions and actions[-1] != "accept":
            actions.append("stop_best")
        # no V3 score exists here: abstention is decided in _post from the routing score (route.score exec / v3 / consistency ...)
        return best, False, actions

    def _run_adaptive(self, item, root, nodes, seed):
        """X7 confidence-routed compute (route.adaptive; policy gated): k extra T = 0.7 samples of the root prompt (route.consistency, one API request
        for k <= n_per_call), executed. consistency (share of samples with the root's result) >= theta_fast -> the root is the answer ("fast");
        otherwise the execution-result majority over root + samples ("vote"; ties -> root, or the V3 score of the tied groups with
        tiebreak v3). NO revision call. Everything needed afterwards (sample keys for the routing consistency, sample nodes for cands.jsonl,
        public preds fields) is stored in root["adaptive"]."""
        pipe, ac = self.pipe, self.pipe.ad_cfg
        sqls = pipe.draw_samples(item, root, seed)
        res = pipe.sample_results(item, sqls)
        skeys = [db.rows_key(res[q]["rows"]) if res[q]["ok"] else None for q in sqls]
        keys = [root["rkey"] if root["res"]["ok"] else None] + skeys
        empties = [root["critic"]["status"] == "ok_empty"] + [(M5.exec_status(res[q]) == "ok_empty") if res[q]["ok"] else False for q in sqls]
        d = adaptive_decide(keys, empties, ac["theta_fast"], ac["tiebreak"], ac["empty"])
        snodes = [pipe.adopt_sample(item, root, q, seed) for q in sqls] if self.rcfg["record_candidates"] else []
        chosen, v3 = d["chosen"], None
        def node_of(i):
            if i == 0:
                return root
            if snodes:
                return snodes[i - 1]
            return pipe.adopt_sample(item, root, sqls[i - 1], seed)
        if d["path"] == "vote" and d["tied"]:            # tiebreak v3: one critic call per tied group representative
            v3 = {}
            for i in d["tied"]:
                nd = node_of(i)
                v3[i] = pipe.critic.score(item, nd["sql"], nd["ctx"], nd["graph"], nd["res"], seed, pipe.res.db_store(item).schema["row_counts"])["score"]
            chosen = max(d["tied"], key=lambda i: (v3[i], -i if i != 0 else 1))          # best V3; ties -> root, then the earliest sample
        final = node_of(chosen)           # a picked sample is NOT a revision: `nodes` keeps only the root, the samples go to cands.jsonl (action "sample")
        actions = ["adaptive_" + d["path"]] + (["pick_root" if chosen == 0 else "pick_sample"] if d["path"] == "vote" else [])
        pub = {"adaptive_path": d["path"], "adaptive_cons": d["cons"], "adaptive_n": d["n"], "adaptive_choice": "root" if chosen == 0 else "sample",
               "adaptive_sizes": d["sizes"], "adaptive_tie": d["tie"]}
        if v3 is not None:
            pub["adaptive_v3"] = {str(i): v3[i] for i in v3}
        print("[lineage] adaptive %s: path=%s cons=%s choice=%s" % (item.qid, d["path"], d["cons"], pub["adaptive_choice"]), flush=True)
        root["adaptive"] = {"public": pub, "sample_keys": skeys, "sample_nodes": snodes}
        return final, False, actions

    def _post(self, item, root, nodes, final, abstain, actions, seed):
        """X5 post-processing of the loop result: optional arbitration, V3 score of the final answer (gated), consistency sampling, routing
        score, abstention from the routing score. With every switch at its default this is the identity and adds no key (legacy rows)."""
        pipe, m5, rcfg = self.pipe, self.rcfg["m5"], self.rcfg["route"]
        rt = pipe.router
        extra = {}
        swapped = False
        if m5.get("arbitrate") and m5["enabled"] and self.controller is None:
            chosen, info = pipe.arbitrate(item, root, final, seed)
            extra.update(info)
            if info.get("arbitrated"):
                actions.append("arbitrate_" + info["arb_choice"])
            swapped = chosen is not final
            final = chosen
        k = rt.cons_k
        if rt.legacy and not swapped and not k and not pipe.gated:
            return final, abstain, extra
        if pipe.gated and "v3" in rt.needs and final["critic"].get("semantic") is None:
            pipe.score_final_v3(item, final, seed)
        cons = None
        if k:
            if root.get("adaptive"):        # X7: the samples were drawn and executed by _run_adaptive; consistency is relative to the FINAL answer
                cons, n, agree, bits = pipe.agreement(final["rkey"] if final["res"]["ok"] else None, root["adaptive"]["sample_keys"])
            else:
                cons, n, agree, bits = pipe.consistency(item, root, final, seed)
            extra.update({"consistency": cons, "n_samples": n, "n_agree": agree, "cons_bits": bits})
        crit = final["critic"]
        feats = {"v3": crit["score"] if crit.get("semantic") is not None else None, "exec": crit["syntax"], "consistency": cons,
                 "nonempty": 1.0 if crit.get("status") == "ok_nonempty" else 0.0}
        if rt.legacy and not swapped and not pipe.gated:
            return final, abstain, extra                    # consistency recorded only; routing score stays the V3 score
        rs = rt.score(feats) if (rt.coef is not None or all(feats.get(p) is not None for p in rt.parts)) else None
        if rs is None:                                      # a component is unavailable (e.g. the API returned no consistency sample)
            extra.update({"route_score": None, "route_kind": rt.kind})
            return final, abstain, extra
        extra.update({"route_score": rs, "route_kind": rt.kind, "confidence": rs})
        theta = rcfg.get("theta_h")
        if theta is None and rt.kind == "v3":
            theta = m5["theta_h"]
        abstain = bool(m5["route_human"] and theta is not None and rs < float(theta))
        return final, abstain, extra

    def _run_rl(self, item, root, nodes, seed, corpus):
        ctl = self.rcfg["m7"]
        cur = root
        actions = []
        budget = ctl["call_budget"]
        mask_escalate = ctl["actions"] == 4 or not self.pipe.tier2
        for step in range(ctl["horizon"]):
            st = self.pipe.state(cur, budget)
            can_gen = cur["k"] < ctl["max_gen_depth"] and cur["calls_used"] + 2 <= budget
            mask = [True, can_gen, can_gen, can_gen and not mask_escalate, True]
            a = F.ACTIONS[self.controller.act(st, mask)]
            actions.append(a)
            if a == "accept":
                return cur, False, actions
            if a == "route_human":
                best = max(nodes, key=lambda n: (n["critic"]["score"], -n["k"]))
                return best, True, actions
            child = self.pipe.act(item, cur, a, T=self.rcfg["temperature"], seed=seed, corpus=corpus)
            nodes.append(child)
            cur = child
        best = max(nodes, key=lambda n: (n["critic"]["score"], -n["k"]))
        actions.append("forced_best")
        return best, False, actions

    def plan_requests(self, item, seed=0):
        return None

    def warm(self, items):
        seen = set()
        for it in items:
            if it.db_path not in seen:
                seen.add(it.db_path)
                self.pipe.res.db_store(it)
            self.pipe.res.corpus_for(it)
            self.pipe.res.ensure_terms(it)
        if self.pipe.vg_on:              # X9: build the value indexes up front (one database after the other; a 5 GB database costs up to budget_s once per worker machine cache)
            done = set()
            for it in items:
                if it.db_path not in done:
                    done.add(it.db_path)
                    try:
                        VG.get_index(it.db_path, self.pipe.res.db_store(it).schema, self.pipe.vg_params)
                    except Exception as e:
                        print("[lineage] value index failed %s: %s" % (it.db_id, type(e).__name__), flush=True)


class FullFixed(FullSystem):
    name = "full_fixed"
    force_p0 = True


class FullGated(FullSystem):
    """M1-M6 + execution-gated loop (X5): config m5.policy gated is forced; the other X5 switches come from the config."""
    name = "full_gated"
    force_p0 = True
    cfg_override = {"m5": {"policy": "gated"}}


class FullGatedBlind(FullSystem):
    """X6, P5 control (SAP 13.3): identical to full_gated (same gate, same call count, same roots) but the revision prompt carries NO execution
    error text: blind retry. Config m5.revise_feedback: blind is forced."""
    name = "full_gated_blind"
    force_p0 = True
    cfg_override = {"m5": {"policy": "gated", "revise_feedback": "blind"}}


class FullRL(FullSystem):
    name = "full_rl"


class FullX(FullSystem):
    name = "full_x"
    cfg_override = {"models": {"critic": "dsv41flash"}}


class ZeroShot(System):
    """DDL + sample values + evidence -> SQL, one call. Registered as zs_ours (B1's `zs` takes the name `zs` if present)."""
    name = "zs_ours"

    def __init__(self, llm, model, cfg):
        super().__init__(llm, model, cfg)
        self.cfg = C.deep_update(DEFAULTS, cfg or {})

    def _req(self, item, seed):
        if self.cfg.get("no_evidence"):
            item = dataclasses.replace(item, evidence="")
        sc = data.get_schema(item)
        qk = bool((self.cfg.get("ddl") or {}).get("quote_keywords"))      # X5: SQLite reserved words quoted in the schema + one rule line
        user = P.render("gen_zs.user", schema=data.render_ddl(sc, quote_keywords=True) if qk else sc["ddl"], question=item.question,
                        evidence=item.evidence or "(none)", quote_rule=P.QUOTE_RULE if qk else "")
        return [{"role": "system", "content": P.render("common.system")}, {"role": "user", "content": user}]

    def plan_requests(self, item, seed=0):
        return [{"messages": self._req(item, seed), "temperature": self.cfg["temperature"], "seed": seed,
                 "max_tokens": self.cfg["max_tokens"]["generate"], "tag": "gen"}]

    def run(self, item, seed=0):
        r = self.llm.chat(self.model, self._req(item, seed), temperature=self.cfg["temperature"], seed=seed,
                          max_tokens=self.cfg["max_tokens"]["generate"], tag="gen")
        return {"final_sql": strip_sql(r["text"]), "abstained": False, "critic": None, "iterations": 0, "actions": [],
                "ctx_tables": None, "ctx_full_schema": True}
