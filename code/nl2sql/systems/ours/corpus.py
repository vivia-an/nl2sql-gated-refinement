"""M6 value-weighted experience corpus (+ the E4 policy variants).

Entry e = (q, evidence, SQL, db_id, Cbar, Fbar, n, WS). WS = alpha*Cbar + beta*Fbar + gamma*Phi, Phi = 10*min(1, ln(1+n)/ln(1+20)),
(alpha,beta,gamma)=(0.4,0.5,0.1). Online estimate: EMA x <- (1-eta)x + eta*x_obs (first observation sets x), eta=0.3; priors
Cbar=Fbar=7.0, n=0 -> WS0 = 6.3 (>= tau). Eviction: WS < tau (6.0) and n_fb >= 1. Priority recall: top-20 by hybrid similarity
(RRF of BM25 over the masked question and dense cosine), rerank, rank by sim_norm + rho*WS/10 (rho=0.2), top k_e.
Policies (E4): static | append | ws_evict | ws_full | memrl, all configuration of this one class (no code fork)."""
import hashlib
import json
import math
import sqlite3
import threading

import numpy as np

from ... import paths
from .text import BM25, mask_question, rrf, tokenize

POLICIES = {
    "static":   {"write": "none",     "evict": False, "rho": 0.0, "rank": "sim"},
    "append":   {"write": "accepted", "evict": False, "rho": 0.0, "rank": "sim"},
    "ws_evict": {"write": "accepted", "evict": True,  "rho": 0.0, "rank": "sim"},
    "ws_full":  {"write": "accepted", "evict": True,  "rho": 0.2, "rank": "ws"},
    "memrl":    {"write": "accepted", "evict": False, "rho": 0.0, "rank": "utility", "lambda_m": 0.5},
    "raw":      {"write": "none",     "evict": False, "rho": 0.0, "rank": "sim"},     # -M6 ablation on dev: raw pool
}


def phi(n, k_max=20):
    return 10.0 * min(1.0, math.log(1 + n) / math.log(1 + k_max))


def u01(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


class Entry:
    __slots__ = ("id", "kind", "db_id", "question", "evidence", "sql", "C", "F", "n", "n_fb", "WS", "Q", "polluted", "origin", "alive", "step")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        self.alive = True if self.alive is None else self.alive
        self.polluted = bool(self.polluted)
        self.n = self.n or 0
        self.n_fb = self.n_fb or 0

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


class ExperienceCorpus:
    def __init__(self, llm, embed_model, policy="ws_full", params=None, seed=0):
        self.llm, self.embed_model = llm, embed_model
        p = dict(POLICIES[policy] if isinstance(policy, str) else policy)
        self.policy_name = policy if isinstance(policy, str) else "custom"
        d = {"alpha": 0.4, "beta": 0.5, "gamma": 0.1, "eta": 0.3, "tau": 6.0, "phi": 0.5, "prior_C": 7.0, "prior_F": 7.0,
             "k_max": 20, "top_cand": 20, "memrl_eta": 0.3, "memrl_q0": 0.5}
        d.update(p)
        d.update(params or {})
        self.p = d
        self.seed = seed
        self.entries = {}          # id -> Entry
        self.order = []            # ids in insertion order (vector rows)
        self._vecs = []            # per-row vectors
        self._mat = None
        self._row = {}
        self.bm = BM25(1.2, 0.75)
        self.log = {"evicted": [], "writes": 0}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ construction
    def _ws(self, e):
        p = self.p
        return p["alpha"] * e.C + p["beta"] * e.F + p["gamma"] * phi(e.n, p["k_max"])

    def _insert(self, e, vec):
        with self._lock:
            self.entries[e.id] = e
            self._row[e.id] = len(self.order)
            self.order.append(e.id)
            self._vecs.append(vec.astype(np.float32))
            self._mat = None
            self.bm.add(e.id, tokenize(mask_question(e.question)))

    def add_history(self, items, vecs=None):
        """Pool (question, evidence, gold SQL) pairs as kind=history entries at the prior WS."""
        if vecs is None:
            vecs = self._embed([it.question for it in items])
        for it, v in zip(items, vecs):
            e = Entry(id=it.qid, kind="history", db_id=it.db_id, question=it.question, evidence=it.evidence, sql=it.gold_sql,
                      C=self.p["prior_C"], F=self.p["prior_F"], n=0, n_fb=0, Q=self.p["memrl_q0"], origin="pool", step=0)
            e.WS = self._ws(e)
            self._insert(e, v)

    def _embed(self, texts):
        v = self.llm.embed_np(texts, self.embed_model)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def add_entry(self, eid, db_id, question, evidence, sql, critic_score, ex_self=None, qvec=None, polluted=False, origin="stream",
                  feedback=None, step=0, kind="exp", seed_entry=False):
        """New entry from a stream item (or a seed-set item). feedback: None -> decided by phi; True/False forces."""
        with self._lock:
            if eid in self.entries:
                return self.entries[eid]
            fb_obs = None
            if ex_self is not None and not seed_entry:
                take = (u01(self.seed, "create", eid) < self.p["phi"]) if feedback is None else feedback
                fb_obs = 10.0 * ex_self if take else None
            C = critic_score if (critic_score is not None and not seed_entry) else self.p["prior_C"]
            e = Entry(id=eid, kind=kind, db_id=db_id, question=question, evidence=evidence, sql=sql, C=C,
                      F=fb_obs if fb_obs is not None else self.p["prior_F"], n=0, n_fb=1 if fb_obs is not None else 0,
                      Q=self.p["memrl_q0"], polluted=polluted, origin=origin, step=step)
            e.WS = self._ws(e)
            if qvec is None:
                qvec = self._embed([question])[0]
            self._insert(e, qvec)
            self.log["writes"] += 1
            return e

    # ------------------------------------------------------------------ recall
    def _matrix(self):
        if self._mat is None or self._mat.shape[0] != len(self._vecs):
            self._mat = np.stack(self._vecs) if self._vecs else np.zeros((0, 1024), dtype=np.float32)
        return self._mat

    def candidates(self, question, qvec, k=None, exclude=()):
        k = k or self.p["top_cand"]
        ex = set(exclude)
        mat = self._matrix()
        sims = mat @ qvec
        alive = np.array([self.entries[i].alive and i not in ex for i in self.order], dtype=bool)
        sims = np.where(alive, sims, -9.0)
        dense = [self.order[i] for i in np.argsort(-sims)[:50] if alive[i]]
        bs = self.bm.scores(tokenize(mask_question(question)))
        bl = [d for d, _ in sorted(bs.items(), key=lambda x: -x[1])[:50] if d not in ex and self.entries[d].alive]
        fused = rrf([bl, dense], k=60, top=k)
        ids = [i for i, _ in fused]
        return ids, {i: float(sims[self._row[i]]) for i in ids}

    def recall(self, rerank_model, question, qvec, k_e, exclude=(), rerank=True, hybrid=True):
        """Return (entries, n_rerank_calls). hybrid=False: dense-only top-k_e (ablation -M3)."""
        if not self.order:
            return [], 0
        if not hybrid:
            mat = self._matrix()
            sims = mat @ qvec
            ex = set(exclude)
            idx = [i for i in np.argsort(-sims) if self.entries[self.order[i]].alive and self.order[i] not in ex][:k_e]
            return [self.entries[self.order[i]] for i in idx], 0
        cand, dsim = self.candidates(question, qvec, exclude=exclude)
        if not cand:
            return [], 0
        nrr = 0
        if rerank:
            docs = [(self.entries[c].question + (" | " + self.entries[c].evidence if self.entries[c].evidence else ""))[:400] for c in cand]
            sims = self.llm.rerank(question, docs, rerank_model)
            nrr = 1
        else:
            sims = [dsim.get(c, 0.0) for c in cand]
        sims = np.asarray(sims, dtype=np.float64)
        mode = self.p["rank"]
        if mode == "sim":
            score = sims
        elif mode == "ws":
            score = sims + self.p["rho"] * np.array([self.entries[c].WS for c in cand]) / 10.0
        elif mode == "utility":
            z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
            score = z(sims) + self.p["lambda_m"] * z(np.array([self.entries[c].Q for c in cand]))
        else:
            raise ValueError(mode)
        order = np.argsort(-score, kind="stable")[:k_e]
        return [self.entries[cand[i]] for i in order], nrr

    # ------------------------------------------------------------------ updates
    def on_use(self, ids, gen_key, ex_final, update_ws=True):
        """The entries `ids` were exemplars of one generation whose final EX is ex_final (0/1, or None if unobservable).
        n += 1; with probability phi the feedback 10*EX updates Fbar (shared credit). memrl: Q update at the same rate."""
        with self._lock:
            for i in ids:
                e = self.entries.get(i)
                if e is None or not e.alive:
                    continue
                e.n += 1
                fb = ex_final is not None and u01(self.seed, "use", gen_key, i) < self.p["phi"]
                if fb and update_ws:
                    x = 10.0 * ex_final
                    e.F = x if e.n_fb == 0 else (1 - self.p["eta"]) * e.F + self.p["eta"] * x
                    e.n_fb += 1
                if fb and self.p["rank"] == "utility":
                    e.Q += self.p["memrl_eta"] * (ex_final - e.Q)
                e.WS = self._ws(e)

    def evict(self):
        if not self.p["evict"]:
            return []
        gone = []
        with self._lock:
            for i in list(self.order):
                e = self.entries[i]
                if e.alive and e.WS < self.p["tau"] and e.n_fb >= 1:
                    e.alive = False
                    self.bm.remove(i)
                    gone.append(i)
        self.log["evicted"] += gone
        return gone

    # ------------------------------------------------------------------ diagnostics / persistence
    def stats(self):
        alive = [e for e in self.entries.values() if e.alive]
        pol = [e for e in alive if e.polluted]
        cl = [e for e in alive if not e.polluted and e.kind == "exp"]
        ev = [self.entries[i] for i in self.log["evicted"]]
        return {"size": len(alive), "size_exp": sum(1 for e in alive if e.kind == "exp"), "polluted_alive": len(pol),
                "share_polluted_exp": len(pol) / max(1, sum(1 for e in alive if e.kind == "exp")),
                "evicted_total": len(ev), "evicted_polluted": sum(1 for e in ev if e.polluted),
                "evicted_clean": sum(1 for e in ev if not e.polluted),
                "mean_ws_polluted": float(np.mean([e.WS for e in pol])) if pol else None,
                "mean_ws_clean_exp": float(np.mean([e.WS for e in cl])) if cl else None}

    def save(self, path):
        c = sqlite3.connect(path)
        c.execute("DROP TABLE IF EXISTS entries")
        c.execute("CREATE TABLE entries (id TEXT PRIMARY KEY, row INTEGER, meta TEXT, vec BLOB)")
        for i, eid in enumerate(self.order):
            e = self.entries[eid]
            if not e.alive:
                continue
            c.execute("INSERT INTO entries VALUES (?,?,?,?)", (eid, i, json.dumps(e.to_dict(), ensure_ascii=False), self._vecs[i].tobytes()))
        c.execute("CREATE TABLE IF NOT EXISTS params (k TEXT, v TEXT)")
        c.execute("DELETE FROM params")
        c.execute("INSERT INTO params VALUES ('p', ?)", (json.dumps(self.p, sort_keys=True),))
        c.commit()
        c.close()

    @classmethod
    def load(cls, path, llm, embed_model, policy="ws_full", params=None, seed=0):
        o = cls(llm, embed_model, policy, params, seed)
        c = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
        for eid, row, meta, vec in c.execute("SELECT id,row,meta,vec FROM entries ORDER BY row"):
            d = json.loads(meta)
            e = Entry(**d)
            o._insert(e, np.frombuffer(vec, dtype=np.float32))
        c.close()
        return o
