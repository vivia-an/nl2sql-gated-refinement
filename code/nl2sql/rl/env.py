"""Replayable environment over cached trees (E6). Deterministic given the tree; transitions pick a stored child uniformly
(by multiplicity). Actions: 0 accept, 1 revise, 2 re_retrieve, 3 escalate, 4 route_human.
Reward: accept -> +1 if gold EX else -1; route_human -> -c_h; every generating step -> -lambda * tokens_step/1000 * m_tier.
Forced termination at H steps (or when no generating action is available and the policy tries one): accept the best-scoring
candidate on the path (standard reward). gamma = 1."""
import json

import numpy as np

from .features import ACTIONS, STATE_DIM

A_ACCEPT, A_REVISE, A_RERETR, A_ESC, A_HUMAN = range(5)
GEN = {"revise": A_REVISE, "re_retrieve": A_RERETR, "escalate": A_ESC}


class TreeSet:
    """All trees as flat arrays for fast vectorised stepping."""

    def __init__(self, trees, m_tier2=4.0):
        self.qids = [t["qid"] for t in trees]
        self.db = [t["db_id"] for t in trees]
        self.fold = [t.get("fold") for t in trees]
        S, EX, SC, TOK, CALLS, TIER = [], [], [], [], [], []
        self.root = []
        self.children = []          # per global node: dict action -> (child_ids array, weights array)
        self.q_of = []
        for qi, t in enumerate(trees):
            base = len(S)
            self.root.append(base)
            nodes = t["nodes"]
            ch = [dict() for _ in nodes]
            for n in nodes:
                if n["parent"] is not None:
                    a = GEN[n["action"]]
                    ch[n["parent"]].setdefault(a, ([], []))
                    ch[n["parent"]][a][0].append(base + n["id"])
                    ch[n["parent"]][a][1].append(n.get("weight", 1))
            for n in nodes:
                S.append(n["state"])
                EX.append(1 if n.get("ex") == 1 else 0)
                SC.append(n["score"])
                # step token cost (only meaningful for non-root): tier of the actor that produced it
                m = m_tier2 if n["action"] in ("escalate",) or (n["tier"] == 1 and n["action"] != "root") else 1.0
                TOK.append((n["tok_actor"] * m + n["tok_crit"]) / 1000.0)
                CALLS.append(n["calls"])
                TIER.append(n["tier"])
                self.q_of.append(qi)
            for d in ch:
                self.children.append({a: (np.array(c, dtype=np.int64), np.array(w, dtype=np.float64) / np.sum(w)) for a, (c, w) in d.items()})
        self.state = np.asarray(S, dtype=np.float32)
        self.ex = np.asarray(EX, dtype=np.float32)
        self.score = np.asarray(SC, dtype=np.float32)
        self.tok = np.asarray(TOK, dtype=np.float32)
        self.calls = np.asarray(CALLS, dtype=np.float32)
        self.tier = np.asarray(TIER)
        self.root = np.asarray(self.root)
        self.root_calls = np.asarray([self.calls[r] for r in self.root])
        assert self.state.shape[1] == STATE_DIM

    @classmethod
    def load(cls, path, folds=None, dbs=None, **kw):
        trees = []
        for l in open(path):
            t = json.loads(l)
            if "nodes" not in t:
                continue
            if folds is not None and t.get("fold") not in folds:
                continue
            if dbs is not None and t["db_id"] not in dbs:
                continue
            trees.append(t)
        return cls(trees, **kw)

    def mask(self, node):
        m = np.zeros(5, dtype=bool)
        m[A_ACCEPT] = m[A_HUMAN] = True
        for a in self.children[node]:
            m[a] = True
        return m


class VecEnv:
    def __init__(self, ts: TreeSet, n_envs, c_h, lam, horizon=6, a4=True, seed=0, q_subset=None):
        self.ts, self.n, self.c_h, self.lam, self.H, self.a4 = ts, n_envs, c_h, lam, horizon, a4
        self.rs = np.random.RandomState(seed)
        self.qs = np.arange(len(ts.qids)) if q_subset is None else np.asarray(q_subset)
        self.node = np.zeros(n_envs, dtype=np.int64)
        self.step_i = np.zeros(n_envs, dtype=np.int64)
        self.best = np.zeros(n_envs, dtype=np.int64)
        self.reset_all()

    def reset_idx(self, i):
        q = self.qs[self.rs.randint(len(self.qs))]
        self.node[i] = self.ts.root[q]
        self.best[i] = self.node[i]
        self.step_i[i] = 0

    def reset_all(self):
        for i in range(self.n):
            self.reset_idx(i)

    def obs(self):
        return self.ts.state[self.node]

    def masks(self):
        m = np.stack([self.ts.mask(n) for n in self.node])
        if self.a4:
            m[:, A_ESC] = False
        return m

    def step(self, act):
        ts = self.ts
        r = np.zeros(self.n, dtype=np.float32)
        done = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            a = int(act[i])
            n = self.node[i]
            if a == A_ACCEPT:
                r[i] = 1.0 if ts.ex[n] else -1.0
                done[i] = True
            elif a == A_HUMAN:
                r[i] = -self.c_h
                done[i] = True
            else:
                c, w = ts.children[n][a]
                ch = c[self.rs.choice(len(c), p=w)] if len(c) > 1 else c[0]
                r[i] = -self.lam * ts.tok[ch]
                self.node[i] = ch
                self.step_i[i] += 1
                if ts.score[ch] > ts.score[self.best[i]]:
                    self.best[i] = ch
                if self.step_i[i] >= self.H:
                    b = self.best[i]
                    r[i] += 1.0 if ts.ex[b] else -1.0
                    done[i] = True
        for i in np.where(done)[0]:
            self.reset_idx(i)
        return r, done


# ----------------------------------------------------------------------------- exact policy evaluation on trees
def evaluate_policy(ts: TreeSet, policy, qidx=None, c_h=0.5, lam=0.0, a4=True, horizon=6, per_question=True):
    """Exact expected metrics of a deterministic policy over uniformly-weighted children.
    policy(state, mask) -> action. Returns dict(ret, ex_all, deferral, calls, tokens_k, accept_rate, actions, per_question).
    per_question (X4; needed for paired, db-clustered CIs): list aligned with `qidx`, one dict per question
      {"qid", "db", "fold", "ex", "calls", "deferred", "ret", "accept"}: expected values for that question under the policy
      (ex = P(final answer is accepted AND correct) = the question's contribution to ex_all, deferred = P(route_human),
      calls = expected LLM calls incl. the root's, ret = expected return); the aggregate keys are the means of these lists.
    Compare two policies on the same qidx by pairing the lists on `qid`; resample databases (`db`), not questions."""
    qidx = range(len(ts.qids)) if qidx is None else list(qidx)
    tot = {"ret": 0.0, "ex": 0.0, "defer": 0.0, "calls": 0.0, "tok": 0.0, "acc": 0.0}
    hist = np.zeros(5)

    def rec(n, step, best, p, calls0):
        mask = ts.mask(n)
        if a4:
            mask[A_ESC] = False
        a = policy(ts.state[n], mask)
        a = a if mask[a] else A_ACCEPT
        hist[a] += p
        if a == A_ACCEPT:
            ret = 1.0 if ts.ex[n] else -1.0
            return p * ret, p * float(ts.ex[n]), 0.0, p * (ts.calls[n] and 0), p * 1.0, p * 0.0
        if a == A_HUMAN:
            return p * (-c_h), 0.0, p * 1.0, 0.0, 0.0, 0.0
        c, w = ts.children[n][a]
        R = EXs = D = Ca = Ac = Tk = 0.0
        for ci, wi in zip(c, w):
            cost = -lam * float(ts.tok[ci])
            if step + 1 >= horizon:
                b = ci if ts.score[ci] > ts.score[best] else best
                r_ = cost + (1.0 if ts.ex[b] else -1.0)
                R += p * wi * r_
                EXs += p * wi * float(ts.ex[b])
                Ac += p * wi
                Ca += p * wi * float(ts.calls[ci])
                Tk += p * wi * float(ts.tok[ci])
            else:
                b = ci if ts.score[ci] > ts.score[best] else best
                r2, e2, d2, c2, a2, t2 = rec(ci, step + 1, b, p * wi, 0)
                R += p * wi * cost + r2
                EXs += e2
                D += d2
                Ca += p * wi * float(ts.calls[ci]) + c2
                Ac += a2
                Tk += p * wi * float(ts.tok[ci]) + t2
        return R, EXs, D, Ca, Ac, Tk

    pq = []
    for q in qidx:
        root = ts.root[q]
        R, E, D, Ca, Ac, Tk = rec(root, 0, root, 1.0, 0)
        pq.append({"qid": ts.qids[q], "db": ts.db[q], "fold": ts.fold[q], "ex": float(E), "calls": float(Ca + float(ts.calls[root])),
                   "deferred": float(D), "ret": float(R), "accept": float(Ac)})
        tot["ret"] += R
        tot["ex"] += E
        tot["defer"] += D
        tot["calls"] += Ca + float(ts.calls[root])
        tot["tok"] += Tk + float(ts.tok[root])
        tot["acc"] += Ac
    n = float(len(list(qidx)))
    if n == 0:   # empty fold (only possible with tiny smoke tree sets)
        nan = float('nan')
        return {"ret": nan, "ex_all": nan, "deferral": nan, "calls": nan, "tokens_k": nan, "accept_rate": nan, "actions": {a: nan for a in ACTIONS},
                "per_question": []}
    return {"ret": tot["ret"] / n, "ex_all": tot["ex"] / n, "deferral": tot["defer"] / n, "calls": tot["calls"] / n,
            "tokens_k": tot["tok"] / n, "accept_rate": tot["acc"] / n,
            "actions": {ACTIONS[i]: float(hist[i] / n) for i in range(5)}, "per_question": pq if per_question else []}
