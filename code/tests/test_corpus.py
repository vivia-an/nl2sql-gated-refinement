import hashlib

import numpy as np

from nl2sql.data import Item
from nl2sql.systems.ours.corpus import ExperienceCorpus, phi, POLICIES


class FakeLLM:
    def embed_np(self, texts, model):
        out = []
        for t in texts:
            rs = np.random.RandomState(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
            out.append(rs.randn(32).astype(np.float32))
        return np.stack(out)

    def rerank(self, q, docs, model):
        # similarity = word overlap
        qs = set(q.lower().split())
        return [len(qs & set(d.lower().split())) / (len(qs) + 1.0) for d in docs]


def mk(policy="ws_full", params=None):
    c = ExperienceCorpus(FakeLLM(), "m", policy, params, seed=1)
    items = [Item("h:%d" % i, "db%d" % (i % 3), "how many items %d of kind %s" % (i, "ab"[i % 2]), "", "SELECT %d" % i, "", "pool", {}) for i in range(30)]
    c.add_history(items)
    return c


def test_prior_and_ws_formula():
    c = mk()
    e = c.entries["h:0"]
    assert abs(e.WS - 6.3) < 1e-9               # 0.4*7 + 0.5*7 + 0.1*0
    assert phi(0) == 0 and abs(phi(20) - 10) < 1e-9 and phi(100) == 10


def test_feedback_ema_and_eviction():
    c = mk("ws_full", {"phi": 1.0})
    # first observation sets F; wrong answers drive WS below tau=6 -> evicted
    c.on_use(["h:0"], "g1", 0)
    e = c.entries["h:0"]
    assert e.n == 1 and e.n_fb == 1 and e.F == 0.0
    assert abs(e.WS - (0.4 * 7 + 0.5 * 0 + 0.1 * phi(1))) < 1e-9
    gone = c.evict()
    assert gone == ["h:0"] and not e.alive
    # EMA after the first observation
    c.on_use(["h:1"], "g1", 1)
    assert c.entries["h:1"].F == 10.0
    c.on_use(["h:1"], "g2", 0)
    assert abs(c.entries["h:1"].F - 7.0) < 1e-9          # 0.7*10 + 0.3*0
    # good entries stay
    assert c.entries["h:1"].alive and c.evict() == []


def test_no_eviction_without_feedback_and_policy_flags():
    c = mk("ws_full", {"phi": 0.0})
    for _ in range(5):
        c.on_use(["h:2"], "g%d" % _, 0)
    assert c.entries["h:2"].n == 5 and c.entries["h:2"].n_fb == 0 and c.evict() == []
    s = mk("append", {"phi": 1.0})
    s.on_use(["h:0"], "g", 0)
    assert s.evict() == [] and s.entries["h:0"].alive           # append-only never evicts
    st = mk("static")
    assert POLICIES["static"]["write"] == "none"


def test_add_entry_creation_feedback():
    c = mk("ws_full", {"phi": 1.0})
    e = c.add_entry("s:1", "db0", "q new", "", "SELECT 1", critic_score=9.0, ex_self=1)
    assert e.C == 9.0 and e.F == 10.0 and e.n_fb == 1
    bad = c.add_entry("s:2", "db0", "q bad", "", "SELECT 2", critic_score=9.0, ex_self=0)
    assert bad.WS == 0.4 * 9 + 0.5 * 0 and bad.WS < 6 and c.evict() == ["s:2"]
    seed = c.add_entry("s:3", "db0", "q seed", "", "SELECT 3", critic_score=9.0, ex_self=1, seed_entry=True)
    assert seed.C == 7.0 and seed.F == 7.0 and seed.n_fb == 0           # seed-set entries start at the prior


def test_priority_recall_prefers_high_ws():
    c = mk("ws_full", {"phi": 1.0, "rho": 5.0, "top_cand": 30})
    q = "how many items 4 of kind a"
    vec = c._embed([q])[0]
    base, _ = c.recall("r", q, vec, 3, rerank=True)
    for e in c.entries.values():
        e.WS = 0.0
    c.entries["h:29"].WS = 10.0
    top, _ = c.recall("r", q, vec, 3, rerank=True)
    assert "h:29" in [e.id for e in top]
    c0 = mk("static")
    t0, _ = c0.recall("r", q, vec, 3, rerank=True)
    assert len(t0) == 3
    # exclusion (leave-one-out)
    t1, _ = c0.recall("r", q, vec, 3, exclude={e.id for e in t0}, rerank=True)
    assert not ({e.id for e in t0} & {e.id for e in t1})


def test_memrl_utility_update_and_persist(tmp_path):
    c = mk("memrl", {"phi": 1.0})
    c.on_use(["h:3"], "g", 1)
    assert abs(c.entries["h:3"].Q - (0.5 + 0.3 * 0.5)) < 1e-9
    c.evict()
    p = str(tmp_path / "c.sqlite")
    c.save(p)
    d = ExperienceCorpus.load(p, FakeLLM(), "m", "memrl")
    assert len(d.entries) == 30 and abs(d.entries["h:3"].Q - c.entries["h:3"].Q) < 1e-9
