"""M2 multi-source knowledge fusion (public-data realisation) and the table/column side of M3.

Sources (EXPERIMENT_SPEC M2): schema+descriptions+samples (per db, kind=schema), historical SQL log = pool
(question, evidence, gold SQL) of other databases (kind=history; lives in corpus.ExperienceCorpus), lineage = FK graph
+ inferred joins (same column name and type, >=30 % value overlap on a 1000-row sample; kind=lineage), business
metadata = evidence clauses of the pool split into (term -> definition) (kind=term). History/terms are
cross-database by construction (target dbs are disjoint from the pool). The paper must say this is simulated."""
import hashlib
import json
import os
import re
from collections import defaultdict

import networkx as nx
import numpy as np

from ... import data, paths
from ...db import _run
from ...util import est_tokens, sha256_text, tmp_suffix
from .text import BM25, rrf, tokenize

# ---------------------------------------------------------------- terms (business metadata)
_TERM_RE = re.compile(r"^(.*?)\s+(?:refers? to|means?|is defined as|stands? for|represents?)\s+(.*)$", re.I | re.S)


def split_evidence(ev):
    """Deterministic normalisation: strip, split on ';' / newlines."""
    return [c.strip() for c in re.split(r";|\n", ev or "") if c.strip()]


def extract_terms(pool_items):
    seen, out = set(), []
    for it in pool_items:
        for cl in split_evidence(it.evidence):
            key = cl.lower()
            if key in seen:
                continue
            seen.add(key)
            m = _TERM_RE.match(cl)
            if m:
                term, defin = m.group(1).strip(), m.group(2).strip()
            elif "=" in cl and len(cl.split("=")[0].split()) <= 8:
                term, defin = cl.split("=", 1)[0].strip(), cl.split("=", 1)[1].strip()
            else:
                term, defin = " ".join(cl.split()[:6]), cl
            if term and defin:
                out.append({"id": "term:%d" % len(out), "term": term, "definition": defin, "db_id": it.db_id, "text": cl})
    return out


class TermIndex:
    def __init__(self, llm, embed_model, pool_items, name):
        self.terms = extract_terms(pool_items)
        self.bm = BM25()
        for t in self.terms:
            self.bm.add(t["id"], tokenize(t["text"]))
        self.idx = {t["id"]: i for i, t in enumerate(self.terms)}
        self.vecs = _embed_cached(llm, embed_model, [t["text"] for t in self.terms], "terms_" + name)

    def top(self, llm, embed_model, qtext, qvec, k=3):
        if not self.terms:
            return []
        bs = self.bm.scores(tokenize(qtext))
        bl = [d for d, _ in sorted(bs.items(), key=lambda x: -x[1])[:30]]
        sims = self.vecs @ qvec
        dl = [self.terms[i]["id"] for i in np.argsort(-sims)[:30]]
        fused = rrf([bl, dl], top=k)
        return [self.terms[self.idx[i]] for i, _ in fused]


def _embed_cached(llm, model, texts, name):
    """float32 L2-normalised matrix; persisted to NL2SQL_CACHE/kstore for fast process start."""
    h = hashlib.sha256((model + "\x1f" + "\x1e".join(texts)).encode()).hexdigest()[:16]
    d = paths.CACHE / "kstore"
    d.mkdir(parents=True, exist_ok=True)
    f = d / ("%s_%s.npy" % (name, h))
    if f.exists():
        try:
            return np.load(f)
        except Exception:
            pass
    v = llm.embed_np(texts, model) if texts else np.zeros((0, 1024), dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True) if len(v) else v
    v = v / np.maximum(n, 1e-9) if len(v) else v
    tmp = str(f) + ".%s.tmp.npy" % tmp_suffix()
    np.save(tmp, v)
    os.replace(tmp, f)
    return v


# ---------------------------------------------------------------- per-database store
class DbStore:
    """Units (tables + columns) with BM25 and dense indexes, FK/inferred-join graph, table domains."""

    def __init__(self, llm, embed_model, rerank_model, item, cfg):
        self.llm, self.embed_model, self.rerank_model = llm, embed_model, rerank_model
        self.db_id = item.db_id
        self.db_path = item.db_path
        self.schema = data.get_schema(item)
        self.cfg = cfg
        sc = self.schema
        self.units, texts = [], []
        for t, cols in sc["tables"].items():
            ttext = "table %s: columns %s" % (t, ", ".join(c["name"] + (" (%s)" % c["desc"][:60] if c["desc"] else "") for c in cols))
            self.units.append({"id": "T:" + t, "table": t, "column": None, "text": ttext[:700]})
            for c in cols:
                ct = "%s.%s" % (t, c["name"])
                if c["desc"]:
                    ct += ": " + c["desc"][:200]
                if c["examples"]:
                    ct += " | samples: " + ", ".join(c["examples"])
                self.units.append({"id": "C:%s.%s" % (t, c["name"]), "table": t, "column": c["name"], "text": ct[:500]})
        self.uidx = {u["id"]: i for i, u in enumerate(self.units)}
        self.vecs = _embed_cached(llm, embed_model, [u["text"] for u in self.units], "units_" + item.qid.split(":")[0] + "_" + self.db_id)
        self.bm = BM25(1.2, 0.75)
        for u in self.units:
            self.bm.add(u["id"], tokenize(u["text"] + " " + u["table"] + " " + (u["column"] or "")))
        self.table_vec = {}
        for i, u in enumerate(self.units):
            if u["column"] is None:
                self.table_vec[u["table"]] = self.vecs[i]
        self.edges = self._lineage()
        self.domains = self._domains()
        self.full_tokens = est_tokens(sc["ddl"])

    # lineage: FK + inferred joins ------------------------------------------------
    def _lineage(self):
        sc = self.schema
        edges = [{"a": f["table"], "ac": f["column"], "b": f["ref_table"], "bc": f["ref_column"], "kind": "fk", "overlap": 1.0}
                 for f in sc["fks"] if f["ref_column"]]
        fkset = {(e["a"], e["ac"], e["b"], e["bc"]) for e in edges} | {(e["b"], e["bc"], e["a"], e["ac"]) for e in edges}
        cf = paths.CACHE / "kstore" / ("lineage_%s.json" % sha256_text(self.db_path + str(os.stat(self.db_path).st_size))[:16])
        inferred = None
        if cf.exists():
            try:
                inferred = json.load(open(cf))
            except Exception:
                inferred = None
        if inferred is None:
            inferred = self._infer_joins(fkset)
            tmp = str(cf) + ".%s.tmp" % tmp_suffix()
            json.dump(inferred, open(tmp, "w"))
            os.replace(tmp, cf)
        for e in inferred:
            if (e["a"], e["ac"], e["b"], e["bc"]) not in fkset:
                edges.append(e)
        return edges

    def _infer_joins(self, fkset):
        sc = self.schema
        cols = [(t, c) for t, cs in sc["tables"].items() for c in cs]
        by_name = defaultdict(list)
        for t, c in cols:
            by_name[c["name"].lower()].append((t, c))
        samples = {}

        def sample(t, c):
            k = (t, c["name"])
            if k not in samples:
                r = _run(self.db_path, 'SELECT DISTINCT "%s" FROM "%s" WHERE "%s" IS NOT NULL LIMIT 1000'
                         % (c["name"].replace('"', '""'), t.replace('"', '""'), c["name"].replace('"', '""')), 20, 1000)
                samples[k] = {str(x[0]) for x in r["rows"]} if r["ok"] else set()
            return samples[k]

        out = []
        for name, lst in by_name.items():
            if len(lst) < 2 or len(lst) > 12:
                continue
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    (t1, c1), (t2, c2) = lst[i], lst[j]
                    if t1 == t2 or (c1["type"] or "").upper()[:3] != (c2["type"] or "").upper()[:3]:
                        continue
                    A, B = sample(t1, c1), sample(t2, c2)
                    if not A or not B:
                        continue
                    ov = len(A & B) / min(len(A), len(B))
                    if ov >= 0.3:
                        out.append({"a": t1, "ac": c1["name"], "b": t2, "bc": c2["name"], "kind": "inferred", "overlap": round(ov, 3)})
        return out

    def graph(self):
        g = nx.Graph()
        g.add_nodes_from(self.schema["tables"].keys())
        for e in self.edges:
            g.add_edge(e["a"], e["b"])
        return g

    def _domains(self):
        tabs = list(self.schema["tables"].keys())
        if len(tabs) <= 6:
            return [tabs]
        g = nx.Graph()
        g.add_nodes_from(tabs)
        for i, a in enumerate(tabs):
            for b in tabs[i + 1:]:
                w = 0.5 * max(0.0, float(self.table_vec[a] @ self.table_vec[b]))
                if any((e["a"], e["b"]) in ((a, b), (b, a)) and e["kind"] == "fk" for e in self.edges):
                    w += 1.0
                if w > 0:
                    g.add_edge(a, b, weight=w)
        comms = nx.community.louvain_communities(g, weight="weight", seed=0)
        return [sorted(c) for c in sorted(comms, key=lambda c: (-len(c), sorted(c)[0]))]

    # retrieval -------------------------------------------------------------------
    def retrieve_tables(self, qtext, qvec, rerank_query, k_t, mode="full", n_domains=2, rerank=True):
        """-> (ranked tables [(table, score)], n_rerank_calls). mode 'dense_only' = ablation -M3."""
        if mode == "dense_only":
            tv = sorted(self.table_vec.items(), key=lambda kv: -float(kv[1] @ qvec))
            ucs = {}
            sims = self.vecs @ qvec
            for i, u in enumerate(self.units):
                ucs[u["table"]] = max(ucs.get(u["table"], -1), float(sims[i]))
            return sorted(ucs.items(), key=lambda x: -x[1])[:k_t], 0
        sims = self.vecs @ qvec
        # route: top-n_domains domains by max cosine to member tables
        dscore = []
        for d in self.domains:
            dscore.append((max(float(self.table_vec[t] @ qvec) for t in d), d))
        dscore.sort(key=lambda x: -x[0])
        picked = [d for _, d in dscore[:n_domains]]
        tscore = {}
        nrr = 0
        qtok = tokenize(qtext)
        for dom in picked:
            ids = {u["id"] for u in self.units if u["table"] in dom}
            bs = self.bm.scores(qtok, allowed=ids)
            bl = [d for d, _ in sorted(bs.items(), key=lambda x: -x[1])[:50]]
            dl = [self.units[i]["id"] for i in np.argsort(-sims) if self.units[i]["id"] in ids][:50]
            fused = [i for i, _ in rrf([bl, dl], k=60, top=20)]
            if rerank and fused:
                rs = self.llm.rerank(rerank_query, [self.units[self.uidx[i]]["text"] for i in fused], self.rerank_model)
                nrr += 1
            else:
                rs = [1.0 / (r + 1) for r in range(len(fused))]
            for i, s in zip(fused, rs):
                t = self.units[self.uidx[i]]["table"]
                tscore[t] = max(tscore.get(t, -1), s)
        ranked = sorted(tscore.items(), key=lambda x: -x[1])[:k_t]
        return ranked, nrr

    def expand_fk_paths(self, tables, max_extra_hops=2):
        g = self.graph()
        sel = set(tables)
        for i, a in enumerate(tables):
            for b in tables[i + 1:]:
                try:
                    p = nx.shortest_path(g, a, b)
                except Exception:
                    continue
                if 2 <= len(p) <= max_extra_hops + 2:
                    sel.update(p)
        order = list(tables) + sorted(sel - set(tables))
        return order

    def lineage_text(self, tables, quote_keywords=False):
        ts = set(tables)
        q = (lambda n: data._q(n, True)) if quote_keywords else (lambda n: n)
        lines = []
        for e in self.edges:
            if e["a"] in ts and e["b"] in ts:
                tag = "FK" if e["kind"] == "fk" else "inferred, overlap %.2f" % e["overlap"]
                lines.append("%s.%s = %s.%s  (%s)" % (q(e["a"]), q(e["ac"]), q(e["b"]), q(e["bc"]), tag))
        return "\n".join(lines)
