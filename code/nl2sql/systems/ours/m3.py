"""M3 domain-routed retrieval -> Context. Deterministic workers (BM25 + dense -> RRF -> rerank), not LLM agents."""
from dataclasses import dataclass, field

import numpy as np

from ...data import render_ddl
from ...util import est_tokens
from . import values as VG
from .m1 import mentions


@dataclass
class Context:
    tables: list = field(default_factory=list)
    ddl: str = ""
    lineage: str = ""
    examples: list = field(default_factory=list)       # corpus Entry objects
    ex_sims: list = field(default_factory=list)
    terms: list = field(default_factory=list)
    full_schema: bool = False
    n_rerank: int = 0
    k_t: int = 6
    exemplar_ids: list = field(default_factory=list)
    value_hints: list = field(default_factory=list)    # X9 (m2.value_grounding): [{table, column, literal, kind, surface, sim, identical}]
    vh_text: str = ""                                   # X9: the rendered `# Value hints` block ("" = none) for the optional placeholder <<value_hints?>>
    vh_info: dict = field(default_factory=dict)

    def sections_core(self):
        parts = ["# Schema\n" + self.ddl]
        if self.lineage:
            parts.append("# Lineage hints (join keys)\n" + self.lineage)
        return "\n\n".join(parts)

    def sections(self):
        parts = [self.sections_core()]
        if self.examples:
            blocks = []
            for e in self.examples:
                blocks.append("Question: %s\n%sSQL: %s" % (e.question, ("Evidence: %s\n" % e.evidence) if e.evidence else "", e.sql))
            parts.append("# Examples\n" + "\n\n".join(blocks))
        if self.terms:
            parts.append("# Business terms\n" + "\n".join("- %s: %s" % (t["term"], t["definition"]) for t in self.terms))
        return "\n\n".join(parts)


def retrieve(res, item, graph, cfg, corpus, extra_query="", exclude_ids=(), extra_tables=0):
    """res: Resources (db stores, term index, models). Returns Context."""
    m3, m2 = cfg["m3"], cfg["m2"]
    qk = bool((cfg.get("ddl") or {}).get("quote_keywords"))
    st = res.db_store(item)
    llm, em = res.llm, cfg["models"]["embed"]
    ment = mentions(graph) if cfg["m1"]["enabled"] else []
    qtext = " ".join([item.question, item.evidence or "", " ".join(ment), extra_query]).strip()
    rq = " ".join([item.question, item.evidence or "", " ".join(ment), extra_query]).strip()[:800]
    qvec = res.embed_one(qtext)
    k_t = m3["k_t"] + extra_tables
    ranked, nrr = st.retrieve_tables(qtext, qvec, rq, k_t, mode=m3["mode"], n_domains=m3["n_domains"], rerank=m3["rerank"])
    tables = [t for t, _ in ranked]
    if m3["mode"] == "full" and m3["expand_fk"]:
        tables = st.expand_fk_paths(tables)
    sc = st.schema
    full = False
    if m3["mode"] == "full" and st.full_tokens < m3["schema_fallback_tokens"]:
        full = True
        ddl = render_ddl(sc, flag=set(tables[:k_t]), quote_keywords=qk)
        shown = list(sc["tables"].keys())
    else:
        shown = [t for t in tables if t in sc["tables"]] or list(sc["tables"].keys())
        ddl = render_ddl(sc, tables=shown, quote_keywords=qk)
    lineage = st.lineage_text(shown, quote_keywords=qk) if m2["lineage"] else ""
    ctx = Context(tables=shown, ddl=ddl, lineage=lineage, full_schema=full, n_rerank=nrr, k_t=k_t)
    if m2["history"] and corpus is not None:
        qv = res.embed_one(item.question)
        ex = set(exclude_ids) | {item.qid}
        entries, nr2 = corpus.recall(cfg["models"]["rerank"], item.question, qv, m3["k_e"], exclude=ex,
                                     rerank=m3["rerank"] and m3["mode"] == "full", hybrid=(m3["mode"] == "full"))
        ctx.examples = entries
        ctx.exemplar_ids = [e.id for e in entries]
        ctx.n_rerank += nr2
        ctx.ex_sims = [float(corpus._matrix()[corpus._row[e.id]] @ qv) for e in entries]
    if m2["terms"] and res.terms is not None:
        qv = res.embed_one(item.question)
        ctx.terms = res.terms.top(llm, em, item.question + " " + " ".join(ment), qv, m3["k_terms"])
    if m2.get("value_grounding"):        # X9: stored literals of the shown tables that resemble words of the question / evidence (no LLM call, read-only)
        ctx.value_hints, ctx.vh_text, ctx.vh_info = VG.context_hints(item, sc, shown, cfg)
        if ctx.value_hints:
            print("[lineage] value hints %s: %d (%s)" % (item.qid, len(ctx.value_hints), ctx.vh_info.get("ms")), flush=True)
    return ctx
