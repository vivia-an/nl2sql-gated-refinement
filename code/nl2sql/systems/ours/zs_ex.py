"""R3 control `zs_ex` (internal note R3EXP, SAP section 26): the single call of `zs_ours` plus the few-shot examples that the root-only arm of the
SAME backbone and seed retrieved for the same question. Nothing else is added: no task graph, no routed schema, no lineage hints, no business terms.

The example ids are read from the `exemplar_ids` field of the stored predictions of the root-only run (config key `zs_ex.sources`:
{model key: {seed: run directory relative to the runs root}}), and the entries (question, evidence, SQL) are looked up in the frozen M6 store that the
pipeline used (`m6.path`). The examples block has the format of `m3.Context.sections()` and sits after the schema, as in the pipeline prompt.
A question without stored example ids is a system error (never a silent run without examples)."""
import json
import os
import threading

from ... import config as C
from ... import data, paths
from ... import prompts_util as P
from ...util import strip_sql
from .corpus import ExperienceCorpus
from .pipeline import ZeroShot


def examples_block(entries):
    """"# Examples" block in the format of m3.Context.sections() (empty string for no entries)."""
    if not entries:
        return ""
    blocks = []
    for e in entries:
        blocks.append("Question: %s\n%sSQL: %s" % (e.question, ("Evidence: %s\n" % e.evidence) if e.evidence else "", e.sql))
    return "# Examples\n" + "\n\n".join(blocks)


class ZeroShotExamples(ZeroShot):
    name = "zs_ex"

    def __init__(self, llm, model, cfg):
        super().__init__(llm, model, cfg)
        zc = (cfg or {}).get("zs_ex") or {}
        self.mkey = C.model_key(model)
        self.sources = {str(k): {str(s): v for s, v in d.items()} for k, d in (zc.get("sources") or {}).items()}
        m6 = (cfg or {}).get("m6") or {}
        self.store_path = m6.get("path")
        if not self.store_path:
            raise ValueError("zs_ex needs m6.path (the frozen experience store the pipeline used)")
        self._lock = threading.Lock()
        self._corpus = None
        self._ids = {}                 # seed -> {qid: [exemplar ids]}

    def _corp(self):
        with self._lock:
            if self._corpus is None:
                p = self.store_path if str(self.store_path).startswith("/") else str(paths.REPO / self.store_path)
                self._corpus = ExperienceCorpus.load(p, self.llm, self.cfg["models"]["embed"], "ws_full", None)
            return self._corpus

    def _ids_for(self, seed):
        with self._lock:
            d = self._ids.get(seed)
            if d is None:
                rel = (self.sources.get(self.mkey) or {}).get(str(seed))
                if not rel:
                    raise ValueError("zs_ex: no example source for model %s seed %s" % (self.mkey, seed))
                p = os.path.join(str(paths.RUNS), rel, "preds.jsonl")
                d = {}
                with open(p) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            r = json.loads(line)
                            d[r["qid"]] = r.get("exemplar_ids")
                self._ids[seed] = d
            return d

    def _req(self, item, seed):
        ids = self._ids_for(seed).get(item.qid)
        if not ids:
            raise ValueError("zs_ex: no stored example ids for %s (model %s, seed %s)" % (item.qid, self.mkey, seed))
        corp = self._corp()
        entries = [corp.entries[i] for i in ids]
        sc = data.get_schema(item)
        qk = bool((self.cfg.get("ddl") or {}).get("quote_keywords"))      # same renderer and rule line as zs_ours (config ddl.quote_keywords)
        schema = (data.render_ddl(sc, quote_keywords=True) if qk else sc["ddl"]) + "\n\n" + examples_block(entries)
        user = P.render("gen_zs.user", schema=schema, question=item.question, evidence=item.evidence or "(none)",
                        quote_rule=P.QUOTE_RULE if qk else "")
        return [{"role": "system", "content": P.render("common.system")}, {"role": "user", "content": user}]

    def plan_requests(self, item, seed=0):
        return None

    def run(self, item, seed=0):
        r = self.llm.chat(self.model, self._req(item, seed), temperature=self.cfg["temperature"], seed=seed,
                          max_tokens=self.cfg["max_tokens"]["generate"], tag="gen")
        return {"final_sql": strip_sql(r["text"]), "abstained": False, "critic": None, "iterations": 0, "actions": [],
                "ctx_tables": None, "ctx_full_schema": True, "exemplar_ids": list(self._ids_for(seed).get(item.qid) or [])}
