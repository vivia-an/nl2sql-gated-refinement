"""Post hoc control `zs_retry` (X2DSR): the single call of `zs_ours` plus the execution gate of the gated loop, and nothing else.

Root = the zero-shot call (same request as zs_ours: full schema DDL with sample values, question, evidence; no task graph, no retrieved examples, no business
terms, no routed schema, no lineage hints). Then, exactly as `FullSystem._run_gated`: while the current SQL fails to execute or times out (30 s, db.execute with
max_rows 10000) and fewer than m5.t_max (3) revisions were made, the SQL and the database error text go back to the same model (prompt rev_exec.user, the template of
the gated loop; its `sections` slot holds the zero-shot schema block, graph / value hints are empty); a revision identical to its parent ends the loop (revise_noop);
final = best node by execution status (non-empty > empty > error), ties to the earliest (M5.node_rank rule). No semantic critic call, no abstention.
With the locked zero-shot archive imported into the call cache (tools/cache_export.py import), the root call is a cache hit that returns the stored answer of the locked
single-call run (pairs question by question); only the revisions of failing roots are new calls. No prompt file is added or changed (prompts_sha unchanged)."""
from ... import data, db
from ... import prompts_util as P
from ...util import strip_sql
from . import m5 as M5
from .pipeline import ZeroShot


class ZeroShotRetry(ZeroShot):
    name = "zs_retry"

    def _node(self, item, sql, k):
        res = db.execute(item.db_path, sql, timeout_s=30, max_rows=10000)
        return {"sql": sql, "res": res, "status": M5.exec_status(res), "k": k}

    def _schema_block(self, item):
        sc = data.get_schema(item)
        qk = bool((self.cfg.get("ddl") or {}).get("quote_keywords"))
        return "# Schema\n" + (data.render_ddl(sc, quote_keywords=True) if qk else sc["ddl"])

    def _revise(self, item, node, seed):
        qk = bool((self.cfg.get("ddl") or {}).get("quote_keywords"))
        user = P.render("rev_exec.user", sections=self._schema_block(item), question=item.question, evidence=item.evidence or "(none)", graph="",
                        sql=node["sql"], preview=M5.exec_summary(node["res"]), quote_rule=P.QUOTE_RULE if qk else "",
                        feedback=M5.exec_feedback_text(node["res"], node["status"]), value_hints="")
        msgs = [{"role": "system", "content": P.render("common.system")}, {"role": "user", "content": user}]
        r = self.llm.chat(self.model, msgs, temperature=self.cfg["temperature"], seed=seed, max_tokens=self.cfg["max_tokens"]["revise"], tag="rev")
        return strip_sql(r["text"])

    def run(self, item, seed=0):
        m5 = self.cfg["m5"]
        r = self.llm.chat(self.model, self._req(item, seed), temperature=self.cfg["temperature"], seed=seed,
                          max_tokens=self.cfg["max_tokens"]["generate"], tag="gen")
        root = self._node(item, strip_sql(r["text"]), 0)
        nodes, actions, cur = [root], [], root
        while True:
            if not M5.gate_trigger(cur["status"], m5.get("gate_empty", False)):
                actions.append("accept")
                break
            if cur["k"] >= m5["t_max"]:
                break
            sql = self._revise(item, cur, seed)
            child = self._node(item, sql, cur["k"] + 1)
            actions.append("revise")
            nodes.append(child)
            if sql.strip() == cur["sql"].strip():
                actions.append("revise_noop")
                break
            cur = child
        best = max(nodes, key=lambda n: (M5.STATUS_RANK.get(n["status"], 0), -n["k"]))
        if actions and actions[-1] != "accept":
            actions.append("stop_best")
        return {"final_sql": best["sql"], "abstained": False, "critic": None, "iterations": len(nodes) - 1, "actions": actions,
                "ctx_tables": None, "ctx_full_schema": True, "final_status": best["status"], "root_sql": root["sql"], "root_status": root["status"],
                "n_candidates": len(nodes)}
