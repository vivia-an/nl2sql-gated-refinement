"""Post hoc control `zs_think` (W7, 2026-10-29): the single call of `zs_ours` with the provider's thinking mode on, and nothing else.

Same request as zs_ours (full schema DDL with sample values, question, evidence; same prompt files, same DDL rendering), sent with enable_thinking=True (the endpoint
then streams; only the answer content is kept, the reasoning text is not stored). Completion tokens as reported by the provider include the reasoning tokens, so costs
count them. One call per question, no execution gate, no front end. The thinking budget is the provider default; max_tokens.generate is raised so that the answer is not
cut after the reasoning (config posthoc/zs_think.yaml)."""
from ...util import strip_sql
from .pipeline import ZeroShot


class ZeroShotThink(ZeroShot):
    name = "zs_think"

    def run(self, item, seed=0):
        r = self.llm.chat(self.model, self._req(item, seed), temperature=self.cfg["temperature"], seed=seed,
                          max_tokens=self.cfg["max_tokens"]["generate"], enable_thinking=True, tag="gen")
        return {"final_sql": strip_sql(r["text"]), "abstained": False, "critic": None, "iterations": 0, "actions": [],
                "ctx_tables": None, "ctx_full_schema": True}
