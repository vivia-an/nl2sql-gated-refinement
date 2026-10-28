"""AC-SQL (Zheng et al., arXiv 2410.22082): Actor + binary Critic, regenerate on reject.  No official code was found
(paper has no code link; GitHub repository search on 2026-10-25 returned nothing), so this is a reimplementation.

Prompts quoted from the paper (Sec. 4.3 Listings 1-2 and Appendix A):
  Actor : "<schema SQL>\nCreate a SQL query only for the given questions using database schema above without explanation: <question>"
  Regen : "Please provide a new SQL query to the question only without explanation: <question>"   (same Actor dialogue)
  Critic: "<schema SQL>\nAnswer True if the SQL query is correct and False if incorrect without explanation. Question: <q> SQL: <sql>"
Procedure (Sec. 3.2): Execution Critic first (False on any execution error; True otherwise even if the result is empty), then
the LLM Critic; the Actor regenerates on False; stops at True or after `max_regen` regenerations, the last regenerated SQL is
output WITHOUT a Critic check.  Paper: max_regen = 5, same LLM for Actor and Critic, zero-shot.
Our adaptations: BIRD evidence appended to the question; schema = DDL + column notes; max_regen = 3 in the compute-matched
setting (EXPERIMENT_SPEC section 4; 5 = native); `acsql_x` uses a different Critic model.
"""
import re

from ._common import BaselineSystem, extract_sql, question_block, schema_text

ACTOR = "{schema}\nCreate a SQL query only for the given questions using database schema above without explanation: {question}"
REGEN = "Please provide a new SQL query to the question only without explanation: {question}"
CRITIC = ("{schema}\nAnswer True if the SQL query is correct and False if incorrect without explanation. "
          "Question: {question} SQL: {sql}")


def parse_bool(text):
    t = (text or "").strip().lower()
    m = re.match(r"\W*(true|false)", t)
    if m:
        return m.group(1) == "true"
    return ("true" in t) and ("false" not in t)


class ACSQL(BaselineSystem):
    name = "ac_sql"
    default_cfg = {"max_regen": 3, "critic_model": None, "max_tokens": 1024, "critic_max_tokens": 16, "notes": False}

    def solve(self, item, seed, ctx):
        schema = schema_text(item, notes=self.cfg["notes"], quote_keywords=self.qk)
        qb = question_block(item)
        critic_model = self.cfg.get("critic_model") or self.model
        actor_msgs = [{"role": "user", "content": ACTOR.format(schema=schema, question=qb)}]
        sql = extract_sql(ctx.chat(actor_msgs, tag="acsql.actor"))
        actor_msgs.append({"role": "assistant", "content": sql})
        actions = ["generate"]
        regen = 0
        verdicts = []
        while True:
            res = ctx.execute(sql)
            if not res["ok"]:
                ok = False
                verdicts.append("exec_error")
            else:
                reply = ctx.user(CRITIC.format(schema=schema, question=qb, sql=" ".join(sql.split())), model=critic_model,
                                 max_tokens=self.cfg["critic_max_tokens"], tag="acsql.critic")
                ok = parse_bool(reply)
                verdicts.append("true" if ok else "false")
            if ok:
                actions.append("accept")
                break
            if regen >= int(self.cfg["max_regen"]):
                actions.append("budget_exhausted")
                break
            regen += 1
            actor_msgs.append({"role": "user", "content": REGEN.format(question=qb)})
            sql = extract_sql(ctx.chat(actor_msgs, tag="acsql.actor"))
            actor_msgs.append({"role": "assistant", "content": sql})
            actions.append("regenerate")
            if regen >= int(self.cfg["max_regen"]):
                actions.append("emit_unchecked")   # paper: SQL of the last allowed regeneration is output directly
                break
        return sql, {"iterations": regen + 1, "actions": actions, "extra": {"verdicts": verdicts, "critic_model": critic_model}}


class ACSQLCross(ACSQL):
    name = "ac_sql_x"
    default_cfg = dict(ACSQL.default_cfg, critic_model="deepseek-v4.1-flash-1017")
