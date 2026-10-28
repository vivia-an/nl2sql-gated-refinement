"""Self-Debug (Chen et al., ICLR 2024, arXiv 2304.05128) for text-to-SQL with a chat model.

Prompts are the paper's zero-shot chat prompts (Appendix E.6 "Chat Simple Feedback" and E.7 "Chat Explanation Feedback"),
quoted verbatim where the paper gives text:
  System: "You are an expert programming assistant."
  User:   "Write the SQL query for the given question.\n<schema and question>"
  User:   "Summarize the return type of the SQL execution.\n<schema and SQL>"          (+ execution result, first 2 rows, Sec. 4.1)
  User:   "Infer the return type of the question.\n<question>"
  User:   "Does the SQL match the question? If not, generate the fixed SQL."
Loop (Sec. 4.1): explain execution -> compare with question -> fix; stops when the model judges the SQL correct or after
`max_rounds` debugging turns.  One conversation is kept across turns ("[...]" in E.6/E.7).
Our adaptations: (1) evidence is appended to the question; (2) when the SQL raises an execution error we skip the
explanation step and feed the error text (the paper's code-translation setting feeds runtime errors; Spider has none);
(3) the initial generation turn is part of the conversation.
"""
import re

from ._common import BaselineSystem, extract_sql, fmt_rows, one_line, question_block, schema_text

SYSTEM = "You are an expert programming assistant."
Q_GEN = "Write the SQL query for the given question.\n{schema}\n\n{question}"
Q_EXEC = "Summarize the return type of the SQL execution.\n{schema}\n\nSQL: {sql}\n\nExecution:\n{execution}"
Q_QUES = "Infer the return type of the question.\n{question}"
Q_FIX = "Does the SQL match the question? If not, generate the fixed SQL."
Q_ERR = ("Execution of the SQL failed with the error below.\nSQL: {sql}\nError: {error}\n"
         "Does the SQL match the question? If not, generate the fixed SQL.")  # our adaptation (error feedback)
Q_SIMPLE = "Execution:\n{execution}\n" + Q_FIX                                    # simple-feedback variant

_ROWS = 2   # paper: only the first 2 rows of the returned table are shown


def _has_sql(reply):
    return bool(re.search(r"```|^\s*(?:select|with)\b", reply, re.I | re.M))


class SelfDebug(BaselineSystem):
    name = "self_debug"
    default_cfg = {"max_rounds": 3, "feedback": "explain", "max_tokens": 1024, "notes": False}

    def solve(self, item, seed, ctx):
        schema = schema_text(item, notes=self.cfg["notes"], quote_keywords=self.qk)
        qblock = question_block(item)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": Q_GEN.format(schema=schema, question=qblock)}]
        reply = ctx.chat(msgs, tag="sdebug.gen")
        msgs.append({"role": "assistant", "content": reply})
        sql = extract_sql(reply)
        actions = ["generate"]
        rounds = 0
        q_explained = False
        mode = self.cfg["feedback"]
        while rounds < int(self.cfg["max_rounds"]):
            res = ctx.execute(sql)
            if not res["ok"]:
                msgs.append({"role": "user", "content": Q_ERR.format(sql=one_line(sql), error=(res["error"] or "")[:500])})
                reply = ctx.chat(msgs, tag="sdebug.fix")
            elif mode == "simple":
                msgs.append({"role": "user", "content": Q_SIMPLE.format(execution=fmt_rows(res["rows"], _ROWS) or "(empty result)")})
                reply = ctx.chat(msgs, tag="sdebug.fix")
            else:
                exe = fmt_rows(res["rows"], _ROWS) or "(empty result)"
                msgs.append({"role": "user", "content": Q_EXEC.format(schema=schema, sql=one_line(sql), execution=exe)})
                msgs.append({"role": "assistant", "content": ctx.chat(msgs, tag="sdebug.explain_sql")})
                if not q_explained:
                    msgs.append({"role": "user", "content": Q_QUES.format(question=qblock)})
                    msgs.append({"role": "assistant", "content": ctx.chat(msgs, tag="sdebug.explain_q")})
                    q_explained = True
                msgs.append({"role": "user", "content": Q_FIX})
                reply = ctx.chat(msgs, tag="sdebug.fix")
            msgs.append({"role": "assistant", "content": reply})
            rounds += 1
            new_sql = extract_sql(reply) if _has_sql(reply) else ""
            if not new_sql or one_line(new_sql) == one_line(sql):
                actions.append("accept")      # model judged the SQL correct (or produced no change)
                break
            sql = new_sql
            actions.append("revise")
        else:
            actions.append("budget_exhausted")
        return sql, {"iterations": rounds + 1, "actions": actions, "extra": {"rounds": rounds}}
