"""Zero-shot baseline: schema (DDL + column notes with sample values) + evidence + question -> one SQLite query.
Prompt text is B1's own design ("our adaptation"); no original exists for a generic zero-shot baseline."""
from ._common import BaselineSystem, extract_sql, quote_flag, schema_text, question_block

SYSTEM = "You write one SQLite query that answers the question. Return only the SQL in a ```sql code block, with no explanation."


def build_prompt(item, cfg):
    return "# Database schema (SQLite)\n%s\n\n# Request\n%s" % (
        schema_text(item, notes=cfg.get("notes", False), quote_keywords=quote_flag(cfg)), question_block(item))


class ZeroShot(BaselineSystem):
    name = "zero_shot"
    default_cfg = {"max_tokens": 1024, "notes": False}

    def solve(self, item, seed, ctx):
        text = ctx.user(build_prompt(item, self.cfg), system=SYSTEM, tag="zs.gen")
        ctx.act("generate")
        return extract_sql(text), {"iterations": 1, "actions": ["generate"]}
