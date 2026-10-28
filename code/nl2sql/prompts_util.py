"""Prompt templates in code/prompts/*.txt. Placeholders are <<name>>; sha of the directory goes in the lock.
`<<name?>>` is an OPTIONAL placeholder: renders to "" when the caller does not pass it (X5: lets a template carry an extra rule line that
is empty - byte-identical output - for every legacy configuration)."""
import re
from functools import lru_cache

from . import paths


@lru_cache(maxsize=None)
def _read(name):
    return (paths.PROMPTS_DIR / (name + ".txt")).read_text()


def render(name: str, **kw) -> str:
    t = _read(name)
    def rep(m):
        k = m.group(1)
        if k not in kw:
            if m.group(2):
                return ""
            raise KeyError("prompt %s: missing placeholder %s" % (name, k))
        return str(kw[k])
    return re.sub(r"<<(\w+)(\?)?>>", rep, t)


# One-line generation rule added when config `ddl.quote_keywords` is true (HEADROOM row 1). Leading space: it is appended to a sentence.
QUOTE_RULE = " Quote SQLite reserved words used as table or column names with double quotes."
