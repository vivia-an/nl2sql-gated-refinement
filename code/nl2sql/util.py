"""Small helpers: hashing, fenced-SQL stripping, tolerant JSON parsing, token estimates."""
import hashlib
import json
import re

_FENCE = re.compile(r"```(?:sql|sqlite|SQL)?\s*\n?(.*?)```", re.S)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def strip_sql(text: str) -> str:
    """Extract one SQL statement from an LLM reply (fences, prose, trailing ';')."""
    if text is None:
        return ""
    t = text.strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    elif t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("sql"):
            t = t[3:].strip()
    # drop leading prose such as "Here is the query:" before the first SELECT/WITH
    if not re.match(r"(?is)^\s*(select|with|\()", t):
        m2 = re.search(r"(?is)\b(select|with)\b", t)
        if m2:
            t = t[m2.start():]
    # keep the first statement only
    t = t.split(";\n")[0] if ";\n" in t else t
    t = t.strip().rstrip(";").strip()
    return t


def parse_json(text: str):
    """json.loads -> outermost {...} regex fallback. Returns dict or None."""
    if text is None:
        return None
    t = text.strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            v = json.loads(t[i:j + 1])
            return v if isinstance(v, dict) else None
        except Exception:
            return None
    return None


def est_tokens(text: str) -> int:
    """Rough token estimate (no tokenizer for qwen): ~3.6 chars/token for mixed English/SQL."""
    return int(len(text) / 3.6) + 1


def trunc(s, n=40):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def tmp_suffix():
    """unique per process+thread+call (atomic-rename temp files)."""
    import os, threading, random
    return "%d.%d.%06d" % (os.getpid(), threading.get_ident() % 1000000, random.randrange(1000000))


def json_default(o):
    """json.dumps(default=...) that coerces numpy scalars/arrays, sets and tuples so one odd value cannot kill a result writer."""
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)
