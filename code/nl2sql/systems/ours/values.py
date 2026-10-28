"""X9 value grounding = the M2 "business metadata" source realised as a per-database value index (config `m2.value_grounding`, default off).

Why: X7 (docs/HEADROOM_ADDENDUM.md A1) found that 80 of the 1401 gated HR + HT answers (5.7 pp) are wrong because a filter literal is
approximate or absent in the question / evidence (case, punctuation, stem, code, a longer stored string) and that for 49 of them (3.5 pp)
the gold literal is stored verbatim in its column. The model only sees three sample values per column in the schema block.

What this module does (no LLM call, read-only database access):
  1. OFFLINE INDEX (`get_index`): per database, the distinct values of every SHORT TEXT column (declared TEXT / CHAR / VARCHAR / CLOB or no
     declared type, `typeof(col) = 'text'`, `length(col) <= max_len`), built lazily and thread-safely at the first use and cached as a
     gzip-JSON file under the machine-local cache (`paths.CACHE/vstore`, key = db path + size + mtime + build parameters). Limits that keep big
     databases (bike_share_1: 5 GB, 71 M rows in `status`) cheap - every limit is recorded per column in the cache file (`status`):
       * only the first `scan_rows` rows of a table are read (`SELECT DISTINCT c FROM (SELECT c FROM t LIMIT scan_rows) ...`);
       * at most `max_distinct` distinct values per column (more -> `truncated`; the kept ones are the first found, an arbitrary subset);
       * at most `max_total` values per database (columns are taken in order of increasing table size, so small tables are never starved);
       * each statement is interrupted after `stmt_timeout_s` (column skipped, status `timeout`); the whole build stops after `budget_s`
         (remaining columns skipped, status `budget`).
     Numeric, date-typed and BLOB columns, and values longer than `max_len`, are never indexed.
  2. NORMALISATION (`norm_tokens`): accent stripping, case folding, punctuation and spaces dropped (tokens = alphanumeric runs), a light
     suffix stemmer (ies -> y, sses/shes/ches/xes -> -es, s, ing, ed) applied to both sides; the compact key is the concatenation of the
     stemmed tokens ('E-mail' = 'email', 'NOBLE' = 'Noble', 'Community Post Office ' = 'community post office').
  3. MATCHING at question time (`ValueIndex.hints`): every n-gram (1..`max_ngram` tokens) of question + evidence is looked up
       exact    - equal compact key (plus, with `siblings`, the longer stored values that contain it)    (tier 3)
       contains - the n-gram is a contiguous token run of a longer stored value (<= `contains_extra` extra tokens), only when it is specific
                  (<= `contains_max_values` such values) and covers the value (`contains_min_cover`) or is a capitalised / quoted name of
                  >= `contains_min_chars` characters: 'Hartzler' -> 'Vicky Hartzler', 'High income' -> 'High income: OECD'              (tier 2)
       fuzzy    - no exact / contains hit: trigram candidates, difflib ratio >= `fuzzy_threshold` on the compact keys                 (tier 1)
     Noise guards: an n-gram that spells a table / column name of the database refers to the schema and is skipped; a bare lower-case word matches
     only categorical columns (<= `cat_max` complete distinct values: 'female', 'rain'), never a name / city / comment column; contains / fuzzy need
     a capitalised or quoted word (or several words).
     Stop words, one-character tokens and (by default) bare numbers are no n-gram boundaries. Spans are chosen greedily (tier, tokens, capitalised or
     quoted, length), a span covered by a better one is dropped; per span at most `max_cols_per_value` columns (columns whose name shares a word with the
     request first) and `max_variants` stored spellings per column; only columns of tables that are SHOWN in the prompt are hinted; at most `max_hints`
     (column, literal) pairs in total.
  4. PROMPT BLOCK (`render_block`): `- table.column: 'stored literal', ...`, one line per column, placed after the schema / examples by the optional
     template placeholder `<<value_hints?>>` (empty = prompt byte-identical to a run without the switch).
Gold SQL is never an input of anything in this module."""
import difflib
import gzip
import json
import os
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict

from ... import db, paths
from ...util import est_tokens, sha256_text, tmp_suffix
from . import probe as PR

VG_DEFAULTS = {
    "max_hints": 10,            # (column, literal) pairs in the prompt
    "max_len": 60,              # values longer than this (characters) are not indexed
    "max_distinct": 10000,      # distinct values kept per column (more -> truncated)
    "max_total": 150000,        # values kept per database
    "scan_rows": 1000000,       # rows read per table
    "stmt_timeout_s": 15.0,     # per index statement
    "budget_s": 90.0,           # per database build
    "max_ngram": 6,
    "min_chars": 3,             # single-token n-grams shorter than this (compact characters) are skipped, except 2-letter all-capital tokens (US, NY)
    "numbers": False,           # True: n-grams that are bare digits (>= 2 digits) may match text columns (years, codes)
    "contains": True,
    "siblings": True,           # an exact hit also lists longer stored values that contain it, in any shown column ('High income' -> 'High income: OECD', 'New York' -> 'New York City')
    "contains_extra": 4,        # a stored value may have at most this many tokens more than the n-gram
    "contains_max_values": 6,   # an n-gram contained in more distinct stored values than this is generic: no hint
    "contains_min_cover": 0.5,  # a contained n-gram must make up at least this share of the stored value (compact characters) ...
    "contains_min_chars": 6,    # ... unless it is capitalised / quoted and has at least this many compact characters ('Hartzler' in 'Vicky Hartzler')
    "cat_max": 50,              # a single lower-case word that is not quoted matches only columns with at most this many (complete) distinct values: categorical columns
                                #   ('female', 'rain'); in name / city / comment columns a stray 'Max', 'Divide', 'Price' is noise unless the request spells it as a name
    "loose_single": False,       # False: a single lower-case word may only match exactly; contains / fuzzy need a capitalised (not sentence-initial) or quoted word
    "fuzzy": True,
    "fuzzy_threshold": 0.85,
    "fuzzy_min_chars": 6,
    "fuzzy_max_ngrams": 10,     # n-grams per question that may go through the trigram search
    "max_cols_per_value": 2,
    "max_variants": 3,          # stored spellings per column (NOBLE, Noble)
    "identical": "keep",        # keep | drop | multi: hints whose literal equals the question text verbatim (they only say WHICH column holds it)
                                #   drop = never shown, multi = shown only when the value sits in >= 2 columns (disambiguation)
}

_STOP = frozenset("""
a about above after again against all also am an and any are as at be because been before being below between both but by can could did do does doing
down during each few for from further had has have having he her here hers him his how i if in into is it its just me more most my no nor not now of off
on once only or other our out over own per same she should so some such than that the their them then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your
many much list give show tell name names number numbers total average percentage percent ratio difference highest lowest top least count calculate
describe indicate state states refer refers referred mean means defined define identify id ids among whose
max min sum avg divide divided multiply subtract add true false null desc asc distinct group order limit
""".split())
_TOK = re.compile(r"[^\W_]+", re.U)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_VG_LOCK = threading.Lock()
_MEM = {}          # cache file path -> ValueIndex (process-wide)
_BUILD_LOCKS = {}


# ------------------------------------------------------------------------------------------------ normalisation
def strip_accents(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def stem(w):
    """Light suffix stemmer on a case-folded token; applied identically to question and stored values, so only consistency matters."""
    n = len(w)
    if not w.isalpha():
        return w
    if n > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if n > 4 and w.endswith(("sses", "shes", "ches", "xes", "zes")):
        return w[:-2]
    if n > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    if n > 5 and w.endswith("ing"):
        return w[:-3]
    if n > 4 and w.endswith("ed"):
        return w[:-2]
    return w


def norm_word(w):
    return stem(strip_accents(w).casefold())


def norm_tokens(s):
    """Normalised token tuple of a string: accent-stripped, case-folded, stemmed alphanumeric runs."""
    return tuple(norm_word(w) for w in _TOK.findall(s or ""))


def compact(tokens):
    return "".join(tokens)


def sql_quote(s):
    return "'" + str(s).replace("'", "''") + "'"


# ------------------------------------------------------------------------------------------------ parameters
def params(cfg=None):
    """VG_DEFAULTS overridden by `m2.vg` of a resolved config; validated."""
    p = dict(VG_DEFAULTS)
    vg = ((cfg or {}).get("m2") or {}).get("vg")
    if isinstance(vg, dict):
        unknown = set(vg) - set(VG_DEFAULTS)
        if unknown:
            raise ValueError("unknown m2.vg keys: %s" % sorted(unknown))
        p.update(vg)
    if p["identical"] not in ("keep", "drop", "multi"):
        raise ValueError("m2.vg.identical must be keep, drop or multi, got %r" % (p["identical"],))
    for k in ("max_hints", "max_len", "max_distinct", "max_total", "scan_rows", "max_ngram"):
        if int(p[k]) < 1:
            raise ValueError("m2.vg.%s must be >= 1" % k)
    if not 0.5 <= float(p["fuzzy_threshold"]) <= 1.0:
        raise ValueError("m2.vg.fuzzy_threshold must be in [0.5, 1]")
    return p


def _build_key(p):
    keys = ("max_len", "max_distinct", "max_total", "scan_rows", "stmt_timeout_s", "budget_s")
    return sha256_text(json.dumps([p[k] for k in keys]))[:8]


# ------------------------------------------------------------------------------------------------ building
def _bq(name):
    return PR._bq(name)


def build_columns(db_path, schema, p, now=time.time):
    """-> (cols, meta). cols = [{"table","column","values":[str],"status":"ok|truncated|empty|timeout|error|budget|total_cap","scanned":int}]
    (one entry per text-like column, in build order), meta = {"elapsed_s","n_values","n_cols_ok",...}. Read-only, time-guarded."""
    t0 = now()
    deadline = t0 + float(p["budget_s"])
    rc = schema.get("row_counts") or {}
    items = []
    order = 0
    for t, cols in schema["tables"].items():
        for c in cols:
            order += 1
            if PR.is_text_type(c.get("type")):
                items.append((max(int(rc.get(t, 0) or 0), 0), order, t, c["name"]))
    items.sort()                                           # small tables first
    out, total = [], 0
    L, cap, scan = int(p["max_len"]), int(p["max_distinct"]), int(p["scan_rows"])
    for rows, _o, t, c in items:
        rec = {"table": t, "column": c, "values": [], "status": "ok", "scanned": min(rows, scan)}
        left = deadline - now()
        if left <= 0:
            rec["status"] = "budget"
        elif total >= int(p["max_total"]):
            rec["status"] = "total_cap"
        else:
            q, qc = _bq(t), _bq(c)
            keep = min(cap, int(p["max_total"]) - total)
            sql = ("SELECT DISTINCT %s FROM (SELECT %s FROM %s LIMIT %d) WHERE typeof(%s) = 'text' AND length(%s) <= %d LIMIT %d"
                   % (qc, qc, q, scan, qc, qc, L, keep + 1))
            r = db.execute(db_path, sql, timeout_s=min(float(p["stmt_timeout_s"]), max(left, 1.0)), max_rows=keep + 1)
            if not r["ok"]:
                rec["status"] = "timeout" if str(r.get("error") or "").startswith("timeout") else "error"
            else:
                vals = [str(x[0]) for x in r["rows"] if str(x[0]).strip() != ""]
                if len(r["rows"]) > keep:
                    rec["status"] = "truncated"
                    vals = vals[:keep]
                elif not vals:
                    rec["status"] = "empty"
                rec["values"] = vals
                total += len(vals)
        out.append(rec)
    meta = {"elapsed_s": round(now() - t0, 3), "n_values": total, "n_cols": len(out),
            "status": dict(Counter(r["status"] for r in out)), "params": {k: p[k] for k in ("max_len", "max_distinct", "max_total", "scan_rows", "stmt_timeout_s", "budget_s")}}
    return out, meta


def _cache_file(db_path, p):
    st = os.stat(db_path)
    key = sha256_text("%s|%d|%d" % (db_path, st.st_size, int(st.st_mtime)))[:16]
    return paths.CACHE / "vstore" / ("vidx_%s_%s.json.gz" % (key, _build_key(p)))


def get_index(db_path, schema, p=None):
    """The ValueIndex of a database, built at the first call (thread-safe: one builder per file, the others wait; a finished build is written
    atomically to the machine-local cache and re-read by later processes). Raises on an unreadable database file."""
    p = p or params()
    f = _cache_file(db_path, p)
    key = str(f)
    with _VG_LOCK:
        idx = _MEM.get(key)
        if idx is not None:
            return idx
        lk = _BUILD_LOCKS.setdefault(key, threading.Lock())
    with lk:
        idx = _MEM.get(key)
        if idx is not None:
            return idx
        cols = meta = None
        if f.exists():
            try:
                with gzip.open(f, "rt", encoding="utf-8") as fh:
                    d = json.load(fh)
                cols, meta = d["cols"], d["meta"]
            except Exception:
                cols = meta = None
        if cols is None:
            cols, meta = build_columns(db_path, schema, p)
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(f) + ".%s.tmp" % tmp_suffix()
            with gzip.open(tmp, "wt", encoding="utf-8") as fh:
                json.dump({"db_path": db_path, "cols": cols, "meta": meta}, fh, ensure_ascii=False)
            os.replace(tmp, f)
            print("[lineage] value index built %s: %d values in %d columns, %.1fs, status %s" % (os.path.basename(db_path), meta["n_values"], meta["n_cols"],
                                                                                             meta["elapsed_s"], meta["status"]), flush=True)
        idx = ValueIndex(cols, meta, db_path)
        idx.set_names(schema)
        idx._prepare()                    # normalise / index the values now (seconds for 150 k values), not inside the first question
        with _VG_LOCK:
            _MEM[key] = idx
        return idx


def clear_memory():
    """Drop the process-wide index cache (tests)."""
    with _VG_LOCK:
        _MEM.clear()
        _BUILD_LOCKS.clear()


# ------------------------------------------------------------------------------------------------ the index
class ValueIndex:
    def __init__(self, cols, meta=None, db_path=None):
        self.cols, self.meta, self.db_path = cols, meta or {}, db_path
        self.names = set()
        self._ready = False
        self._lock = threading.Lock()

    def set_names(self, schema):
        """Compact normalised table / column names of the schema: an n-gram that spells one of them refers to the schema, not to a stored value."""
        names = set()
        for t, cols in (schema.get("tables") or {}).items():
            names.add(compact(norm_tokens(t)))
            for c in cols:
                names.add(compact(norm_tokens(c["name"])))
        names.discard("")
        self.names = names

    @classmethod
    def from_values(cls, mapping):
        """{(table, column): [values]} -> index (tests, tools)."""
        return cls([{"table": t, "column": c, "values": list(v), "status": "ok", "scanned": len(v)} for (t, c), v in mapping.items()])

    # -- lazy structures
    def _prepare(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self.vcol, self.vlit, self.vtok, self.vcmp = [], [], [], []
            self.by_cmp = defaultdict(list)
            self.by_tok = defaultdict(list)
            for ci, c in enumerate(self.cols):
                for lit in c["values"]:
                    tk = norm_tokens(lit)
                    cm = compact(tk)
                    if not cm:
                        continue
                    v = len(self.vlit)
                    self.vcol.append(ci)
                    self.vlit.append(lit)
                    self.vtok.append(tk)
                    self.vcmp.append(cm)
                    self.by_cmp[cm].append(v)
                    if len(tk) <= 12:
                        for t in set(tk):
                            self.by_tok[t].append(v)
            self.col_size = [len(c["values"]) for c in self.cols]
            self.col_complete = [c["status"] != "truncated" for c in self.cols]
            self.col_words = [set(norm_tokens(_CAMEL.sub(" ", c["column"]))) for c in self.cols]       # 'IncomeGroup' / 'income_group' -> {income, group}
            self._tri = None
            self._uniq = None
            self._ready = True

    def _trigrams(self):
        if self._tri is None:
            with self._lock:
                if self._tri is None:
                    uniq = sorted(self.by_cmp)
                    tri = defaultdict(list)
                    for u, k in enumerate(uniq):
                        for g in {k[i:i + 3] for i in range(len(k) - 2)}:
                            tri[g].append(u)
                    self._uniq, self._tri = uniq, tri
        return self._uniq, self._tri

    def _fuzzy(self, key, thr):
        """[(ratio, compact key)] of stored compact keys similar to `key` (best first)."""
        uniq, tri = self._trigrams()
        grams = {key[i:i + 3] for i in range(len(key) - 2)}
        use = sorted((len(tri[g]), g) for g in grams if g in tri)[:8]
        cnt = Counter()
        for ln, g in use:
            if ln > 4000:
                continue
            for u in tri[g]:
                cnt[u] += 1
        need = max(2, int(0.4 * len(use)))
        out = []
        for u, c in cnt.most_common(60):
            if c < need:
                break
            k2 = uniq[u]
            if abs(len(k2) - len(key)) > max(3, 0.25 * len(key)):
                continue
            r = difflib.SequenceMatcher(None, key, k2, autojunk=False).ratio()
            if r >= thr:
                out.append((r, k2))
        out.sort(key=lambda x: -x[0])
        return out[:3]

    # -- matching
    def hints(self, text, shown=None, p=None):
        """Hints for one request text. shown: iterable of table names that appear in the prompt (None = all). -> list of dicts
        {table, column, literal, kind (exact|contains|fuzzy), surface, sim, identical} in rank order, at most p['max_hints']."""
        p = p or params()
        self._prepare()
        text = text or ""
        toks = [(m.group(), m.start(), m.end()) for m in _TOK.finditer(text)]
        N = len(toks)
        if not N or not self.vlit:
            return []
        nt = [norm_word(w) for w, _, _ in toks]
        shown_l = None if shown is None else {s.lower() for s in shown}
        okcol = [(shown_l is None or c["table"].lower() in shown_l) for c in self.cols]
        tw = set(nt)
        capital = []
        for k, (w, s, e) in enumerate(toks):
            quoted = s > 0 and text[s - 1] in "'\"`" and e < len(text) and text[e] in "'\"`"
            capital.append(bool(quoted or (w[:1].isupper() and k > 0 and not _sentence_start(text, s))))
        schema_tok = [False] * N                      # tokens inside an n-gram that spells a table / column name ('Unit Cost' for UnitCost)
        if self.names:
            for n in range(1, min(int(p["max_ngram"]), N) + 1):
                for i in range(N - n + 1):
                    if compact(nt[i:i + n]) in self.names:
                        for k in range(i, i + n):
                            schema_tok[k] = True
        cands = []
        n_fuzzy = 0
        for n in range(min(int(p["max_ngram"]), N), 0, -1):
            for i in range(N - n + 1):
                j = i + n
                if not _boundary_ok(toks, nt, i, j, p):
                    continue
                key = tuple(nt[i:j])
                cm = compact(key)
                if all(schema_tok[i:j]):
                    continue
                span_cap = any(capital[i:j])
                vids = [v for v in self.by_cmp.get(cm, ()) if okcol[self.vcol[v]]]
                kind, sim = "exact", 1.0
                if vids and n == 1 and not span_cap:           # a bare lower-case word: only categorical columns (see cat_max)
                    vids = [v for v in vids if self.col_complete[self.vcol[v]] and self.col_size[self.vcol[v]] <= int(p["cat_max"])]
                sib = []
                if vids and p["contains"] and p["siblings"] and (n > 1 or span_cap or p["loose_single"]) and (n > 1 or len(cm) >= 4):
                    sib = [v for v in self._contains(key, okcol, p, span_cap) if v not in vids]
                loose = n > 1 or span_cap or p["loose_single"]
                if not vids and p["contains"] and loose and (n > 1 or len(cm) >= 4):
                    vids = self._contains(key, okcol, p, span_cap)
                    kind = "contains"
                if not vids and p["fuzzy"] and loose and len(cm) >= int(p["fuzzy_min_chars"]) and n_fuzzy < int(p["fuzzy_max_ngrams"]):
                    n_fuzzy += 1
                    for r, k2 in self._fuzzy(cm, float(p["fuzzy_threshold"])):
                        vv = [v for v in self.by_cmp[k2] if okcol[self.vcol[v]]]
                        if vv:
                            vids, kind, sim = vv, "fuzzy", r
                            break
                if not vids:
                    continue
                tier = {"exact": 3, "contains": 2, "fuzzy": 1}[kind]
                cands.append(((tier, n, int(span_cap), len(cm), sim), i, j, kind, sim, vids, sib))
        cands.sort(key=lambda c: c[0], reverse=True)
        taken, out, seen = [False] * N, [], set()
        for _score, i, j, kind, sim, vids, sib in cands:
            if any(taken[i:j]):
                continue
            surface = text[toks[i][1]:toks[j - 1][2]]
            ents = self._pick(vids, sib, tw, surface, p)
            if not ents:
                continue
            for k in range(i, j):
                taken[k] = True
            for v, ident in ents:
                c = self.cols[self.vcol[v]]
                hk = (c["table"], c["column"], self.vlit[v])
                if hk in seen:
                    continue
                seen.add(hk)
                out.append({"table": c["table"], "column": c["column"], "literal": self.vlit[v], "kind": kind, "surface": surface,
                            "sim": round(sim, 3), "identical": ident})
                if len(out) >= int(p["max_hints"]):
                    return out
        return out

    def _contains(self, key, okcol, p, span_cap=True):
        post = [self.by_tok.get(t) for t in key]
        if any(x is None for x in post):
            return []
        rare = min(post, key=len)
        if len(rare) > 5000:
            return []
        n = len(key)
        hit = []
        for v in rare:
            vt = self.vtok[v]
            if n < len(vt) <= n + int(p["contains_extra"]) and okcol[self.vcol[v]] and _contig(vt, key):
                kl = len(compact(key))
                if kl >= float(p["contains_min_cover"]) * len(self.vcmp[v]) or (span_cap and kl >= int(p["contains_min_chars"])):
                    hit.append(v)
        if not hit or len({self.vcmp[v] for v in hit}) > int(p["contains_max_values"]):
            return []
        return hit

    def _pick(self, vids, sib, text_words, surface, p):
        """Choose columns and spellings of one matched span -> [(vid, identical)]. `sib` = siblings (longer values containing an exact hit): after the
        matched spellings of their column."""
        by_col = defaultdict(list)
        for v in vids:
            by_col[self.vcol[v]].append(v)
        sibset = set(sib)
        for v in sib:
            by_col[self.vcol[v]].append(v)
        exact_cols = {self.vcol[v] for v in vids}
        order = sorted(by_col, key=lambda ci: (-len(self.col_words[ci] & text_words), ci not in exact_cols, self.col_size[ci], ci))
        multi = len(by_col) >= 2
        sur = surface.strip()
        out = []
        for ci in order[: int(p["max_cols_per_value"])]:
            vs = sorted(by_col[ci], key=lambda v: (v in sibset, self.vlit[v] != sur, self.vlit[v]))[: int(p["max_variants"])]
            for v in vs:
                ident = self.vlit[v].strip() == sur
                if ident and (p["identical"] == "drop" or (p["identical"] == "multi" and not multi)):
                    continue
                out.append((v, ident))
        return out


def _sentence_start(text, s):
    """True when the token at char offset s starts a sentence (or the text)."""
    k = s - 1
    while k >= 0 and text[k] in " \t\"'(":
        k -= 1
    return k < 0 or text[k] in ".?!\n;:"


def _contig(vt, key):
    n = len(key)
    return any(vt[k:k + n] == key for k in range(len(vt) - n + 1))


def _boundary_ok(toks, nt, i, j, p):
    """An n-gram must start and end on a content token (no stop word, no single character); a single token must be specific enough."""
    for k in (i, j - 1):
        w, t = toks[k][0], nt[k]
        if t in _STOP or w.casefold() in _STOP or len(t) < 2:
            return False
    if j - i == 1:
        w, t = toks[i][0], nt[i]
        if t.isdigit():
            return bool(p["numbers"])
        if len(t) < int(p["min_chars"]):
            return len(w) == 2 and w.isupper()
    return True


# ------------------------------------------------------------------------------------------------ prompt block + logging
HEADER = ("# Value hints\nStored values (exact spelling) that resemble words of the request. Use the stored spelling in a filter when the request "
          "refers to one of them; ignore hints that do not fit.")


def render_block(hints, quote_keywords=False):
    """The text that replaces `<<value_hints?>>`: "" for no hints, else "\\n\\n# Value hints ..." (leading blank line: it follows <<sections>>)."""
    if not hints:
        return ""
    from ...data import _q
    lines, order = {}, []
    for h in hints:
        k = (h["table"], h["column"])
        if k not in lines:
            lines[k] = []
            order.append(k)
        if h["literal"] not in lines[k]:
            lines[k].append(h["literal"])
    body = "\n".join("- %s.%s: %s" % (_q(t, quote_keywords), _q(c, quote_keywords), ", ".join(sql_quote(v) for v in lines[(t, c)])) for t, c in order)
    return "\n\n" + HEADER + "\n" + body


def context_hints(item, schema, shown, cfg):
    """M3 entry point: -> (hints, block text, info). Never raises: any failure gives no hints (the question is answered without them)."""
    p = params(cfg)
    t0 = time.time()
    try:
        idx = get_index(item.db_path, schema, p)
        hints = idx.hints("%s\n%s" % (item.question or "", item.evidence or ""), shown, p)
    except Exception as e:
        print("[lineage] value hints failed %s: %s" % (item.qid, type(e).__name__), flush=True)
        return [], "", {"error": type(e).__name__}
    qk = bool((cfg.get("ddl") or {}).get("quote_keywords"))
    block = render_block(hints, qk)
    return hints, block, {"ms": round(1000 * (time.time() - t0), 1), "tokens": est_tokens(block) if block else 0}


def literal_in_sql(literal, sql):
    """True when the stored literal occurs in the SQL as a string (equality / IN, or inside a LIKE pattern): exact, case-sensitive."""
    s = sql or ""
    return sql_quote(literal) in s or str(literal).replace("'", "''") in s


def log_record(ctx, root_sql, final_sql):
    """preds.jsonl `value_hints`: count, columns, literals, how many of the hinted literals reached the root / final SQL, block size."""
    hs = ctx.value_hints or []
    lits = []
    for h in hs:
        if h["literal"] not in lits:
            lits.append(h["literal"])
    cols = []
    for h in hs:
        c = "%s.%s" % (h["table"], h["column"])
        if c not in cols:
            cols.append(c)
    return {"n": len(hs), "cols": cols, "lits": lits[:12],
            "used": sum(1 for x in lits if literal_in_sql(x, final_sql)), "used_root": sum(1 for x in lits if literal_in_sql(x, root_sql)),
            "tokens": (ctx.vh_info or {}).get("tokens", 0), "ms": (ctx.vh_info or {}).get("ms", 0.0)}
