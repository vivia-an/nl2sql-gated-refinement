"""DAIL-SQL (Gao et al., VLDB 2024, arXiv 2308.15363), few-shot variant "DAIL_S selection + DAIL_O organisation".

Source: https://github.com/BeachWang/DAIL-SQL  (commit 2061f68112222083134a0c9e2877961ff315ff44, Apache-2.0 per LICENSE.txt).

Identical to the original (prompt/PromptReprTemplate.py SQLPrompt, ExampleFormatTemplate.py QuestionSqlExampleStyle,
ExampleSelectorTemplate.py EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdSelector, utils/utils.py sql2skeleton
post-processing and jaccard_similarity):
  * two stages: a preliminary query from the masked-question-similarity selector, then the final query whose examples are
    re-selected by masked-question Euclidean distance with a query-skeleton Jaccard filter (threshold 0.85), padded with the
    remaining nearest examples;
  * target prompt = "/* Given the following database schema: */" + CREATE TABLE statements + "/* Answer the following: Q */";
  * examples = "/* Answer the following: Q */\n<SQL>" under "/* Some SQL examples are provided based on similar problems: */";
  * BIRD protocol of data_preprocess.py: evidence is appended to the question text.
Adapted (our adaptation):
  * sentence encoder: all-mpnet-base-v2 -> DashScope `qwen3.7-text-embedding` (L2-normalised, so Euclidean order = cosine order);
  * schema-linking mask (CoreNLP + RAT-SQL matcher in the original) -> regex n-gram matcher over table/column names ("<mask>")
    and numbers / years / quoted or evidence-quoted strings ("<unk>"); no cell-value DB lookups;
  * query skeleton from sqlparse + schema -> regex tokenizer with the original post-processing rules;
  * a one-line system message asks a chat model to complete the SQL (the original used completion-style `SELECT ` suffixes);
  * k = 5 (EXPERIMENT_SPEC section 4; the repo's BIRD scripts use 7);
  * `schema_style`: "rich" (default) = the same DDL with column descriptions and sample values every other baseline sees
    (data.get_schema()["ddl"]); "plain" = bare CREATE TABLE statements as in the original SQLPrompt;
  * example pool = shared train pool only (`bird_train_pool` = POOL57), never the target database.
`organization`: "dail" (DAIL_O, question+SQL pairs, the DAIL-SQL system) or "full" (FI_O: schema+question+SQL per example,
which EXPERIMENT_SPEC section 4 literally names "full information"; costs many more tokens).
"""
import re
import threading

import numpy as np

from ._common import (BaselineSystem, DEFAULT_EMBED_MODEL, datamod, embed_matrix, embed_one, extract_sql,
                      get_schema_cached, one_line, raw_ddl, schema_text)

SYSTEM = "Complete the SQLite query for the last request. Output only the SQL query."
PREFIX = "/* Some SQL examples are provided based on similar problems: */\n"
TH = 0.85

# ----------------------------------------------------------------------------- skeleton (utils.py sql2skeleton, regex tokenizer)
_KW = set("""select from where group by order having limit offset join inner left right outer cross natural on using as and or not in
like between is null exists distinct union intersect except all case when then else end asc desc count sum avg min max cast
substr substring strftime date datetime julianday round abs length lower upper replace ifnull coalesce iif instr trim total
integer real text numeric float int varchar with over partition row_number glob true false current_date""".split())
_TOK = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"])*\"|`[^`]*`|\[[^\]]*\]|\d+(?:\.\d+)?|[A-Za-z_][\w$]*(?:\.[A-Za-z_*][\w$]*)*|<=|>=|!=|<>|==|\|\||[^\s\w]")


def sql_skeleton(sql):
    s = sql.strip().rstrip(";").lower()
    toks = []
    for t in _TOK.findall(s):
        if t[0] in "'\"`[":
            toks.append("_")
        elif re.fullmatch(r"\d+(?:\.\d+)?", t):
            toks.append("_")
        elif re.fullmatch(r"[a-z_][\w$]*(?:\.[a-z_*][\w$]*)*", t):
            toks.append(t if t in _KW else "_")
        else:
            toks.append(t)
    # drop aliases "as _"
    out, i = [], 0
    while i < len(toks):
        if toks[i] == "as" and i + 1 < len(toks) and toks[i + 1] == "_":
            i += 2
            continue
        out.append(toks[i])
        i += 1
    sk = " ".join(out)
    # --- post-processing copied from the original
    sk = sk.replace("on _ = _ and _ = _", "on _ = _")
    sk = sk.replace("on _ = _ or _ = _", "on _ = _")
    sk = sk.replace(" on _ = _", "")
    sk = re.sub(re.compile(r"_ (?:join _ ?)+"), "_ ", sk)
    while "_ , _" in sk:
        sk = sk.replace("_ , _", "_")
    for op in ["=", "!=", ">", ">=", "<", "<="]:
        if "_ {} _".format(op) in sk:
            sk = sk.replace("_ {} _".format(op), "_")
    while "where _ and _" in sk or "where _ or _" in sk:
        if "where _ and _" in sk:
            sk = sk.replace("where _ and _", "where _")
        if "where _ or _" in sk:
            sk = sk.replace("where _ or _", "where _")
    while "  " in sk:
        sk = sk.replace("  ", " ")
    parts = sk.split(" ")
    for i in range(2, len(parts)):
        if parts[i - 2] == "order" and parts[i - 1] == "by" and parts[i] != "_":
            parts[i] = "_"
    return " ".join(parts)


def jaccard(a, b):
    """utils.jaccard_similarity (multiset Jaccard over whitespace tokens), verbatim semantics."""
    t1, t2 = a.strip().split(" "), b.strip().split(" ")
    d1, d2 = {}, {}
    for t in t1:
        d1[t] = d1.get(t, 0) + 1
    for t in t2:
        d2[t] = d2.get(t, 0) + 1
    inter = sum(min(v, d2[k]) for k, v in d1.items() if k in d2)
    union = len(t1) + len(t2) - inter
    return inter / union if union else 0.0


# ----------------------------------------------------------------------------- question masking (our adaptation)
_STOP = set("the of a an and or in on at to for by with from is are was were be been what which who whom whose how many much "
            "list name give show tell find count number total all any each per than that this these those its their his her "
            "have has had do does did not no yes id".split())
_QTOK = re.compile(r"'[^']+'|\"[^\"]+\"|\d+(?:[.,/\-:]\d+)*%?|\w+|[^\w\s]")


def _norm_word(w):
    w = w.lower()
    for suf in ("ies", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[:-len(suf)] + ("y" if suf == "ies" else "")
    return w


def _name_words(name):
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return [_norm_word(w) for w in re.split(r"[\s_\-/()]+", name) if w]


def schema_vocab(item):
    """(phrases, words): normalised multiword phrases and single words of table/column names."""
    sch = get_schema_cached(item)
    phrases, words = set(), set()
    for t, cols in (sch.get("tables") or {}).items():
        names = [t] + [c["name"] for c in cols]
        for n in names:
            ws = _name_words(n)
            if ws:
                phrases.add(" ".join(ws))
            for w in ws:
                if len(w) >= 4 and w not in _STOP:
                    words.add(w)
    return phrases, words


def mask_question(text, evidence, phrases, words, mask="<mask>", unk="<unk>"):
    toks = _QTOK.findall(text)
    norm = [_norm_word(t) for t in toks]
    out = list(toks)
    used = [False] * len(toks)
    # value mask: numbers, quoted strings, strings quoted in the evidence
    ev_vals = {v.strip("'\" ").lower() for v in re.findall(r"'[^']+'|\"[^\"]+\"", evidence or "") if len(v) > 3}
    for i, t in enumerate(toks):
        if re.fullmatch(r"\d+(?:[.,/\-:]\d+)*%?", t) or t[0] in "'\"" or t.strip("'\" ").lower() in ev_vals:
            out[i], used[i] = unk, True
    # schema mask: longest phrase first (3..1 grams)
    for n in (3, 2, 1):
        for i in range(len(toks) - n + 1):
            if any(used[i:i + n]):
                continue
            ph = " ".join(norm[i:i + n])
            if ph in phrases and not (n == 1 and ph in _STOP):
                for j in range(i, i + n):
                    out[j], used[j] = mask, True
    for i, w in enumerate(norm):
        if not used[i] and len(w) >= 4 and w in words:
            out[i], used[i] = mask, True
    return " ".join(out)


# ----------------------------------------------------------------------------- pool index
_POOL = {}
_POOL_LOCK = threading.Lock()


class PoolIndex:
    def __init__(self, items, masked, skeletons, emb):
        self.items, self.masked, self.skeletons, self.emb = items, masked, skeletons, emb


def get_pool(llm, cfg, embed_model):
    split = cfg.get("pool_split", "bird_train_pool")
    key = (split, embed_model, cfg.get("pool_limit"))
    with _POOL_LOCK:
        if key in _POOL:
            return _POOL[key]
        items = datamod.load_split(split)
        if cfg.get("pool_limit"):
            items = items[: int(cfg["pool_limit"])]
        vocab = {}
        masked, skels = [], []
        for it in items:
            if it.db_id not in vocab:
                vocab[it.db_id] = schema_vocab(it)
            ph, wd = vocab[it.db_id]
            q = (it.question + " " + (it.evidence or "")).strip()
            masked.append(mask_question(q, it.evidence, ph, wd))
            skels.append(sql_skeleton(it.gold_sql))
        emb = embed_matrix(llm, embed_model, masked, cache_dir=cfg.get("emb_cache_dir"), name="dail_pool_" + split)
        _POOL[key] = PoolIndex(items, masked, skels, emb)
        return _POOL[key]


def select_examples(pool, q_emb, k, *, exclude_db, exclude_q, pre_skel=None, th=TH):
    d = np.sqrt(np.maximum(2.0 - 2.0 * (pool.emb @ q_emb), 0.0))
    order = [i for i in np.argsort(d) if pool.items[i].db_id != exclude_db and pool.items[i].question != exclude_q]
    if pre_skel is None:
        return order[:k]
    top = [i for i in order if jaccard(pool.skeletons[i], pre_skel) >= th][:k]
    if len(top) < k:
        top += [i for i in order if i not in set(top)][: k - len(top)]
    return top


# ----------------------------------------------------------------------------- prompts
def target_block(ddl, question_text):
    return "/* Given the following database schema: */\n%s\n\n/* Answer the following: %s */" % (ddl, question_text)


def ddl_of(item, style, qk=False):
    return schema_text(item, quote_keywords=qk) if style == "rich" else raw_ddl(item, quote_keywords=qk)


def example_block(ex, org, style="rich", qk=False):
    q = (ex.question + " " + (ex.evidence or "")).strip()
    if org == "full":
        ddl = ddl_of(ex, style, qk)
        return "%s\n%s" % (target_block(ddl, q), one_line(ex.gold_sql))
    return "/* Answer the following: %s */\n%s" % (q, one_line(ex.gold_sql))


def complete(sql_text):
    s = extract_sql(sql_text)
    if s and not re.match(r"(?is)^\s*(select|with|\()", s):
        s = "SELECT " + s
    return s


class DailSQL(BaselineSystem):
    name = "dail_sql"
    default_cfg = {"k": 5, "organization": "dail", "threshold": TH, "max_tokens": 1024,
                   "embedding_model": DEFAULT_EMBED_MODEL, "emb_cache_dir": None, "pool_split": "bird_train_pool",
                   "schema_style": "rich"}

    def _prompt(self, item, ddl, qtext, examples, org):
        tgt = target_block(ddl, qtext)
        if not examples:
            return tgt + "\nSELECT "
        blocks = [example_block(e, org, self.cfg["schema_style"], self.qk) for e in examples]
        return PREFIX + "\n\n".join(blocks + [tgt]) + "\nSELECT "

    def solve(self, item, seed, ctx):
        emodel = self.cfg["embedding_model"]
        pool = get_pool(self.llm, self.cfg, emodel)
        ph, wd = schema_vocab(item)
        qtext = (item.question + " " + (item.evidence or "")).strip()
        q_emb = embed_one(self.llm, emodel, mask_question(qtext, item.evidence, ph, wd))
        ddl = ddl_of(item, self.cfg["schema_style"], self.qk)
        k, org = int(self.cfg["k"]), self.cfg["organization"]
        # stage 1: preliminary query (EUCDISQUESTIONMASK)
        ex1 = [pool.items[i] for i in select_examples(pool, q_emb, k, exclude_db=item.db_id, exclude_q=item.question)]
        pre = complete(ctx.user(self._prompt(item, ddl, qtext, ex1, org), system=SYSTEM, tag="dail.pre"))
        # stage 2: EUCDISMASKPRESKLSIMTHR
        idx2 = select_examples(pool, q_emb, k, exclude_db=item.db_id, exclude_q=item.question,
                               pre_skel=sql_skeleton(pre) if pre else None, th=float(self.cfg["threshold"]))
        ex2 = [pool.items[i] for i in idx2]
        final = complete(ctx.user(self._prompt(item, ddl, qtext, ex2, org), system=SYSTEM, tag="dail.final"))
        return final or pre, {"iterations": 2, "actions": ["preliminary", "final"],
                              "extra": {"pre_sql": pre, "example_qids": [e.qid for e in ex2], "organization": org}}
