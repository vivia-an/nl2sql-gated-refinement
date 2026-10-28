"""MAC-SQL (Wang et al., COLING 2025, arXiv 2312.11242): Selector / Decomposer / Refiner.

Source: https://github.com/wbbeyourself/MAC-SQL, commit 31a9df5e0d520be4769be57a4b9022e5e34a14f4 (2025-02-27).
The upstream repository carries NO licence file.
Prompts: `_vendor/macsql_const.py` is a byte-identical copy of core/const.py (selector_template, decompose_template_bird,
refiner_template are used unchanged).  The schema renderer, the "need prune" rule, the value-example heuristics, the
Decomposer's last-```sql``` parsing and the Refiner's refine criteria for BIRD (empty result, a None in the first 5 rows, or an
execution error) are ported from core/agents.py / core/utils.py.
Adapted (our adaptation):
  * LLM client: openai==0.28 `ChatCompletion` at T=0.1 -> `LLMClient.chat` at the run temperature (T=0 for headline runs);
    one user message per call, as in the original;
  * the Refiner loop runs up to `max_refine` rounds (paper appendix: "a maximum of three rounds"); the public code shares
    MAX_ROUND=3 with Selector and Decomposer, which allows exactly ONE refinement and does not re-execute it -> use
    `max_refine: 1` to reproduce the repository behaviour;
  * if the Refiner reply has no ```sql block we keep the previous SQL (the original would emit an "error:" string);
  * BIRD `*_tables.json` is located next to the databases; if absent a Spider-style entry is synthesised from PRAGMAs;
  * per-database value-example statistics are cached on disk (`cache_dir`); the frequency-ordered distinct-value scan is capped
    at 5000 values per column (original: unbounded).
"""
import json
import os
import re
import sqlite3
import threading
import time
from copy import deepcopy

from ._common import BaselineSystem, qname
try:
    from ._vendor import macsql_const as C
except ImportError:                      # the upstream file has no licence and is not redistributed: python code/tools/fetch_macsql_const.py
    class _MissingConst(object):
        def __getattr__(self, name):
            raise ImportError("nl2sql/systems/baselines/_vendor/macsql_const.py is not part of this release; run: python code/tools/fetch_macsql_const.py")
    C = _MissingConst()

MAX_ROUND_REPO = 3

# ----------------------------------------------------------------------------- utils ported from core/utils.py
_SQL_PAT = re.compile(r"```sql(.*?)```", re.DOTALL)


def parse_sql_from_string(s):
    allq = [m.group(1).strip() for m in _SQL_PAT.finditer(s)]
    return allq[-1] if allq else "error: No SQL found in the input string"


def is_valid_date(date_str):
    if not isinstance(date_str, str):
        return False
    parts = date_str.split()
    if not parts:
        return False
    date_str = parts[0]
    if len(date_str) != 10:
        return False
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        year, month, day = map(int, date_str.split("-"))
        return not (year < 1 or month < 1 or month > 12 or day < 1 or day > 31)
    return False


def is_valid_date_column(col_value_lst):
    return all(is_valid_date(v) for v in col_value_lst)


def is_email(string):
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", string))


def check_selector_response(json_data):
    for k, v in json_data.items():
        if isinstance(v, str):
            if v not in ("keep_all", "drop_all"):
                return False
        elif not isinstance(v, list):
            return False
    return True


def parse_json(text):
    start = text.find("```json")
    end = text.find("```", start + 7)
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start + 7:end])
            return data if check_selector_response(data) else {}
        except Exception:
            return {}
    return {}


def add_prefix(sql):
    if not sql.startswith("SELECT") and not sql.startswith("select"):
        sql = "SELECT" + sql
    return sql


# ----------------------------------------------------------------------------- database info (core/agents.py Selector)
_TABLES_JSON = {}
_DBINFO = {}
_LOCK = threading.Lock()


def _find_tables_json(db_path):
    root = os.path.dirname(os.path.dirname(os.path.dirname(db_path)))
    for cand in os.listdir(root) if os.path.isdir(root) else []:
        if cand.endswith("tables.json"):
            return os.path.join(root, cand)
    return None


def _synth_entry(db_id, db_path):
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    tabs = [r[0] for r in con.execute("select name from sqlite_master where type='table' and name!='sqlite_sequence'")]
    cols, pks, fks, colidx = [(-1, "*")], [], [], {}
    for ti, t in enumerate(tabs):
        for _, name, typ, _, _, pk in con.execute("pragma table_info(`%s`)" % t):
            colidx[(t.lower(), name.lower())] = len(cols)
            cols.append((ti, name))
            if pk:
                pks.append(len(cols) - 1)
    for t in tabs:
        for r in con.execute("pragma foreign_key_list(`%s`)" % t):
            a, b = colidx.get((t.lower(), str(r[3]).lower())), colidx.get((str(r[2]).lower(), str(r[4]).lower()))
            if a is not None and b is not None:
                fks.append([a, b])
    con.close()
    return {"db_id": db_id, "table_names_original": tabs, "table_names": [t.replace("_", " ") for t in tabs],
            "column_names_original": cols, "column_names": [(i, n.replace("_", " ")) for i, n in cols],
            "primary_keys": pks, "foreign_keys": fks}


def get_tables_entry(item):
    path = _find_tables_json(item.db_path)
    with _LOCK:
        if path and path not in _TABLES_JSON:
            _TABLES_JSON[path] = {d["db_id"]: d for d in json.load(open(path))}
        entry = _TABLES_JSON.get(path, {}).get(item.db_id) if path else None
    if entry is None:
        entry = _synth_entry(item.db_id, item.db_path)
    item_ = deepcopy(entry)
    names = item_["table_names"]
    item_["table_count"] = len(names)
    cnt = [0] * len(names)
    for tb_idx, _ in item_["column_names"]:
        if tb_idx >= 0:
            cnt[tb_idx] += 1
    item_["max_column_count"] = max(cnt)
    item_["total_column_count"] = sum(cnt)
    item_["avg_column_count"] = sum(cnt) // len(names)
    return item_


def _value_examples_str(values, col_type, dataset="bird"):
    if not values:
        return ""
    if len(values) > 10 and col_type in ["INTEGER", "REAL", "NUMERIC", "FLOAT", "INT"]:
        return ""
    vals, has_null = [], False
    for v in values:
        if v is None:
            has_null = True
        else:
            if str(v).strip() == "":
                continue
            vals.append(v)
    if not vals:
        return ""
    if col_type in ["TEXT", "VARCHAR"]:
        new_values = []
        for v in vals:
            if not isinstance(v, str):
                new_values.append(v)
            else:
                if dataset == "spider":
                    v = v.strip()
                if v == "":
                    continue
                elif ("https://" in v) or ("http://" in v):
                    return ""
                elif is_email(v):
                    return ""
                else:
                    new_values.append(v)
        vals = new_values
        lens = [len(str(a)) for a in vals]
        if not lens:
            return ""
        if max(lens) > 50:
            return ""
    if not vals:
        return ""
    vals = vals[:6]
    if is_valid_date_column(vals):
        vals = vals[:1]
    if has_null:
        vals.insert(0, None)
    return str(vals)


def load_db_info(item, tables_entry, cache_dir=None):
    """Port of Selector._load_single_db_info (value examples by frequency, pk/fk dicts)."""
    db_id = item.db_id
    with _LOCK:
        if db_id in _DBINFO:
            return _DBINFO[db_id]
    cpath = os.path.join(cache_dir, "macsql_dbinfo_%s.json" % db_id) if cache_dir else None
    if cpath and os.path.exists(cpath):
        info = json.load(open(cpath))
        with _LOCK:
            _DBINFO[db_id] = info
        return info
    db_dict = tables_entry
    important = []
    for col_id in db_dict["primary_keys"] + db_dict["foreign_keys"]:
        important.extend(col_id if isinstance(col_id, list) else [col_id])
    conn = sqlite3.connect("file:%s?mode=ro" % item.db_path, uri=True)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    cur = conn.cursor()
    desc, value, pk_d, fk_d = {}, {}, {}, {}
    tnames = db_dict["table_names_original"]
    allcols = db_dict["column_names_original"]
    for tb_idx, tb in enumerate(tnames):
        col2 = []
        names, keyflags = [], []
        for ci, (rt, oc) in enumerate(allcols):
            if rt != tb_idx:
                continue
            names.append(oc)
            keyflags.append(ci in important)
            full = db_dict["column_names"][ci][1].replace("_", " ")
            col2.append([oc, full, ""])
        desc[tb] = col2
        fk_d[tb], pk_d[tb] = [], []
        cur.execute("PRAGMA table_info(`%s`)" % tb)
        pinfo = cur.fetchall()
        sq_names, sq_types = [p[1] for p in pinfo], [p[2] for p in pinfo]
        key_cols = [names[i] for i, f in enumerate(keyflags) if f]
        col_vals = {}
        for idx, cname in enumerate(sq_names):
            if cname in key_cols:
                continue
            low = cname.lower()
            if low.endswith("id") or low.endswith("email") or low.endswith("url"):
                col_vals[cname] = ""
                continue
            try:
                cur.execute("SELECT `%s` FROM `%s` GROUP BY `%s` ORDER BY COUNT(*) DESC LIMIT 5000" % (cname, tb, cname))
                values = [v[0] for v in cur.fetchall()]
                col_vals[cname] = _value_examples_str(values, sq_types[idx])
            except Exception:
                col_vals[cname] = ""
        value[tb] = [[n, ("" if keyflags[i] else col_vals.get(n, ""))] for i, n in enumerate(names)]
    for fa, fb in db_dict["foreign_keys"]:
        ft, fc = tnames[allcols[fa][0]], allcols[fa][1]
        tt, tc = tnames[allcols[fb][0]], allcols[fb][1]
        fk_d[ft].append([fc, tt, tc])
    for pk in db_dict["primary_keys"]:
        for cp in (pk if isinstance(pk, list) else [pk]):
            pk_d[tnames[allcols[cp][0]]].append(allcols[cp][1])
    cur.close()
    conn.close()
    info = {"desc_dict": desc, "value_dict": value, "pk_dict": pk_d, "fk_dict": fk_d}
    if cpath:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cpath + ".tmp%d_%d" % (os.getpid(), threading.get_ident())
        json.dump(info, open(tmp, "w"))
        os.replace(tmp, cpath)
    with _LOCK:
        _DBINFO[db_id] = info
    return info


def _table_str(table_name, cols_desc, cols_val, qk=False):
    s = "# Table: %s\n" % qname(table_name, qk)
    infos = []
    for (cn, full, extra), (_, vstr) in zip(cols_desc, cols_val):
        extra = ("And " + str(extra)) if extra != "" and str(extra) != "nan" else ""
        extra = extra[:100]
        line = "  (" + "%s," % qname(cn, qk)
        if full != "":
            line += " %s." % full.strip()
        if vstr != "":
            line += " Value examples: %s." % vstr
        if extra != "":
            line += " %s" % extra
        line += "),"
        infos.append(line)
    return s + "[\n" + "\n".join(infos).strip(",") + "\n]" + "\n"


def desc_str(info, extracted, qk=False):
    """Port of Selector._get_db_desc_str (BIRD list format).  qk (X6, config ddl.quote_keywords): table / column names that are SQLite
    reserved words are shown double-quoted (display only; the selector's keep/drop lookups use the raw names).  False = original text."""
    out, fks = "", []
    for tb in info["desc_dict"]:
        cd, cv = info["desc_dict"][tb], info["value_dict"][tb]
        fk, pk = info["fk_dict"][tb], info["pk_dict"][tb]
        decision = extracted.get(tb, "")
        all_cols = [n for n, _, _ in cd]
        important = list(pk) + [n for n, _, _ in fk]
        if decision == "drop_all":
            ncd, ncv = deepcopy(cd[:6]), deepcopy(cv[:6])
        elif decision == "keep_all" or decision == "":
            ncd, ncv = deepcopy(cd), deepcopy(cv)
        else:
            ncd, ncv, app = [], [], []
            for i, col in enumerate(all_cols):
                if col in important or col in decision:
                    ncd.append(cd[i]); ncv.append(cv[i]); app.append(col)
            if len(all_cols) > 6 and len(ncv) < 6:
                for i, col in enumerate(all_cols):
                    if len(app) >= 6:
                        break
                    if col not in app:
                        ncd.append(cd[i]); ncv.append(cv[i]); app.append(col)
        out += _table_str(tb, ncd, ncv, qk)
        for cn, tt, tc in fk:
            cn_ = "`%s`" % cn if "`" not in str(cn) else cn
            tc_ = "`%s`" % tc if "`" not in str(tc) else tc
            link = "%s.%s = %s.%s" % (qname(tb, qk), cn_, qname(tt, qk), tc_)
            if link not in fks:
                fks.append(link)
    return out.strip(), "\n".join(fks).strip()


def need_prune(entry):
    return not (entry["avg_column_count"] <= 6 and entry["total_column_count"] <= 30)


# ----------------------------------------------------------------------------- Refiner execution (agents.py Refiner)
def exec_for_refiner(db_path, sql, timeout_s=30):
    conn = None
    try:
        conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        t0 = time.time()
        conn.set_progress_handler(lambda: 1 if time.time() - t0 > timeout_s else 0, 10000)
        cur = conn.cursor()
        cur.execute(sql)
        return {"sql": str(sql), "data": cur.fetchall()[:5], "sqlite_error": "", "exception_class": ""}
    except sqlite3.Error as er:
        return {"sql": str(sql), "sqlite_error": str(" ".join(str(a) for a in er.args)), "exception_class": str(er.__class__)}
    except Exception as e:
        return {"sql": str(sql), "sqlite_error": str(e.args), "exception_class": str(type(e).__name__)}
    finally:
        if conn is not None:
            conn.close()


def need_refine_bird(r):
    data = r.get("data", None)
    if data is None:
        return True
    if len(data) == 0:
        r["sqlite_error"] = "no data selected"
        return True
    for t in data:
        for n in t:
            if n is None:
                r["sqlite_error"] = "exist None value, you can add `NOT NULL` in SQL"
                return True
    return False


class MACSQL(BaselineSystem):
    name = "mac_sql"
    default_cfg = {"max_refine": 3, "max_tokens": 2048, "cache_dir": None, "force_selector": False}

    def solve(self, item, seed, ctx):
        entry = get_tables_entry(item)
        info = load_db_info(item, entry, self.cfg.get("cache_dir"))
        schema, fk = desc_str(info, {}, self.qk)
        evidence = item.evidence or ""
        actions, pruned = [], False
        if need_prune(entry) or self.cfg.get("force_selector"):
            prompt = C.selector_template.format(db_id=item.db_id, query=item.question, evidence=evidence, desc_str=schema, fk_str=fk)
            try:
                raw = parse_json(ctx.user(prompt, tag="macsql.selector"))
            except Exception:
                raw = {}
            schema, fk = desc_str(info, raw, self.qk)
            pruned = True
            actions.append("select")
        prompt = C.decompose_template_bird.format(query=item.question, desc_str=schema, fk_str=fk, evidence=evidence)
        reply = ctx.user(prompt, tag="macsql.decomposer").strip()
        sql = parse_sql_from_string(reply)
        actions.append("decompose")
        rounds = 0
        while rounds < int(self.cfg["max_refine"]):
            if "error" in sql:       # original: SQL containing "error" is never refined
                break
            r = exec_for_refiner(item.db_path, sql)
            if not need_refine_bird(r):
                break
            p = C.refiner_template.format(query=item.question, evidence=evidence, desc_str=schema, fk_str=fk,
                                          sql=add_prefix(r.get("sql")), sqlite_error=r.get("sqlite_error"),
                                          exception_class=r.get("exception_class"))
            new_sql = parse_sql_from_string(ctx.user(p, tag="macsql.refiner"))
            rounds += 1
            actions.append("refine")
            if new_sql.startswith("error:"):
                break
            sql = new_sql
        if sql.startswith("error:"):          # decomposer produced no ```sql block: lenient fallback
            from ._common import extract_sql
            sql = extract_sql(reply, "last")
        return sql, {"iterations": 1 + rounds, "actions": actions, "extra": {"pruned": pruned, "refine_rounds": rounds}}
