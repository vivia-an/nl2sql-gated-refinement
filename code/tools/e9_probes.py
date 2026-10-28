"""E9 contamination probes (EXPERIMENT_SPEC E9), 8 models x 2 probes.
Probe-Q: db_id + first 50 % of the question words -> "Continue the question exactly as in the dataset" (T=0, max_tokens 128);
         reproduced iff normalised Levenshtein similarity to the true remainder >= 0.9 (0.7 also reported).
Probe-SQL: db_id + question + evidence, NO schema; exact match of the sqlglot-normalised SQL to gold, plus table-name overlap.
python tools/e9_probes.py --split bird_dev --model qwen37plus --out <dir> [--limit N] [--sample 300 --seed 20261027]
Locked splits need an exp/dev_lock.json entry for system `e9_probes` (config = this tool's yaml, e.g. exp/configs/e9.yaml)."""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nl2sql import config as C, data, lock, paths  # noqa: E402
from nl2sql.llm import LLMClient, LLMError  # noqa: E402
from nl2sql.util import strip_sql  # noqa: E402


def lev(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def sim(a, b):
    a, b = " ".join(a.lower().split()), " ".join(b.lower().split())
    m = max(len(a), len(b), 1)
    return 1.0 - lev(a, b) / m


def norm_sql(s):
    try:
        import sqlglot
        return sqlglot.parse_one(s, read="sqlite").sql(dialect="sqlite", normalize=True).lower()
    except Exception:
        return " ".join(s.lower().split())


def tables_of(s):
    try:
        import sqlglot
        from sqlglot import exp
        return {t.name.lower() for t in sqlglot.parse_one(s, read="sqlite").find_all(exp.Table) if t.name}
    except Exception:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", default="exp/configs/e9.yaml")
    a = ap.parse_args()
    cfg = a.config if os.path.isabs(a.config) else str(paths.REPO / a.config)
    ok, msg = lock.verify("e9_probes", a.split, 0, cfg)
    if not ok:
        sys.exit("REFUSED: " + msg)
    model = C.resolve_model(a.model)
    os.makedirs(a.out, exist_ok=True)
    llm = LLMClient(str(paths.CACHE / "llm"), os.path.join(a.out, "usage.jsonl"), paths.KEY_FILE)
    items = data.load_split(a.split)
    gold = data.load_gold(a.split, run_id="e9-" + a.model)
    if a.sample:
        items = random.Random(a.seed).sample(items, min(a.sample, len(items)))
    if a.limit:
        items = items[: a.limit]

    def one(it):
        words = it.question.split()
        h = len(words) // 2
        head, tail = " ".join(words[:h]), " ".join(words[h:])
        r = {"qid": it.qid}
        try:
            q = llm.chat(model, [{"role": "user", "content": "Database: %s\nThe following is the beginning of a question from a text-to-SQL dataset. Continue the question exactly as in the dataset. Return only the continuation.\n\n%s" % (it.db_id, head)}],
                         temperature=0.0, max_tokens=128, tag="probe_q")["text"].strip()
            r["q_sim"] = sim(q, tail)
            s = llm.chat(model, [{"role": "user", "content": "Database: %s\nQuestion: %s\nEvidence: %s\nWrite the SQLite query. You do not see the schema; use the table and column names you expect. Return only the SQL." % (it.db_id, it.question, it.evidence or "(none)")}],
                         temperature=0.0, max_tokens=512, tag="probe_sql")["text"]
            sql = strip_sql(s)
            g = gold.get(it.qid, "")
            r["sql_exact"] = int(bool(g) and norm_sql(sql) == norm_sql(g))
            tg, tp = tables_of(g), tables_of(sql)
            r["table_overlap"] = (len(tg & tp) / len(tg | tp)) if (tg | tp) else 0.0
        except LLMError as e:
            r["error"] = str(e)[:100]
        return r

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, items))
    with open(os.path.join(a.out, "probe_rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok_rows = [r for r in rows if "error" not in r]
    n = max(1, len(ok_rows))
    summ = {"split": a.split, "model": model, "n": len(rows), "n_ok": len(ok_rows),
            "q_repro_0.9": sum(r["q_sim"] >= 0.9 for r in ok_rows) / n, "q_repro_0.7": sum(r["q_sim"] >= 0.7 for r in ok_rows) / n,
            "sql_exact": sum(r["sql_exact"] for r in ok_rows) / n, "table_overlap": sum(r["table_overlap"] for r in ok_rows) / n}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
