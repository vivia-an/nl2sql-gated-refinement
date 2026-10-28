"""Cross-check db.ex_match / db.execute against the OFFICIAL BIRD execute_sql (DAMO-ConvAI evaluation.py, vendored copy in
NL2SQL_DATA/eval_tools/bird_official) on 200 gold/pred pairs: 60 gold==gold, 60 gold vs other-question gold (same db),
40 gold vs hand-made mutations, 20 syntactically broken, 20 timeouts-ish (cross joins are skipped; the official script
has no per-statement timeout so we only use fast queries). Writes a JSON report next to the script output.
Usage: python tests/crosscheck_bird_official.py [out.json]"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nl2sql import data, db, paths  # noqa: E402

sys.path.insert(0, str(paths.DATA / "eval_tools" / "bird_official"))
import evaluation as official  # noqa: E402


def official_ex(pred, gold, dbp):
    try:
        return official.execute_sql(pred, gold, dbp)
    except Exception:
        return 0


def ours_ex(pred, gold, dbp):
    p = db.execute(dbp, pred, timeout_s=30, max_rows=None)
    g = db.execute(dbp, gold, timeout_s=30, max_rows=None)
    if not p["ok"] or not g["ok"]:
        return 0
    return db.ex_match(p["rows"], g["rows"])


def main():
    rnd = random.Random(7)
    items = data.load_split("bird_dev", with_gold=True)         # offline cross-check; not a pipeline run
    gold = {i.qid: i for i in items}
    pairs = []
    sample = rnd.sample(items, 200)
    for k, it in enumerate(sample):
        g = it.gold_sql
        if k < 60:
            pred, kind = g, "gold_gold"
        elif k < 120:
            other = rnd.choice([x for x in items if x.db_id == it.db_id])
            pred, kind = other.gold_sql, "other_gold"
        elif k < 160:
            m = rnd.choice(["limit", "distinct", "swap", "where"])
            if m == "limit":
                pred = "SELECT * FROM (%s) LIMIT 1" % g.rstrip(";")
            elif m == "distinct":
                pred = "SELECT DISTINCT * FROM (%s)" % g.rstrip(";")
            elif m == "swap":
                pred = g.replace(" ASC", " DESC").replace(">", "<")
            else:
                pred = "SELECT * FROM (%s) WHERE 1=0" % g.rstrip(";")
            kind = "mutation_" + m
        elif k < 180:
            pred, kind = g.replace("SELECT", "SELEC", 1), "broken"
        else:
            pred, kind = "SELECT * FROM (%s) ORDER BY 1 DESC" % g.rstrip(";"), "reordered"
        pairs.append((it, pred, kind))
    agree = 0
    rows = []
    for it, pred, kind in pairs:
        a, b = official_ex(pred, it.gold_sql, it.db_path), ours_ex(pred, it.gold_sql, it.db_path)
        agree += int(a == b)
        rows.append({"qid": it.qid, "kind": kind, "official": a, "ours": b})
    rep = {"n": len(rows), "agree": agree, "disagree": [r for r in rows if r["official"] != r["ours"]],
           "positives_official": sum(r["official"] for r in rows), "positives_ours": sum(r["ours"] for r in rows)}
    out = sys.argv[1] if len(sys.argv) > 1 else None
    if out:
        json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "disagree"}), "disagreements:", len(rep["disagree"]))
    assert agree == len(rows), rep["disagree"][:5]


if __name__ == "__main__":
    main()
