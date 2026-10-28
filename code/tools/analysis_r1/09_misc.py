"""Misc numbers: (a) sizes of the decisive pairwise-judge / selection test sets on HR trees; (b) reserved-word identifiers in BIRD *train* schemas
(dev schemas deliberately not inspected); (c) root repeatability (live vs tree root); (d) MDE."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from execdata import *
Qs = load_trees(); annotate(Qs)
res = {}
# (a)
dis = one_correct = 0; dis_d1 = one_d1 = 0; rev_wins = rev_loses = 0
for q in Qs:
    r0 = q.first_child(0, "revise")
    if r0 is None: continue
    if r0["rk"] != q.root["rk"]:
        dis += 1
        if r0["ex"] != q.root["ex"]:
            one_correct += 1
            rev_wins += int(r0["ex"] == 1); rev_loses += int(r0["ex"] == 0)
    alts = {c["rk"] for c in q.depth1()} | {q.root["rk"]}
    if len(alts) > 1:
        dis_d1 += 1
        ex = {c["ex"] for c in q.depth1()} | {q.root["ex"]}
        if ex == {0, 1}: one_d1 += 1
res["root_vs_revise_T0"] = {"disagree_questions": dis, "exactly_one_correct": one_correct, "revision_correct": rev_wins, "root_correct": rev_loses}
res["depth1_pool"] = {"questions_with_disagreement": dis_d1, "with_both_a_correct_and_a_wrong_candidate": one_d1}
# per-db
res["pool_disagree_by_db"] = {d: sum(1 for q in Qs if q.db == d and len({c["rk"] for c in q.depth1()} | {q.root["rk"]}) > 1) for d in sorted({q.db for q in Qs})}
# (b) reserved words
KW = set("""abort action add after all alter always analyze and as asc attach autoincrement before begin between by cascade case cast check collate column commit conflict constraint create cross current current_date current_time current_timestamp database default deferrable deferred delete desc detach distinct do drop each else end escape except exclude exclusive exists explain fail filter first following for foreign from full generated glob group groups having if ignore immediate in index indexed initially inner insert instead intersect into is isnull join key last left like limit match materialized natural no not nothing notnull null nulls of offset on or order others outer over partition plan pragma preceding primary query raise range recursive references regexp reindex release rename replace restrict returning right rollback row rows savepoint select set table temp temporary then ties to transaction trigger unbounded union unique update using vacuum values view virtual when where window with without""".split())
tabs = json.load(open(R + "/data/bird/train/train/train_tables.json"))
train = json.load(open(TRAIN))
nq = collections.Counter(r["db_id"] for r in train)
hits = {}
for t in tabs:
    th = [x for x in t["table_names_original"] if x.lower() in KW]
    ch = [c for _, c in t["column_names_original"] if c != "*" and c.lower() in KW]
    if th or ch:
        hits[t["db_id"]] = {"tables": th, "columns": sorted(set(ch)), "n_questions": nq[t["db_id"]]}
res["train_dbs_total"] = len(tabs); res["train_dbs_with_keyword_identifiers"] = len(hits)
res["train_dbs_with_keyword_TABLE_names"] = {d: v for d, v in hits.items() if v["tables"]}
res["train_questions_in_dbs_with_keyword_table"] = int(sum(v["n_questions"] for v in hits.values() if v["tables"]))
res["train_questions_total"] = len(train)
res["keyword_column_dbs"] = {d: v["columns"] for d, v in hits.items() if v["columns"]}
# (c) repeatability
live = by_qid(jl(HR_RUN + "/cands.jsonl"))
# (d) MDE
def mde(pd, n, deff, z=1.96 + 0.84): return z * math.sqrt(pd * deff / n)
res["MDE_pp"] = {"n1101_pd0.03_deff2": 100 * mde(0.03, 1101, 2), "n1101_pd0.05_deff2": 100 * mde(0.05, 1101, 2), "n1534_pd0.05_deff2.4": 100 * mde(0.05, 1534, 2.4), "n1534_pd0.03_deff2.4": 100 * mde(0.03, 1534, 2.4), "n300_pd0.08_deff2": 100 * mde(0.08, 300, 2)}
dump("q9_misc.json", res)
print(json.dumps(res, indent=1))
