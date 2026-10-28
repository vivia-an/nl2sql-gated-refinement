"""Q3 on HT-300: cross-prompt voting between independent-ish systems on the same model (zs, zs_ours, full_fixed root/final), execution-result plurality.
The three members use different prompts (B1 zero-shot, our zero-shot, M1-M4 RAG prompt) but the same model (qwen3.7-plus, T=0): not independent samples."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from execdata import *
zb = by_qid(jl(HT_ZS_B1 + "/preds.jsonl")); zo = by_qid(jl(HT_ZS + "/preds.jsonl")); ff = by_qid(jl(HT_RUN + "/preds.jsonl"))
cands = collections.defaultdict(list)
for c in jl(HT_RUN + "/cands.jsonl"): cands[c["qid"]].append(c)
qs = sorted(ff); d = np.array([ff[q]["db_id"] for q in qs])
def K(q, sql): return key(ff[q]["db_id"], sql)
gk = {q: K(q, gold_sql(q)) for q in qs}
mem = {"zs": {q: (K(q, zb[q]["final_sql"]), zb[q]["ex"]) for q in qs}, "zs_ours": {q: (K(q, zo[q]["final_sql"]), zo[q]["ex"]) for q in qs},
       "full_root": {q: (K(q, cands[q][0]["sql"]), cands[q][0]["ex"]) for q in qs}, "full_final": {q: (K(q, ff[q]["final_sql"]), ff[q]["ex"]) for q in qs}}
# sanity: recomputed ex vs stored
sane = {m: float(np.mean([int(v[0][0] == "H" and gk[q][0] == "H" and v[0][1] == gk[q][1]) == v[1] for q, v in mem[m].items()])) for m in mem}
def vote(names, tiebreak):
    ex = []
    for q in qs:
        c = collections.Counter(mem[m][q][0] for m in names)
        c = {k: v for k, v in c.items() if k[0] != "E"} or c
        top = max(c.values()); winners = [k for k, v in c.items() if v == top]
        if len(winners) == 1: k = winners[0]
        else: k = mem[tiebreak][q][0] if mem[tiebreak][q][0] in winners else winners[0]
        # correct iff the winning key is the gold key
        ex.append(int(k[0] == "H" and gk[q][0] == "H" and k[1] == gk[q][1]))
    return np.array(ex)
res = {"sanity_recomputed_vs_stored_EX": sane}
rows = []
def add(nm, ex, calls):
    e = cluster_boot(d, lambda idx: ex[idx].mean(), B=2000)
    rows.append({"policy": nm, "EX": e, "calls": calls})
for m, cl in (("zs", 1.0), ("zs_ours", 1.0), ("full_root", 3.0), ("full_final", 3.25)):
    add(m, np.array([mem[m][q][1] for q in qs]), cl)
add("vote3(zs, zs_ours, full_final) tie->full_final", vote(["zs", "zs_ours", "full_final"], "full_final"), 1 + 1 + 3.25)
add("vote3(zs, zs_ours, full_final) tie->zs_ours", vote(["zs", "zs_ours", "full_final"], "zs_ours"), 1 + 1 + 3.25)
add("vote3(zs, zs_ours, full_root) tie->zs_ours", vote(["zs", "zs_ours", "full_root"], "zs_ours"), 1 + 1 + 3.0)
add("oracle(any of zs, zs_ours, full_final)", np.array([max(mem[m][q][1] for m in ("zs", "zs_ours", "full_final")) for q in qs]), 5.25)
add("oracle(any of zs, zs_ours)", np.array([max(mem[m][q][1] for m in ("zs", "zs_ours")) for q in qs]), 2.0)
# full pool of stored HT candidates (root + revised) exec vote
def vote_cands():
    ex = []
    for q in qs:
        cs = cands[q]; c = collections.Counter(K(q, x["sql"]) for x in cs)
        k = max(c, key=lambda k: (c[k], k == K(q, cs[0]["sql"])))
        ex.append(int(k[0] == "H" and gk[q][0] == "H" and k[1] == gk[q][1]))
    return np.array(ex)
res["policies"] = rows
dis = np.array([len({mem[m][q][0] for m in ("zs", "zs_ours", "full_final")}) for q in qs])
res["distinct_results_among_3_systems"] = {str(k): int((dis == k).sum()) for k in (1, 2, 3)}
agree_all = dis == 1
ex_full = np.array([mem["full_final"][q][1] for q in qs])
res["EX_when_all_three_agree"] = float(ex_full[agree_all].mean()); res["EX_when_not_all_agree"] = float(ex_full[~agree_all].mean()); res["share_all_agree"] = float(agree_all.mean())
# pairwise agreement rates (result-level)
res["pairwise_result_agreement"] = {"%s~%s" % (a, b): float(np.mean([mem[a][q][0] == mem[b][q][0] for q in qs])) for a, b in (("zs", "zs_ours"), ("zs", "full_final"), ("zs_ours", "full_final"), ("full_root", "zs_ours"))}
dump("q3_ht_vote.json", res)
print(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
