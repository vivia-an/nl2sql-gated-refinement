"""HT-300 counterpart of Q1: root vs final (live), vs zero-shot, revision gain/harm by root status. 4 dbs only (CIs are wide)."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
hp = by_qid(jl(HT_RUN + "/preds.jsonl")); zo = by_qid(jl(HT_ZS + "/preds.jsonl")); zb = by_qid(jl(HT_ZS_B1 + "/preds.jsonl"))
cands = collections.defaultdict(list)
for c in jl(HT_RUN + "/cands.jsonl"):
    cands[c["qid"]].append(c)
qs = sorted(hp)
d = np.array([hp[q]["db_id"] for q in qs])
f = np.array([hp[q]["ex"] for q in qs]); r = np.array([cands[q][0]["ex"] for q in qs]); a = np.array([zo[q]["ex"] for q in qs]); b = np.array([zb[q]["ex"] for q in qs])
res = {}
def pc(x, y):
    dd = (x - y).astype(float)
    return cluster_boot(d, lambda idx: dd[idx].mean())
res["EX"] = {"full_final": float(f.mean()), "full_root": float(r.mean()), "zs_ours": float(a.mean()), "zs_b1": float(b.mean())}
res["delta_full_vs_zs_ours"] = pc(f, a); res["delta_full_vs_zs"] = pc(f, b); res["delta_final_vs_root"] = pc(f, r); res["delta_root_vs_zs_ours"] = pc(r, a)
res["mcnemar_full_vs_zs_ours"] = [int(((f == 1) & (a == 0)).sum()), int(((f == 0) & (a == 1)).sum())]
res["mcnemar_final_vs_root"] = [int(((f == 1) & (r == 0)).sum()), int(((f == 0) & (r == 1)).sum())]
res["mcnemar_root_vs_zs_ours"] = [int(((r == 1) & (a == 0)).sum()), int(((r == 0) & (a == 1)).sum())]
# agreement between full root and zs_ours (same model, different prompt) -> noise floor of prompt choice
res["root_vs_zs_ours_sql_equal"] = float(np.mean([cands[q][0]["sql"].strip() == zo[q]["final_sql"].strip() for q in qs]))
res["root_vs_zs_ours_ex_disagree"] = float((r != a).mean())
res["zs_vs_zs_ours_ex_disagree"] = float((b != a).mean())
tab = collections.defaultdict(collections.Counter)
for q in qs:
    cs = cands[q]; root = cs[0]
    st = "exec_error" if root["status"].startswith("err") or root["status"] == "timeout" else root["status"]
    tab[st]["n"] += 1; tab[st]["gain"] += int(root["ex"] == 0 and hp[q]["ex"] == 1); tab[st]["harm"] += int(root["ex"] == 1 and hp[q]["ex"] == 0); tab[st]["revised"] += int(len(cs) > 1)
res["by_root_status"] = {k: dict(v) for k, v in tab.items()}
res["root_status_counts"] = dict(collections.Counter(cands[q][0]["status"] for q in qs))
per = {}
for dbn in sorted(set(d)):
    m = d == dbn
    per[dbn] = {"n": int(m.sum()), "full": float(f[m].mean()), "root": float(r[m].mean()), "zs_ours": float(a[m].mean()), "zs": float(b[m].mean())}
res["per_db"] = per
dump("q1_ht.json", res)
print(json.dumps(res, indent=1))
