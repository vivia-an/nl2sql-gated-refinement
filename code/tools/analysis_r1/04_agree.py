"""Q1-noise + Q2b: execution-agreement signals on HR trees. Needs out/exec_results.jsonl complete for HR trees.
 (a) sanity: recomputed EX (result hash == gold hash) vs stored node ex
 (b) stable-wrong / annotation-noise proxies
 (c) agreement of the root result with its depth-1 revisions as a confidence signal vs V3 score (AUROC, risk-coverage)"""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from execdata import *

Qs = load_trees()
miss = annotate(Qs)
res = {"missing_exec_results": miss}
dbs = np.array([q.db for q in Qs])
# (a) sanity
agree = tot = 0; bad = []
for q in Qs:
    for n in q.nodes:
        if n["rk"] == ("?",) or q.gkey in (("?",),):
            continue
        mine = int(n["rk"][0] == "H" and q.gkey[0] == "H" and n["rk"][1] == q.gkey[1])
        tot += 1; agree += int(mine == n["ex"])
        if mine != n["ex"] and len(bad) < 5:
            bad.append((q.qid, n["id"], n["ex"], mine, q.gkey[0]))
res["recomputed_EX_agreement"] = {"nodes": tot, "agree": agree, "rate": agree / max(1, tot), "examples_disagree": bad}
# (b) noise proxies
root_ex = np.array([q.root["ex"] for q in Qs])
gold_empty = np.array([bool(q.gold and q.gold["ok"] and q.gold["empty"]) for q in Qs])
gold_fail = np.array([bool(q.gold is None or not q.gold["ok"]) for q in Qs])
res["gold_empty_or_null_n"] = int(gold_empty.sum()); res["gold_exec_fail_n"] = int(gold_fail.sum())
res["root_EX_when_gold_empty"] = float(root_ex[gold_empty].mean()) if gold_empty.sum() else None
unreach = np.array([not any(n["ex"] == 1 for n in q.nodes) for q in Qs])
res["unreachable_all_nodes_n"] = int(unreach.sum()); res["unreachable_share_of_all"] = float(unreach.mean())
res["unreachable_share_of_wrong_roots"] = float(unreach[root_ex == 0].mean())
res["unreachable_given_gold_empty"] = float(unreach[gold_empty].mean()) if gold_empty.sum() else None
# modal share among nodes (weights) for questions that are wrong at root


def modal(q, nodes):
    c = collections.Counter()
    for n in nodes:
        c[n["rk"]] += n.get("weight", 1)
    k, v = c.most_common(1)[0]
    return k, v / sum(c.values()), len(c)


stable = []
for q in Qs:
    k, share, nd = modal(q, q.nodes)
    mod_correct = (k[0] == "H" and q.gkey[0] == "H" and k[1] == q.gkey[1])
    stable.append((share, nd, mod_correct))
stable = np.array(stable, dtype=float)
res["modal_answer_correct_share_of_all"] = float(stable[:, 2].mean())
res["modal_vote_EX_whole_tree"] = float(stable[:, 2].mean())
for th in (0.5, 0.8, 0.95):
    m = stable[:, 0] >= th
    res["modal_share>=%.2f" % th] = {"frac_questions": float(m.mean()), "EX_of_modal": float(stable[m, 2].mean()), "n_wrong": int((stable[m, 2] == 0).sum())}
sw = (stable[:, 0] >= 0.8) & (stable[:, 2] == 0)
res["stable_wrong_(>=80%_nodes_agree_on_a_wrong_result)"] = {"n": int(sw.sum()), "share_of_all": float(sw.mean()), "share_of_wrong_roots": float(sw[root_ex == 0].mean()),
                                                          "share_of_unreachable": float(sw[unreach].mean())}
# auto-class of wrong live finals
live = by_qid(jl(HR_RUN + "/preds.jsonl"))
ac = collections.Counter(live[q.qid]["auto_class"] for q in Qs if live[q.qid]["ex"] == 0)
res["live_final_wrong_by_auto_class"] = dict(ac)
res["live_final_wrong_n"] = int(sum(ac.values()))
amb = {"shape_mismatch", "column_permutation", "row_superset", "row_subset"}
res["ambiguity_type_classes_share_of_wrong"] = float(sum(v for k, v in ac.items() if k in amb) / sum(ac.values()))
res["ambiguity_type_classes_pp_of_all"] = float(100 * sum(v for k, v in ac.items() if k in amb) / len(Qs))
res["certain_data_problem_pp_of_all"] = float(100 * (ac.get("gold_fail", 0) + ac.get("timeout", 0)) / len(Qs))
# per-db unreachable
res["unreachable_by_db"] = {d: float(unreach[dbs == d].mean()) for d in sorted(set(dbs))}
dump_tmp = {}
# (c) agreement signals
y = root_ex


def wfrac(q, nodes):
    tot = sum(n["weight"] for n in nodes)
    return sum(n["weight"] for n in nodes if n["rk"] == q.root["rk"]) / tot if tot else float("nan")


sig = {}
sig["V3_score"] = np.array([q.root["score"] for q in Qs], float)
sig["agree_rev_T0"] = np.array([float((q.first_child(0, "revise") or {"rk": None})["rk"] == q.root["rk"]) for q in Qs])
sig["agree_frac_revise_d1"] = np.array([wfrac(q, q.children(0, "revise")) for q in Qs])
sig["agree_frac_tier1_d1(rev+rr)"] = np.array([wfrac(q, q.children(0, "revise") + q.children(0, "re_retrieve")) for q in Qs])
sig["agree_frac_all_d1"] = np.array([wfrac(q, q.depth1()) for q in Qs])
sig["agree_frac_whole_tree"] = np.array([wfrac(q, q.nodes[1:]) for q in Qs])
sig["agree_escalate"] = np.array([float((q.first_child(0, "escalate") or {"rk": None})["rk"] == q.root["rk"]) for q in Qs])
sig["exec_nonempty"] = np.array([float(q.root["status"] == "ok_nonempty") for q in Qs])
A = []
for k, s in sig.items():
    s2 = np.nan_to_num(s, nan=0.0)
    a = cluster_boot(dbs, lambda idx: auroc(s2[idx], y[idx]), B=1000)
    A.append({"signal": k, "AUROC": a[0], "lo": a[1], "hi": a[2]})
# combos: score + agree (rank sum) via LODO logistic
from sklearn.linear_model import LogisticRegression
def lodo(F):
    p = np.zeros(len(y))
    for d in np.unique(dbs):
        tr, te = dbs != d, dbs == d
        m = LogisticRegression(C=1.0, max_iter=500).fit(F[tr], y[tr]); p[te] = m.predict_proba(F[te])[:, 1]
    return p
combos = {"V3+agree_rev_T0": ["V3_score", "agree_rev_T0"], "V3+agree_frac_revise_d1": ["V3_score", "agree_frac_revise_d1"],
          "V3+agree_frac_all_d1": ["V3_score", "agree_frac_all_d1"], "V3+nonempty+agree_all_d1": ["V3_score", "exec_nonempty", "agree_frac_all_d1"],
          "agree_frac_all_d1+nonempty": ["agree_frac_all_d1", "exec_nonempty"]}
for nm, ks in combos.items():
    F = np.column_stack([np.nan_to_num(sig[k]) for k in ks]); p = lodo(F)
    a = cluster_boot(dbs, lambda idx: auroc(p[idx], y[idx]), B=1000)
    A.append({"signal": "LR(" + nm + ") LODO", "AUROC": a[0], "lo": a[1], "hi": a[2]})
res["auroc_root_signals_HR"] = A
# risk-coverage at natural thresholds for agreement
rc = []
for nm, s in (("agree_rev_T0=1", sig["agree_rev_T0"] >= 1), ("agree_all_d1=1.0", sig["agree_frac_all_d1"] >= 1.0), ("agree_all_d1>=0.6", sig["agree_frac_all_d1"] >= 0.6),
              ("agree_all_d1=1 & V3>=9.5", (sig["agree_frac_all_d1"] >= 1.0) & (sig["V3_score"] >= 9.5)), ("V3>=9.5", sig["V3_score"] >= 9.5), ("V3==10", sig["V3_score"] >= 10)):
    e = cluster_boot(dbs, lambda idx: (y[idx][s[idx]].mean() if s[idx].sum() else float("nan")), B=1000)
    c = cluster_boot(dbs, lambda idx: s[idx].mean(), B=1000)
    rc.append({"rule": nm, "coverage": c, "EX_accepted": e, "n": int(s.sum())})
res["risk_coverage_agreement"] = rc
# conditional: among questions where all depth-1 agree with root, what is EX; where they disagree
dis = sig["agree_frac_all_d1"] < 1.0
res["root_EX_if_all_d1_agree"] = float(y[~dis].mean()); res["root_EX_if_any_d1_disagrees"] = float(y[dis].mean()); res["share_any_d1_disagree"] = float(dis.mean())
dump("q2b_agree.json", res)
print(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
