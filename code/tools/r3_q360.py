#!/usr/bin/env python3
"""R3 / E-Q: the selection of the 360-question corrected-reference subset (S_same of the primary gold variant G1), and what the stored data say about it.
Post hoc, descriptive, no model call (SAP section 26).

    worker machine:  PYTHONPATH=<code> python code/tools/r3_q360.py --runs $R/runs --data $R/data --rescore <results/robust/rescore.json> --out results/r3

1. The selection chain, reconstructed with the code that defined the subset (`rescore_corrected.build_sets`): dev 1 534 -> ids of the corrected release (498) ->
   minus databases whose gold differs between the dev and the Mini-Dev SQLite file (19) -> S_all (479) -> question and evidence text unchanged (360).
   The function takes only the dev rows, the corrected-release rows and the db-discordant list: no prediction, no EX, no model output enters it.
2. What the subset is, in stored original-reference outcomes: EX of every arm inside the 360, in the 1 174 other dev questions, in the 1 036 dev questions outside the
   corrected release; gated minus fixed on each; databases of the subset.
3. A bound on the full-set shift of the gated-minus-fixed contrast: shift on the 360 (from results/robust/rescore.json) scaled to the 1 534 under two assumptions
   about the 1 174 questions that have no corrected reference (no shift there; the same per-question shift as in the subset), plus the item-level interval of the shift."""
import argparse
import collections
import inspect
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3_common as C  # noqa: E402
from r3_common import S  # noqa: E402
import rescore_corrected as RC  # noqa: E402

ARM_LABELS = (("plus", "zs"), ("plus", "rootonly"), ("plus", "gated"), ("plus", "fixed"),
              ("flash", "zs"), ("flash", "rootonly"), ("flash", "gated"), ("flash", "fixed"),
              ("27b", "zs"), ("27b", "rootonly"), ("27b", "gated"), ("27b", "fixed"))


def selection_chain(data_dir, rescore):
    dev_by_id, variants = RC.load_variants(data_dir)
    var_by_id = variants["G1"]
    dev = dev_by_id
    var = var_by_id
    discord = set(rescore["variants"]["G1"]["db_discordant"])
    st = RC.build_sets(dev_by_id, var_by_id, discord)
    chain = collections.OrderedDict([
        ("dev_questions", len(dev_by_id)),
        ("release_rows", len(var)),
        ("release_distinct_ids", len(var_by_id)),
        ("release_ids_in_dev", sum(1 for q in var_by_id if q in dev_by_id)),
        ("db_discordant_removed", len(st["db_discordant"])),
        ("S_all", len(st["all"])),
        ("question_changed_in_S_all", sum(1 for q in st["all"] if st["q_changed"][q])),
        ("evidence_changed_in_S_all", sum(1 for q in st["all"] if st["e_changed"][q])),
        ("S_same", len(st["same"])),
    ])
    params = list(inspect.signature(RC.build_sets).parameters)
    return st, chain, params, dev_by_id, var_by_id


def run(runs_root, data_dir, rescore_path, B):
    rescore = json.load(open(rescore_path))
    st, chain, params, dev_by_id, var_by_id = selection_chain(data_dir, rescore)
    assert chain["S_same"] == rescore["variants"]["G1"]["sets"]["S_same"]["n"], "S_same differs from results/robust/rescore.json"
    assert chain["S_all"] == rescore["variants"]["G1"]["sets"]["S_all"]["n"], "S_all differs from results/robust/rescore.json"
    same, allq = set(st["same"]), set(st["all"])
    mini_ids = set(RC.qid_of(dev_by_id[q]) for q in var_by_id if q in dev_by_id)
    rows = {}
    for b, a in ARM_LABELS:
        r = C.load_preds(runs_root, C.run_path(b, a, 0))
        ok, why = C.check_run(r)
        if not ok:
            raise SystemExit("run %s %s seed 0: %s" % (b, a, why))
        rows[(b, a)] = r
    qids = sorted(set.intersection(*[set(r) for r in rows.values()]))
    ex = {k: np.array([C.ex_of(r[q]) for q in qids], float) for k, r in rows.items()}
    dbs = np.array([rows[("plus", "zs")][q]["db_id"] for q in qids])
    in_same = np.array([q in same for q in qids])
    in_mini = np.array([q in mini_ids for q in qids])
    sets = collections.OrderedDict([("S_same", in_same), ("other_dev", ~in_same), ("outside_release", ~in_mini)])
    out = {"chain": chain, "build_sets_params": params, "n_qids": len(qids), "sets": {k: int(v.sum()) for k, v in sets.items()}}
    # EX per set and arm
    out["ex"] = {}
    for (b, a), v in ex.items():
        out["ex"]["%s|%s" % (b, a)] = {sn: 100.0 * float(v[m].mean()) for sn, m in sets.items()}
        out["ex"]["%s|%s" % (b, a)]["all"] = 100.0 * float(v.mean())
    # gated minus fixed, loop effect, per set (original reference)
    out["contrast"] = {}
    for b in C.BK:
        for cid, x, y in (("g_minus_f", "gated", "fixed"), ("loop_g", "gated", "rootonly"), ("stack", "rootonly", "zs")):
            d = ex[(b, x)] - ex[(b, y)]
            for sn, m in sets.items():
                est, ci, ic = C.boot_ci(d[m], dbs[m], B=B)
                out["contrast"]["%s|%s|%s" % (cid, b, sn)] = {"est": est, "ci": ci, "item_ci": ic, "n": int(m.sum()), "n_dbs": int(len(set(dbs[m])))}
    # discordance rate gated vs fixed
    out["discordance"] = {}
    for b in C.BK:
        disc = (ex[(b, "gated")] != ex[(b, "fixed")])
        out["discordance"][b] = {sn: 100.0 * float(disc[m].mean()) for sn, m in sets.items()}
    # databases of the subset
    cnt_all = collections.Counter(dbs.tolist())
    cnt_same = collections.Counter(dbs[in_same].tolist())
    out["by_db"] = {d: {"dev": cnt_all[d], "S_same": cnt_same.get(d, 0)} for d in sorted(cnt_all)}
    # bound on the shift of gated - fixed
    N = len(qids)
    nS = int(in_same.sum())
    out["bound"] = {}
    con = rescore["variants"]["G1"]["sets"]["S_same"]["analysis"]["contrasts"]
    for name, key in (("pooled3", "H1.pooled3"), ("plus", "H1.plus"), ("flash", "H1.flash"), ("27b", "H1.q27b")):
        c = con[key]
        s = c["shift"]
        lo, hi = c["item_ci_shift"]
        clo, chi = c["ci_shift"]
        f = nS / float(N)
        out["bound"][name] = {"shift_subset": s, "ci_shift_cluster": [clo, chi], "ci_shift_item": [lo, hi], "weight": f,
                              "full_if_no_shift_elsewhere": s * f, "full_if_same_rate_elsewhere": s,
                              "full_item_lo": min(lo * f, lo), "full_item_hi": max(hi * f, hi),
                              "full_cluster_lo": min(clo * f, clo), "full_cluster_hi": max(chi * f, chi),
                              "orig_d0": c["d0"], "corr_d1": c["d1"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rescore", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=C.B_DEFAULT)
    a = ap.parse_args(argv)
    out = run(a.runs, a.data, a.rescore, a.B)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "q360.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    rows = [[k.split("|")[0], k.split("|")[1], k.split("|")[2], e["est"], e["ci"][0], e["ci"][1], e["item_ci"][0], e["item_ci"][1], e["n"], e["n_dbs"]] for k, e in sorted(out["contrast"].items())]
    C.write_csv(os.path.join(a.out, "q360_contrasts.csv"), ["contrast", "backbone", "set", "est_pp", "cl_lo", "cl_hi", "item_lo", "item_hi", "n", "n_dbs"], rows)
    C.write_csv(os.path.join(a.out, "q360_ex.csv"), ["backbone_arm", "S_same", "other_dev", "outside_release", "all"], [[k, e["S_same"], e["other_dev"], e["outside_release"], e["all"]] for k, e in sorted(out["ex"].items())])
    print("[r3_q360] chain", json.dumps(out["chain"]))


if __name__ == "__main__":
    main()
