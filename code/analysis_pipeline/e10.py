"""E10 error analysis: automatic classes from preds.jsonl, annotator-run agreement (kappa, AC1), and L1-L8 label prevalence from
LLM annotation labels (two independent Claude Sonnet 5.5 runs + adjudicator; stratum-weighted, stratified bootstrap).

Inputs (all optional; whatever exists is used):
  <annot_dir>/annotation_key.csv     item_id, qid, system, auto_class, stratum, stratum_population, stratum_sample, weight   (make_annotation_sheet.py)
  <annot_dir>/labels_A1.csv, labels_A2.csv (or annotation_A1/A2.csv filled in)   item_id, primary_label, ...    (independent LLM annotator runs)
  <annot_dir>/final_labels.csv       item_id, final_label (adjudicator output; or adjudicated.csv with primary_label)
  <annot_dir>/llm_labels.jsonl|csv   qid, system (optional), label (L1..L8)  or  item_id, label   (only if A1/A2 are absent)
  <annot_dir>/spot_check.csv         item_id, primary_label   (optional: authors' blind human spot check -> E10.spot.*)
Final label per item = adjudicated if present, else the shared label of A1/A2 when they agree, else missing."""
import csv
import json
import os

import numpy as np

from . import runs as R, stats as S

LABELS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"]


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_key(d):
    p = os.path.join(d, "annotation_key.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    for r in _read_csv(p):
        out[r["item_id"]] = r
    return out


def load_labels(path):
    """-> dict item_id-or-(qid,system) -> label."""
    out = {}
    if path.endswith(".jsonl"):
        for r in R.jl(path):
            lab = r.get("label") or r.get("primary_label") or r.get("final_label")
            if r.get("item_id"):
                out[r["item_id"]] = lab
            else:
                out[(r["qid"], r.get("system"))] = lab
    else:
        for r in _read_csv(path):
            lab = r.get("primary_label") or r.get("label") or r.get("final_label")
            if r.get("item_id"):
                out[r["item_id"]] = lab
            else:
                out[(r["qid"], r.get("system"))] = lab
    return out


def _resolve(labels, key):
    """map (qid, system)-keyed labels onto item_ids using the key."""
    out = {}
    by_q = {}
    for iid, r in key.items():
        by_q.setdefault((r["qid"], r["system"]), iid)
        by_q.setdefault((r["qid"], None), iid)
    for k, v in labels.items():
        if isinstance(k, tuple):
            iid = by_q.get(k) or by_q.get((k[0], None))
            if iid:
                out[iid] = v
        else:
            out[k] = v
    return {k: v for k, v in out.items() if v in LABELS}


def prevalence(items, labels, key, B=2000, seed=20261101):
    """items: list of item_ids with a label; stratified (system, stratum) weighting by stratum_population. -> {system: {label: (est, lo, hi, n)}}"""
    out = {}
    systems = sorted(set(key[i]["system"] for i in items))
    rng = np.random.RandomState(seed)
    for sysn in systems:
        its = [i for i in items if key[i]["system"] == sysn]
        strata = {}
        for i in its:
            strata.setdefault(key[i]["stratum"], []).append(i)
        pop = {s: float(key[v[0]].get("stratum_population") or len(v)) for s, v in strata.items()}
        tot = sum(pop.values())

        def est(sample):
            r = {l: 0.0 for l in LABELS}
            for s, ids in sample.items():
                if not ids:
                    continue
                w = pop[s] / tot
                for l in LABELS:
                    r[l] += w * np.mean([1.0 if labels[i] == l else 0.0 for i in ids])
            return r

        point = est(strata)
        bs = {l: [] for l in LABELS}
        for _ in range(B):
            smp = {s: [ids[j] for j in rng.randint(0, len(ids), len(ids))] for s, ids in strata.items()}
            e = est(smp)
            for l in LABELS:
                bs[l].append(e[l])
        out[sysn] = {l: (point[l],) + S.pctl_ci(bs[l]) + (sum(1 for i in its if labels[i] == l),) for l in LABELS}
        out[sysn]["_n"] = len(its)
    return out


def agreement(a, b, B=2000, seed=20261101):
    ids = [i for i in a if i in b]
    if len(ids) < 5:
        return None
    x, y = [a[i] for i in ids], [b[i] for i in ids]
    k = S.cohen_kappa(x, y)
    ac1 = S.gwet_ac1(x, y)
    pa = float(np.mean([1.0 if p == q else 0.0 for p, q in zip(x, y)]))
    x5, y5 = ["L5" if v == "L5" else "o" for v in x], ["L5" if v == "L5" else "o" for v in y]
    k5 = S.cohen_kappa(x5, y5)
    rng = np.random.RandomState(seed)
    kb, ab, k5b = [], [], []
    n = len(ids)
    for _ in range(B):
        j = rng.randint(0, n, n)
        xs, ys = [x[t] for t in j], [y[t] for t in j]
        kb.append(S.cohen_kappa(xs, ys)); ab.append(S.gwet_ac1(xs, ys))
        k5b.append(S.cohen_kappa(["L5" if v == "L5" else "o" for v in xs], ["L5" if v == "L5" else "o" for v in ys]))
    cm = [[sum(1 for p, q in zip(x, y) if p == l1 and q == l2) for l2 in LABELS] for l1 in LABELS]
    return {"n": n, "kappa": k, "kappa_ci": S.pctl_ci([v for v in kb if v is not None]), "ac1": ac1, "ac1_ci": S.pctl_ci([v for v in ab if v is not None]),
            "pct": pa, "kappa_l5": k5, "kappa_l5_ci": S.pctl_ci([v for v in k5b if v is not None]), "confusion": cm}


def auto_class_tables(res, labelled_runs, model):
    """counts of the automatic class of failed items per system (preds.auto_class / error_class)."""
    for lab, run in labelled_runs.items():
        rows = [r for r in run.rows.values() if R.ex_value(r) == 0]
        n_all = max(len(run.rows), 1)
        cnt = {}
        for r in rows:
            c = r.get("auto_class") or r.get("error_class") or "other"
            cnt[c] = cnt.get(c, 0) + 1
        res.set("E10.auto.%s.n_fail" % lab, len(rows)); res.set("E10.auto.%s.fail_rate.mean" % lab, len(rows) / float(n_all))
        for c, v in cnt.items():
            res.set("E10.auto.%s.%s.n" % (lab, c), v)
            res.set("E10.auto.%s.%s.share.mean" % (lab, c), v / float(max(len(rows), 1)))


def _discordant_summary(res, annot_dir):
    """<annot_dir>/discordant_summary.json (written by the annotation session, results/e10/) -> E10.revision.* / E10.repair_nonexec.* / E10.discordant.*.
    E10.revision.{n,helped,hurt} = the revising system `full_fixed` on all dev items (n = items whose final answer differs from the root answer; helped = wrong->right,
    hurt = right->wrong); the same for every system as E10.revision.<system>.*, for the discordant items only as E10.revision.discordant.<system>.*."""
    import json
    p = os.path.join(annot_dir, "discordant_summary.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    def put(pre, blk):
        res.set(pre + ".n", blk.get("n_changed")); res.set(pre + ".helped", blk.get("helped")); res.set(pre + ".hurt", blk.get("hurt"))
        res.set(pre + ".neutral", blk.get("neutral_both_correct_or_both_wrong"))
    for sysn, blk in (d.get("revision_effect_all_dev_items") or {}).items():
        put("E10.revision.%s" % sysn, blk)
        if sysn == "full_fixed":
            put("E10.revision", blk)
    for sysn, blk in (d.get("revision_effect_discordant_items") or {}).items():
        put("E10.revision.discordant.%s" % sysn, blk)
    for sysn, blk in (d.get("repair_of_nonexecutable_root_all_dev_items") or {}).items():
        put("E10.repair_nonexec.%s" % sysn, blk)
    for sysn, blk in (d.get("repair_of_nonexecutable_root_discordant_items") or {}).items():
        put("E10.repair_nonexec.discordant.%s" % sysn, blk)
    dc = d.get("discordant") or {}
    for k in ("n", "fixed_correct_gated_wrong", "fixed_wrong_gated_correct"):
        if k in dc:
            res.set("E10.discordant.%s" % k, dc[k])
    res.set("E10.revision.basis", "revision effect of full_fixed on all dev items (changed = final answer differs from the root answer; helped wrong->right, hurt right->wrong); source discordant_summary.json")


def run_e10(reg, res, annot_dir, llm_labels=None, labelled_runs=None, B=2000):
    """Annotators are two independent Claude Sonnet 5.5 runs (A1, A2; SAP Amendment 2); disagreements are adjudicated by a third run.
    Nothing here is human annotation. Keys: E10.kappa / ac1 / percent_agreement (A1 vs A2, before adjudication), E10.llm.<system>.<Lk>.prevalence
    (stratum-weighted, final = adjudicated label), E10.spot.* only if the authors supplied <annot_dir>/spot_check.csv (optional human spot-check)."""
    model = reg.primary
    if labelled_runs:
        auto_class_tables(res, labelled_runs, model)
    if not annot_dir or not os.path.isdir(annot_dir):
        res.set("E10.status", "partial: automatic classes only (no annotation directory)" if labelled_runs else "not run: no annotation directory")
        return
    key = load_key(annot_dir)
    final = None
    def first(*names):
        for n_ in names:
            if os.path.exists(os.path.join(annot_dir, n_)):
                return os.path.join(annot_dir, n_)
        return os.path.join(annot_dir, names[0])
    a1p, a2p = first("labels_A1.csv", "annotation_A1.csv"), first("labels_A2.csv", "annotation_A2.csv")     # tools/annotation_kappa.py merge writes labels_A*.csv
    if os.path.exists(a1p) and os.path.exists(a2p):
        a1, a2 = _resolve(load_labels(a1p), key) if key else load_labels(a1p), _resolve(load_labels(a2p), key) if key else load_labels(a2p)
        ag = agreement(a1, a2, B)
        if ag:
            res.set("E10.n_annotated", ag["n"])
            res.ci("E10.kappa", ag["kappa"], *ag["kappa_ci"]); res.ci("E10.ac1", ag["ac1"], *ag["ac1_ci"]); res.set("E10.percent_agreement.mean", ag["pct"])
            res.ci("E10.kappa_l5", ag["kappa_l5"], *ag["kappa_l5_ci"]); res.set("E10.confusion", ag["confusion"])
            res.note("E10: kappa/AC1 are the agreement of two LLM runs, computed BEFORE adjudication; there is no human annotation")
        final = {}
        adjp = first("final_labels.csv", "adjudicated.csv")
        adj = _resolve(load_labels(adjp), key) if os.path.exists(adjp) and key else {}
        for i in set(a1) | set(a2):
            if i in adj:
                final[i] = adj[i]
            elif a1.get(i) == a2.get(i) and a1.get(i):
                final[i] = a1[i]
        res.set("E10.n_final_labels", len(final)); res.set("E10.n_pending_adjudication", len(set(a1) & set(a2)) - len(final))
    if final is None:                                             # alternative single-file source (earlier design): already final labels
        for cand in ([llm_labels] if llm_labels else []) + [os.path.join(annot_dir, "llm_labels.jsonl"), os.path.join(annot_dir, "llm_labels.csv")]:
            if cand and os.path.exists(cand):
                final = _resolve(load_labels(cand), key) if key else load_labels(cand)
                res.set("E10.n_final_labels", len(final))
                res.note("E10: labels taken from %s (single file, no A1/A2 agreement available)" % os.path.basename(cand))
                break
    if final and key:
        ids = [i for i in final if i in key]
        pv = prevalence(ids, final, key, B)
        for sysn, d in pv.items():
            res.set("E10.llm.%s.n" % sysn, d["_n"])
            for l in LABELS:
                est, lo, hi, cnt = d[l]
                res.ci("E10.llm.%s.%s.prevalence" % (sysn, l), est, lo, hi)
                res.set("E10.llm.%s.%s.count" % (sysn, l), cnt)
            res.ci("E10.l6.prevalence.%s" % sysn, *d["L6"][:3])
            res.ci("E10.l5.prevalence.%s" % sysn, *d["L5"][:3])
    sp = os.path.join(annot_dir, "spot_check.csv")                  # optional: authors' blind spot check (item_id, primary_label)
    if final and os.path.exists(sp):
        spot = _resolve(load_labels(sp), key) if key else load_labels(sp)
        ag3 = agreement(spot, final, B)
        if ag3:
            res.set("E10.spot.n", ag3["n"]); res.ci("E10.spot.kappa", ag3["kappa"], *ag3["kappa_ci"]); res.set("E10.spot.percent_agreement.mean", ag3["pct"])
    _discordant_summary(res, annot_dir)
    res.set("E10.status", "ok" if final else "partial: no labels found")
    res.set("E10.primary_label_source", "llm_adjudicated" if final else "none")
