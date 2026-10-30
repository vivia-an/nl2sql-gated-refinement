#!/usr/bin/env python3
"""Post hoc: where the questions lost by the empty trigger come from. No model call.

    python3 code/tools/x2ds_w8_roots.py --w <host work tree> --out results/x2ds/w8_roots.json

(1) Pipeline + empty gate (W8) against the pipeline: the root query of every question is read from cands.jsonl (action "root") of both runs. A root
differs when its request missed the call cache and was answered afresh (example retrieval runs live, so a different example list changes the request;
a few identical requests also missed). Changed final queries, and the questions turned correct (up) or wrong (down), are counted inside and outside the
regenerated roots; the gain on the questions with the same root isolates the trigger.
(2) Empty-gate retry against the single call: the questions turned wrong, and whether one of them is the question whose reference is one row of NULLs
(results/x2ds/empty_gold.json). (3) Empty-gate retry against the plain single call + retry: the questions turned wrong, with the status of their shared root."""
import argparse
import json
import os

BK = ("plus", "flash", "27b", "dsv41flash")


def _preds(p):
    return {json.loads(l)["qid"]: json.loads(l) for l in open(p)}


def _roots(p):
    out = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("action") == "root":
            out[r["qid"]] = r["sql"]
    return out


def gated_dir(W, bk):
    if bk == "plus":
        return W + "/analysis_runs/MIN/M1/gated-s0"
    if bk == "dsv41flash":
        return W + "/runs/X2DS/gated-dsv41flash-s0"
    return "%s/analysis_runs/MIN/M5/gated-%s-s0" % (W, bk)


def zs_dir(W, bk):
    if bk == "plus":
        return W + "/analysis_runs/MIN/M1/zs-s0"
    if bk == "dsv41flash":
        return W + "/runs/X2DS/zs-dsv41flash-s0"
    return "%s/analysis_runs/MIN/M5/zs-%s-s0" % (W, bk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", required=True)
    ap.add_argument("--empty-gold", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "x2ds", "empty_gold.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eg = json.load(open(a.empty_gold))
    nullrow = set(eg.get("qids_null") or [])
    out = {"_meta": {"scope": "posthoc", "what": "W8 regenerated roots; empty-gate retry losses"}, "backbones": {}}
    for bk in BK:
        gd, ed = gated_dir(a.w, bk), "%s/runs/W8/gatedempty-%s-s0" % (a.w, bk)
        G, E = _preds(gd + "/preds.jsonl"), _preds(ed + "/preds.jsonl")
        RG, RE = _roots(gd + "/cands.jsonl"), _roots(ed + "/cands.jsonl")
        regen = {q for q in G if RG.get(q) != RE.get(q)}
        ch = [q for q in G if G[q]["final_sql"] != E[q]["final_sql"]]
        up = [q for q in ch if E[q]["ex"] > G[q]["ex"]]
        down = [q for q in ch if E[q]["ex"] < G[q]["ex"]]
        same = [q for q in G if q not in regen]
        r = {"n": len(G), "n_regen": len(regen), "n_changed": len(ch), "n_changed_regen": sum(q in regen for q in ch),
             "n_up": len(up), "n_up_regen": sum(q in regen for q in up), "n_down": len(down), "n_down_regen": sum(q in regen for q in down),
             "n_same_root": len(same), "gain_same_root": 100.0 * sum(E[q]["ex"] - G[q]["ex"] for q in same) / len(same),
             "down_qids": sorted(down)}
        # latency: reused requests keep their stored latency, retrieval and new requests are timed live; the change on questions whose request count
        # did not change shows how far the two runs' conditions differ
        sc = [q for q in G if G[q]["calls"] == E[q]["calls"]]
        r["lat_delta"] = sum(E[q]["latency_s"] - G[q]["latency_s"] for q in G) / len(G)
        r["lat_delta_samecalls"] = sum(E[q]["latency_s"] - G[q]["latency_s"] for q in sc) / len(sc)
        r["n_samecalls"] = len(sc)
        Z = _preds(zs_dir(a.w, bk) + "/preds.jsonl")
        Y = _preds("%s/runs/W7/zsretryempty-%s-s0/preds.jsonl" % (a.w, bk))
        zd = sorted(q for q in Z if Z[q]["final_sql"] != Y[q]["final_sql"] and Y[q]["ex"] < Z[q]["ex"])
        r["retry_empty_down_qids"] = zd
        # (3) empty-gate retry against the plain single call + retry: they share the stored root; a loss must come from a revision
        X = _preds("%s/runs/X2DSR/zsretry-%s-s0/preds.jsonl" % (a.w, bk))
        xd = sorted(q for q in X if X[q]["final_sql"] != Y[q]["final_sql"] and Y[q]["ex"] < X[q]["ex"])
        r["empty_vs_retry_down_qids"] = xd
        r["empty_vs_retry_down_root_status"] = {q: Z[q].get("final_status") for q in xd}   # the shared root is the stored single-call answer
        r["empty_vs_retry_down_actions"] = {q: (X[q].get("actions"), Y[q].get("actions"), X[q].get("final_status"), Y[q].get("final_status")) for q in xd}
        r["retry_empty_down_nullrow"] = sum(q in nullrow for q in zd)
        out["backbones"][bk] = r
        print(bk, {k: v for k, v in r.items() if not k.endswith("qids")}, zd, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
