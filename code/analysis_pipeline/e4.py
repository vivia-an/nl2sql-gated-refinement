"""E4 (corpus evolution / pollution): loading of stream units and the analyses behind P4 and S-E.

Unit directory = <policy>-p<pp>-perm<k>-<db>/ with events.jsonl (probe checkpoints + stream events), pollution.json, done (nl2sql.stream e4).
Outcome per probe item = `ex_all` (abstained counts as wrong, as in stream.summarize_trial)."""
import os
import re

import numpy as np

from . import glmm, stats as S
from .runs import jl

UNIT_RE = re.compile(r"^(?P<policy>[a-z][a-z_0-9]*)-p(?P<p>\d+)-perm(?P<perm>\d+)-(?P<db>.+)$")


def find_units(roots, require_done=True):
    """-> list of dicts(policy, p, perm, db, path). Duplicate unit keys: the later path (sorted) wins, a note is returned."""
    seen, notes = {}, []
    for root in roots:
        for dirpath, dirs, files in os.walk(root):
            if "events.jsonl" not in files:
                continue
            m = UNIT_RE.match(os.path.basename(dirpath))
            if not m:
                continue
            if require_done and "done" not in files:
                notes.append("unit without done marker skipped: %s" % dirpath)
                continue
            key = (m.group("policy"), int(m.group("p")), int(m.group("perm")), m.group("db"))
            if key in seen:
                notes.append("duplicate unit %s: %s replaces %s" % (key, dirpath, seen[key]["path"]))
            seen[key] = {"policy": key[0], "p": key[1], "perm": key[2], "db": key[3], "path": dirpath}
    return [seen[k] for k in sorted(seen)], notes


def load_unit(u):
    ev = jl(os.path.join(u["path"], "events.jsonl"))
    probes = [e for e in ev if e.get("type") == "probe"]
    stream = [e for e in ev if e.get("type") == "stream"]
    pj = {}
    pp = os.path.join(u["path"], "pollution.json")
    if os.path.exists(pp):
        import json
        pj = json.load(open(pp))
    return dict(u, probes=probes, stream=stream, n_seed=pj.get("n_seed"), n_polluted=pj.get("n_polluted"))


def load_all(roots, require_done=True):
    us, notes = find_units(roots, require_done)
    return [load_unit(u) for u in us], notes


def probe_rows(units, policy, p, which="final"):
    """-> dict(y, db, qid, perm) of the probe items at the first or final checkpoint."""
    y, db, qid, perm = [], [], [], []
    for u in units:
        if u["policy"] != policy or u["p"] != p or not u["probes"]:
            continue
        pr = u["probes"][-1] if which == "final" else u["probes"][0]
        for it in pr["items"]:
            v = it.get("ex_all")
            if v is None:
                continue
            y.append(int(v))
            db.append(u["db"])
            qid.append(it["qid"])
            perm.append(u["perm"])
    return {"y": np.array(y, int), "db": np.array(db), "qid": np.array(qid), "perm": np.array(perm)}


def db_ratio_boot(sa, na, sb, nb, B=10000, seed=S.SEED_BOOT):
    """bootstrap of mean_a - mean_b with databases as clusters; arrays are per-db sums / counts."""
    k = len(sa)
    rng = np.random.RandomState(seed)
    d = rng.randint(0, k, size=(B, k))
    sa, na, sb, nb = [np.asarray(v, float) for v in (sa, na, sb, nb)]
    return sa[d].sum(1) / np.maximum(na[d].sum(1), 1) - sb[d].sum(1) / np.maximum(nb[d].sum(1), 1)


def policy_contrast(units, pa, pb, p, B=10000, margin_pp=None, fit=True):
    """pa vs pb at pollution p on the final-checkpoint probes: GLMM log-odds contrast (pa - pb), pp difference with db-clustered bootstrap CI,
    pooled-pair exact McNemar. Returns dict or None when a policy has no units."""
    ra, rb = probe_rows(units, pa, p), probe_rows(units, pb, p)
    if len(ra["y"]) == 0 or len(rb["y"]) == 0:
        return None
    out = {}
    dbs = sorted(set(ra["db"].tolist()) & set(rb["db"].tolist()))
    sa = [ra["y"][ra["db"] == d].sum() for d in dbs]
    na = [(ra["db"] == d).sum() for d in dbs]
    sb = [rb["y"][rb["db"] == d].sum() for d in dbs]
    nb = [(rb["db"] == d).sum() for d in dbs]
    bt = db_ratio_boot(sa, na, sb, nb, B)
    est = sum(sa) / float(max(sum(na), 1)) - sum(sb) / float(max(sum(nb), 1))
    lo, hi = S.pctl_ci(bt)
    out.update({"diff_pp": 100 * est, "diff_pp_ci_lo": 100 * lo, "diff_pp_ci_hi": 100 * hi, "n_dbs": len(dbs), "ex_a": sum(sa) / float(max(sum(na), 1)),
                "ex_b": sum(sb) / float(max(sum(nb), 1)), "n_a": int(sum(na)), "n_b": int(sum(nb))})
    if margin_pp is not None:
        out["p_ni"] = S.p_le(100 * bt, -margin_pp)            # H0: diff <= -margin
    # pooled paired discordance on (qid, perm) pairs present for both policies
    ka = {(q, pm): y for q, pm, y in zip(ra["qid"], ra["perm"], ra["y"])}
    kb = {(q, pm): y for q, pm, y in zip(rb["qid"], rb["perm"], rb["y"])}
    both = [k for k in ka if k in kb]
    b = sum(1 for k in both if ka[k] == 1 and kb[k] == 0)
    c = sum(1 for k in both if ka[k] == 0 and kb[k] == 1)
    mc = S.mcnemar_exact(b, c)
    out.update({"b": b, "c": c, "p_mcnemar": mc["p_two"], "n_pairs": len(both)})
    if fit:
        y = np.concatenate([ra["y"], rb["y"]])
        x = np.concatenate([np.ones(len(ra["y"])), np.zeros(len(rb["y"]))])
        grp = [np.concatenate([ra["db"], rb["db"]]), np.concatenate([ra["qid"], rb["qid"]]), np.concatenate([ra["perm"], rb["perm"]])]
        try:
            g = glmm.contrast(y, x, grp, label="policy")
            out.update({"logodds": g["beta"], "logodds_se": g["se"], "logodds_ci_lo": g["ci_lo"], "logodds_ci_hi": g["ci_hi"], "odds_ratio": g["odds_ratio"],
                        "or_ci_lo": g["or_lo"], "or_ci_hi": g["or_hi"], "p_wald": g["p_wald"], "p_lrt": g["p_lrt"], "sigma_db": g["sigma"][0], "sigma_qid": g["sigma"][1],
                        "sigma_perm": g["sigma"][2], "glmm_converged": g["converged"]})
        except Exception as e:                                 # fit failure is reported, never hidden
            out["glmm_error"] = "%s: %s" % (type(e).__name__, e)
    return out


def slope_model(units, policies, levels, which="final"):
    """GLMM  ex ~ pollution_c (+ policy + policy:pollution_c if two policies) + (1|db)+(1|qid)+(1|perm), pollution_c = p / 10.
    -> dict with per-policy slopes (log-odds per 10 pp) and the interaction for exactly two policies."""
    out = {}
    for pol in policies:
        ys, pcs, dbs, qids, perms = [], [], [], [], []
        for p in levels:
            r = probe_rows(units, pol, p, which)
            ys.append(r["y"]); pcs.append(np.full(len(r["y"]), p / 10.0)); dbs.append(r["db"]); qids.append(r["qid"]); perms.append(r["perm"])
        y = np.concatenate(ys)
        if len(y) == 0 or len(set(np.concatenate(pcs).tolist())) < 2:
            continue
        pc = np.concatenate(pcs)
        try:
            g = glmm.contrast(y, pc, [np.concatenate(dbs), np.concatenate(qids), np.concatenate(perms)], label="pollution_c")
            out[pol] = {"slope": g["beta"], "ci_lo": g["ci_lo"], "ci_hi": g["ci_hi"], "p": g["p_lrt"], "converged": g["converged"]}
        except Exception as e:
            out[pol] = {"error": "%s: %s" % (type(e).__name__, e)}
    if len(policies) == 2 and all(p in out and "slope" in out[p] for p in policies):
        pa, pb = policies
        ys, pcs, xs, dbs, qids, perms = [], [], [], [], [], []
        for k, pol in enumerate(policies):
            for p in levels:
                r = probe_rows(units, pol, p, which)
                ys.append(r["y"]); pcs.append(np.full(len(r["y"]), p / 10.0)); xs.append(np.full(len(r["y"]), 1.0 if k == 0 else 0.0))
                dbs.append(r["db"]); qids.append(np.array(["%s" % q for q in r["qid"]])); perms.append(r["perm"])
        y = np.concatenate(ys)
        pc, x = np.concatenate(pcs), np.concatenate(xs)
        try:
            g = glmm.contrast(y, x, [np.concatenate(dbs), np.concatenate(qids), np.concatenate(perms)], extra={"pollution_c": pc, "policy:pollution_c": x * pc},
                              label="policy", test_cols=(3,))
            out["_diff"] = {"pair": [pa, pb], "interaction": g["coef_all"]["policy:pollution_c"], "se": g["se_all"]["policy:pollution_c"], "p_lrt": g["p_lrt"]}
            z = S.norm_ppf(0.975)
            out["_diff"]["ci_lo"] = out["_diff"]["interaction"] - z * out["_diff"]["se"]
            out["_diff"]["ci_hi"] = out["_diff"]["interaction"] + z * out["_diff"]["se"]
        except Exception as e:
            out["_diff"] = {"error": "%s: %s" % (type(e).__name__, e)}
    return out


def eviction_diag(units, policy, p):
    """pooled eviction diagnostics at the final checkpoint. recall = evicted polluted seed entries / polluted seed entries; precision =
    evicted polluted / evicted; fpr = evicted clean / clean entries ever stored (clean seed + non-abstained stream outputs)."""
    ep = ec = npol = nclean_seed = nwritten = 0
    for u in units:
        if u["policy"] != policy or u["p"] != p or not u["probes"]:
            continue
        d = u["probes"][-1].get("diag") or {}
        ep += d.get("evicted_polluted", 0) or 0
        ec += d.get("evicted_clean", 0) or 0
        npol += u.get("n_polluted") or 0
        nclean_seed += (u.get("n_seed") or 0) - (u.get("n_polluted") or 0)
        nwritten += sum(1 for e in u["stream"] if not e.get("abstained"))
    if npol + nclean_seed == 0:
        return None
    ev = ep + ec
    return {"recall": (ep / float(npol)) if npol else None, "precision": (ep / float(ev)) if ev else None, "clean_share_of_evicted": (ec / float(ev)) if ev else None,
            "fpr": ec / float(max(nclean_seed + nwritten, 1)), "evicted_polluted": ep, "evicted_clean": ec, "n_polluted_seed": npol}


def curves(units, policy, p, window=15, B=2000):
    """checkpoint curve (fraction of stream, probe EX with db-clustered CI) and prequential online EX per window (mean over units)."""
    us = [u for u in units if u["policy"] == policy and u["p"] == p and u["probes"]]
    if not us:
        return None
    ncp = min(len(u["probes"]) for u in us)
    xs, ys, los, his = [], [], [], []
    dbs = sorted(set(u["db"] for u in us))
    for j in range(ncp):
        frac = float(np.mean([(u["probes"][j]["after"] / float(max(u["probes"][j]["n_stream"], 1))) for u in us]))
        sa = [sum((it.get("ex_all") or 0) for u in us if u["db"] == d for it in u["probes"][j]["items"]) for d in dbs]
        na = [sum(1 for u in us if u["db"] == d for it in u["probes"][j]["items"]) for d in dbs]
        k = len(dbs)
        rng = np.random.RandomState(S.SEED_BOOT + j)
        dr = rng.randint(0, k, size=(B, k))
        sa_, na_ = np.asarray(sa, float), np.asarray(na, float)
        bt = sa_[dr].sum(1) / np.maximum(na_[dr].sum(1), 1)
        lo, hi = S.pctl_ci(bt)
        xs.append(frac); ys.append(sum(sa) / float(max(sum(na), 1))); los.append(lo); his.append(hi)
    # online (prequential) EX
    nstream = min(len(u["stream"]) for u in us)
    wx, wy = [], []
    for s in range(0, nstream - window + 1, window):
        vals = [0 if e.get("abstained") else (e.get("ex") or 0) for u in us for e in u["stream"][s:s + window]]
        wx.append(s + window); wy.append(float(np.mean(vals)))
    auc = float(sum((xs[i + 1] - xs[i]) * (ys[i + 1] + ys[i]) / 2.0 for i in range(len(xs) - 1))) if len(xs) > 1 else None
    return {"x": xs, "y": ys, "ci_lo": los, "ci_hi": his, "online_x": wx, "online_y": wy, "auc": auc, "n_units": len(us)}
