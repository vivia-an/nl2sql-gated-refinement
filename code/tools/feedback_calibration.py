#!/usr/bin/env python3
"""Simulated oracle-feedback calibration of the Critic (SAP Amendment 5, docs/SAP.md section 17).

scope = diagnostic-posthoc, label = exploratory-posthoc.  Offline, no API call, pure Python standard library (works on the host python 3.6).
Reads only results/diag/records.jsonl.gz (the stored Critic Scores of the Amendment 4 run).  The label of an item is taken from gold
(injected = incorrect, gold control = correct): an UPPER BOUND for an ideal user, not human feedback.

Usage:  python tools/feedback_calibration.py [--records ../results/diag/records.jsonl.gz] [--out ../results/diag] [--seeds 20] [--B 2000] [--Bauc 500]
"""
import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys

SCOPE = "diagnostic-posthoc"
LABEL = "exploratory-posthoc"
BUDGETS = [0, 25, 50, 100, 200, 400]
N_SEEDS = 20
N_FOLDS = 5
B_RATE = 2000
B_AUC = 500
BOOT_SEED = 20261102
THETA_ACC = 8.5
C_LR = 1.0
LEARNERS = ["threshold", "logreg"]
RULES = ["fp", "det"]
HARD = ["distinct_drop", "count_distinct"]
MARGIN = {"det": 0.05, "fpr": 0.05, "auroc": 0.02}
CATS = ["distinct_drop", "limit_change", "order_flip", "count_distinct", "agg_maxmin", "agg_sumavg", "cmp_flip", "where_drop", "col_swap"]


# ----------------------------------------------------------------------------------------------------------- data
def load_items(path):
    """-> list of dict(id, cat, db, y, x=[syntax, semantic, efficiency], total).  y=1 injected (incorrect), y=0 gold control (correct)."""
    items = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            for var, y in (("pert", 1), ("gold", 0)):
                v = r["variants"].get(var)
                if not v:
                    continue
                c = v["critic"]
                items.append({"id": r["id"], "cat": r["cat"], "db": r["db_id"], "y": y,
                              "x": [float(c["syntax"]), float(c["semantic"]), float(c["efficiency"])], "total": float(c["score"])})
    return items


# ------------------------------------------------------------------------------------------------------- learners
def candidate_thresholds(vals):
    u = sorted(set(vals))
    if not u:
        return [0.0]
    c = [u[0] - 1.0]
    c += [(u[i] + u[i + 1]) / 2.0 for i in range(len(u) - 1)]
    c.append(u[-1] + 1.0)
    return c


def fit_threshold(s, y, rule, alpha0, d0):
    """flag = s < t.  Returns t chosen on the labelled sample by rule 'fp' (max detection s.t. sample FPR <= alpha0) or 'det'
    (min FPR s.t. sample detection >= d0).  None if the sample lacks a class."""
    pos = sorted(v for v, l in zip(s, y) if l == 1)
    neg = sorted(v for v, l in zip(s, y) if l == 0)
    if not pos or not neg:
        return None
    best, best_key = None, None
    for t in candidate_thresholds(s):
        det = bisect.bisect_left(pos, t) / float(len(pos))
        fpr = bisect.bisect_left(neg, t) / float(len(neg))
        if rule == "fp":
            if fpr > alpha0 + 1e-12:
                continue
            key = (det, -fpr, -t)
        else:
            if det < d0 - 1e-12:
                continue
            key = (-fpr, det, -t)
        if best_key is None or key > best_key:
            best, best_key = t, key
    return best


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        piv = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= piv
        for r in range(n):
            if r != i and M[r][i] != 0.0:
                f = M[r][i]
                for j in range(i, n + 1):
                    M[r][j] -= f * M[i][j]
    return [M[i][n] for i in range(n)]


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _objective(w, b, X, y, C):
    loss = 0.0
    for xi, yi in zip(X, y):
        z = b + sum(wj * xj for wj, xj in zip(w, xi))
        loss += max(z, 0.0) - z * yi + math.log1p(math.exp(-abs(z)))
    return 0.5 * sum(wj * wj for wj in w) + C * loss


def fit_logreg(X, y, C=C_LR, tol=1e-8, max_iter=100):
    """L2 logistic regression: minimise 0.5*|w|^2 + C*sum logloss, intercept unpenalised (sklearn convention).  Newton with backtracking."""
    d = len(X[0])
    w, b = [0.0] * d, 0.0
    obj = _objective(w, b, X, y, C)
    for _ in range(max_iter):
        g = [wj for wj in w] + [0.0]
        H = [[0.0] * (d + 1) for _ in range(d + 1)]
        for j in range(d):
            H[j][j] = 1.0
        H[d][d] = 1e-10
        for xi, yi in zip(X, y):
            p = _sigmoid(b + sum(wj * xj for wj, xj in zip(w, xi)))
            r = C * (p - yi)
            v = C * p * (1.0 - p)
            xe = list(xi) + [1.0]
            for j in range(d + 1):
                g[j] += r * xe[j]
                for k in range(d + 1):
                    H[j][k] += v * xe[j] * xe[k]
        step = _solve(H, g)
        t = 1.0
        while True:
            w2 = [wj - t * sj for wj, sj in zip(w, step[:d])]
            b2 = b - t * step[d]
            o2 = _objective(w2, b2, X, y, C)
            if o2 <= obj + 1e-12 or t < 1e-6:
                break
            t /= 2.0
        done = max(abs(t * sj) for sj in step) < tol
        w, b, obj = w2, b2, o2
        if done:
            break
    return w, b


def lr_score(w, b, x):
    """Lower = more likely incorrect (so that 'flag = s < t' is shared with the total score)."""
    return -(b + sum(wj * xj for wj, xj in zip(w, x)))


# ---------------------------------------------------------------------------------------------------- splits, AUROC
def make_folds(dbs_with_counts, seed, k=N_FOLDS):
    """dbs_with_counts: dict db -> item count.  Shuffle with the seed, assign greedily to the currently smallest fold.  -> dict db -> fold."""
    dbs = sorted(dbs_with_counts)
    random.Random("folds-%d" % seed).shuffle(dbs)
    size = [0] * k
    out = {}
    for d in dbs:
        f = min(range(k), key=lambda i: (size[i], i))
        out[d] = f
        size[f] += dbs_with_counts[d]
    return out


def auroc(s, y):
    """P(score of an injected item < score of a control) + 0.5 ties (lower = injected)."""
    return auroc_weighted(tie_groups(s), y, list(range(len(y))), [1] * len(y))


def tie_groups(s):
    order = sorted(range(len(s)), key=lambda i: s[i])
    groups, cur, last = [], [], None
    for i in order:
        if last is not None and s[i] != last:
            groups.append(cur)
            cur = []
        cur.append(i)
        last = s[i]
    if cur:
        groups.append(cur)
    return groups


def auroc_weighted(groups, y, dbidx, m):
    """Weighted by the multiplicity m[dbidx[i]] of the bootstrap resample.  Returns None if a class is empty."""
    totpos = totneg = 0
    for g in groups:
        for i in g:
            w = m[dbidx[i]]
            if y[i]:
                totpos += w
            else:
                totneg += w
    if totpos == 0 or totneg == 0:
        return None
    cum, num = 0, 0.0
    for g in groups:
        p = q = 0
        for i in g:
            w = m[dbidx[i]]
            if y[i]:
                p += w
            else:
                q += w
        num += p * (totneg - cum - q) + 0.5 * p * q
        cum += q
    return num / float(totpos * totneg)


# --------------------------------------------------------------------------------------------------------- the run
def run_cv(items, seeds=N_SEEDS, budgets=BUDGETS, alpha0=None, d0=None, coef_log=None):
    """-> flags[(learner, rule, n)][seed] = list of 0/1 per item (out-of-fold), scores[(learner, n)][seed] = list of floats (lower = injected)."""
    N = len(items)
    y = [it["y"] for it in items]
    tot = [it["total"] for it in items]
    X = [it["x"] for it in items]
    cnt = {}
    for it in items:
        cnt[it["db"]] = cnt.get(it["db"], 0) + 1
    flags, scores = {}, {}
    for L in LEARNERS:
        for n in budgets:
            for R in RULES:
                flags[(L, R, n)] = {}
            scores[(L, n)] = {}
    base = [1 if tot[i] < THETA_ACC else 0 for i in range(N)]
    for seed in range(seeds):
        fold_of = make_folds(cnt, seed)
        fidx = [fold_of[it["db"]] for it in items]
        for n in budgets:
            if n == 0:
                for L in LEARNERS:
                    for R in RULES:
                        flags[(L, R, 0)][seed] = list(base)
                    scores[(L, 0)][seed] = list(tot)
                continue
            fl = {(L, R): [0] * N for L in LEARNERS for R in RULES}
            sc = {L: [0.0] * N for L in LEARNERS}
            for f in range(N_FOLDS):
                train = [i for i in range(N) if fidx[i] != f]
                test = [i for i in range(N) if fidx[i] == f]
                rng = random.Random("draw-%d-%d-%d" % (seed, f, n))
                samp = rng.sample(train, min(n, len(train)))
                ys = [y[i] for i in samp]
                have_both = (1 in ys) and (0 in ys)
                # (a) threshold on the total score
                for i in test:
                    sc["threshold"][i] = tot[i]
                for R in RULES:
                    t = fit_threshold([tot[i] for i in samp], ys, R, alpha0, d0) if have_both else None
                    t = THETA_ACC if t is None else t
                    for i in test:
                        fl[("threshold", R)][i] = 1 if tot[i] < t else 0
                # (b) logistic regression on the components
                if have_both:
                    w, b = fit_logreg([X[i] for i in samp], ys)
                    if coef_log is not None:
                        coef_log.append((n, w, b))
                    s_s = [lr_score(w, b, X[i]) for i in samp]
                    for i in test:
                        sc["logreg"][i] = lr_score(w, b, X[i])
                    for R in RULES:
                        t = fit_threshold(s_s, ys, R, alpha0, d0)
                        for i in test:
                            fl[("logreg", R)][i] = 1 if sc["logreg"][i] < t else 0
                else:
                    for i in test:
                        sc["logreg"][i] = tot[i]
                        for R in RULES:
                            fl[("logreg", R)][i] = base[i]
            for L in LEARNERS:
                scores[(L, n)][seed] = sc[L]
                for R in RULES:
                    flags[(L, R, n)][seed] = fl[(L, R)]
    return flags, scores


# ----------------------------------------------------------------------------------------------------- aggregation
def pct(vals, q):
    v = sorted(vals)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def ci(vals):
    return [pct(vals, 0.025), pct(vals, 0.975)]


def make_resamples(dbs, B, seed=BOOT_SEED):
    rng = random.Random(seed)
    D = len(dbs)
    out = []
    for _ in range(B):
        m = [0] * D
        for _ in range(D):
            m[rng.randrange(D)] += 1
        out.append(m)
    return out


def analyze(items, flags, scores, seeds=N_SEEDS, B=B_RATE, Bauc=B_AUC):
    N = len(items)
    y = [it["y"] for it in items]
    dbs = sorted(set(it["db"] for it in items))
    dbi = {d: i for i, d in enumerate(dbs)}
    dbidx = [dbi[it["db"]] for it in items]
    D = len(dbs)
    resamples = make_resamples(dbs, B)
    groups_def = {"ALL": list(range(N))}
    for c in CATS:
        groups_def[c] = [i for i in range(N) if items[i]["cat"] == c]

    def den(group):
        pos, neg = [0] * D, [0] * D
        for i in group:
            (pos if y[i] else neg)[dbidx[i]] += 1
        return pos, neg

    dens = {g: den(idx) for g, idx in groups_def.items()}
    rs_den = {}
    for g, (pos, neg) in dens.items():
        rs_den[g] = ([sum(m[d] * pos[d] for d in range(D)) for m in resamples], [sum(m[d] * neg[d] for d in range(D)) for m in resamples])

    def series(flagset, g):
        """flagset: {seed: flags}.  -> det/fpr: mean over seeds, seed list, db-level mean numerators."""
        idx = groups_def[g]
        pos, neg = dens[g]
        tp = [0.0] * D
        fp = [0.0] * D
        per_seed_det, per_seed_fpr = [], []
        for s in range(seeds):
            f = flagset[s]
            a = b = 0
            for i in idx:
                if f[i]:
                    if y[i]:
                        tp[dbidx[i]] += 1.0 / seeds
                        a += 1
                    else:
                        fp[dbidx[i]] += 1.0 / seeds
                        b += 1
            P, Q = sum(pos), sum(neg)
            per_seed_det.append(a / float(P) if P else None)
            per_seed_fpr.append(b / float(Q) if Q else None)
        return tp, fp, per_seed_det, per_seed_fpr

    def boot(num, g, which):
        dn = rs_den[g][0 if which == "det" else 1]
        out = []
        for b in range(B):
            if dn[b] == 0:
                out.append(None)
                continue
            m = resamples[b]
            out.append(sum(m[d] * num[d] for d in range(D)) / float(dn[b]))
        return out

    def summarize(point, bootvals, base_point, base_boot, seedvals):
        ok = [v for v in bootvals if v is not None]
        res = {"mean": point, "ci": ci(ok), "seed_range": [pct(seedvals, 0.025), pct(seedvals, 0.975)] if seedvals else None}
        if base_point is not None:
            dl = [a - b for a, b in zip(bootvals, base_boot) if a is not None and b is not None]
            res["delta_vs_n0"] = {"mean": point - base_point, "ci": ci(dl)}
        return res

    out = {"pooled": {}, "per_category": {}}
    ns = sorted(set(k[2] for k in flags))
    for L in LEARNERS:
        for R in RULES:
            for g in ["ALL"] + CATS:
                base = None
                for n in ns:
                    tp, fp, sd, sf = series(flags[(L, R, n)], g)
                    pos, neg = dens[g]
                    P, Q = float(sum(pos)), float(sum(neg))
                    det_pt = sum(tp) / P if P else None
                    fpr_pt = sum(fp) / Q if Q else None
                    det_b = boot(tp, g, "det") if P else [None] * B
                    fpr_b = boot(fp, g, "fpr") if Q else [None] * B
                    if n == 0:
                        base = (det_pt, det_b, fpr_pt, fpr_b)
                    cell = {"detection": summarize(det_pt, det_b, base[0], base[1], [v for v in sd if v is not None]),
                            "fpr": summarize(fpr_pt, fpr_b, base[2], base[3], [v for v in sf if v is not None]),
                            "n_inj": int(P), "n_ctrl": int(Q)}
                    tgt = out["pooled"] if g == "ALL" else out["per_category"].setdefault(g, {})
                    tgt.setdefault(L, {}).setdefault(R, {})[str(n)] = cell
    # AUROC (pooled, all injected vs all controls) with db-clustered bootstrap; threshold learner = raw total at every n
    aucs = {}
    auc_cells = {}
    base_boot = None
    for L in LEARNERS:
        for n in ns:
            if L == "threshold" and n != 0:
                continue
            sc = scores[(L, n)]
            gps = {s: tie_groups(sc[s]) for s in range(seeds)}
            point = sum(auroc(sc[s], y) for s in range(seeds)) / seeds
            per_seed = [auroc(sc[s], y) for s in range(seeds)]
            if L == "threshold" or n == 0:
                one = tie_groups(sc[0])      # the raw total score does not depend on the seed
            bvals = []
            for b in range(Bauc):
                m = resamples[b]
                if n == 0 or L == "threshold":
                    v = auroc_weighted(one, y, dbidx, m)
                else:
                    vs = [auroc_weighted(gps[s], y, dbidx, m) for s in range(seeds)]
                    vs = [a for a in vs if a is not None]
                    v = sum(vs) / len(vs) if vs else None
                bvals.append(v)
            aucs[(L, n)] = (point, bvals, per_seed)
    base_pt, base_b, _ = aucs[("threshold", 0)]
    out["auroc"] = {}
    for L in LEARNERS:
        out["auroc"][L] = {}
        for n in ns:
            key = ("threshold", 0) if (L == "threshold" or n == 0) else (L, n)
            pt, bv, ps = aucs[key]
            ok = [v for v in bv if v is not None]
            dl = [a - b for a, b in zip(bv, base_b) if a is not None and b is not None]
            out["auroc"][L][str(n)] = {"mean": pt, "ci": ci(ok), "seed_range": [pct(ps, 0.025), pct(ps, 0.975)],
                                       "delta_vs_n0": {"mean": pt - base_pt, "ci": ci(dl)},
                                       "note": "raw total score, independent of n" if key == ("threshold", 0) and n != 0 else None}
    return out


def per_category_auc(items, scores, seeds=N_SEEDS):
    """AUROC of injected items of one category vs the gold controls of the same category, mean over seeds; seed spread only (no db interval)."""
    y = [it["y"] for it in items]
    res = {}
    for c in CATS:
        idx = [i for i, it in enumerate(items) if it["cat"] == c]
        res[c] = {}
        for L in LEARNERS:
            res[c][L] = {}
            for n in sorted(set(k[1] for k in scores)):
                if L == "threshold" and n != 0:
                    continue
                v = [auroc([scores[(L, n)][s][i] for i in idx], [y[i] for i in idx]) for s in range(seeds)]
                res[c][L][str(n)] = {"mean": sum(v) / len(v), "seed_range": [pct(v, 0.025), pct(v, 0.975)]}
    return res


def coef_summary(coef_log):
    """Descriptive addendum (added after the first full run, see internal note FBK): mean fitted LR coefficients per budget."""
    out = {}
    for n in sorted(set(c[0] for c in coef_log)):
        rows = [c for c in coef_log if c[0] == n]
        out[str(n)] = {"w_syntax_semantic_efficiency": [sum(r[1][j] for r in rows) / len(rows) for j in range(3)],
                       "intercept": sum(r[2] for r in rows) / len(rows), "n_fits": len(rows)}
    return out


def component_table(items):
    """Descriptive addendum: mean component scores per category and label (what a per-component learner can latch onto)."""
    out = {}
    for c in CATS:
        out[c] = {}
        for lab, nm in ((1, "injected"), (0, "gold_control")):
            v = [it["x"] for it in items if it["cat"] == c and it["y"] == lab]
            out[c][nm] = {"n": len(v), "mean_syntax_semantic_efficiency": [sum(a[j] for a in v) / len(v) for j in range(3)]}
    return out


def verdict(res):
    """Apply the binding reading rule of SAP 17.6 to the pooled results."""
    imp = {}
    for L in LEARNERS:
        for R in RULES:
            for n_s, cell in res["pooled"][L][R].items():
                n = int(n_s)
                if n == 0:
                    continue
                m = cell["detection"] if R == "fp" else cell["fpr"]
                dl = m["delta_vs_n0"]
                good = (dl["ci"][0] > 0 and dl["mean"] >= MARGIN["det"]) if R == "fp" else (dl["ci"][1] < 0 and dl["mean"] <= -MARGIN["fpr"])
                if good:
                    imp.setdefault("%s/%s" % (L, R), []).append(n)
    for n_s, cell in res["auroc"]["logreg"].items():
        n = int(n_s)
        if n == 0:
            continue
        dl = cell["delta_vs_n0"]
        if dl["ci"][0] > 0 and dl["mean"] >= MARGIN["auroc"]:
            imp.setdefault("logreg/auroc", []).append(n)
    small = {k: [n for n in v if n <= 200] for k, v in imp.items()}
    small = {k: v for k, v in small.items() if v}
    hard = {}
    for c in HARD:
        hard[c] = {}
        for L in LEARNERS:
            for R in RULES:
                lst = []
                for n_s, cell in res["per_category"][c][L][R].items():
                    n = int(n_s)
                    if n == 0:
                        continue
                    m = cell["detection"] if R == "fp" else cell["fpr"]
                    dl = m["delta_vs_n0"]
                    good = (dl["ci"][0] > 0 and dl["mean"] >= MARGIN["det"]) if R == "fp" else (dl["ci"][1] < 0 and dl["mean"] <= -MARGIN["fpr"])
                    if good:
                        lst.append(n)
                hard["%s:%s/%s" % (c, L, R)] = lst
    hard_any = {k: v for k, v in hard.items() if v}
    if small:
        text = "improved at n<=200 for: %s (apply the reading rule of SAP 17.6: check curve shape, upper-bound feedback)" % sorted(small)
        v = "improved_at_n_le_200"
    elif imp:
        text = ("improved only at n>200 for: %s; not described as 'the longer used the better'" % sorted(imp))
        v = "improved_only_at_large_n"
    else:
        text = "no improvement under the pre-fixed rule; the paper says calibration from this feedback does not help"
        v = "no_help"
    return {"verdict": v, "text": text, "improved_at_n": imp, "hard_categories_improved": hard_any,
            "hard_categories_text": ("hard categories improve for: %s" % sorted(hard_any)) if hard_any else
            "distinct_drop and count_distinct do not improve under the pre-fixed rule (no learner/rule/budget with interval excluding 0 and a gain >= 5 pp)"}


def flatten_csv(res, path):
    rows = []

    def add(scope, cat, L, R, n, metric, cell):
        dl = cell.get("delta_vs_n0") or {"mean": None, "ci": [None, None]}
        rows.append([scope, cat, L, R, n, metric, cell["mean"], cell["ci"][0], cell["ci"][1],
                     (cell.get("seed_range") or [None, None])[0], (cell.get("seed_range") or [None, None])[1], dl["mean"], dl["ci"][0], dl["ci"][1]])

    for L in res["pooled"]:
        for R in res["pooled"][L]:
            for n, cell in res["pooled"][L][R].items():
                add("pooled", "ALL", L, R, n, "detection", cell["detection"])
                add("pooled", "ALL", L, R, n, "fpr", cell["fpr"])
    for c in res["per_category"]:
        for L in res["per_category"][c]:
            for R in res["per_category"][c][L]:
                for n, cell in res["per_category"][c][L][R].items():
                    add("category", c, L, R, n, "detection", cell["detection"])
                    add("category", c, L, R, n, "fpr", cell["fpr"])
    for L in res["auroc"]:
        for n, cell in res["auroc"][L].items():
            add("pooled", "ALL", L, "", n, "auroc", cell)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "category", "learner", "rule", "n", "metric", "mean", "ci_lo", "ci_hi", "seed_lo", "seed_hi", "delta_mean", "delta_lo", "delta_hi"])
        w.writerows(rows)


def figure_json(res):
    fig = {"_meta": {"scope": SCOPE, "label": LABEL, "kind": "learning curve data (figure-ready); x = number of simulated oracle-labelled items",
                     "caveat": "simulated oracle feedback from gold = upper bound for an ideal user, not human feedback; exploratory post-hoc"},
           "x": BUDGETS, "series": []}

    def pack(name, cells, key, extra=None):
        ns = sorted(int(k) for k in cells)
        s = {"name": name, "metric": key, "x": ns, "mean": [cells[str(n)]["mean"] if key == "auroc" else cells[str(n)][key]["mean"] for n in ns]}
        g = (lambda n: cells[str(n)]) if key == "auroc" else (lambda n: cells[str(n)][key])
        s["ci_lo"] = [g(n)["ci"][0] for n in ns]
        s["ci_hi"] = [g(n)["ci"][1] for n in ns]
        s["seed_lo"] = [g(n)["seed_range"][0] if g(n).get("seed_range") else None for n in ns]
        s["seed_hi"] = [g(n)["seed_range"][1] if g(n).get("seed_range") else None for n in ns]
        if extra:
            s.update(extra)
        fig["series"].append(s)

    for L in LEARNERS:
        for R in RULES:
            cells = res["pooled"][L][R]
            pack("%s / rule %s: detection" % (L, R.upper()), cells, "detection", {"learner": L, "rule": R, "panel": "pooled"})
            pack("%s / rule %s: FPR" % (L, R.upper()), cells, "fpr", {"learner": L, "rule": R, "panel": "pooled"})
    for L in LEARNERS:
        pack("%s: AUROC" % L, res["auroc"][L], "auroc", {"learner": L, "panel": "pooled"})
    for c in HARD:
        for L in LEARNERS:
            pack("%s / %s / rule FP: detection" % (c, L), res["per_category"][c][L]["fp"], "detection", {"learner": L, "rule": "fp", "panel": c})
            pack("%s / %s / rule DET: FPR" % (c, L), res["per_category"][c][L]["det"], "fpr", {"learner": L, "rule": "det", "panel": c})
    return fig


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=os.path.join(here, "..", "..", "results", "diag", "records.jsonl.gz"))
    ap.add_argument("--out", default=os.path.join(here, "..", "..", "results", "diag"))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--B", type=int, default=B_RATE)
    ap.add_argument("--Bauc", type=int, default=B_AUC)
    a = ap.parse_args()
    items = load_items(a.records)
    pos = [it for it in items if it["y"] == 1]
    neg = [it for it in items if it["y"] == 0]
    d0 = sum(1 for it in pos if it["total"] < THETA_ACC) / float(len(pos))
    alpha0 = sum(1 for it in neg if it["total"] < THETA_ACC) / float(len(neg))
    print("items %d (inj %d, ctrl %d), %d dbs; d0=%.4f alpha0=%.4f" % (len(items), len(pos), len(neg), len(set(i["db"] for i in items)), d0, alpha0), flush=True)
    coef_log = []
    flags, scores = run_cv(items, a.seeds, BUDGETS, alpha0, d0, coef_log)
    res = analyze(items, flags, scores, a.seeds, a.B, a.Bauc)
    res["per_category_auroc"] = per_category_auc(items, scores, a.seeds)
    res["reading_rule"] = verdict(res)
    res["addendum_descriptive"] = {"note": "added after the first full run (not part of the SAP 17 spec); descriptive only",
                                   "mean_lr_coefficients": coef_summary(coef_log), "component_means_by_category": component_table(items)}
    n0 = res["pooled"]["threshold"]["fp"]["0"]
    res["_meta"] = {
        "scope": SCOPE, "label": LABEL, "sap": "docs/SAP.md section 17 (Amendment 5), committed before this analysis was run",
        "feedback": "SIMULATED ORACLE feedback: label from gold (injected = incorrect, gold control = correct); an UPPER BOUND for an ideal user; not human, expert or user feedback",
        "reading": "exploratory post-hoc analysis of already collected Critic Scores; not evidence of an EX gain on BIRD; injected errors are easier than natural ones; never enters results_macros.tex",
        "n_items": len(items), "n_injected": len(pos), "n_controls": len(neg), "n_databases": len(set(i["db"] for i in items)),
        "budgets": BUDGETS, "seeds": a.seeds, "folds": N_FOLDS, "split": "database-grouped, fold assignment re-drawn per seed",
        "learners": {"threshold": "refitted threshold on the total Critic Score", "logreg": "L2 logistic regression C=%s on (syntax, semantic, efficiency), raw 0-10 scale" % C_LR},
        "rules": {"fp": "max sample detection s.t. sample FPR <= alpha0", "det": "min sample FPR s.t. sample detection >= d0"},
        "alpha0": alpha0, "d0": d0, "theta_acc": THETA_ACC,
        "ci": "95%% percentile, database-clustered bootstrap (B=%d rates, B=%d AUROC, seed %d) of the mean over seeds; seed_range = 2.5/97.5 percentile of per-seed values" % (a.B, a.Bauc, BOOT_SEED),
        "margins": MARGIN, "records_sha256": hashlib.sha256(open(a.records, "rb").read()).hexdigest(),
        "n0_check": {"detection": n0["detection"]["mean"], "fpr": n0["fpr"]["mean"], "expected_detection": d0, "expected_fpr": alpha0,
                     "reproduces": abs(n0["detection"]["mean"] - d0) < 1e-9 and abs(n0["fpr"]["mean"] - alpha0) < 1e-9},
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "feedback_calibration.json"), "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    flatten_csv(res, os.path.join(a.out, "feedback_calibration.csv"))
    with open(os.path.join(a.out, "feedback_calibration_fig.json"), "w") as f:
        json.dump(figure_json(res), f, indent=1)
    print(json.dumps({"n0_check": res["_meta"]["n0_check"], "reading_rule": res["reading_rule"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
