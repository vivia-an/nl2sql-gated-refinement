"""Score quality metrics for routing / selective prediction (P3, E5, E7): AUROC, AURC, accepted-EX at coverage, ECE, Brier,
with db-clustered paired bootstrap on shared draws."""
import math

import numpy as np

from . import stats as S

COVERAGES = (0.4, 0.6, 0.8)


def rc_vector(score, y, covs=COVERAGES):
    """[AURC, accepted-EX at each coverage] (expected over random tie breaks)."""
    cum = S._expected_cum_acc(score, y)
    n = len(cum)
    if n == 0:
        return np.full(1 + len(covs), np.nan)
    out = [float((1.0 - cum).mean())]
    for c in covs:
        out.append(float(cum[max(1, int(round(c * n))) - 1]))
    return np.array(out)


def metric_vector(score, y, covs=COVERAGES):
    """[AUROC, AURC, acc@cov...]"""
    return np.concatenate([[S.auroc(score, y)], rc_vector(score, y, covs)])


METRIC_NAMES = ["auroc", "aurc"] + ["acc_ex_cov%d" % round(100 * c) for c in COVERAGES]


def boot_metrics(scores, y, boot, covs=COVERAGES, B=None):
    """scores: dict label -> array. -> dict label -> array [B, n_metrics] of bootstrap values (same draws for all labels)."""
    B = B or boot.B
    y = np.asarray(y, int)
    out = {k: np.empty((B, 2 + len(covs))) for k in scores}
    for b in range(B):
        idx = np.concatenate([boot.members[j] for j in boot.draws[b]])
        yy = y[idx]
        for k, s in scores.items():
            out[k][b] = metric_vector(np.asarray(s)[idx], yy, covs)
    return out


def point_metrics(scores, y, covs=COVERAGES):
    return {k: metric_vector(np.asarray(s), y, covs) for k, s in scores.items()}


def panel(scores, y, dbs, boot, ref=None, covs=COVERAGES, p_scale=10.0, prefix=""):
    """Flat keys for each score: <prefix><label>.<metric>.{mean,ci_lo,ci_hi}; vs `ref`: <prefix><label>_vs_<ref>.d<metric>.{mean,ci_lo,ci_hi,p_le0 / p_ge0}.
    Also ECE/Brier on score/p_scale (raw), curves are produced by the caller."""
    y = np.asarray(y, int)
    pt = point_metrics(scores, y, covs)
    bs = boot_metrics(scores, y, boot, covs)
    out = {}
    names = ["auroc", "aurc"] + ["acc_ex_cov%d" % round(100 * c) for c in covs]
    for k in scores:
        for j, nm in enumerate(names):
            lo, hi = S.pctl_ci(bs[k][:, j])
            out["%s%s.%s.mean" % (prefix, k, nm)] = float(pt[k][j])
            out["%s%s.%s.ci_lo" % (prefix, k, nm)] = lo
            out["%s%s.%s.ci_hi" % (prefix, k, nm)] = hi
        p = np.clip(np.asarray(scores[k], float) / p_scale, 0, 1)
        out["%s%s.ece_raw.mean" % (prefix, k)] = S.ece_equal_mass(p, y, 15)
        out["%s%s.brier_raw.mean" % (prefix, k)] = S.brier(p, y)
    if ref is not None and ref in scores:
        for k in scores:
            if k == ref:
                continue
            for j, nm in enumerate(names):
                d = bs[k][:, j] - bs[ref][:, j]
                lo, hi = S.pctl_ci(d)
                key = "%s%s_vs_%s.d%s" % (prefix, k, ref, nm)
                out[key + ".mean"] = float(pt[k][j] - pt[ref][j])
                out[key + ".ci_lo"] = lo
                out[key + ".ci_hi"] = hi
                out[key + ".p_le0"] = S.p_le(d, 0.0)          # H0: delta <= 0 (k not better than ref in the 'higher is better' metrics)
                out[key + ".p_ge0"] = S.p_ge(d, 0.0)          # H0: delta >= 0 (for AURC: lower is better)
    return out, bs, pt
