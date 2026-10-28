"""Statistics used by the analysis pipeline (numpy; scipy is NOT required here).

Everything in SAP sections 3, 4, 6 and 13 that is not a model fit: db-clustered paired bootstrap, exact McNemar, Tango score
non-inferiority test, Clopper-Pearson / Wilson intervals, AUROC (tie-aware) with DeLong, expected AURC under random tie breaking,
accepted-EX at coverage, ECE / Brier / isotonic (PAV), Holm / Benjamini-Hochberg, Cochran's Q, Spearman.
Python 3.6 syntax compatible."""
import math

import numpy as np

SEED_BOOT = 20261101
SEED_PERM = 20261102


# ------------------------------------------------------------------------------------------ special functions
def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def norm_ppf(p):
    """Acklam's rational approximation refined by one Halley step (abs error < 1e-12 on (1e-12, 1-1e-12))."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def _gammainc_lower_reg(a, x):
    """Regularised lower incomplete gamma P(a, x) (Numerical Recipes series / continued fraction)."""
    if x <= 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-15:
            break
    return 1.0 - math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x, df):
    if x <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - _gammainc_lower_reg(df / 2.0, x / 2.0)))


# ------------------------------------------------------------------------------------------ binomial machinery
def _logpmf(k, n, p):
    if p <= 0.0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if k == n else -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log(1 - p)


def binom_sf(k, n, p=0.5):
    """P(X >= k)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, sum(math.exp(_logpmf(i, n, p)) for i in range(k, n + 1)))


def binom_cdf(k, n, p=0.5):
    """P(X <= k)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, sum(math.exp(_logpmf(i, n, p)) for i in range(0, k + 1)))


def mcnemar_exact(b, c):
    """b = A right / B wrong, c = A wrong / B right. -> dict(p_two, p_mid_two, p_one_greater [H1: A better], p_one_less, or, or_lo, or_hi).
    Two-sided exact p = min(1, 2 * P(X <= min(b, c))), X ~ Bin(b + c, 1/2); mid-p subtracts half the point mass."""
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_disc": 0, "p_two": 1.0, "p_mid_two": 1.0, "p_one_greater": 1.0, "p_one_less": 1.0, "or": None, "or_lo": None, "or_hi": None}
    m = min(b, c)
    p_two = min(1.0, 2.0 * binom_cdf(m, n, 0.5))
    pm = math.exp(_logpmf(m, n, 0.5))
    p_mid = min(1.0, max(0.0, 2.0 * (binom_cdf(m, n, 0.5) - 0.5 * pm)))
    lo, hi = clopper_pearson(b, n)
    odds = (lambda q: (q / (1 - q)) if q < 1 else None)
    return {"b": b, "c": c, "n_disc": n, "p_two": p_two, "p_mid_two": p_mid, "p_one_greater": binom_sf(b, n, 0.5), "p_one_less": binom_cdf(b, n, 0.5),
            "or": (b / c if c else None), "or_lo": odds(lo), "or_hi": odds(hi)}


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (None, None)
    if k == 0:
        lo = 0.0
    else:
        a, bb = 0.0, 1.0
        for _ in range(60):
            mid = (a + bb) / 2
            if binom_sf(k, n, mid) > alpha / 2:
                bb = mid
            else:
                a = mid
        lo = (a + bb) / 2
    if k == n:
        hi = 1.0
    else:
        a, bb = 0.0, 1.0
        for _ in range(60):
            mid = (a + bb) / 2
            if binom_cdf(k, n, mid) > alpha / 2:
                a = mid
            else:
                bb = mid
        hi = (a + bb) / 2
    return (lo, hi)


def wilson(k, n, alpha=0.05):
    if n == 0:
        return (None, None)
    z = norm_ppf(1 - alpha / 2)
    p = k / float(n)
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, ctr - half), min(1.0, ctr + half))


def tango_ni(n10, n01, n, delta):
    """Tango (1998)-type score test for paired-binary NON-INFERIORITY. n10 = new right / ref wrong, n01 = new wrong / ref right, n = pairs.
    H0: p_new - p_ref <= -delta (delta > 0 as a fraction).  -> dict(diff, z, p_one) with p_one = P(Z >= z).
    The variance is evaluated at the constrained MLE on the H0 boundary (a = d - delta for the two discordant cell probabilities)."""
    if n <= 0:
        return {"diff": None, "z": None, "p_one": None}
    diff = (n10 - n01) / float(n)
    n00 = n - n10 - n01
    lo, hi = max(delta, 1e-12), (1.0 + delta) / 2.0 - 1e-12
    f = lambda d: (n10 / (d - delta) if n10 else 0.0) + n01 / d - 2.0 * n00 / (1.0 - 2.0 * d + delta) if n00 or n10 or n01 else 0.0
    # f is decreasing in d on (delta, (1+delta)/2); bisection for f(d) = 0
    flo, fhi = f(lo + 1e-12), f(hi)
    if flo <= 0:
        d = lo
    elif fhi >= 0:
        d = hi
    else:
        a_, b_ = lo + 1e-12, hi
        for _ in range(100):
            mid = (a_ + b_) / 2
            if f(mid) > 0:
                a_ = mid
            else:
                b_ = mid
        d = (a_ + b_) / 2
    var0 = max(2.0 * d - delta - delta * delta, 1e-12)
    z = (diff + delta) / math.sqrt(var0 / n)
    return {"diff": diff, "z": z, "p_one": norm_sf(z)}


# ------------------------------------------------------------------------------------------ multiplicity
def holm(pvals, m=None):
    """pvals: dict name -> p (None = test not run, treated as p = 1 but still counted in m). -> dict name -> adjusted p (None stays None)."""
    names = list(pvals)
    m = m if m is not None else len(names)
    ps = {k: (1.0 if pvals[k] is None else float(pvals[k])) for k in names}
    order = sorted(names, key=lambda k: ps[k])
    adj, run = {}, 0.0
    pad = max(0, m - len(names))                  # tests of the family that have no p-value at all: p = 1 at the end of the order
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        adj[k] = run
    return {k: (None if pvals[k] is None else adj[k]) for k in names}


def benjamini_hochberg(pvals):
    names = [k for k in pvals if pvals[k] is not None]
    n = len(names)
    order = sorted(names, key=lambda k: pvals[k])
    q, prev = {}, 1.0
    for i in range(n - 1, -1, -1):
        k = order[i]
        prev = min(prev, pvals[k] * n / (i + 1.0))
        q[k] = min(1.0, prev)
    return {k: q.get(k) for k in pvals}


# ------------------------------------------------------------------------------------------ ranks, AUROC, DeLong
def midrank(x):
    x = np.asarray(x, float)
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    mid = cum - (cnt - 1) / 2.0
    return mid[inv]


def auroc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = midrank(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _delong_components(scores, y):
    """scores [k, n]; -> aucs [k], V10 [k, n1], V01 [k, n0]."""
    y = np.asarray(y, int)
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    n1, n0 = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    aucs, v10, v01 = np.empty(k), np.empty((k, n1)), np.empty((k, n0))
    for i in range(k):
        tz = midrank(np.concatenate([pos[i], neg[i]]))
        tx, ty = midrank(pos[i]), midrank(neg[i])
        v10[i] = (tz[:n1] - tx) / n0
        v01[i] = 1.0 - (tz[n1:] - ty) / n1
        aucs[i] = v10[i].mean()
    return aucs, v10, v01


def delong_paired(score_a, score_b, y):
    """DeLong test for AUC(a) - AUC(b) on the same items (item level, no clustering). -> dict(auc_a, auc_b, diff, se, z, p_two, p_one_greater)."""
    y = np.asarray(y, int)
    S = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 2 or n0 < 2:
        return {"auc_a": None, "auc_b": None, "diff": None, "se": None, "z": None, "p_two": None, "p_one_greater": None}
    aucs, v10, v01 = _delong_components(S, y)
    s10, s01 = np.cov(v10), np.cov(v01)
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n1 + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n0
    diff = float(aucs[0] - aucs[1])
    se = math.sqrt(max(var, 0.0))
    if se == 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "se": 0.0, "z": None, "p_two": None, "p_one_greater": None}
    z = diff / se
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff, "se": se, "z": z, "p_two": 2 * norm_sf(abs(z)), "p_one_greater": norm_sf(z)}


# ------------------------------------------------------------------------------------------ selective prediction
def _expected_cum_acc(score, y):
    """Expected cumulative accuracy of the accepted set at positions 1..n when items are accepted in order of decreasing score and ties are
    broken uniformly at random (closed form: inside a tie group of m items with k correct the expected number of correct among the first j is
    j*k/m). -> array of length n."""
    s, y = np.asarray(score, float), np.asarray(y, float)
    n = len(s)
    if n == 0:
        return np.zeros(0)
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    new = np.r_[True, ss[1:] != ss[:-1]]
    gid = np.cumsum(new) - 1
    m = np.bincount(gid)
    k = np.bincount(gid, weights=yy)
    start = np.r_[0, np.cumsum(m)[:-1]]
    kcum_before = np.r_[0.0, np.cumsum(k)[:-1]]
    pos_in_group = np.arange(n) - start[gid] + 1
    exp_correct = kcum_before[gid] + pos_in_group * (k[gid] / m[gid])
    return exp_correct / np.arange(1, n + 1)


def aurc_expected(score, y):
    """Area under the risk-coverage curve (mean risk over accepted-set sizes 1..n, expectation over random tie breaking). Lower is better."""
    cum = _expected_cum_acc(score, y)
    return float((1.0 - cum).mean()) if len(cum) else float("nan")


def acc_at_coverage(score, y, cov):
    """Expected EX of the accepted fraction `cov` (top round(cov * n) by score, ties random)."""
    cum = _expected_cum_acc(score, y)
    n = len(cum)
    if n == 0:
        return float("nan")
    return float(cum[max(1, int(round(cov * n))) - 1])


def risk_coverage_curve(score, y, points=100):
    cum = _expected_cum_acc(score, y)
    n = len(cum)
    cov = np.linspace(1.0 / points, 1.0, points)
    return [float(c) for c in cov], [float(1.0 - cum[max(1, int(round(c * n))) - 1]) for c in cov]


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece_equal_mass(p, y, bins=15):
    """ECE with equal-mass bins (sorted by p, split into `bins` groups)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    n = len(p)
    if n == 0:
        return float("nan")
    o = np.argsort(p, kind="mergesort")
    tot = 0.0
    for chunk in np.array_split(o, bins):
        if len(chunk):
            tot += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return float(tot / n)


def reliability_bins(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        msk = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1.0))
        out.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(msk.sum()),
                    "mean_p": float(p[msk].mean()) if msk.any() else None, "frac_pos": float(y[msk].mean()) if msk.any() else None})
    return out


def isotonic_fit(x, y):
    """Pool-adjacent-violators (increasing). -> (knots_x sorted unique, fitted values) usable with isotonic_apply."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x, kind="mergesort")
    xs, ys = x[o], y[o]
    blocks = []                                    # [sum_y, count, x_lo, x_hi]
    for xi, yi in zip(xs, ys):
        blocks.append([yi, 1.0, xi, xi])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1] = [blocks[-1][0] + b[0], blocks[-1][1] + b[1], blocks[-1][2], b[3]]
    kx, ky = [], []
    for s, c, lo, hi in blocks:
        kx += [lo, hi]
        ky += [s / c, s / c]
    return np.array(kx), np.array(ky)


def isotonic_apply(model, x):
    kx, ky = model
    return np.interp(np.asarray(x, float), kx, ky)


# ------------------------------------------------------------------------------------------ clustered bootstrap
class ClusterBoot(object):
    """Database-clustered bootstrap with shared draws (paired statistics use the same resample for every system).
    `dbs` = cluster label per item (aligned with every vector later passed in)."""

    def __init__(self, dbs, B=10000, seed=SEED_BOOT):
        self.dbs = np.asarray(dbs)
        self.u = sorted(set(self.dbs.tolist()))
        self.members = [np.where(self.dbs == d)[0] for d in self.u]
        self.k = len(self.u)
        self.B = int(B)
        rng = np.random.RandomState(seed)
        self.draws = rng.randint(0, self.k, size=(self.B, self.k)) if self.k else np.zeros((self.B, 0), int)
        self.n = len(self.dbs)

    def _sums(self, v):
        return np.array([np.asarray(v, float)[m].sum() for m in self.members])

    def means(self, v):
        """bootstrap distribution of the mean of v (ratio of resampled sum / resampled count)."""
        S = self._sums(v)
        N = np.array([len(m) for m in self.members], float)
        return S[self.draws].sum(1) / np.maximum(N[self.draws].sum(1), 1)

    def ratio(self, num, den):
        """bootstrap distribution of sum(num)/sum(den)."""
        a, b = self._sums(num), self._sums(den)
        return a[self.draws].sum(1) / np.maximum(b[self.draws].sum(1), 1e-12)

    def stat(self, fn, B=None):
        """generic statistic fn(idx) -> float on the concatenated item indices of a resample (slow path; used for AUROC, AURC)."""
        B = B or self.B
        out = np.empty(B)
        for i in range(B):
            idx = np.concatenate([self.members[j] for j in self.draws[i]]) if self.k else np.zeros(0, int)
            out[i] = fn(idx)
        return out

    def loo(self, fn):
        """leave-one-db-out values of fn(idx): list of (db, value)."""
        out = []
        for j, d in enumerate(self.u):
            idx = np.concatenate([m for i, m in enumerate(self.members) if i != j]) if self.k > 1 else np.zeros(0, int)
            out.append((d, float(fn(idx))))
        return out


def pctl_ci(boot, alpha=0.05):
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return (None, None)
    return (float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2))))


def p_le(boot, null):
    """one-sided bootstrap p for H0: stat <= null  =  (#{boot <= null} + 1) / (B + 1)  (smallest alpha with the (1-2 alpha) interval above null)."""
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    return float((np.sum(b <= null) + 1.0) / (len(b) + 1.0))


def p_ge(boot, null):
    b = np.asarray(boot, float)
    b = b[np.isfinite(b)]
    return float((np.sum(b >= null) + 1.0) / (len(b) + 1.0))


def item_boot_means(v, B=10000, seed=SEED_BOOT):
    v = np.asarray(v, float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(v), size=(B, len(v)))
    return v[idx].mean(1)


def item_boot_ratio(num, den, B=10000, seed=SEED_BOOT):
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(num), size=(B, len(num)))
    return num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)


def signflip_p(d, dbs, reps=10000, seed=SEED_PERM):
    """paired sign-flip permutation test on the mean of per-item differences d, flips drawn per item (stratified by db only in that the
    statistic is reported per db elsewhere); two-sided p."""
    d = np.asarray(d, float)
    rng = np.random.RandomState(seed)
    obs = abs(d.mean())
    cnt = 0
    chunk = 500
    done = 0
    while done < reps:
        r = min(chunk, reps - done)
        sg = rng.choice([-1.0, 1.0], size=(r, len(d)))
        cnt += int(np.sum(np.abs((sg * d).mean(1)) >= obs - 1e-15))
        done += r
    return float((cnt + 1.0) / (reps + 1.0))


# ------------------------------------------------------------------------------------------ misc tests
def cochran_q(M):
    """M: [n_items, k] 0/1 matrix. -> dict(q, df, p)."""
    M = np.asarray(M, float)
    n, k = M.shape
    C = M.sum(0)
    R = M.sum(1)
    T = M.sum()
    den = k * T - (R ** 2).sum()
    if den <= 0 or k < 2:
        return {"q": 0.0, "df": k - 1, "p": 1.0}
    q = (k - 1) * (k * (C ** 2).sum() - T ** 2) / den
    return {"q": float(q), "df": k - 1, "p": chi2_sf(q, k - 1)}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rx, ry = midrank(x), midrank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return None
    labs = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / float(n)
    pe = sum((sum(1 for x in a if x == l) / float(n)) * (sum(1 for y in b if y == l) / float(n)) for l in labs)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def gwet_ac1(a, b):
    a, b = list(a), list(b)
    n = len(a)
    labs = sorted(set(a) | set(b))
    q = len(labs)
    if n == 0 or q < 2:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / float(n)
    pi = [(sum(1 for x in a if x == l) + sum(1 for y in b if y == l)) / (2.0 * n) for l in labs]
    pe = sum(p * (1 - p) for p in pi) / (q - 1)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def verdict(p_adj, cl_ok, item_ok=None, favourable=None, p_adj_item=None):
    """SAP 13.3 decision rule. confirmed: Holm-adjusted p < .05 and the db-clustered CI satisfies the bound; supported at item level: item-level
    Holm p < .05 and the item-level interval satisfies it; direction consistent: only the point estimate is on the right side; else not confirmed."""
    if p_adj is not None and p_adj < 0.05 and cl_ok:
        return "confirmed"
    pi = p_adj_item if p_adj_item is not None else p_adj
    if pi is not None and pi < 0.05 and item_ok:
        return "supported at item level, not at database level"
    if favourable:
        return "direction consistent, not confirmed"
    return "not confirmed"
