"""Small-cluster inference helpers (post hoc, SAP Amendment 11 / internal note S2). numpy (+ scipy for the t quantile) only; no model or API call.

Why: the headline intervals are percentile bootstraps over G = 11 (BIRD dev) or G = 19 (Spider-Realistic) databases, which under-cover. This module
provides the readings a referee asks for with so few clusters:

  cr_t        item-pooled mean (the headline estimate) with the cluster-robust (CR1) standard error and a t(G - 1) interval
  t_db        t(G - 1) interval of the unweighted mean of the database-level mean differences
  sf_p, sf_ci exact sign-flip (randomisation) test over the G database-level values, its one- or two-sided p and the interval obtained by inverting the
              two-sided test at 5 %; all 2^G sign patterns are enumerated (2 048 for G = 11, 524 288 for G = 19), no random draw
  pseudo      delete-one-database jackknife pseudo-values (for statistics that are not item means: ratios of sums, AUROC differences, pooled ratios)
  hull        the widest interval over several intervals (the rule "where readings differ, use the widest")

Weights: w_g = n_g / n reproduces the item-pooled headline estimate; w_g = 1 / G is the unweighted database mean.
"""
import math

import numpy as np

_SIGNS = {}
ALPHA = 0.05


def tcrit(df, alpha=ALPHA):
    from scipy import stats as sps
    return float(sps.t.ppf(1.0 - alpha / 2.0, df))


def t_sf(t, df):
    from scipy import stats as sps
    return float(sps.t.sf(t, df))


def db_level(d, dbs):
    """-> (labels, n_g, m_g): sorted database labels, items per database, mean of d per database."""
    d = np.asarray(d, float)
    dbs = np.asarray(dbs)
    u = sorted(set(dbs.tolist()))
    n_g = np.array([float((dbs == x).sum()) for x in u])
    m_g = np.array([float(d[dbs == x].mean()) for x in u])
    return u, n_g, m_g


def cr_t(d, dbs, theta0=0.0):
    """item-pooled mean of d with the cluster-robust variance  G/(G-1) * sum_g (sum_{i in g} (d_i - mean))^2 / n^2  and t(G-1).
    -> dict est, se, df, lo, hi, t, p2 (two-sided), p_gt (H1: theta > theta0), p_lt (H1: theta < theta0)."""
    d = np.asarray(d, float)
    u, n_g, m_g = db_level(d, dbs)
    G, n = len(u), float(len(d))
    est = float(d.mean())
    s_g = n_g * (m_g - est)
    se = math.sqrt(G / (G - 1.0) * float((s_g ** 2).sum())) / n if G > 1 else float("nan")
    return _t_block(est, se, G - 1, theta0)


def t_db(d, dbs, theta0=0.0):
    """unweighted mean of the database means with se = sd(m_g) / sqrt(G) and t(G-1)."""
    _, _, m_g = db_level(d, dbs)
    G = len(m_g)
    est = float(m_g.mean())
    se = float(m_g.std(ddof=1)) / math.sqrt(G) if G > 1 else float("nan")
    return _t_block(est, se, G - 1, theta0)


def _t_block(est, se, df, theta0):
    q = tcrit(df)
    if not (se > 0):
        t = float("inf") if est != theta0 else 0.0
        p2 = 0.0 if est != theta0 else 1.0
        return {"est": est, "se": se, "df": df, "lo": est, "hi": est, "t": t, "p2": p2, "p_gt": 0.0 if est > theta0 else 1.0, "p_lt": 0.0 if est < theta0 else 1.0}
    t = (est - theta0) / se
    return {"est": est, "se": se, "df": df, "lo": est - q * se, "hi": est + q * se, "t": t, "p2": 2.0 * t_sf(abs(t), df),
            "p_gt": t_sf(t, df), "p_lt": t_sf(-t, df)}


def pseudo(est, loo, G):
    """jackknife pseudo-values  psi_g = G * est - (G - 1) * est_(-g)  from the leave-one-database-out estimates."""
    loo = np.asarray(loo, float)
    return G * float(est) - (G - 1.0) * loo


def sign_matrix(G):
    """all 2^G sign patterns (rows) as float64 +-1; pattern 0 is all +1 (the observed one)."""
    if G not in _SIGNS:
        if G > 22:
            raise ValueError("sign-flip enumeration is limited to G <= 22 (got %d)" % G)
        idx = np.arange(2 ** G, dtype=np.int64)
        bits = (idx[:, None] >> np.arange(G, dtype=np.int64)[None, :]) & 1
        _SIGNS[G] = (1.0 - 2.0 * bits).astype(np.float64)
    return _SIGNS[G]


def _wts(v, w):
    v = np.asarray(v, float)
    if w is None:
        w = np.full(len(v), 1.0 / len(v))
    else:
        w = np.asarray(w, float)
        w = w / w.sum()
    return v, w


def sf_p(v, w=None, theta0=0.0, alt="two"):
    """exact sign-flip p for H0: the database-level values are symmetric about theta0. Statistic T = sum_g w_g s_g (v_g - theta0) over all sign patterns s.
    alt: 'two' (|T| >= |T_obs|), 'gt' (H1 theta > theta0: T >= T_obs), 'lt' (H1 theta < theta0: T <= T_obs). The observed pattern counts, so p >= 2^-G (two-sided 2^(1-G))."""
    v, w = _wts(v, w)
    x = w * (v - theta0)
    T = sign_matrix(len(v)).dot(x)
    obs = float(x.sum())
    tol = 1e-9 * (1.0 + float(np.abs(x).sum()))
    if alt == "two":
        return float(np.mean(np.abs(T) >= abs(obs) - tol))
    if alt == "gt":
        return float(np.mean(T >= obs - tol))
    if alt == "lt":
        return float(np.mean(T <= obs + tol))
    raise ValueError(alt)


def sf_min_p(G, alt="two"):
    return (2.0 if alt == "two" else 1.0) / float(2 ** G)


def sf_ci(v, w=None, alpha=ALPHA, iters=60):
    """interval = {theta0 : two-sided sign-flip p(theta0) > alpha}, found by bisection outward from the weighted mean (p = 1 there).
    Returns (lo, hi) or (None, None) when the test cannot reject at alpha for any theta0 (2^(1-G) > alpha, i.e. G < 6)."""
    v, w = _wts(v, w)
    G = len(v)
    if sf_min_p(G) > alpha:
        return (None, None)
    est = float((w * v).sum())
    span = float(np.abs(v - est).max())
    if span == 0.0:
        return (est, est)
    far = 2.0 * span + 1.0 * abs(est) + 1e-9

    def edge(sign):
        a, b = est, est + sign * far              # a accepted (p > alpha), b rejected
        if sf_p(v, w, b) > alpha:                 # not rejected even far away (cannot happen for G >= 6, kept for safety)
            return b
        for _ in range(iters):
            m = 0.5 * (a + b)
            if sf_p(v, w, m) > alpha:
                a = m
            else:
                b = m
        return a
    return (edge(-1.0), edge(1.0))


def hull(*ivs):
    """widest interval over the given (lo, hi) pairs; pairs with None are skipped."""
    ok = [iv for iv in ivs if iv is not None and iv[0] is not None and iv[1] is not None]
    if not ok:
        return (None, None)
    return (min(x[0] for x in ok), max(x[1] for x in ok))


def deff_cr(d, dbs):
    """design effect from the cluster-robust variance: V_CR / (var(d) / n), var with ddof = 1 (cross-check of the bootstrap ratio)."""
    d = np.asarray(d, float)
    n = float(len(d))
    c = cr_t(d, dbs)
    v_srs = float(d.var(ddof=1)) / n
    return (c["se"] ** 2) / v_srs if v_srs > 0 else None
