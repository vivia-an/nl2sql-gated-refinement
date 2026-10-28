"""Logistic GLMM with crossed random intercepts, Laplace approximation (numpy + scipy.optimize).

Used for the E4 / P4 contrast  ex ~ policy + (1|db) + (1|qid) + (1|perm)  (SAP 5).  R / lme4 is not installed on the worker machines and the SAP's
named fallback (statsmodels BinomialBayesMixedGLM) is not available either, so this module implements the same model class directly:
    eta = X beta + sum_f b_f[level_f],  b_f ~ N(0, sigma_f^2),  y ~ Bernoulli(logit^-1 eta)
Marginal log-likelihood by the Laplace approximation with the inner Newton problem solved exactly (dense, q <= ~600 latent effects).
Outputs: beta, SE (observed information from central finite differences over (beta, log sigma)), Wald p, LRT p against the model without the
tested columns. Validated in selftest (parameter recovery, agreement with plain logistic regression when sigma -> 0)."""
import math

import numpy as np

from . import stats as S


def _prep(y, X, groups):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    codes, sizes = [], []
    for g in groups:
        u, inv = np.unique(np.asarray(g), return_inverse=True)
        codes.append(inv)
        sizes.append(len(u))
    order = sorted(range(len(sizes)), key=lambda i: sizes[i])          # largest factor last (its block of H is diagonal -> Schur complement)
    codes, sizes = [codes[i] for i in order], [sizes[i] for i in order]
    offs = np.cumsum([0] + sizes[:-1])
    inv_order = [0] * len(order)
    for pos, i in enumerate(order):
        inv_order[i] = pos
    Z = {"inv_order": inv_order, "codes": codes, "sizes": sizes, "offs": [int(o) for o in offs], "q": int(sum(sizes)), "n": len(y)}
    return y, X, Z, sizes


class Fit(object):
    pass


def _zdot(Z, b):
    out = np.zeros(Z["n"])
    for c, o in zip(Z["codes"], Z["offs"]):
        out += b[o + c]
    return out


def _ztdot(Z, v):
    out = np.zeros(Z["q"])
    for c, o, sz in zip(Z["codes"], Z["offs"], Z["sizes"]):
        out[o:o + sz] = np.bincount(c, weights=v, minlength=sz)
    return out


def _zwz(Z, w):
    q = Z["q"]
    H = np.zeros((q, q))
    cs, os_, ss = Z["codes"], Z["offs"], Z["sizes"]
    for i in range(len(cs)):
        H[os_[i]:os_[i] + ss[i], os_[i]:os_[i] + ss[i]] += np.diag(np.bincount(cs[i], weights=w, minlength=ss[i]))
        for j in range(i + 1, len(cs)):
            blk = np.bincount(cs[i] * ss[j] + cs[j], weights=w, minlength=ss[i] * ss[j]).reshape(ss[i], ss[j])
            H[os_[i]:os_[i] + ss[i], os_[j]:os_[j] + ss[j]] += blk
            H[os_[j]:os_[j] + ss[j], os_[i]:os_[i] + ss[i]] += blk.T
    return H


def _solve_logdet(H, g, ns):
    """Solve H x = g and return (x, logdet H) using the Schur complement on the last block (diagonal) of size q - ns."""
    q = H.shape[0]
    if ns <= 0 or ns >= q:
        L = np.linalg.cholesky(H)
        x = np.linalg.solve(H, g) if g is not None else None
        return x, 2.0 * float(np.sum(np.log(np.diag(L))))
    d = np.diag(H)[ns:]
    Hss, Hsl = H[:ns, :ns], H[:ns, ns:]
    Dinv = 1.0 / d
    S_ = Hss - (Hsl * Dinv[None, :]).dot(Hsl.T)
    x = None
    if g is not None:
        gs, gl = g[:ns], g[ns:]
        xs = np.linalg.solve(S_, gs - (Hsl * Dinv[None, :]).dot(gl))
        xl = Dinv * (gl - Hsl.T.dot(xs))
        x = np.concatenate([xs, xl])
    try:
        ld = float(np.sum(np.log(d))) + 2.0 * float(np.sum(np.log(np.diag(np.linalg.cholesky(S_)))))
    except np.linalg.LinAlgError:
        ld = float(np.sum(np.log(d))) + float(np.linalg.slogdet(S_)[1])
    return x, ld


def _laplace(beta, logsig, y, X, Z, sizes, b0=None, iters=60):
    """-> (negative Laplace log-likelihood, bhat)."""
    q = Z["q"]
    ns = q - Z["sizes"][-1] if len(Z["sizes"]) > 1 else 0
    dinv = np.concatenate([np.full(sz, math.exp(-2.0 * ls)) for sz, ls in zip(sizes, logsig)])
    xb = X.dot(beta)
    b = np.zeros(q) if b0 is None else b0.copy()
    def pen(bb):
        e = np.clip(xb + _zdot(Z, bb), -35, 35)
        return float(np.sum(y * e - np.logaddexp(0.0, e))) - 0.5 * float(np.sum(dinv * bb * bb))

    cur = pen(b)
    for _ in range(iters):
        eta = np.clip(xb + _zdot(Z, b), -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        g = _ztdot(Z, y - p) - dinv * b
        w = p * (1 - p)
        H = _zwz(Z, w) + np.diag(dinv)
        try:
            step, _ = _solve_logdet(H, g, ns)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        t = 1.0
        while t > 1e-4:                                   # step halving keeps the penalised log-likelihood increasing
            new = pen(b + t * step)
            if np.isfinite(new) and new >= cur - 1e-10:
                break
            t *= 0.5
        b = b + t * step
        cur = pen(b)
        if np.max(np.abs(t * step)) < 1e-8:
            break
    eta = np.clip(xb + _zdot(Z, b), -35, 35)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = p * (1 - p)
    H = _zwz(Z, w) + np.diag(dinv)
    try:
        _, logdetH = _solve_logdet(H, None, ns)
    except np.linalg.LinAlgError:
        logdetH = float(np.linalg.slogdet(H)[1])
    ll = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    logdetD = float(np.sum(np.concatenate([np.full(sz, 2.0 * ls) for sz, ls in zip(sizes, logsig)])))
    val = ll - 0.5 * float(np.sum(dinv * b * b)) - 0.5 * logdetD - 0.5 * logdetH
    return -val, b


def fit_glmm(y, X, groups, names=None, start_beta=None, logsig_bounds=(-5.0, 3.0)):
    """X must contain the intercept column if wanted. groups: list of label arrays (one per random factor). -> Fit object."""
    from scipy.optimize import minimize
    y, X, Z, sizes = _prep(y, X, groups)
    p = X.shape[1]
    nf = len(sizes)
    state = {"b": None}

    def obj(theta):
        beta, ls = theta[:p], np.clip(theta[p:], logsig_bounds[0], logsig_bounds[1])
        v, b = _laplace(beta, ls, y, X, Z, sizes, state["b"])
        if np.isfinite(v):
            state["b"] = b
        return v if np.isfinite(v) else 1e12

    beta0 = np.zeros(p) if start_beta is None else np.asarray(start_beta, float)
    th0 = np.concatenate([beta0, np.full(nf, -1.0)])
    res = minimize(obj, th0, method="L-BFGS-B", bounds=[(None, None)] * p + [logsig_bounds] * nf, options={"maxiter": 200, "ftol": 1e-10})
    if not res.success:
        res2 = minimize(obj, res.x, method="Nelder-Mead", options={"maxiter": 800, "xatol": 1e-5, "fatol": 1e-9})
        if res2.fun <= res.fun:
            res = res2
    th = res.x
    f = Fit()
    f.beta, f.logsig, f.nll, f.p, f.nf = th[:p], th[p:], float(res.fun), p, nf
    f.sigma_internal = np.exp(np.clip(f.logsig, logsig_bounds[0], logsig_bounds[1]))
    f.sigma = f.sigma_internal[Z["inv_order"]]                      # in the order of the `groups` argument
    f.converged = bool(res.success)
    f.names = names or ["b%d" % i for i in range(p)]
    # observed information by central differences on the free parameters (beta, log sigma); sigma at the lower bound is held fixed
    free = list(range(p)) + [p + i for i in range(nf) if logsig_bounds[0] + 1e-3 < th[p + i]]
    h = 1e-3
    m = len(free)
    Hm = np.zeros((m, m))
    f0 = obj(th)
    for a in range(m):
        for b_ in range(a, m):
            ea, eb = np.zeros(len(th)), np.zeros(len(th))
            ea[free[a]] = h
            eb[free[b_]] = h
            if a == b_:
                val = (obj(th + ea) - 2 * f0 + obj(th - ea)) / (h * h)
            else:
                val = (obj(th + ea + eb) - obj(th + ea - eb) - obj(th - ea + eb) + obj(th - ea - eb)) / (4 * h * h)
            Hm[a, b_] = Hm[b_, a] = val
    try:
        cov = np.linalg.inv(Hm)
        se = np.sqrt(np.maximum(np.diag(cov)[:p], 0.0))
    except np.linalg.LinAlgError:
        cov, se = None, np.full(p, np.nan)
    f.se = se
    f.cov_beta = cov[:p, :p] if cov is not None else None
    f.z = f.beta / np.where(se > 0, se, np.nan)
    f.p_wald = np.array([2 * S.norm_sf(abs(z)) if np.isfinite(z) else float("nan") for z in f.z])
    f.n = len(y)
    return f


def lrt(y, X_full, X_red, groups, names=None):
    """Likelihood-ratio test of the columns that are in X_full but not in X_red. -> dict(chi2, df, p, full, red)."""
    ff = fit_glmm(y, X_full, groups, names)
    fr = fit_glmm(y, X_red, groups)
    chi = max(0.0, 2.0 * (fr.nll - ff.nll))
    df = np.asarray(X_full).reshape(len(y), -1).shape[1] - np.asarray(X_red).reshape(len(y), -1).shape[1]
    return {"chi2": float(chi), "df": int(df), "p": S.chi2_sf(chi, df), "full": ff, "red": fr}


def contrast(y, x, groups, extra=None, label="x", test_cols=(1,)):
    """Covariates: intercept, `x` (+ optional extra columns). Reports the coefficient of `x` (column 1) with Wald CI and Wald p, and the LRT p
    for dropping the columns in `test_cols` (default: x only).
    -> dict(beta, se, ci_lo, ci_hi, odds_ratio, p_wald, p_lrt, sigma, converged)."""
    n = len(y)
    cols = [np.ones(n), np.asarray(x, float)]
    names = ["(Intercept)", label]
    if extra is not None:
        for k, v in extra.items():
            cols.append(np.asarray(v, float))
            names.append(k)
    X = np.column_stack(cols)
    Xr = np.column_stack([c for i, c in enumerate(cols) if i not in test_cols])
    r = lrt(y, X, Xr, groups, names)
    ff = r["full"]
    b, se = float(ff.beta[1]), float(ff.se[1])
    z = S.norm_ppf(0.975)
    return {"beta": b, "se": se, "ci_lo": b - z * se, "ci_hi": b + z * se, "odds_ratio": math.exp(b), "or_lo": math.exp(b - z * se), "or_hi": math.exp(b + z * se),
            "p_wald": float(ff.p_wald[1]), "p_lrt": r["p"], "chi2": r["chi2"], "sigma": [float(s) for s in ff.sigma], "converged": bool(ff.converged and r["red"].converged),
            "n_obs": int(n), "coef_all": {nm: float(v) for nm, v in zip(names, ff.beta)}, "se_all": {nm: float(v) for nm, v in zip(names, ff.se)}}
