"""X5 routing score (M7 route-to-human / selective prediction; HEADROOM row 2).

route.score = '+'-joined components of {v3, exec, consistency}:
  v3           final-answer V3 critic score (0-10; the patent score; needs the semantic critic call)
  exec         execution-grounded syntax / field-mismatch sub-score of the final answer (0-10, no LLM call)
  consistency  fraction of k extra T>0 samples of the initial generation whose execution result equals the final result (0-1)
The routing score is on the 0-10 scale (higher = more trustworthy):
  * single component: v3 -> as is; exec -> as is; consistency -> 10 * consistency
  * several components, `route.coef` / `route.coef_path` given: 10 * sigmoid(intercept + sum_i coef_i * feature_i)  (logistic regression
    fitted by tools/fit_route_score.py; features v3 (0-10), exec (0-10), consistency (0-1), nonempty (0/1))
  * several components, no coefficients: weighted mean of the components on the 0-10 scale, `route.weights` (default equal)
The file written by tools/fit_route_score.py is {"features": [...], "coef": [...], "intercept": b, "link": "logistic", ...}; it may use any
subset of the four features above (e.g. LR(V3, consistency, nonempty), the best row of HEADROOM section 2)."""
import json
import math

from ... import paths

COMPONENTS = ("v3", "exec", "consistency")
FEATURES = ("v3", "exec", "consistency", "nonempty")


def parse_kind(kind):
    parts = [p.strip() for p in str(kind or "v3").split("+") if p.strip()]
    bad = [p for p in parts if p not in COMPONENTS]
    if not parts or bad:
        raise ValueError("route.score %r: components must be from %s" % (kind, COMPONENTS))
    return parts


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def load_coef(route_cfg):
    """-> dict(features, coef, intercept) or None."""
    c = route_cfg.get("coef")
    if not c and route_cfg.get("coef_path"):
        p = str(route_cfg["coef_path"])
        p = p if p.startswith("/") else str(paths.REPO / p)
        with open(p) as f:
            c = json.load(f)
    if not c:
        return None
    feats, coef = list(c["features"]), [float(x) for x in c["coef"]]
    if len(feats) != len(coef) or any(f not in FEATURES for f in feats):
        raise ValueError("route coefficients: bad features %r" % (feats,))
    return {"features": feats, "coef": coef, "intercept": float(c.get("intercept", 0.0))}


class RouteScorer:
    def __init__(self, route_cfg):
        self.cfg = route_cfg or {}
        self.kind = str(self.cfg.get("score") or "v3")
        self.parts = parse_kind(self.kind)
        self.coef = load_coef(self.cfg) if len(self.parts) > 1 or self.cfg.get("coef") or self.cfg.get("coef_path") else None
        w = self.cfg.get("weights") or {}
        self.weights = {p: float(w.get(p, 1.0)) for p in self.parts}
        if self.coef is not None:
            self.needs = set(self.coef["features"]) - {"nonempty"}
        else:
            self.needs = set(self.parts)
        self.cons_k = int(((self.cfg.get("consistency") or {}).get("k")) or 0)
        if ("consistency" in self.needs) and self.cons_k <= 0:
            raise ValueError("route.score %r needs route.consistency.k > 0" % self.kind)

    @property
    def legacy(self):
        return self.kind == "v3" and self.coef is None

    def score(self, feats):
        """feats: dict with v3 (float|None), exec (float), consistency (float|None), nonempty (0/1). -> routing score on 0-10."""
        if self.coef is not None:
            z = self.coef["intercept"]
            for f, w in zip(self.coef["features"], self.coef["coef"]):
                v = feats.get(f)
                if v is None:
                    raise ValueError("route feature %s unavailable" % f)
                z += w * float(v)
            return round(10.0 * _sigmoid(z), 4)
        vals = {"v3": feats.get("v3"), "exec": feats.get("exec"),
                "consistency": None if feats.get("consistency") is None else 10.0 * float(feats["consistency"])}
        num = den = 0.0
        for p in self.parts:
            if vals[p] is None:
                raise ValueError("route component %s unavailable" % p)
            num += self.weights[p] * float(vals[p])
            den += self.weights[p]
        return round(num / max(den, 1e-9), 4)


# ----------------------------------------------------------------------------- X7: confidence-routed compute (route.adaptive)
ADAPTIVE_TIEBREAK = ("root", "v3")
ADAPTIVE_EMPTY = ("vote", "demote")


def adaptive_decide(keys, empties=None, theta_fast=1.0, tiebreak="root", empty="vote"):
    """Pure decision of the adaptive path (used by the live system AND by tools/adaptive_replay.py, so the replay is the same logic).

    keys[0] = result key of the root, keys[1:] = result keys of the k extra samples (hashable, equal <=> equal result sets; None = the
    execution failed, a failed execution never agrees and never votes). empties[i] = result i is empty / one all-NULL row (only used by empty='demote').
      consistency = share of the samples whose key equals the root's (0 when the root failed)
      consistency >= theta_fast               -> path 'fast', chosen = 0 (the root)
      otherwise path 'vote': the executing candidates are grouped by key and the largest group wins;
        empty='demote': a non-empty group always outranks an empty one (then by size); empty='vote': size only (spec default)
        tie among the best groups: tiebreak 'root' -> the group of the root if it is one of them, else the group whose first member comes
        first (sample order); tiebreak 'v3' -> `tied` lists one representative per tied group (the root's group is index 0) for the caller to
        score with the V3 critic (`chosen` = the 'root' rule until the caller overrides it); no executing candidate at all -> the root.
    -> {"path", "cons", "n", "agree", "chosen" (0 = root, i = sample i-1), "tied", "sizes" (descending), "tie" (bool)}"""
    if tiebreak not in ADAPTIVE_TIEBREAK:
        raise ValueError("route.adaptive.tiebreak must be one of %s, got %r" % (ADAPTIVE_TIEBREAK, tiebreak))
    if empty not in ADAPTIVE_EMPTY:
        raise ValueError("route.adaptive.empty must be one of %s, got %r" % (ADAPTIVE_EMPTY, empty))
    keys = list(keys)
    empties = list(empties) if empties is not None else [False] * len(keys)
    n = len(keys) - 1
    rk = keys[0]
    agree = sum(1 for k in keys[1:] if rk is not None and k is not None and k == rk)
    cons = (agree / float(n)) if n > 0 else None
    out = {"path": "vote", "cons": cons, "n": n, "agree": agree, "chosen": 0, "tied": [], "sizes": [], "tie": False}
    if n <= 0 or agree >= float(theta_fast) * n - 1e-9:
        out["path"] = "fast"
        return out
    groups, order = {}, []
    for i, k in enumerate(keys):
        if k is None:
            continue
        if k not in groups:
            groups[k] = {"first": i, "idx": [], "empty": bool(empties[i])}
            order.append(k)
        groups[k]["idx"].append(i)
    if not groups:
        return out
    def score(g):
        return ((0 if (empty == "demote" and g["empty"]) else 1), len(g["idx"]))
    best = max(score(groups[k]) for k in order)
    top = [groups[k] for k in order if score(groups[k]) == best]
    out["sizes"] = sorted((len(groups[k]["idx"]) for k in order), reverse=True)
    if len(top) == 1:
        out["chosen"] = top[0]["first"]
        return out
    out["tie"] = True
    root_in = any(0 in g["idx"] for g in top)
    out["chosen"] = 0 if root_in else top[0]["first"]
    if tiebreak == "v3":
        out["tied"] = [g["first"] for g in top]            # includes the root's group (index 0) when it is one of them
    return out
