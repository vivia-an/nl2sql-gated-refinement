"""Run discovery, loading and alignment.

A run = directory with preds.jsonl (+ run_meta.json, config.yaml, cands.jsonl, usage*.jsonl, summary.json) as written by nl2sql.run.
Roles ("labels") such as `full_gated` or `abl_m1` are resolved by (system, config stem, model key, seed, split) from run_meta.json and
config.yaml (`argv.config`), or pinned in exp/analysis_manifest.json (always wins). Incomplete runs are never used silently.
Python 3.6 compatible."""
import glob
import json
import os
import re

import numpy as np

SEEDS = {0: 0, 1: 101, 2: 202}          # repetition index -> seed (EXPERIMENT_SPEC 0)
SEED_TO_REP = {v: k for k, v in SEEDS.items()}
DEV_N = {"bird_dev": 1534}              # SAP 1; every other split: size of the largest run found
PRIMARY_MODEL = "qwen37plus"
BASELINE_SYSTEMS = {"zs": ["zs", "zero_shot", "zs_ours"], "dail": ["dail", "dail_sql"], "sdebug": ["sdebug", "self_debug"], "macsql": ["macsql", "mac_sql"],
                    "acsql_s": ["acsql_s", "ac_sql"], "acsql_x": ["acsql_x", "ac_sql_x"], "memosql": ["memosql", "memo_sql"],
                    "alphasql": ["alphasql", "alpha_sql"], "bon_vote": ["bon_vote", "bon_exec_vote"], "sc_cot": ["sc_cot", "self_consistency"],
                    "bon_critic": ["bon_critic"]}
SYSTEM_TO_BASELINE = {s: k for k, v in BASELINE_SYSTEMS.items() for s in v}
# role label -> list of alternative selectors; a selector = {"systems": [...], "cfgs": [...]} (AND inside, OR between selectors)
LABELS = {
    "full_fixed": [{"systems": ["full_fixed"], "cfgs": ["full_fixed", "full_quote"]}],
    "full_gated": [{"systems": ["full_gated"], "cfgs": ["full_gated", "full_gated_route_ca"]}],            # MIN: r0 = full_gated_route_ca (one run serves P1, P3, P5)
    "full_gated_blind": [{"systems": ["full_gated_blind"]}, {"cfgs": ["full_gated_blind"]}],
    "full_gated_route": [{"systems": ["full_gated"], "cfgs": ["full_gated_cons_lr", "full_gated_cons", "full_gated_route", "full_gated_route_ca"]}],
    "full_gated_arb": [{"cfgs": ["full_gated_arb"]}],
    "full_rl": [{"systems": ["full_rl"]}],
    "full_x": [{"systems": ["full_x"]}],
    "abl_m1": [{"cfgs": ["abl_m1", "gated_noM1"]}], "abl_m2": [{"cfgs": ["abl_m2"]}], "abl_m3": [{"cfgs": ["abl_m3"]}], "abl_m3force": [{"cfgs": ["full_m3force", "full_gated_m3force", "gated_m3force"]}],
    "abl_m5": [{"cfgs": ["abl_m5", "gated_rootonly"]}], "abl_m6": [{"cfgs": ["abl_m6"]}], "abl_q": [{"cfgs": ["abl_q", "full_nq", "full_gated_nq", "gated_legacyddl"]}],
    "crit_v1": [{"cfgs": ["crit_v1"]}], "crit_v2": [{"cfgs": ["crit_v2"]}], "crit_v5": [{"cfgs": ["crit_v5"]}],
}
for _k, _v in BASELINE_SYSTEMS.items():
    LABELS[_k] = [{"systems": _v}]


class RunError(Exception):
    pass


def jl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def read_yaml(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        return None


def model_keys(models_yaml):
    """model id (and key) -> key from exp/models.yaml."""
    out = {}
    d = read_yaml(models_yaml) if models_yaml and os.path.exists(models_yaml) else None
    if d:
        for m in d.get("models", []):
            out[m["id"]] = m["key"]
            out[m["key"]] = m["key"]
    return out


class Run(object):
    def __init__(self, path, meta, cfg_stem, modelkey):
        self.path = path
        self.meta = meta
        self.system = meta.get("system")
        self.split = meta.get("split")
        self.model = meta.get("model")
        self.modelkey = modelkey
        self.seed = meta.get("seed", 0)
        self.rep = SEED_TO_REP.get(self.seed)
        self.cfg = cfg_stem
        self.run_id = meta.get("run_id") or os.path.basename(path)
        self._rows = None
        self._cands = None
        self.label = None
        self.subset = False                 # set by Registry.complete for subset baselines (SAP 10 / SPEC 4 subset rule)

    def __repr__(self):
        return "Run(%s, sys=%s, cfg=%s, model=%s, seed=%s, split=%s)" % (self.run_id, self.system, self.cfg, self.modelkey, self.seed, self.split)

    @property
    def rows(self):
        """qid -> row (last occurrence wins)."""
        if self._rows is None:
            d = {}
            for r in jl(os.path.join(self.path, "preds.jsonl")):
                d[r["qid"]] = r
            self._rows = d
        return self._rows

    @property
    def n_rows(self):
        return len(self.rows)

    @property
    def cands(self):
        """list of candidate rows (cands.jsonl) or None."""
        if self._cands is None:
            p = os.path.join(self.path, "cands.jsonl")
            self._cands = jl(p) if os.path.exists(p) else []
        return self._cands or None

    def roots(self):
        """qid -> root candidate (cand_idx 0) or {}."""
        out = {}
        for c in (self.cands or []):
            if c.get("cand_idx") == 0:
                out[c["qid"]] = c
        return out

    def usage_files(self):
        return sorted(glob.glob(os.path.join(self.path, "usage*.jsonl")))


def scan(runs_root, models_yaml=None):
    """All directories below runs_root that contain preds.jsonl."""
    mk = model_keys(models_yaml)
    out = []
    for dirpath, dirs, files in os.walk(runs_root):
        if "preds.jsonl" not in files:
            continue
        dirs[:] = []                                            # runs do not nest
        meta = {}
        mp = os.path.join(dirpath, "run_meta.json")
        if os.path.exists(mp):
            try:
                meta = json.load(open(mp))
            except ValueError:
                meta = {}
        cfg_stem = None
        cp = os.path.join(dirpath, "config.yaml")
        if os.path.exists(cp):
            y = read_yaml(cp)
            argv = (y or {}).get("argv") or {}
            c = argv.get("config")
            cfg_stem = os.path.splitext(os.path.basename(c))[0] if c else None
            if not meta:
                meta = {"run_id": (y or {}).get("run_id")}
            meta.setdefault("exp", argv.get("exp"))
        if not meta.get("system"):                              # fall back on the first row
            try:
                with open(os.path.join(dirpath, "preds.jsonl")) as f:
                    r0 = json.loads(f.readline())
                meta.update({"system": r0.get("system"), "split": r0.get("split"), "model": r0.get("model"), "seed": r0.get("seed", 0)})
            except Exception:
                continue
        meta["exp"] = meta.get("exp") or os.path.relpath(dirpath, runs_root).split(os.sep)[0]
        mkey = mk.get(meta.get("model"), meta.get("model"))
        out.append(Run(dirpath, meta, cfg_stem, mkey))
    return out


class Registry(object):
    def __init__(self, runs_root, manifest=None, models_yaml=None, split="bird_dev", primary_model=PRIMARY_MODEL, expected_n=None, replay_dirs=None):
        self.root = runs_root
        self.split = split
        self.primary = primary_model
        self.manifest = manifest or {}
        self.all = scan(runs_root, models_yaml) if runs_root and os.path.isdir(runs_root) else []
        if replay_dirs is not None:                              # explicit run selection: only these directories take part (see exp/replay_manifest.json)
            rel = lambda r: os.path.relpath(r.path, runs_root).replace(os.sep, "/")
            have = set(rel(r) for r in self.all)
            missing = sorted(set(replay_dirs) - have)
            if missing:
                raise RunError("exp/replay_manifest.json lists %d run directories that are not under %s (first: %s); unpack the complete run archive there, "
                               "or pass --replay-manifest none to analyse whatever is found" % (len(missing), runs_root, ", ".join(missing[:3])))
            keep = set(replay_dirs)
            self.all = [r for r in self.all if rel(r) in keep]
        self.expected_n = expected_n or DEV_N.get(split)
        if self.expected_n is None:
            ns = [r.n_rows for r in self.all if r.split == split]
            self.expected_n = max(ns) if ns else None
        self.notes = []                                          # human-readable resolution log -> results/<exp>.json _meta

    # -------------------------------------------------------------------------
    def _pinned(self, label, model, rep):
        key = "%s:%s:r%d" % (label, model, rep)
        runs = self.manifest.get("runs", {})
        p = runs.get(key) or (runs.get("%s:r%d" % (label, rep)) if model == self.primary else None) or (runs.get(label) if model == self.primary and rep == 0 else None)
        if not p:
            return None
        cand = p if os.path.isabs(p) else os.path.join(self.root, p)
        for r in self.all:
            if os.path.abspath(r.path) == os.path.abspath(cand):
                return r
        raise RunError("manifest pins %s to %s but no run with preds.jsonl was found there" % (key, p))

    def candidates(self, label, model=None, rep=0, split=None, selectors=None):
        model = model or self.primary
        split = split or self.split
        sels = selectors if selectors is not None else LABELS.get(label)
        if sels is None:
            raise RunError("unknown label %s" % label)
        out = []
        for r in self.all:
            if r.split != split or r.modelkey != model or r.rep != rep:
                continue
            for s in sels:
                if "systems" in s and r.system not in s["systems"]:
                    continue
                if "cfgs" in s and r.cfg not in s["cfgs"]:
                    continue
                out.append(r)
                break
        return out

    def expected_for(self, split):
        if split == self.split and self.expected_n is not None:
            return self.expected_n
        if split in DEV_N:
            return DEV_N[split]
        ns = [r.n_rows for r in self.all if r.split == split]
        return max(ns) if ns else None

    SUBSET_SYSTEMS = ("memosql", "memo_sql", "alphasql", "alpha_sql")     # run on a stratified subset by design (MINIMAL_PLAN M2, SPEC 4): complete at >= 100 rows

    def complete(self, run):
        e = self.expected_for(run.split)
        if e is None or run.n_rows >= e:
            return True
        if run.system in self.SUBSET_SYSTEMS and run.n_rows >= 100:
            run.subset = True
            return True
        return False

    def get(self, label, model=None, rep=0, split=None):
        """-> Run, or None if there is no (complete) run. Raises RunError on ambiguity (pin the run in the manifest)."""
        model = model or self.primary
        pinned = self._pinned(label, model, rep) if (split is None or split == self.split) else None
        if pinned is not None:
            cands = [pinned]
        else:
            cands = self.candidates(label, model, rep, split)
            if label in BASELINE_SYSTEMS:
                cands = [c for c in cands if c.cfg is None or not re.match(r"^(abl_|full_)", c.cfg or "")]
        if not cands:
            return None
        done = [c for c in cands if self.complete(c)]
        if not done:
            c = max(cands, key=lambda r: r.n_rows)
            self.notes.append("%s:%s:r%d incomplete (%d of %s rows in %s)" % (label, model, rep, c.n_rows, self.expected_for(c.split), c.path))
            return None
        if len(done) > 1:
            # identical config sha -> a rerun of the same thing: take the one with most rows, then the one that started last (run_meta.json "started"),
            # then the run id. Never file times: they differ between copies of an archive. Replays avoid the choice altogether (exp/replay_manifest.json).
            shas = set(c.meta.get("config_sha256") for c in done)
            if len(shas) == 1:
                done.sort(key=lambda r: (r.n_rows, str(r.meta.get("started") or ""), r.run_id, r.path))
                return done[-1]
            raise RunError("ambiguous role %s:%s:r%d - %d complete runs (%s); pin one in exp/analysis_manifest.json" % (
                label, model, rep, len(done), "; ".join("%s cfg=%s" % (c.run_id, c.cfg) for c in done)))
        run = done[0]
        run.label = label
        return run

    def baselines(self, model=None, rep=0):
        """all complete baseline runs on the split -> {display label: Run}; same system with several configs -> `system@cfg`."""
        model = model or self.primary
        found = {}
        for lab in BASELINE_SYSTEMS:
            cands = [c for c in self.candidates(lab, model, rep) if self.complete(c) and not re.match(r"^(abl_|full_)", c.cfg or "")]
            pinned = self._pinned(lab, model, rep)
            if pinned is not None:
                cands = [pinned]
            by_cfg = {}
            for c in cands:
                by_cfg.setdefault(c.meta.get("config_sha256"), []).append(c)
            for sha, grp in by_cfg.items():
                grp.sort(key=lambda r: r.n_rows)
            runs = [g[-1] for g in by_cfg.values()]
            for r in runs:
                name = lab if len(runs) == 1 else "%s@%s" % (lab, r.cfg or r.run_id[-8:])
                r.label = name
                found[name] = r
        return found


# ------------------------------------------------------------------------------------------ alignment
def ex_value(r):
    """EX with the SAP 10 rule: api_fail / parse_fail count as 0; rows with ex null (infrastructure errors) are dropped (None)."""
    v = r.get("ex")
    if v is None:
        if r.get("error_class") in ("api_fail", "parse_fail"):
            return 0
        return None
    return int(v)


def align(runs, qids=None, require_ex=True):
    """Items present in all runs (and with a usable EX in all runs if require_ex). -> dict(qids, dbs(np), rows {label: [row]}, ex {label: np}, dropped)."""
    labs = list(runs)
    common = None
    for lab in labs:
        s = set(runs[lab].rows)
        common = s if common is None else (common & s)
    q = sorted(common if qids is None else (set(qids) & common))
    dropped = {"not_in_all": (max(len(r.rows) for r in runs.values()) - len(q)) if runs else 0, "ex_null": 0}
    keep = []
    for x in q:
        if require_ex and any(ex_value(runs[l].rows[x]) is None for l in labs):
            dropped["ex_null"] += 1
            continue
        keep.append(x)
    rows = {l: [runs[l].rows[x] for x in keep] for l in labs}
    ex = {l: np.array([ex_value(r) if ex_value(r) is not None else -1 for r in rows[l]], int) for l in labs}
    dbs = np.array([rows[labs[0]][i].get("db_id") or keep[i].split(":")[1] for i in range(len(keep))]) if labs else np.array([])
    return {"qids": keep, "dbs": dbs, "rows": rows, "ex": ex, "dropped": dropped}


def col(rows, key, default=0.0):
    return np.array([(r.get(key) if r.get(key) is not None else default) for r in rows], float)


def load_manifest(path):
    if path and os.path.exists(path):
        return json.load(open(path))
    return {}


def load_replay_dirs(path):
    """exp/replay_manifest.json -> list of run directories (relative to the runs root) that the analyses use; None if `path` is None or "none"
    (the synthetic self-test and tuning-set work scan everything they find). A path that does not exist is an error."""
    if not path or str(path).lower() == "none":
        return None
    with open(path) as f:
        m = json.load(f)
    return sorted(m["runs"])


def load_decisions(path):
    """exp/selection_decisions.json (S1 selection_rules.py output; may not exist yet). Recognised keys: full ('full_gated'|'full_fixed'),
    route_ca_adopted (bool), bstar (baseline label chosen on HR), m_family (int)."""
    if path and os.path.exists(path):
        return json.load(open(path))
    return {}
