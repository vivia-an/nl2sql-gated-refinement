"""Dev lock (CONTRACT section 3, SPEC 1.3). exp/dev_lock.json = {"entries": [ {system, config_path, config_sha256,
models_yaml_sha256, prompts_sha256, controller_sha256, corpus_sha256, git_hash, date, datasets_allowed, seeds,
sap_commit} ]}. A run on any split outside UNLOCKED needs a matching entry; any hash mismatch refuses the run.

CLI for the main thread:  python -m nl2sql.lock add --system full_fixed --config exp/configs/frozen/full_fixed.yaml
    [--seeds 0,101,202] [--sap-commit HASH]   (appends an entry computed from the current files)"""
import argparse
import fnmatch
import json
import time
from pathlib import Path

from . import config as C
from . import paths
from .data import is_locked, dataset_of
from .util import sha256_file

DEFAULT_DATASETS = ["bird_dev", "bird_minidev", "spider_*", "ambiqt", "livesql_lite"]


def _hash_or_marker(rel):
    p = paths.REPO / rel
    return sha256_file(p) if p.exists() else "MISSING:%s" % rel      # a later-appearing file then mismatches -> run refused


def _artefact_hashes(cfg):
    """Hashes of the frozen artefacts a config points at. The legacy top-level keys (controller.path / corpus.path) are kept; the
    keys the system really reads are m7.controller.path (RL controller), m6.path (burn-in store) and, for the learned critic V5,
    m5.learned.path (X3, 2026-10-26: before this, controller and corpus files did not enter the lock because the keys differed)."""
    out = {"controller_sha256": None, "corpus_sha256": None, "critic_sha256": None, "route_sha256": None}
    ctl = (cfg.get("controller") or {}).get("path") or (((cfg.get("m7") or {}).get("controller")) or {}).get("path")
    if ctl:
        out["controller_sha256"] = _hash_or_marker(ctl)
    cp = (cfg.get("corpus") or {}).get("path") or (cfg.get("m6") or {}).get("path")
    if cp:
        out["corpus_sha256"] = _hash_or_marker(cp)
    m5 = cfg.get("m5") or {}
    if m5.get("variant") == "v5":
        out["critic_sha256"] = _hash_or_marker((m5.get("learned") or {}).get("path") or "exp/frozen/critic_v5.pkl")
    rt = cfg.get("route") or {}
    if rt.get("coef_path"):                      # X5: fitted routing coefficients (tools/fit_route_score.py) enter the lock
        out["route_sha256"] = _hash_or_marker(rt["coef_path"])
    return out


def compute_entry(system, config_path, seeds=(0, 101, 202), sap_commit=None):
    cfg = C.load_config(config_path)
    e = {"system": system, "config_path": str(Path(config_path).resolve().relative_to(paths.REPO)),
         "config_sha256": sha256_file(config_path), "models_yaml_sha256": C.models_sha(),
         "prompts_sha256": C.prompts_sha(), "git_hash": C.git_hash(), "date": time.strftime("%Y-%m-%d"),
         "datasets_allowed": DEFAULT_DATASETS, "seeds": list(seeds), "sap_commit": sap_commit}
    e.update(_artefact_hashes(cfg))
    return e


def verify(system, split, seed, config_path, lock_path=None):
    """(ok, message). Unlocked splits always pass."""
    if not is_locked(split):
        return True, "unlocked split"
    lp = Path(lock_path or paths.DEV_LOCK)
    if not lp.exists():
        return False, "no exp/dev_lock.json: runs on %s are refused until the config is registered" % split
    entries = json.load(open(lp)).get("entries", [])
    cur = compute_entry(system, config_path)
    ds = dataset_of(split)
    why = []
    for e in entries:
        if e.get("system") != system:
            continue
        bad = [k for k in ("config_sha256", "models_yaml_sha256", "prompts_sha256", "controller_sha256", "corpus_sha256", "critic_sha256", "route_sha256")
               if e.get(k) != cur.get(k)]
        if bad:
            why.append("hash mismatch: %s" % ",".join(bad))
            continue
        if not any(fnmatch.fnmatch(ds, pat) for pat in e.get("datasets_allowed", [])):
            why.append("dataset %s not allowed" % ds)
            continue
        if e.get("seeds") and seed not in e["seeds"]:
            why.append("seed %s not registered" % seed)
            continue
        return True, "lock entry ok (git %s)" % e.get("git_hash")
    return False, "dev lock refused: " + ("; ".join(why) if why else "no entry for system %s" % system)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("add")
    a.add_argument("--system", required=True)
    a.add_argument("--config", required=True)
    a.add_argument("--seeds", default="0,101,202")
    a.add_argument("--sap-commit", default=None)
    args = ap.parse_args()
    if args.cmd == "add":
        e = compute_entry(args.system, args.config, [int(x) for x in args.seeds.split(",")], args.sap_commit)
        lp = paths.DEV_LOCK
        d = json.load(open(lp)) if lp.exists() else {"entries": []}
        d["entries"].append(e)
        json.dump(d, open(lp, "w"), indent=1)
        print(json.dumps(e, indent=1))


if __name__ == "__main__":
    main()
