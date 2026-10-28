#!/usr/bin/env python3
"""Gate job of the MIN queue: exits 0 only if EVERY locked-split command of the queue would be accepted by the runner's own lock check.
Runs on a worker machine (needs the venv: nl2sql + yaml), cwd = <root>/code, after `code/scripts/sync_pod.sh` shipped code/, exp/ and the queue file.

  python tools/lock_gate.py --queue $R/exp/queues/MIN.jsonl [--require exp/selection_decisions.json ...] [--lock exp/dev_lock.json]

For every job command it finds the `-m nl2sql.run ...` and `tools/e9_probes.py ...` segments (also behind `&&`), reads --system / --split / --seed / --config
(and, for e9_probes, the default config exp/configs/e9.yaml), and for every split that is locked (bird_dev, minidev, spider_*, ambiqt, livesql_lite)
calls nl2sql.lock.verify(system, split, seed, config): the same function and the same hashes (config, models.yaml, prompts, M6 store, controller,
critic, route coefficients) the runner uses at start-up. Extra checks the runner does NOT do:
  * the config is a frozen file (exp/configs/frozen/...) or the e9 config, never a `min_*` template, and contains no FILLED_AT_FREEZE marker;
  * a config with `memory_path` (Memo-SQL) points at an existing file whose sha256 equals the config's `memory_sha256` (the lock does not hash it);
  * `--qids-file` files exist; `--require` files exist (e.g. exp/selection_decisions.json, which SAP 13.6 wants committed before the lock);
  * every entry of exp/dev_lock.json has a non-empty sap_commit;
  * (warning only) entry git_hash != code/GIT_HASH.txt of the code that is about to run.
A non-zero exit leaves all dependent queue jobs `pending` (queue_run `after`); fix the cause, then `queue_run.py requeue --name MIN --id min-lock-gate`."""
import argparse
import hashlib
import json
import os
import re
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def segments(command):
    """-> list of token lists, one per `&&`-separated simple command that is a runner invocation."""
    out = []
    for part in re.split(r"\s*&&\s*", command):
        try:
            toks = shlex.split(part)
        except ValueError:
            continue
        j = " ".join(toks)
        if "-m nl2sql.run" in j or "e9_probes.py" in j or "nl2sql.stream e4c" in j:
            out.append(toks)
    return out


def flag(toks, name, default=None):
    for i, t in enumerate(toks):
        if t == name and i + 1 < len(toks):
            return toks[i + 1]
        if t.startswith(name + "="):
            return t.split("=", 1)[1]
    return default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--require", nargs="*", default=[], help="files (relative to the repo root) that must exist")
    ap.add_argument("--lock", default=None)
    a = ap.parse_args()

    from nl2sql import config as C, lock, paths
    from nl2sql.data import is_locked

    root = os.environ.get("R", str(paths.ROOT))
    repo = str(paths.REPO)
    problems, warns, checked = [], [], 0

    def sub(s):
        return s.replace("$R", root).replace("${R}", root)

    lp = a.lock or str(paths.DEV_LOCK)
    entries = []
    if not os.path.exists(lp):
        problems.append("no lock file %s" % lp)
    else:
        entries = json.load(open(lp)).get("entries", [])
        if not entries:
            problems.append("lock file has no entries")
        for e in entries:
            if not e.get("sap_commit"):
                problems.append("lock entry %s/%s has no sap_commit (SPEC 1.3 step 7)" % (e.get("system"), e.get("config_path")))
        gh = C.git_hash()
        for e in entries:
            if e.get("git_hash") and e["git_hash"] != gh:
                warns.append("entry %s/%s git_hash %s != code GIT_HASH %s (informational; hashes of config/prompts/models decide)" % (
                    e.get("system"), e.get("config_path"), str(e["git_hash"])[:8], str(gh)[:8]))
    for rf in a.require:
        p = rf if os.path.isabs(rf) else os.path.join(repo, rf)
        if not os.path.exists(p):
            problems.append("required file missing: %s" % rf)

    jobs = []
    for line in open(a.queue):
        line = line.strip()
        if line and not line.startswith("#"):
            jobs.append(json.loads(line))
    seen = set()
    for j in jobs:
        for toks in segments(j["command"]):
            toks = [sub(t) for t in toks]
            joined = " ".join(toks)
            if "e9_probes.py" in joined:
                system, seed = "e9_probes", 0
                cfg = flag(toks, "--config", "exp/configs/e9.yaml")
            else:
                system, seed = flag(toks, "--system"), int(flag(toks, "--seed", "0"))
                cfg = flag(toks, "--config")
            split = flag(toks, "--split")
            if not (system and split and cfg):
                problems.append("%s: cannot parse system/split/config from %r" % (j["id"], joined[:120]))
                continue
            for qf in [flag(toks, "--qids-file")]:
                if qf and not os.path.exists(qf):
                    problems.append("%s: --qids-file %s does not exist" % (j["id"], qf))
            if not is_locked(split):
                continue
            key = (system, split, seed, cfg)
            if key in seen:
                continue
            seen.add(key)
            checked += 1
            cfg_abs = cfg if os.path.isabs(cfg) else os.path.join(repo, cfg)
            rel = os.path.relpath(cfg_abs, repo)
            if not os.path.exists(cfg_abs):
                problems.append("%s: config %s does not exist (freeze_configs.py not run / not synced?)" % (j["id"], rel))
                continue
            if system != "e9_probes" and not rel.startswith(os.path.join("exp", "configs", "frozen") + os.sep):
                problems.append("%s: dev run with a non-frozen config %s" % (j["id"], rel))
            txt = open(cfg_abs).read()
            if "FILLED_AT_FREEZE" in txt:
                problems.append("%s: %s still contains FILLED_AT_FREEZE" % (j["id"], rel))
            cd = C.load_config(cfg_abs)
            mp = cd.get("memory_path")
            if mp:
                mpa = mp if os.path.isabs(mp) else os.path.join(repo, mp)
                if not os.path.exists(mpa):
                    problems.append("%s: memory_path %s missing" % (j["id"], mp))
                elif cd.get("memory_sha256") != sha256_file(mpa):
                    problems.append("%s: memory_sha256 of %s does not match the file" % (j["id"], rel))
            ok, msg = lock.verify(system, split, seed, cfg_abs, lp)
            print("%-8s %-14s %-16s seed %-3s %s  (%s)" % ("OK" if ok else "REFUSED", system, split, seed, rel, msg))
            if not ok:
                problems.append("%s: %s" % (j["id"], msg))
    for w in warns:
        print("WARNING:", w)
    if problems:
        print("\nLOCK GATE FAILED (%d problem(s)):" % len(problems))
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("\nLOCK GATE OK: %d distinct (system, split, seed, config) combinations on locked splits are registered; %d entries in %s" % (checked, len(entries), lp))


if __name__ == "__main__":
    main()
