#!/usr/bin/env python3
"""Gate job of the PF queues (SAP Amendment 9, section 22). Same check as tools/lock_gate.py (the runner's own nl2sql.lock.verify with the same hashes)
for every locked-split runner command of the given queue files, except that the PF configs live in exp/configs/pf/ (not exp/configs/frozen/, which the
generic gate hard-codes and which is shared with other tasks). Also requires: every lock entry has a sap_commit; no FILLED_AT_FREEZE; exp/pf_prompts/FROZEN.json exists.
Exit 0 only if everything is accepted.   python tools/pf_gate.py --queue <file> [--queue <file> ...]"""
import argparse
import json
import os
import shlex
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from lock_gate import flag, segments  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue", action="append", required=True)
    ap.add_argument("--lock", default=None)
    a = ap.parse_args()
    from nl2sql import lock, paths
    from nl2sql.data import is_locked
    root, repo = os.environ.get("R", str(paths.ROOT)), str(paths.REPO)
    lp = a.lock or str(paths.DEV_LOCK)
    problems, checked = [], 0
    ents = json.load(open(lp)).get("entries", [])
    problems += ["lock entry %s has no sap_commit" % e.get("config_path") for e in ents if not e.get("sap_commit")]
    if not os.path.exists(os.path.join(repo, "exp", "pf_prompts", "FROZEN.json")):
        problems.append("exp/pf_prompts/FROZEN.json missing")
    seen = set()
    for q in a.queue:
        for line in open(q):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            j = json.loads(line)
            for toks in segments(j["command"]):
                toks = [t.replace("$R", root) for t in toks]
                system, split, cfg, seed = flag(toks, "--system"), flag(toks, "--split"), flag(toks, "--config"), int(flag(toks, "--seed", "0"))
                if not (system and split and cfg):
                    problems.append("%s: cannot parse" % j["id"])
                    continue
                if not is_locked(split) or (system, split, seed, cfg) in seen:
                    continue
                seen.add((system, split, seed, cfg))
                ca = cfg if os.path.isabs(cfg) else os.path.join(repo, cfg)
                rel = os.path.relpath(ca, repo)
                if not os.path.exists(ca):
                    problems.append("%s: config %s missing" % (j["id"], rel))
                    continue
                if system != "patent_fb" or not rel.startswith(os.path.join("exp", "configs", "pf") + os.sep):
                    problems.append("%s: not a PF config (%s %s)" % (j["id"], system, rel))
                if "FILLED_AT_FREEZE" in open(ca).read():
                    problems.append("%s: FILLED_AT_FREEZE in %s" % (j["id"], rel))
                ok, msg = lock.verify(system, split, seed, ca, lp)
                checked += 1
                print("%-8s %-10s %s seed %d  (%s)" % ("OK" if ok else "REFUSED", system, rel, seed, msg))
                if not ok:
                    problems.append("%s: %s" % (j["id"], msg))
    if problems or not checked:
        print("PF GATE FAILED (%d problem(s), %d checked):" % (len(problems), checked))
        for p in problems:
            print("  - " + p)
        return 1
    print("PF GATE OK: %d locked-split commands accepted" % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
