#!/usr/bin/env python3
"""Writes the FROZEN configs of the MIN plan (exp/configs/frozen/*.yaml) from the templates exp/configs/min_*.yaml + the frozen artefacts, and prints the exact
lock commands. Host side, stdlib only, Python >= 3.6. Run AFTER the burn-in store, the route coefficients and (if used) the Memo-SQL memory exist
under exp/frozen/ on the host, and AFTER pick_trial.py chose the baseline trials. See the freeze checklist (not released).

  python3 code/tools/freeze_configs.py --m6-path exp/frozen/corpus_burnin_ab12cd34.sqlite --route-coef exp/frozen/route_coef_hr.json \\
         --memo-memory exp/frozen/memo_memory_v1.json --baseline-trials sdebug=1,acsql=1,macsql=1,bon_vote=1 --sap-commit <hash> [--dry-run] [--force]

For every entry of FROZEN below it copies the template (or, for a baseline, the chosen exp/configs/min_tune_<b>_t<k>.yaml) to exp/configs/frozen/<name>.yaml with a
provenance header; configs that use M6 get the line  m6: {policy: ws_full, path: <m6-path>, update: false}  (the same line tools/set_m6_path.py writes; that
tool is NOT used because it patches the legacy configs in place and does not know full_gated); the Memo-SQL config gets memory_sha256 filled in.
A frozen file that already exists is never overwritten unless --force (a frozen config must not change silently; a change is a new lock entry v2, SPEC 1.3 step 6).
Also writes exp/configs/frozen/MANIFEST.json and prints the `python -m nl2sql.lock add ...` commands (run them on a worker machine, see the freeze checklist (not released))."""
import argparse
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CFG = os.path.join(REPO, "exp", "configs")
FROZEN_DIR = os.path.join(CFG, "frozen")

# name -> (system for the lock, template stem, uses M6 store, seeds registered, baseline key for --baseline-trials)
FROZEN = [
    ("zs", "zs_ours", "min_zs", False, "0,101,202", None),
    ("full_fixed", "full_fixed", "min_full_fixed", True, "0", None),
    ("full_gated", "full_gated", "min_full_gated", True, "0,101,202", None),
    ("full_gated_route_ca", "full_gated", "min_full_gated_route_ca", True, "0", None),
    ("gated_noM1", "full_gated", "min_gated_noM1", True, "0", None),
    ("gated_m3force", "full_gated", "min_gated_m3force", True, "0", None),
    ("gated_rootonly", "full_gated", "min_gated_rootonly", True, "0", None),
    ("gated_legacyddl", "full_gated", "min_gated_legacyddl", True, "0", None),
    ("gated_blind", "full_gated_blind", "min_gated_blind", True, "0", None),
    ("sdebug", "sdebug", "min_sdebug", False, "0", "sdebug"),
    ("acsql_s", "acsql_s", "min_acsql", False, "0", "acsql"),
    ("macsql", "macsql", "min_macsql", False, "0", "macsql"),
    ("bon_vote", "bon_vote", "min_bon_vote", False, "0", "bon_vote"),
    ("bon_vote_n3", "bon_vote", "min_bon_vote_n3", False, "0", None),
    ("memosql_light", "memosql", "min_memosql_light", False, "0", None),
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m6-path", default=None, help="repo-relative path of the frozen burn-in store (name must contain its sha256[:8])")
    ap.add_argument("--route-coef", default="exp/frozen/route_coef_hr.json")
    ap.add_argument("--memo-memory", default="exp/frozen/memo_memory_v1.json")
    ap.add_argument("--baseline-trials", default="", help="e.g. sdebug=1,acsql=1,macsql=1,bon_vote=1 (output of pick_trial.py); default = trial 1 = the template")
    ap.add_argument("--sap-commit", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    only = set(a.only.split(",")) if a.only else None
    sel = [r for r in FROZEN if not only or r[0] in only]
    need_m6 = any(r[3] for r in sel)
    need_rc = any(r[0] == "full_gated_route_ca" for r in sel)
    need_mm = any(r[0] == "memosql_light" for r in sel)
    errs = []
    m6 = os.path.join(REPO, a.m6_path) if a.m6_path else None
    if need_m6:                       # F1: staged freeze (--only) needs an artefact only for the configs that use it
        if not m6 or not os.path.exists(m6):
            errs.append("burn-in store %s not found on the host (pull it from the worker machine first)" % a.m6_path)
        else:
            sh = sha256_file(m6)
            if sh[:8] not in os.path.basename(a.m6_path):
                errs.append("file name %s does not contain the sha256 prefix %s of the file" % (os.path.basename(a.m6_path), sh[:8]))
    rc = os.path.join(REPO, a.route_coef)
    if need_rc and not os.path.exists(rc):
        errs.append("route coefficients %s not found" % a.route_coef)
    mm = os.path.join(REPO, a.memo_memory)
    if need_mm and not os.path.exists(mm):
        errs.append("Memo-SQL memory %s not found (copy worker machine b1_work/out/memo_memory_v1.json)" % a.memo_memory)
    trials = dict(kv.split("=") for kv in a.baseline_trials.split(",") if "=" in kv)
    if errs:
        sys.exit("cannot freeze:\n  - " + "\n  - ".join(errs))
    os.makedirs(FROZEN_DIR, exist_ok=True)
    mpath = os.path.join(FROZEN_DIR, "MANIFEST.json")
    old = json.load(open(mpath)) if os.path.exists(mpath) else {}
    manifest = {"date": time.strftime("%F %T"), "m6_path": a.m6_path if need_m6 else old.get("m6_path"), "m6_sha256": sha256_file(m6) if need_m6 else old.get("m6_sha256"),
                "route_coef": a.route_coef, "route_coef_sha256": sha256_file(rc) if os.path.exists(rc) else None,
                "memo_memory": a.memo_memory, "memo_memory_sha256": sha256_file(mm) if os.path.exists(mm) else None, "sap_commit": a.sap_commit,
                "configs": dict(old.get("configs", {}))}
    cmds = []
    for name, system, tmpl, uses_m6, seeds, bkey in FROZEN:
        if only and name not in only:
            continue
        src_stem = tmpl
        if bkey and bkey in trials:
            src_stem = "min_tune_%s_t%d" % (bkey, int(trials[bkey]))
        src = os.path.join(CFG, src_stem + ".yaml")
        if not os.path.exists(src):
            sys.exit("template %s missing" % src)
        body = open(src).read()
        if "FILLED_AT_FREEZE" in body:
            body = body.replace("FILLED_AT_FREEZE", manifest["memo_memory_sha256"])
        if name == "full_gated_route_ca" and a.route_coef not in body:
            sys.exit("%s: coef_path in the template differs from --route-coef %s" % (src_stem, a.route_coef))
        if name == "memosql_light" and a.memo_memory not in body:
            sys.exit("%s: memory_path in the template differs from --memo-memory %s" % (src_stem, a.memo_memory))
        head = "# FROZEN %s by code/tools/freeze_configs.py from exp/configs/%s.yaml (sha256 %s). Do not edit: a change is a new lock entry (SPEC 1.3 step 6).\n" % (
            time.strftime("%F"), src_stem, sha256_file(src)[:16])
        text = head + body.rstrip("\n") + "\n"
        if uses_m6:
            text = re.sub(r"^m6:.*\n?", "", text, flags=re.M)
            text += "m6: {policy: ws_full, path: %s, update: false}\n" % a.m6_path
        dst = os.path.join(FROZEN_DIR, name + ".yaml")
        if os.path.exists(dst) and open(dst).read() != text and not a.force:
            sys.exit("%s exists and differs: frozen configs are not overwritten (use --force only before the lock entry is made)" % dst)
        if not a.dry_run:
            open(dst, "w").write(text)
        manifest["configs"][name] = {"system": system, "template": src_stem, "frozen_sha256": hashlib.sha256(text.encode()).hexdigest(), "seeds": seeds,
                                     "m6": uses_m6}
        cmds.append("$PY -m nl2sql.lock add --system %s --config $R/exp/configs/frozen/%s.yaml --seeds %s%s" % (
            system, name, seeds, (" --sap-commit " + a.sap_commit) if a.sap_commit else " --sap-commit <SAP_COMMIT>"))
        print("%s %-22s <- %-28s system %-10s seeds %s%s" % ("(dry) " if a.dry_run else "wrote", name + ".yaml", src_stem, system, seeds, "  +m6" if uses_m6 else ""))
    cmds.append("$PY -m nl2sql.lock add --system e9_probes --config $R/exp/configs/e9.yaml --seeds 0%s" % ((" --sap-commit " + a.sap_commit) if a.sap_commit else " --sap-commit <SAP_COMMIT>"))
    if not a.dry_run:
        json.dump(manifest, open(os.path.join(FROZEN_DIR, "MANIFEST.json"), "w"), indent=1, sort_keys=True)
    print("\n# lock commands (run on ONE worker machine in this order, after sync_pod.sh; shell: R=<NL2SQL_ROOT>; cd $R/code; export NL2SQL_ROOT=$R PY=$R/venv/bin/python)")
    for c in cmds:
        print(c)


if __name__ == "__main__":
    main()
