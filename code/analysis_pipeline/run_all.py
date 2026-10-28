"""One command for every analysis:   python -m analysis_pipeline.run_all --runs $R/runs --results $R/results_out [options]

Order: primary (P1-P5, family of 13.3) -> W1 (SAP 15.3: W1, P1 per backbone, P1', P5', family P*) -> E1 (+E11) -> E2 -> E3 -> E4 -> Psec (P2 in S-A, P4 in S-E) -> E5 -> E7 -> E8 -> E9 -> E10 -> setup -> tuning -> E6 status.
Each experiment writes results/<name>.json (flat keys). A missing run never raises: the keys are simply absent and `<name>.status` says why.
Afterwards (host): code/tools/make_macros.py, code/tools/make_tables.py; (worker machine): code/analysis_pipeline/figures/make_all.py."""
import argparse
import os
import sys
import time
import traceback

from . import e10 as E10, e7 as E7, misc, primary, runs as R, secondary as SEC, weak
from .util import REPO, Result, detect_scope, set_default_scope

ALL = ["primary", "W1", "E1", "E2", "E3", "E4", "Psec", "E5", "E7", "E8", "E9", "E10", "setup", "tuning", "E6", "X1"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, help="runs root (worker machine: $R/runs); scanned recursively for preds.jsonl")
    ap.add_argument("--results", default=os.path.join(REPO, "results"))
    ap.add_argument("--split", default="bird_dev")
    ap.add_argument("--model", default=R.PRIMARY_MODEL)
    ap.add_argument("--expected-n", type=int, default=None)
    ap.add_argument("--B", type=int, default=10000, help="bootstrap replicates (SAP 3: 10000)")
    ap.add_argument("--only", nargs="*", default=None, choices=ALL)
    ap.add_argument("--e4-roots", nargs="*", default=[], help="directories containing E4 stream units (events.jsonl)")
    ap.add_argument("--e9-dirs", nargs="*", default=[])
    ap.add_argument("--annot-dir", default=None, help="E10 annotation directory (see e10.py)")
    ap.add_argument("--llm-labels", default=None)
    ap.add_argument("--calib-preds", default=None, help="HR preds.jsonl of the routing run: isotonic calibration source for E7")
    ap.add_argument("--route-coef", default=None)
    ap.add_argument("--route-k", type=int, default=None)
    ap.add_argument("--manifest", default=os.path.join(REPO, "exp", "analysis_manifest.json"))
    ap.add_argument("--decisions", default=os.path.join(REPO, "exp", "selection_decisions.json"))
    ap.add_argument("--replay-manifest", default=os.path.join(REPO, "exp", "replay_manifest.json"), help="run directories (relative to --runs) that take part; 'none' = every run found. Default: exp/replay_manifest.json (the run set of the released result files)")
    ap.add_argument("--models-yaml", default=os.path.join(REPO, "exp", "models.yaml"))
    ap.add_argument("--prices", default=os.path.join(REPO, "exp", "prices.yaml"))
    ap.add_argument("--plan-commit", default=None, help="short hash of the commit that froze docs/SAP.md (host: git log -1 --format=%%h -- docs/SAP.md)")
    ap.add_argument("--scope", default=None, choices=["confirmatory", "tuning-set", "synthetic"], help="default: synthetic if <runs>/SYNTHETIC exists, tuning-set if --split is not bird_dev, else confirmatory")
    ap.add_argument("--tuning-results", default=os.path.join(REPO, "results"), help="directory holding r1/ and p2/ tuning jsons")
    a = ap.parse_args(argv)
    todo = a.only or ALL
    scope = a.scope or detect_scope(a.runs, a.split)
    set_default_scope(scope)
    os.makedirs(a.results, exist_ok=True)
    try:
        reg = R.Registry(a.runs, R.load_manifest(a.manifest), a.models_yaml, a.split, a.model, a.expected_n, R.load_replay_dirs(a.replay_manifest))
    except R.RunError as e:
        raise SystemExit("[run_all] %s" % e)
    decisions = R.load_decisions(a.decisions)
    prices = SEC.load_prices(a.prices)
    coef = primary.load_coef(a.route_coef)
    full_label = decisions.get("full") or "full_gated"
    print("[S1] runs found: %d under %s; split=%s expected_n=%s B=%d scope=%s" % (len(reg.all), a.runs, a.split, reg.expected_n, a.B, scope), flush=True)
    status = {}

    def stage(name, fn):
        if name not in todo:
            return None
        t = time.time()
        try:
            out = fn()
            status[name] = "ok (%.0fs)" % (time.time() - t)
            return out
        except Exception as e:                                   # a failing stage must not hide the others, but must be loud
            traceback.print_exc()
            status[name] = "FAILED: %s: %s" % (type(e).__name__, e)
            return None

    bstar = None
    pres = {}

    def do_primary():
        res, _ = primary.run_primary(reg, a.e4_roots, decisions, a.B, coef, a.route_k, os.path.join(a.results, "primary.json"))
        pres["r"] = res
    stage("primary", do_primary)
    if "r" in pres:
        bstar = pres["r"].d.get("P2.bstar")
    e1runs = {}

    def do_e1():
        res = Result("E1")
        runs = SEC.run_e1(reg, res, a.B, prices, decisions, bstar)
        e1runs.update(runs or {})
        pres["e1"] = res
        res.write(os.path.join(a.results, "E1.json"))
        r11 = Result("E11")
        SEC.run_e11(reg, r11, a.B, prices, e1runs)
        r11.write(os.path.join(a.results, "E11.json"))
    stage("E1", do_e1)

    own_scope = {"E4": "synthetic" if scope == "synthetic" else "confirmatory",          # E4 streams are BIRD-train HR by design (no dev data)
                 "tuning": "tuning-export"}                                             # T.* keys: tuning-set numbers the paper may cite only as such

    def simple(name, fn):
        def go():
            res = Result(name, own_scope.get(name))
            fn(res)
            pres["out_" + name] = res
            res.write(os.path.join(a.results, name + ".json"))
        stage(name, go)

    simple("W1", lambda r: weak.run_w1_stage(reg, r, a.B, pres.get("r")))        # needs the primary result for P3 (family P*); `--only W1` alone writes W1 without p_adj
    simple("E2", lambda r: SEC.run_e2(reg, r, a.B, prices=prices))
    # ablations were run on full_gated (MINIMAL_PLAN M3); the matched base is full_gated whatever rule 3 selected as primary
    simple("E3", lambda r: (SEC.run_e3(reg, r, a.B, "full_gated"), SEC.run_e3_backbones(reg, r, a.B)))
    simple("E4", lambda r: misc.run_e4(a.e4_roots, r, a.B))
    simple("Psec", lambda r: weak.run_secondary_rows(r, pres["r"], pres.get("e1"), pres.get("out_E4")) if "r" in pres else r.note("Psec: primary stage did not run"))   # P2 -> S-A, P4 -> S-E (15.2)
    simple("E5", lambda r: E7.run_e5(reg, r, a.B, full_label))
    simple("E7", lambda r: E7.run_e7(reg, r, a.B, decisions, coef, a.route_k, a.calib_preds, full_label))
    simple("E8", lambda r: misc.run_e8(reg, r, a.B, full_label=full_label))
    simple("E9", lambda r: misc.run_e9(a.e9_dirs, r))
    labelled = {}
    for lab in ("zs", "full_fixed", full_label):
        try:
            x = reg.get(lab)
        except R.RunError:
            x = None
        if x is not None:
            labelled[lab] = x
    simple("E10", lambda r: E10.run_e10(reg, r, a.annot_dir, a.llm_labels, labelled))
    simple("setup", lambda r: misc.run_setup(reg, r, plan_commit=a.plan_commit, B=a.B, e4_roots=a.e4_roots))
    simple("X1", lambda r: weak.run_x1(reg, r, a.B, prices))
    simple("tuning", lambda r: misc.run_tuning(r, a.tuning_results))
    simple("E6", misc.run_e6)
    print("\n[S1] stage status:")
    for k in ALL:
        if k in status:
            print("  %-8s %s" % (k, status[k]))
    for n in reg.notes:
        print("[S1] note:", n)
    return 1 if any(v.startswith("FAILED") for v in status.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
