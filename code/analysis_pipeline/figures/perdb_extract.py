"""Per-database delta, gated minus fixed loop, for fig_perdb_delta (final mode).

results/E1.json uses the fixed loop as the E1 reference system (selection_decisions.json: full = full_fixed), so it holds E1.perdb_delta.full_fixed_vs_*
and no gated-minus-fixed block.  This script calls the pipeline's own function `secondary.run_e1` once with full = full_gated into a SCRATCH Result (nothing
is written to results/), and keeps only the keys E1.perdb_delta.full_gated_vs_full_fixed.* (per-database delta, item-level 95% interval, leave-one-database-out, n).
No statistics stage of the confirmatory analysis is rerun or changed; the estimator is the one of secondary.py (item bootstrap 2000 within the database).

    python -m analysis_pipeline.figures.perdb_extract --runs $R/runs --out figs/data/perdb_gated_vs_fixed.json   (worker machine; needs the runs tree)
"""
import argparse
import json
import os
import time

from .. import runs as R, secondary as SEC
from ..util import REPO, Result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="bird_dev")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--manifest", default=os.path.join(REPO, "exp", "analysis_manifest.json"))
    ap.add_argument("--decisions", default=os.path.join(REPO, "exp", "selection_decisions.json"))
    ap.add_argument("--replay-manifest", default=os.path.join(REPO, "exp", "replay_manifest.json"), help="run directories (relative to --runs) that take part; 'none' = every run found. Default: exp/replay_manifest.json (the run set of the released result files)")
    ap.add_argument("--models-yaml", default=os.path.join(REPO, "exp", "models.yaml"))
    ap.add_argument("--prices", default=os.path.join(REPO, "exp", "prices.yaml"))
    a = ap.parse_args(argv)
    reg = R.Registry(a.runs, R.load_manifest(a.manifest), a.models_yaml, a.split, R.PRIMARY_MODEL, None, R.load_replay_dirs(a.replay_manifest))
    decisions = dict(R.load_decisions(a.decisions))
    decisions["full"] = "full_gated"
    res = Result("E1_perdb_gated_vs_fixed")
    SEC.run_e1(reg, res, a.B, SEC.load_prices(a.prices), decisions, None)
    pre = "E1.perdb_delta.full_gated_vs_full_fixed."
    keys = {k: v for k, v in res.d.items() if k.startswith(pre)}
    if not keys:
        raise SystemExit("no %s* keys (are full_gated and full_fixed r0 runs complete?)" % pre)
    runs = {l: reg.get(l).run_id for l in ("full_gated", "full_fixed")}
    json.dump({"_meta": {"scope": "confirmatory", "name": "perdb_gated_vs_fixed", "created": time.strftime("%Y-%m-%d %H:%M"), "B": a.B, "runs": runs,
                         "note": "secondary.run_e1 with decisions.full = full_gated into a scratch Result; only the perdb_delta keys are kept; results/E1.json is untouched"}, **keys},
              open(a.out, "w"), indent=1, sort_keys=True)
    print("[perdb_extract] %d keys -> %s" % (len(keys), a.out))


if __name__ == "__main__":
    main()
