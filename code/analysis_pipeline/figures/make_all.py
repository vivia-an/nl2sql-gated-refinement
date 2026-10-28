"""Paper figures F1-F7 (PDF + 300-dpi PNG, MDPI sizes, Okabe-Ito palette).

    # reference mode: BIRD-train (tuning-set) run directories -> figures with the corner label 'REFERENCE (tuning set)' (worker machine; needs the runs tree)
    python -m analysis_pipeline.figures.make_all --mode reference --runs $R/runs --out figs
    # final mode: results/*.json (scope confirmatory) -> the same figures, no corner label (one command once the confirmatory results exist)
    python -m analysis_pipeline.figures.make_all --mode final --results results --out figs
    # layout previews from synthetic results (watermarked, output below a directory starting with '_')
    python -m analysis_pipeline.figures.make_all --results /tmp/synth_results --out figs/_preview --synthetic

Files (stem -> paper \\figslot name; panel contents; sources):
  fig1_architecture   F1  (a) system overview, M1-M7 with the execution gate, (b) worked example (round 3: bar panel dropped)   paper/sections/03_method.tex, docs/PATENT_METHOD.md (no data)
  fig_cost_accuracy  F2  (RETIRED, not in the default run) EX against chat-model cost, 3 backbones x {zero-shot, fixed, gated}        ref: runs X8/*, X5/1b-ht-*, P2_base (HT-300) | final: E2.<model>.<system>.*, P1.<model>.delta_ex
  fig_repair         F3  (a) calls by stage, (b) root vs final EX (round 3: the quoting panel moved to fig_quoting)    ref: X5/1b-{hr,ht}-*, X8/*, P2_HR, P2_base | final: E2.*.calls_<stage>, E1.rootstatus.*
  fig_quoting        S   supplement figure: root execution failures with the legacy renderer vs quoted keywords (tuning set only, both modes)    ref: X5/1b-{hr,ht}-*, X5/1a-beer-quote | final: committed tuning results (figdata.negative_tuning / saved repair_tuning)
  fig_risk_coverage  F4  risk-coverage, V3 vs consistency-augmented score                           ref: X5/2-hr-cons (LODO), X5/2-ht-cons + X5/2-fit coefficients | final: E7.curve.*, E7.band.*, P3.dauroc
  fig_pollution      F5  final-probe EX at 40 % pollution, eviction recall / precision               ref: runs/MIN/E4 | final: E4.<policy>.p40.*, E4.evict.ws_full.p40.*
  fig_negative       F6  controller cells vs P0, arbitration, forced M3 / routed compute, value bounds   results/{p2,x5,x7}/*.json (tuning set in BOTH modes)
  fig_errors         F8  automatic error classes of failed answers; L1-L8 LLM-label prevalence (empty until E10 labels exist)   ref: preds auto_class | final: E10.auto.*, E10.llm.*
  fig_perdb_delta   F7  per-database delta gated - fixed                                            ref: X5/1b-hr-gated minus X5/1b-hr-quote | final: E1.perdb_delta.full_gated_vs_full_fixed.*

Rules: every plotted value is computed from a run directory or read from a results file by figures/figdata.py (provenance strings in D['...']['src'], written next to the
data in figs/data/figdata_reference.json); nothing is typed in.  Final mode accepts results files of scope 'confirmatory' (and the 'tuning-export' T.* files, which no
figure reads) and exits on any other scope.  Panels whose evidence exists only on tuning data (F3c, all of F6) carry the stamp 'tuning set' in every mode.
Python 3.12 + matplotlib + numpy + scipy: worker machine venv only (host Python 3.6 cannot import matplotlib)."""
import argparse
import glob
import json
import os
import sys

from . import figdata

OK_SCOPES = ("confirmatory", "tuning-export")
STEMS = ["fig1_architecture", "fig_repair", "fig_quoting", "fig_risk_coverage", "fig_pollution", "fig_negative", "fig_perdb_delta", "fig_errors"]


RETIRED = {"fig_cost_accuracy": "retired in favour of fig_frontier (code/tools/make_xfigs.py); still drawable with --only fig_cost_accuracy"}


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_posthoc(results_dir):
    """results/posthoc2/rv2_summary.json -> its flat key table (scope posthoc-secondary; read only for the interval of fig_repair (c)); {} if absent."""
    f = os.path.join(results_dir, "posthoc2", "rv2_summary.json")
    if not os.path.exists(f):
        return {}
    return json.load(open(f)).get("keys", {})


def load(results_dir, allow_scopes):
    vals, scopes = {}, set()
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(f).startswith("_"):
            continue
        d = json.load(open(f))
        sc = (d.get("_meta") or {}).get("scope", "confirmatory")
        scopes.add(sc)
        if sc not in allow_scopes:
            raise SystemExit("%s has scope %r; final-mode figures are drawn from confirmatory results only (refusing; use --mode reference for tuning-set data or --synthetic for previews)" % (f, sc))
        for k, v in d.items():
            if not k.startswith("_"):
                vals[k] = v
    return vals, scopes


def build_figure(stem, D, mode, synthetic):
    from . import draw_arch, draw_data
    if stem == "fig1_architecture":
        return draw_arch.draw(mode, synthetic, D.get("case"), D.get("repair"), D.get("cost"))
    if stem == "fig_errors":
        return draw_data.fig_errors(D.get("errors"), mode, synthetic)
    if stem == "fig_cost_accuracy":
        return draw_data.fig_cost(D.get("cost"), mode, synthetic)
    if stem == "fig_repair":
        return draw_data.fig_repair(D.get("repair"), D.get("repair_tuning"), mode, synthetic)
    if stem == "fig_quoting":
        return draw_data.fig_quoting(D.get("repair_tuning"), mode, synthetic)
    if stem == "fig_risk_coverage":
        return draw_data.fig_rc(D.get("rc"), mode, synthetic)
    if stem == "fig_pollution":
        return draw_data.fig_pollution(D.get("pollution"), mode, synthetic)
    if stem == "fig_negative":
        return draw_data.fig_negative(D.get("negative"), mode, synthetic)
    if stem == "fig_perdb_delta":
        return draw_data.fig_perdb(D.get("perdb"), mode, synthetic)
    raise KeyError(stem)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["reference", "final"], default="final")
    ap.add_argument("--results", default=None, help="final mode: directory with results/*.json")
    ap.add_argument("--runs", default=None, help="reference mode: runs tree ($R/runs on the worker machine)")
    ap.add_argument("--repo", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=None, help="where figdata_reference.json is written / read (default <out>/data)")
    ap.add_argument("--from-data", default=None, help="reference mode: redraw from a saved figdata_reference.json instead of the runs")
    ap.add_argument("--synthetic", action="store_true", help="results come from synthetic runs: watermark, --out must be below a directory starting with '_'")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    data_dir = a.data_dir or os.path.join(a.out, "data")
    mode = a.mode
    if a.synthetic:
        if not any(part.startswith("_") for part in os.path.abspath(a.out).split(os.sep)):
            raise SystemExit("--synthetic output must go to a directory whose name starts with '_' (e.g. figs/_preview)")
        mode = "final"
    saved = os.path.join(data_dir, "figdata_reference.json")
    tuning = {}
    if os.path.exists(saved):
        sv = json.load(open(saved))
        tuning = {k: sv.get(k) for k in ("repair_tuning", "negative", "case")}
    if mode == "reference":
        if a.from_data:
            D = json.load(open(a.from_data))
        else:
            if not a.runs:
                raise SystemExit("reference mode needs --runs (the BIRD-train run directories) or --from-data")
            D = figdata.reference(a.runs, a.repo)
            os.makedirs(data_dir, exist_ok=True)
            json.dump(D, open(saved, "w"), indent=1, sort_keys=True)
            print("[figures] wrote %s" % saved)
    else:
        if not a.results:
            raise SystemExit("final mode needs --results")
        vals, scopes = load(a.results, ("synthetic", "tuning-export", "confirmatory") if a.synthetic else OK_SCOPES)
        if a.synthetic and "synthetic" not in scopes:
            raise SystemExit("--synthetic given but %s holds no synthetic result file (scopes: %s)" % (a.results, ", ".join(sorted(scopes))))
        if "negative" not in tuning or not tuning["negative"]:
            tuning["negative"] = figdata.negative_tuning(a.repo)                 # committed tuning results; needs no runs tree
        pdb = os.path.join(data_dir, "perdb_gated_vs_fixed.json")                   # written by figures/perdb_extract.py (E1.json has no gated-minus-fixed per-database block)
        if not any(k.startswith("E1.perdb_delta.full_gated_vs_full_fixed.") for k in vals) and os.path.exists(pdb):
            pv = json.load(open(pdb))
            vals.update({k: v for k, v in pv.items() if not k.startswith("_")})
            print("[figures] per-database block read from %s" % pdb)
        D = figdata.final(vals, tuning, load_posthoc(a.results))
        D["_meta"]["sources"] = {os.path.basename(f): _sha(f) for f in sorted(glob.glob(os.path.join(a.results, "*.json"))) + [pdb, os.path.join(a.results, "posthoc2", "rv2_summary.json")] if os.path.exists(f)}
        D["_meta"]["results_dir"] = "results"
        D["_meta"]["note"] = "final mode: BIRD development set; figures drawn by analysis_pipeline.figures.make_all from results/*.json (confirmatory) + posthoc2 interval for fig_repair (c)"
        json.dump(D, open(os.path.join(data_dir, "figdata_final.json"), "w"), indent=1, sort_keys=True)
        print("[figures] wrote %s" % os.path.join(data_dir, "figdata_final.json"))
    from . import style
    for stem in (list(STEMS) + [x for x in (a.only or []) if x in RETIRED and x not in STEMS]):
        if a.only and stem not in a.only:
            continue
        fig = build_figure(stem, D, mode, a.synthetic)
        if isinstance(fig, str):
            print("[figures] SKIP %s: %s" % (stem, fig))
            continue
        paths = style.save(fig, a.out, stem)
        import matplotlib.pyplot as plt
        plt.close(fig)
        print("[figures] %s -> %s" % (stem, ", ".join(paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
