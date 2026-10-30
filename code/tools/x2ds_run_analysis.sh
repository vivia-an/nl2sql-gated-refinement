#!/bin/sh
# Post hoc analyses of SAP Sections 28 to 30 from the stored predictions (no model call, no API key). Needs python3 with numpy and scipy (versions of code/requirements.txt).
#   sh code/tools/x2ds_run_analysis.sh [WORK_DIR] [OUT]
# WORK_DIR = the directory that holds runs/ of the unpacked run archives: the 1.0.0 archive (MIN/..., R4T0/...: the locked s0 runs and the temperature-0 reruns of the
#            Qwen backbones) and the 1.1.0 add-on (X2DS, X2DSR, X2DSRR, W7, W7RR, W8, W9). Default: $NL2SQL_ROOT.
# OUT      = output directory (default: ./x2ds_results). Then compare with results/x2ds/ of the repository (Level 2 of the README):
#            OUT/<bk>/x2ds_summary.json, OUT/pooled/x2ds_pooled.json, OUT/x2ds_rerun.json, OUT/w9_iv.json.
# Not covered here, because they need data that the archives do not hold: results/x2ds/x2ds_rescore.json (corrected reference queries, see DATA_LICENCES.md),
# results/x2ds/empty_gold.json and results/x2ds/w9_harm.json (they execute the reference queries on the BIRD databases; code/tools/x2ds_empty_gold.py, x2ds_w9.py).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
W=${1:-${NL2SQL_ROOT:-$HOME/nl2sql_work}}
RUNS=$W/runs
OUT=${2:-./x2ds_results}
PY=${PYTHON:-python3}
M=$RUNS/MIN
need() { [ -f "$1/preds.jsonl" ] || { echo "missing run directory: $1 (unpack the run archives, see README Level 2)" >&2; exit 2; }; }
for d in M1/zs-s0 M1/fixed-s0 M1/gated-s0 M2/macsql-s0 M3/rootonly-s0; do need "$M/$d"; done
for b in flash 27b; do for a in zs fixed gated rootonly; do need "$M/M5/$a-$b-s0"; done; done
for a in zs fixed gated rootonly; do need "$RUNS/X2DS/$a-dsv41flash-s0"; done
for b in plus flash 27b dsv41flash; do need "$RUNS/X2DSR/zsretry-$b-s0"; need "$RUNS/W7/zsretryempty-$b-s0"; need "$RUNS/W8/gatedempty-$b-s0"; done
mkdir -p "$OUT"
for bk in plus flash 27b dsv41flash; do
  case $bk in
    plus) A="--arm zs=$M/M1/zs-s0/preds.jsonl --arm fixed=$M/M1/fixed-s0/preds.jsonl --arm gated=$M/M1/gated-s0/preds.jsonl --arm rootonly=$M/M3/rootonly-s0/preds.jsonl"; P="--price-in 2.0 --price-out 8.0";;
    flash) A="--arm zs=$M/M5/zs-flash-s0/preds.jsonl --arm fixed=$M/M5/fixed-flash-s0/preds.jsonl --arm gated=$M/M5/gated-flash-s0/preds.jsonl --arm rootonly=$M/M5/rootonly-flash-s0/preds.jsonl"; P="--price-in 0.2 --price-out 0.8";;
    27b) A="--arm zs=$M/M5/zs-27b-s0/preds.jsonl --arm fixed=$M/M5/fixed-27b-s0/preds.jsonl --arm gated=$M/M5/gated-27b-s0/preds.jsonl --arm rootonly=$M/M5/rootonly-27b-s0/preds.jsonl"; P="--price-in 0.6 --price-out 4.8";;
    dsv41flash) A=""; for a in zs fixed gated rootonly; do A="$A --arm $a=$RUNS/X2DS/$a-dsv41flash-s0/preds.jsonl"; done; P="--price-in 2.0 --price-out 8.0";;
  esac
  # W7 arms (SAP Section 29): thinking-on single call and MAC-SQL (Qwen; the Plus MAC-SQL arm is the locked run of the 1.0.0 archive), empty-gate retry (all four)
  case $bk in
    plus) A="$A --arm zsthink=$RUNS/W7/zsthink-plus-s0/preds.jsonl --arm macsql=$M/M2/macsql-s0/preds.jsonl";;
    flash|27b) A="$A --arm zsthink=$RUNS/W7/zsthink-$bk-s0/preds.jsonl --arm macsql=$RUNS/W7/macsql-$bk-s0/preds.jsonl";;
  esac
  A="$A --arm zsretryempty=$RUNS/W7/zsretryempty-$bk-s0/preds.jsonl --arm gatedempty=$RUNS/W8/gatedempty-$bk-s0/preds.jsonl"
  mkdir -p "$OUT/$bk"
  $PY "$HERE/x2ds_analyze.py" $A --arm zsretry=$RUNS/X2DSR/zsretry-$bk-s0/preds.jsonl --out "$OUT/$bk" --B 10000 $P --cache-ratio 0.2 > "$OUT/$bk/stdout.json"
done
POOLARGS="--arm plus:zs=$M/M1/zs-s0/preds.jsonl --arm plus:fixed=$M/M1/fixed-s0/preds.jsonl --arm plus:gated=$M/M1/gated-s0/preds.jsonl --arm plus:rootonly=$M/M3/rootonly-s0/preds.jsonl --arm plus:zsretry=$RUNS/X2DSR/zsretry-plus-s0/preds.jsonl"
for bk in flash 27b; do
  for a in zs fixed gated rootonly; do POOLARGS="$POOLARGS --arm $bk:$a=$M/M5/$a-$bk-s0/preds.jsonl"; done
  POOLARGS="$POOLARGS --arm $bk:zsretry=$RUNS/X2DSR/zsretry-$bk-s0/preds.jsonl"
done
mkdir -p "$OUT/pooled"
$PY "$HERE/x2ds_pool.py" $POOLARGS --out "$OUT/pooled" --B 10000 > "$OUT/pooled/stdout.json"
# reading lines of every step from one rerun of each arm (X2DSRR, W7RR, the W8 reruns, and the R4T0 reruns of the 1.0.0 archive)
$PY "$HERE/x2ds_rerun.py" --w "$W" --out "$OUT/x2ds_rerun.json" --B 10000 > "$OUT/x2ds_rerun.stdout"
# W9 (SAP Section 30, tuning split HR): intervals and the decomposition by trigger set; reads the stored harm counts of results/x2ds/w9_harm.json
$PY "$HERE/x2ds_w9_iv.py" --runs "$RUNS/W9" --qids "$REPO/exp/posthoc/w9_qids.txt" --harm "$REPO/results/x2ds/w9_harm.json" --out "$OUT/w9_iv.json" --B 10000 > "$OUT/w9_iv.stdout"
