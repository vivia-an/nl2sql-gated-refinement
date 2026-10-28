#!/bin/bash
# usage: e4_lane.sh <config> <policy> <conc> <trial:json-params> ...   runs HT mini-stream trials sequentially (resumable)
source <NL2SQL_ROOT>/snap_r0/r0env.sh
CFG=$1; POL=$2; CONC=$3; shift 3
for t in "$@"; do
  NAME=${t%%:*}; PARAMS=${t#*:}
  if [ -f $R/runs/${OUTD:-P2_e4tune}/$NAME/trial_summary.json ]; then echo "skip $NAME"; continue; fi
  $PY tools_r0/e4_ht_tune.py --trial $NAME --policy $POL --params "$PARAMS" --out $R/runs/${OUTD:-P2_e4tune} --config $CFG --conc $CONC --workers 2
done
