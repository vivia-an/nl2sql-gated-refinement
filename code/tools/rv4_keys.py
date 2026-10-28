"""RV4 key grammar: round-4 post hoc statistics (internal note R4A, SAP section 27). Pure Python, imported by make_xmacros.py (host).
The summary file results/r4/rv4_summary.json (written by code/tools/r4_merge.py) must hold exactly the base keys of rv4_grammar.json (macro prefix `RV4.`), each with its kind:

  wide.<rep|samex>.<cid>.<scope>.*   single widest-interval definition (hull of cluster, item-level, t(G-1), sign-flip) for the replication and the locked run, readings, agrees flag
  regen.*   root regeneration between the gated and the fixed run;  ccost.<b>.*  Critic increment and its cost per extra correct answer;  rmargin.*  margin ratio, share kept
  mde4.*    minimum detectable effects at question and database level;  p5u.*  P5' with one unit per question;  silent.*  silent-error rate, share the gate can reach
  mult.*    breakdown of the post hoc contrast count

The grammar file is generated once from a complete run (`python3 code/tools/r4_merge.py --write-grammar`) and committed."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("pp", "pp2", "pctv", "pctv2", "pct", "x1", "x2", "x3", "x3s", "p", "p4", "int", "str", "cny2", "cny4")


def expected_keys():
    with open(os.path.join(HERE, "rv4_grammar.json")) as f:
        g = json.load(f)
    out = [(k, kind) for k, kind in g]
    for k, kind in out:
        if kind not in KINDS:
            raise ValueError("rv4_grammar.json: unknown kind %r at %s" % (kind, k))
    return out
