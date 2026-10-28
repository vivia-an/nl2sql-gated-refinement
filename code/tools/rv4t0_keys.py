"""RV4T0 key grammar: the temperature-0 non-determinism check of the locked run (internal note R4T0, SAP section 27). Pure Python, imported by make_xmacros.py (host).
The summary file results/r4_t0/rv4t0_summary.json (written by code/tools/r4_t0.py) must hold exactly the base keys of rv4t0_grammar.json (macro prefix `RV4T0.`), each with its kind:

  nd.<b>.<arm>.*   s0 against the rerun r1 of one arm: questions compared, EX of both, difference with cluster / item-level / widest interval, share of changed final SQL and of changed
                   correctness, b / c counts and the exact McNemar p, root and example-set differences
  nd.complete.*    the complete reruns together;  pair.f_minus_zs.<b>.*  a contrast re-estimated in the rerun;  lock.<cid>.<scope>.*  the paper's key contrasts in the locked run
  cmp.*            size of the run-to-run change against those contrasts;  req.<b>.<arm>.*  identity of the rerun requests with s0;  cost.*  cost of the R4T0 queue;  lock_file.*  the lock file

The grammar file is generated once from a complete run (`python3 code/tools/rv4t0_keys.py --write-grammar results/r4_t0/rv4t0_summary.json`) and committed."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("pp", "pp2", "pctv", "pctv2", "pct", "x1", "x2", "x3", "x3s", "p", "p4", "int", "str", "cny2", "cny4")
GRAMMAR = os.path.join(HERE, "rv4t0_grammar.json")


def expected_keys():
    with open(GRAMMAR) as f:
        g = json.load(f)
    out = [(k, kind) for k, kind in g]
    for k, kind in out:
        if kind not in KINDS:
            raise ValueError("rv4t0_grammar.json: unknown kind %r at %s" % (kind, k))
    return out


def write_grammar(summary_path, out=GRAMMAR):
    keys = json.load(open(summary_path))["keys"]
    g = [[k, e["kind"]] for k, e in keys.items()]
    with open(out, "w") as f:
        json.dump(g, f, indent=0)
    return len(g)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--write-grammar":
        print("wrote %d keys" % write_grammar(sys.argv[2]))
    else:
        print(__doc__)
