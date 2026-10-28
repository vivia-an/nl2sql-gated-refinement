"""RV3 key grammar: round-3 post hoc analyses (internal note R3EXP, SAP section 26). Pure Python, imported by make_xmacros.py (host).
The summary file results/r3/rv3_summary.json (written by code/tools/r3_merge.py) must hold exactly the base keys of rv3_grammar.json (macro prefix `RV3.`), each with its kind:

  seed.arm.<b>.<arm>.*       EX of every arm and run (s0 = T 0 headline, s101 / s202 = T 0.7), mean, SD with its interval
  seed.<cid>.<scope>.*       paired contrasts (stack, loop_g, loop_f, g_minus_f, g_minus_zs, f_minus_zs; scopes plus, flash, 27b, pooled3; w1.weak2): seed-averaged estimate with the
                             cluster and item-level interval, per-run estimates, range, t interval, widest interval, readings of the single run and of the average
  seed.p1p.*, seed.w1.*, seed.calls.<scope>.*   P1' margin and W1 claims; gated / fixed calls ratio
  seed.split.<b>.*           stack versus loop; seed.regen.<b>.<seed>.*  root-regeneration audit
  samex.*                    same-examples single-call control (arms, contrasts, share of the stack carried by the examples, discordance)
  ret.*                      retrieval activity on the development set; route.*  routing-score controls and the equal-cost vote; p5.*  P5' re-run
  q360.*                     selection and shift bound of the 360-question corrected-reference subset
  inc.<b>.<cfg>.vs_<ref>.*   incremental cost per additional correct answer; mde.*  minimum detectable effects; mult.*  multiplicity counts;
  critic.<b>.*               Critic increment on shared roots; stack.pooled.*  exact interval ends; lat.*  latency; cost.*  cost of the round-3 runs

The grammar file is generated once from a complete run (`python3 code/tools/r3_merge.py --write-grammar`) and committed; any later change of a key, a kind or the key set
therefore needs a deliberate regeneration (the strict gate of make_xmacros compares the summary against it)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = ("pp", "pp2", "pctv", "pctv2", "pct", "x1", "x2", "x3", "x3s", "p", "p4", "int", "str", "cny2", "cny4")


def expected_keys():
    with open(os.path.join(HERE, "rv3_grammar.json")) as f:
        g = json.load(f)
    out = [(k, kind) for k, kind in g]
    for k, kind in out:
        if kind not in KINDS:
            raise ValueError("rv3_grammar.json: unknown kind %r at %s" % (kind, k))
    return out
