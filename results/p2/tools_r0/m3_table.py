"""Table of the M3 grid on HT-300: EX, calls/q, tokens/q, cost, paired b/c vs default (k_t 6, k_e 4). python m3_table.py <runs_root> [out.json]"""
import sys, os, glob, json, yaml, math
R = sys.argv[1]
rows = []
runs = glob.glob(R + "/P2_m3/*/summary.json") + glob.glob(R + "/P2_base/P2_base-full_fixed-*/summary.json")
def load(d):
    return {json.loads(l)["qid"]: json.loads(l) for l in open(d + "/preds.jsonl")}
base = None
res = []
for f in runs:
    d = os.path.dirname(f)
    cfg = yaml.safe_load(open(d + "/config.yaml"))["resolved"]
    m3 = cfg.get("m3", {})
    kt, ke = m3.get("k_t", 6), m3.get("k_e", 4)
    s = json.load(open(f))
    p = load(d)
    res.append((kt, ke, s, p, os.path.basename(d)))
b = [r for r in res if r[0] == 6 and r[1] == 4]
base = b[0][3] if b else None
out = []
for kt, ke, s, p, name in sorted(res, key=lambda r: (r[0], r[1])):
    n = len(p)
    ex = sum((r["ex"] or 0) for r in p.values())
    calls = sum(r["calls"] for r in p.values()) / n
    pt = sum(r["prompt_tokens"] for r in p.values()) / n
    ct = sum(r["completion_tokens"] for r in p.values()) / n
    bb = cc = 0
    if base:
        for q, r in p.items():
            x, y = r["ex"] or 0, base[q]["ex"] or 0
            bb += int(x == 1 and y == 0); cc += int(x == 0 and y == 1)
    out.append({"k_t": kt, "k_e": ke, "n": n, "ex": round(ex / n, 4), "ex_correct": ex, "calls_per_q": round(calls, 3), "prompt_tok_q": round(pt), "compl_tok_q": round(ct),
                "wins_vs_default": bb, "losses_vs_default": cc, "run": name, "api_fail": s.get("api_fail"), "coverage": round(s.get("coverage", 0), 3)})
for o in out:
    print("kt=%d ke=%d  n=%d EX=%.4f (%d)  calls/q=%.3f  prompt/q=%d  compl/q=%d  +%d/-%d vs default  fail=%s  %s" % (o["k_t"], o["k_e"], o["n"], o["ex"], o["ex_correct"], o["calls_per_q"], o["prompt_tok_q"], o["compl_tok_q"], o["wins_vs_default"], o["losses_vs_default"], o["api_fail"], o["run"]))
if len(sys.argv) > 2:
    json.dump(out, open(sys.argv[2], "w"), indent=1)
