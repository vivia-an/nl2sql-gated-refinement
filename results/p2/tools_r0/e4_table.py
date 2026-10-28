"""E4 HT mini-stream tuning table: v1 (first pass, shared cache) and v2 (clean private cache) replicates; decision metric = EX of the first pass v1 (the only complete set: 11 trials; the clean-cache replicate v2 was cut short by the
DashScope Arrearage error and covers 5 trials; used for the noise estimate); ties -> lower calls/q. python e4_table.py <runs> <out.json>"""
import sys, os, json, glob, collections
R, out = sys.argv[1:3]
T = collections.defaultdict(dict)
for tag, d in (("v1", "P2_e4tune"), ("v2", "P2_e4tune_v2")):
    for f in sorted(glob.glob("%s/%s/*/trial_summary.json" % (R, d))):
        s = json.load(open(f))
        T[s["trial"]][tag] = s
rows = []
for t, v in sorted(T.items()):
    s0 = next(iter(v.values()))
    r = {"trial": t, "system": s0["system"], "params": s0["params"]}
    for tag in ("v1", "v2"):
        if tag in v:
            m = v[tag]["metric"]
            ev = sum(x["diag"]["evicted_total"] for x in v[tag]["per_db"].values())
            evp = sum(x["diag"]["evicted_polluted"] for x in v[tag]["per_db"].values())
            r[tag] = {"ex": m["ex"], "probe_final": m["ex_probe_final"], "stream": m["ex_stream"], "calls_per_q": m["calls_per_q"], "evicted": ev, "evicted_polluted": evp}
    exs = [r[t_]["ex"] for t_ in ("v1", "v2") if t_ in r]
    cs = [r[t_]["calls_per_q"] for t_ in ("v1", "v2") if t_ in r]
    r["ex_mean"] = round(sum(exs) / len(exs), 4)
    r["calls_mean"] = round(sum(cs) / len(cs), 3)
    r["n_rep"] = len(exs)
    rows.append(r)
choice = {}
for sysname in ("e4_ws_full", "e4_memrl"):
    rs = [r for r in rows if r["system"] == sysname and "v1" in r]
    if rs:
        best = sorted(rs, key=lambda r: (-r["v1"]["ex"], r["v1"]["calls_per_q"]))[0]
        choice[sysname] = best["trial"]
        # best among rho>0 for ws_full (reporting only)
        if sysname == "e4_ws_full":
            pos = [r for r in rs if r["params"].get("rho", 0) > 0]
            if pos:
                choice["e4_ws_full_rho_positive"] = sorted(pos, key=lambda r: (-r["v1"]["ex"], r["v1"]["calls_per_q"]))[0]["trial"]
json.dump({"rows": rows, "choice": choice}, open(out, "w"), indent=1)
for r in rows:
    print("%-22s %-10s %-26s v1=%s v2=%s mean=%.4f calls=%.3f n=%d" % (r["trial"], r["system"], json.dumps(r["params"]),
          r.get("v1", {}).get("ex"), r.get("v2", {}).get("ex"), r["ex_mean"], r["calls_mean"], r["n_rep"]))
print(choice)
