#!/usr/bin/env python3
"""Cost estimate of the MINIMAL plan (docs/MINIMAL_PLAN.md run list M1-M6) from MEASURED unit costs. Host side, stdlib only, Python >= 3.6.

  python3 exp/budget.py                       table of every job of exp/queues/MIN.jsonl + the pre-lock items + subtotals + total + stop rule
  python3 exp/budget.py --dev-factor 1.5      dev prompts 1.5x longer than HT/HR prompts (dev schemas were never measured: see NOTE)
  python3 exp/budget.py --refresh             re-read the unit costs from the measured files (X5 summaries, results/p2) if present
  python3 exp/budget.py --json out.json       machine-readable table
  python3 exp/budget.py actual --runs results/raw-ignored/MIN [--queue exp/queues/MIN.jsonl]
                                              realised list-price cost per job (usage*.jsonl, own-cache hits free as in cost_report.py) vs estimate
  python3 exp/budget.py legacy --calls-per-q 3.3 --tokens-per-call 2200 --throughput 3.1      the old SPEC section 8 table (kept for reference)

Unit cost per question = ((prompt - cached) * in + cached * in * cache_hit_ratio + completion * out + embed_tokens * embed_in) / 1e6
with the list prices of exp/prices.yaml (CNY per million tokens). PROFILES below hold per-question token counts; `src` says where each came from
(measured = read from a summary/usage file of a real run; derived = arithmetic on measured stage rows; smoke = 10-question smoke of B1; assumed).

NOTE (what is NOT measured): BIRD dev was never loaded, so dev prompt lengths are unknown; the HR/HT numbers are used as they are (--dev-factor 1.0) and the
sensitivity at 1.5x is printed. Baseline profiles come from 10-question smokes (HT) with the legacy DDL and zero implicit-cache hits (an upper bound for
the cached part). Weaker backbones (M5) get a x1.3 token margin (more revisions / longer answers are expected; unmeasured).

STOP RULE (CONTRACT section 7 soft cap stays at CNY 8000 in total; the minimal plan has its own flag):
  * the minimal-plan estimate (queue jobs + pre-lock items, optional jobs excluded) > CNY 800  -> exit code 2 and a FLAG line: do not start, ask the user;
  * spent + estimate > CNY 8000 -> exit code 3;
  * at run time (`actual`): a stage whose realised cost exceeds 1.5x its estimate, or a realised total above CNY 800 -> exit code 2 (stop the queue:
    `queue_run.py cancel` for running jobs, leave the rest pending) and report to the author."""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PRICES = os.path.join(HERE, "prices.yaml")
QUEUE = os.path.join(HERE, "queues", "MIN.jsonl")
SPENT_DEFAULT = 290.0          # CNY already spent (MINIMAL_PLAN: "about CNY 290")
FLAG_MIN_PLAN = 800.0          # flag threshold of the minimal plan (user decision 2026-10-26)
SOFT_CAP = 8000.0              # CONTRACT section 7
OVERRUN = 1.5                  # realised/estimated stage ratio that stops the queue

MODEL_ID = {"qwen37plus": "qwen3.7-plus-2026-05-26", "qwen37flash": "qwen3.7-flash-2026-07-15", "qwen3527b": "qwen3.5-27b",
            "qwen38max": "qwen3.8-max-0902", "dsv41flash": "deepseek-v4.1-flash-1017"}
EMBED_MODEL = "qwen3.7-text-embedding"

# per-question profiles on qwen37plus: calls, prompt tokens, completion tokens, implicit-cache-hit prompt tokens, embedding tokens
PROFILES = {
    "zs": dict(calls=1.0, prompt=1708.3, completion=54.08, cached=1095.68, embed=0.0,
               src="measured: results/p2/base_ht300.json zs_ours (HT-300)"),
    "fixed_q": dict(calls=3.2888, prompt=7190.86, completion=397.91, cached=4958.75, embed=121.0,
                    src="measured: X5 x5-1b-hr-quote (HR-1101, full_fixed + quoting)"),
    "gated": dict(calls=2.0209, prompt=3448.32, completion=293.28, cached=2145.54, embed=631.0,
                  src="measured: X5 x5-1b-hr-gated (HR-1101, full_gated)"),
    # gated + ONE V3 critic call on the final answer (stage crit of fixed_q divided by its 1.16 calls/q) + ONE n=2 consistency request
    # (prompt = the generation prompt, completion = 2 x the generation completion)
    "gated_route_ca": dict(calls=4.0209, prompt=9446.8, completion=489.1, cached=6428.2, embed=631.0,
                           src="derived: gated + crit stage/1.16 + 1 x gen prompt, completion 2 x gen (X5 HR stage rows)"),
    "sdebug": dict(calls=2.2, prompt=3892.0, completion=530.0, cached=0.0, embed=0.0, src="smoke: B1 self_debug_simple (HT-10), max_rounds 3"),
    "acsql": dict(calls=3.0, prompt=5316.0, completion=83.0, cached=0.0, embed=0.0,
                  src="smoke: B1 ac_sql max_regen 3 (HT-10); max_regen 1 is cheaper but UNMEASURED -> upper bound"),
    "macsql": dict(calls=2.0, prompt=5013.0, completion=873.0, cached=0.0, embed=0.0, src="smoke: B1 mac (HT-10), max_refine 3"),
    "bon_n2": dict(calls=2.0, prompt=3272.0, completion=111.0, cached=0.0, embed=0.0, src="smoke x 2/5: B1 bon_exec_vote n=5 per sample 1636/55.6"),
    "bon_n3": dict(calls=3.0, prompt=4908.0, completion=167.0, cached=0.0, embed=0.0, src="smoke x 3/5: B1 bon_exec_vote"),
    "memo_light": dict(calls=18.1, prompt=35224.0, completion=1736.0, cached=0.0, embed=300.0, src="smoke: B1 memo_light (HT-10)"),
    "e9_probe": dict(calls=2.0, prompt=270.0, completion=260.0, cached=0.0, embed=0.0,
                     src="assumed: probe-Q ~120/60 tokens + probe-SQL ~150/200 tokens (max_tokens 128 / 512); not measured"),
    "tune_trial": dict(calls=0.0, prompt=0.0, completion=0.0, cached=0.0, embed=0.0, src="n/a"),
}
WEAK_MARGIN = 1.3

# items that are not queue jobs: (id, plan, label, profile, model key, n_q, note)
PRE_ITEMS = [
    # X5 remainder (exp/queues/X5_live.jsonl; 1a, 1b-hr-* and cmp already finished): required for the freeze = 1b-ht (rule 3) and 2-hr-cons + 2-ht-cons (rule 4)
    ("pre-x5-1b-ht", "PRE", "X5 1b HT-300 quote+gated (required, rule 3; running)", ["fixed_q", "gated"], "qwen37plus", 300, "required"),
    ("pre-x5-2-cons", "PRE", "X5 2 HR+HT gated+V3+consistency k=4 (required, rule 4)", ["gated_route_ca"], "qwen37plus", 1401, "required; k=4 completes slightly dearer than k=2"),
    ("pre-x5-3-4", "PRE", "X5 3 arbitration probe + 4 M3 sensitivity on HT (NOT needed for MIN: rules 5 and 6 keep their defaults)", ["gated"], "qwen37plus", 3000, "skippable"),
    ("pre-burnin", "PRE", "M6 burn-in store, n = 1500 questions of POOL56 through P0 (stream burnin)", ["fixed_q"], "qwen37plus", 1500, "required"),
]
SKIPPABLE = {"pre-x5-3-4"}


def load_prices(path=PRICES):
    out = {"models": {}, "cache_hit_ratio": 0.2, "batch_ratio": 0.5}
    for line in open(path):
        s = line.split("#")[0].rstrip()
        if not s.strip():
            continue
        if s.startswith("cache_hit_ratio"):
            out["cache_hit_ratio"] = float(s.split(":")[1])
        elif s.startswith("batch_ratio"):
            out["batch_ratio"] = float(s.split(":")[1])
        elif "{in:" in s:
            name, rest = s.strip().split("{", 1)
            d = {}
            for kv in rest.rstrip("}").split(","):
                k, v = kv.split(":")
                d[k.strip()] = float(v)
            out["models"][name.strip().rstrip(":").strip()] = d
    return out


def unit_cost(profile, model_key, P, weak=False, dev_factor=1.0):
    """CNY per question for one profile on one model (list price, implicit cache at cache_hit_ratio, embedding included)."""
    pr = P["models"][MODEL_ID[model_key]]
    em = P["models"][EMBED_MODEL]
    f = dev_factor * (WEAK_MARGIN if weak else 1.0)
    pt, ct, ca = profile["prompt"] * f, profile["completion"] * f, profile["cached"] * f
    y = ((pt - ca) * pr["in"] + ca * pr["in"] * P["cache_hit_ratio"] + ct * pr["out"] + profile["embed"] * em["in"]) / 1e6
    return y


def read_queue(path):
    jobs = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            jobs.append(json.loads(line))
    return jobs


def e4_n_q():
    """pipeline runs of one E4 job (all 8 HR dbs): stream + 3 probe checkpoints (0 %, 40 %, 100 %)."""
    sp = json.load(open(os.path.join(HERE, "splits", "train_holdout.json")))
    r = sp["hr_roles"]
    return sum(len(v["stream"]) for v in r.values()) + 3 * sum(len(v["probe"]) for v in r.values())


def refresh_profiles():
    """Overwrite PROFILES from measured files when they exist (host copies under results/)."""
    changed = []
    base = os.path.join(REPO, "results", "raw-ignored", "X5")
    for prof, d in (("fixed_q", "x5-1b-hr-quote"), ("gated", "x5-1b-hr-gated")):
        f = os.path.join(base, d, "summary.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        emb = 0.0
        u = os.path.join(base, d, "usage.jsonl")
        if os.path.exists(u):
            tot = 0
            for line in open(u):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("kind") == "embed" and not r.get("cached"):
                    tot += r.get("pt", 0) or 0
            emb = tot / float(s["n"])
        PROFILES[prof].update(calls=s["calls_per_q"], prompt=s["prompt_tokens_per_q"], completion=s["completion_tokens_per_q"],
                              cached=s.get("cached_tokens_per_q", 0.0), embed=emb, src="measured (refreshed): %s" % os.path.relpath(f, REPO))
        changed.append(prof)
    fz = os.path.join(REPO, "results", "p2", "base_ht300.json")
    if os.path.exists(fz):
        for k, v in json.load(open(fz)).items():
            if k.startswith("P2_base-zs_ours-"):
                PROFILES["zs"].update(calls=v["calls_per_q"], prompt=v["prompt_tokens_per_q"], completion=v["completion_tokens_per_q"],
                                      cached=v.get("cached_tokens_per_q", 0.0), src="measured (refreshed): results/p2/base_ht300.json")
                changed.append("zs")
    # derived profile follows the two measured ones
    g = PROFILES["gated"]
    PROFILES["gated_route_ca"].update(calls=g["calls"] + 2.0, prompt=g["prompt"] + 2886.2 + 3112.3, completion=g["completion"] + 80.3 + 115.5,
                                      cached=g["cached"] + 2149.8 + 2132.9, embed=g["embed"])
    return changed


def build_rows(queue, P, dev_factor):
    rows = []
    for j in queue:
        b = j.get("budget")
        if not b:
            continue
        mk = j.get("model") or "qwen37plus"
        weak = mk in ("qwen37flash", "qwen3527b") and b["profile"] in ("gated", "gated_route_ca")
        prof = PROFILES[b["profile"]]
        df = dev_factor if b.get("dev", False) else 1.0
        uc = unit_cost(prof, mk, P, weak=weak, dev_factor=df)
        rows.append({"id": j["id"], "plan": b["plan"], "label": b.get("label", j["id"]), "model": mk, "profile": b["profile"], "n_q": b["n_q"],
                     "calls": b["n_q"] * prof["calls"],
                     "unit": uc, "cny": uc * b["n_q"], "optional": bool(b.get("optional")), "note": b.get("note", "")})
    return rows


def build_pre(P, dev_factor):
    rows = []
    for (iid, plan, label, profs, mk, n, note) in PRE_ITEMS:
        uc = sum(unit_cost(PROFILES[p], mk, P) for p in profs)      # several profiles = several systems on the same n questions
        calls = sum(PROFILES[p]["calls"] for p in profs)
        rows.append({"id": iid, "plan": plan, "label": label, "model": mk, "profile": "+".join(profs), "n_q": n, "calls": n * calls, "unit": uc,
                     "cny": uc * n, "optional": iid in SKIPPABLE, "note": note})
    return rows


def tune_rows(tune_queue, P):
    rows = []
    for j in tune_queue:
        b = j.get("budget")
        if not b:
            continue
        prof = PROFILES[b["profile"]]
        uc = unit_cost(prof, "qwen37plus", P)
        rows.append({"id": j["id"], "plan": "TUNE", "label": b.get("label", j["id"]), "model": "qwen37plus", "profile": b["profile"], "n_q": b["n_q"],
                     "calls": b["n_q"] * prof["calls"], "unit": uc, "cny": uc * b["n_q"], "optional": False, "note": "HT-300 second pass"})
    return rows


def print_table(rows, title):
    print("\n== %s" % title)
    print("%-34s %-5s %-11s %-14s %6s %9s %9s %9s  %s" % ("job", "plan", "model", "profile", "n_q", "calls", "CNY/q", "CNY", "note"))
    for r in rows:
        print("%-34s %-5s %-11s %-14s %6d %9.0f %9.4f %9.2f  %s%s" % (r["id"][:34], r["plan"], r["model"], r["profile"][:14], r["n_q"], r["calls"],
                                                                      r["unit"], r["cny"], "[optional] " if r["optional"] else "", r["note"][:60]))


def summary(rows_all, spent, flag=FLAG_MIN_PLAN):
    core = [r for r in rows_all if not r["optional"]]
    by_plan = {}
    for r in core:
        by_plan[r["plan"]] = by_plan.get(r["plan"], 0.0) + r["cny"]
    return core, by_plan, sum(r["cny"] for r in core)


def cmd_estimate(a):
    P = load_prices()
    if a.refresh:
        print("refreshed profiles from files:", refresh_profiles() or "none found")
    queue = read_queue(a.queue)
    tune = read_queue(a.tune_queue) if os.path.exists(a.tune_queue) else []
    # E4 n_q is read from the split file (overrides the generator value if the split changed)
    for j in queue:
        if j.get("budget", {}).get("plan") == "M4" and "natfail" not in j["id"]:
            j["budget"]["n_q"] = e4_n_q()
    rows = build_pre(P, a.dev_factor) + tune_rows(tune, P) + build_rows(queue, P, a.dev_factor)
    order = {"PRE": 0, "TUNE": 1, "M1": 2, "M2": 3, "M3": 4, "M4": 5, "M5": 6, "M6": 7}
    rows.sort(key=lambda r: (order.get(r["plan"], 9), r["id"]))
    print_table(rows, "estimate (list price, qwen37plus unless stated; dev-factor %.2f applies to dev jobs only)" % a.dev_factor)
    core, by_plan, total = summary(rows, a.spent)
    print("\nsubtotals (non-optional): " + ", ".join("%s %.1f" % (k, by_plan[k]) for k in sorted(by_plan, key=lambda x: order.get(x, 9))))
    opt = sum(r["cny"] for r in rows if r["optional"])
    calls = sum(r["calls"] for r in core)
    print("TOTAL minimal plan (non-optional): CNY %.1f   | optional extras not in the total: CNY %.1f | calls %.0f" % (total, opt, calls))
    # sensitivity: dev prompts longer than HR/HT
    for f in (1.5,):
        if abs(f - a.dev_factor) > 1e-9:
            rr = build_pre(P, a.dev_factor) + tune_rows(tune, P) + build_rows(queue, P, f)
            _, _, t2 = summary(rr, a.spent)
            print("sensitivity dev-factor %.1f: TOTAL CNY %.1f" % (f, t2))
    print("with %d%% contingency for failed attempts / reruns (max_attempts 2, sys_error retries): TOTAL CNY %.1f" % (int(round(a.contingency * 100 - 100)), total * a.contingency))
    hours = calls / 3.1 / 3600.0
    print("wall clock if the API sustained 3.1 calls/s for qwen37plus (measured at conc 24, exp/models.yaml) and nothing else ran: %.1f h (jobs overlap on several worker machines; the model cap of 12 in-flight calls binds)" % hours)
    print("\nSTOP RULE")
    print("  spent so far (arg --spent): CNY %.0f; soft cap CONTRACT section 7: CNY %.0f" % (a.spent, SOFT_CAP))
    rc = 0
    if total > FLAG_MIN_PLAN:
        print("  FLAG: minimal-plan estimate %.1f > %.0f CNY -> do not start the queue; report to the author / user." % (total, FLAG_MIN_PLAN))
        rc = 2
    else:
        print("  ok: minimal-plan estimate %.1f <= %.0f CNY flag line (%.0f%% of it)." % (total, FLAG_MIN_PLAN, 100.0 * total / FLAG_MIN_PLAN))
    if a.spent + total > SOFT_CAP:
        print("  FLAG: spent + estimate = %.1f > soft cap %.0f." % (a.spent + total, SOFT_CAP))
        rc = 3
    print("  target of MINIMAL_PLAN: CNY 300-500 on top of the 290 spent -> estimate is %s that band." %
          ("inside" if 300 <= total <= 500 else ("below" if total < 300 else "ABOVE")))
    print("  run-time rule: after each plan stage run `exp/budget.py actual --runs results/raw-ignored/MIN`; a stage above %.1fx its estimate, or a realised total above %.0f CNY, stops the queue." % (OVERRUN, FLAG_MIN_PLAN))
    if a.json:
        json.dump({"rows": rows, "total_nonoptional": total, "by_plan": by_plan, "profiles": PROFILES, "dev_factor": a.dev_factor}, open(a.json, "w"), indent=1)
    return rc


def realised(runs_dir, P):
    """-> {job_id: (calls, cny)} from <runs_dir>/<job_id>/**/usage*.jsonl."""
    out = {}
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        if not os.path.isdir(d) or os.path.basename(d) in ("queue",):
            continue
        n, y = 0, 0.0
        for f in glob.glob(os.path.join(d, "**", "usage*.jsonl"), recursive=True):
            for line in open(f):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("cached"):
                    continue
                pr = P["models"].get(r.get("model", "?"))
                n += 1
                if pr is None:
                    continue
                pt, ct = r.get("pt", 0) or 0, r.get("ct", 0) or 0
                ca = min(r.get("cached_tokens", 0) or 0, pt)
                v = ((pt - ca) * pr["in"] + ca * pr["in"] * P["cache_hit_ratio"] + ct * pr["out"]) / 1e6
                y += v * (P["batch_ratio"] if r.get("mode") == "batch" else 1.0)
        out[os.path.basename(d)] = (n, y)
    return out


def cmd_actual(a):
    P = load_prices()
    real = realised(a.runs, P)
    queue = read_queue(a.queue)
    plan_of = {j["id"]: (j.get("budget") or {}).get("plan") for j in queue}
    est = {r["id"]: r["cny"] for r in build_rows(queue, P, 1.0)}
    by_plan_real, by_plan_est = {}, {}
    print("%-34s %-5s %8s %10s %10s" % ("job", "plan", "calls", "CNY real", "CNY est"))
    for jid, (n, y) in sorted(real.items()):
        pl = plan_of.get(jid) or "?"
        print("%-34s %-5s %8d %10.2f %10.2f" % (jid[:34], pl, n, y, est.get(jid, 0.0)))
        by_plan_real[pl] = by_plan_real.get(pl, 0.0) + y
        by_plan_est[pl] = by_plan_est.get(pl, 0.0) + (est.get(jid, 0.0))
    tot = sum(by_plan_real.values())
    rc = 0
    print("")
    for pl in sorted(by_plan_real):
        e = by_plan_est.get(pl, 0.0)
        flag = e > 0 and by_plan_real[pl] > OVERRUN * e
        print("stage %-4s real %8.2f   estimate of the jobs that have outputs %8.2f%s" % (pl, by_plan_real[pl], e, "   <-- over %.1fx" % OVERRUN if flag else ""))
        rc = 2 if flag else rc
    print("TOTAL realised under %s: CNY %.2f" % (a.runs, tot))
    if tot > FLAG_MIN_PLAN:
        print("FLAG: realised total above %.0f CNY -> stop the queue and report." % FLAG_MIN_PLAN)
        rc = 2
    return rc


def cmd_legacy(a):
    ROWS = [("tuning HT-300 (13 systems x <=10 trials)", 300 * 13 * 10, 1.0), ("smoke-50 (14 pipelines)", 50 * 14, 1.0),
            ("baseline matching on HR (8 x ~1100)", 8 * 1100, 1.0), ("E4 M6 parameters (HT mini-stream)", 4 * 40 * 8 * 5, 1.0),
            ("E1 (13 systems x 3 runs + 4 native)", 1534 * (13 * 3 + 4), 1.0), ("E2 (7 models x 8 calls)", 1534 * 7, 1.6),
            ("E4 main (20 configs x 5 perms x 8 dbs x 56 q)", 20 * 5600, 1.0)]
    tc = tt = 0.0
    print("%-52s %12s %10s %8s" % ("item", "calls", "tokens(B)", "hours"))
    for name, nq, mult in ROWS:
        calls = nq * a.calls_per_q * mult
        tok = calls * a.tokens_per_call / 1e9
        tc += calls
        tt += tok
        print("%-52s %12.0f %10.3f %8.1f" % (name, calls, tok, calls / a.throughput / 3600))
    print("%-52s %12.0f %10.3f %8.1f" % ("TOTAL", tc, tt, tc / a.throughput / 3600))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--queue", default=QUEUE)
    ap.add_argument("--tune-queue", default=os.path.join(HERE, "queues", "MIN_tune.jsonl"))
    ap.add_argument("--dev-factor", type=float, default=1.0)
    ap.add_argument("--spent", type=float, default=SPENT_DEFAULT)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--contingency", type=float, default=1.15, help="multiplier shown as an extra line (reruns, retries)")
    ap.add_argument("--json", default=None)
    s = sub.add_parser("actual")
    s.add_argument("--runs", required=True)
    s.add_argument("--queue", default=QUEUE)
    g = sub.add_parser("legacy")
    g.add_argument("--calls-per-q", type=float, required=True)
    g.add_argument("--tokens-per-call", type=float, required=True)
    g.add_argument("--throughput", type=float, required=True)
    a = ap.parse_args()
    if a.cmd == "actual":
        sys.exit(cmd_actual(a))
    if a.cmd == "legacy":
        return cmd_legacy(a)
    sys.exit(cmd_estimate(a))


if __name__ == "__main__":
    main()
