#!/usr/bin/env python3
"""X2DS run metadata (post hoc; pure Python, no model call). Reads <W>/runs/X2DS/*/, <W>/runs/X2DSR/*/, <W>/runs/X2DSRR/*-r1/ and <W>/runs/W7/*/ (run_meta.json, config.yaml hash, usage.jsonl) and writes results/x2ds/x2ds_runs.json:
start and end of every run (Beijing time, UTC+8), number of chat requests billed (usage rows with cached = false) and answered from the call cache, config / prompts / models.yaml hashes, git hash.
    python3 code/tools/x2ds_runmeta.py [W] [OUT]"""
import glob
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
W = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else os.path.join(REPO, "results", "raw-ignored", "r4t0b_host")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "results", "x2ds", "x2ds_runs.json")


def bj(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts + 8 * 3600))


out = {}
for d in sorted(glob.glob(os.path.join(W, "runs", "X2DS", "*-s0")) + glob.glob(os.path.join(W, "runs", "X2DSR", "*-s0"))
                + glob.glob(os.path.join(W, "runs", "X2DSRR", "*-r1")) + glob.glob(os.path.join(W, "runs", "W7", "*-s0")) + glob.glob(os.path.join(W, "runs", "W7RR", "*-r1"))
                + glob.glob(os.path.join(W, "runs", "W8", "*-s0")) + glob.glob(os.path.join(W, "runs", "W8", "*-r1"))):   # W7: reruns and the three further arms; W8: pipeline + empty gate
    name = os.path.basename(os.path.dirname(d)) + "/" + os.path.basename(d)
    meta = json.load(open(os.path.join(d, "run_meta.json")))
    billed = cached = 0
    ts = []
    for line in open(os.path.join(d, "usage.jsonl")):
        r = json.loads(line)
        if r.get("kind") != "chat":
            continue
        ts.append(r["ts"])
        if r.get("cached"):
            cached += 1
        else:
            billed += 1
    out[name] = {"system": meta["system"], "model": meta["model"], "seed": meta["seed"], "first_request_bj": bj(min(ts)), "last_request_bj": bj(max(ts)), "chat_billed": billed,
                 "chat_from_cache": cached, "config_sha256": meta["config_sha256"][:12], "prompts_sha256": meta["prompts_sha256"][:12], "models_yaml_sha256": meta["models_yaml_sha256"][:12],
                 "git_hash": meta["git_hash"][:10]}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
for k, v in out.items():
    print(k, v["first_request_bj"], v["last_request_bj"], v["chat_billed"], v["chat_from_cache"])
