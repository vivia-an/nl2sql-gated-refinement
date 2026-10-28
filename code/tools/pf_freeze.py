#!/usr/bin/env python3
"""Freeze step of task PF (SAP 22.7): writes exp/pf_prompts/FROZEN.json (sha256 of the two prompt files and of the tuning item list) and the four dev
configs exp/configs/pf/patent_fb_{n0,n250_q100,n1000_q100,n1000_q085}.yaml (prompt and verdict-file hashes inside, so the config hash that the dev lock
records pins them). Host side, stdlib only. Run once, after tuning, before `python -m nl2sql.lock add`; commit the result before the lock entries."""
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FORBID = ["results/diag/items.json", "results/diag/tuning/items_tune.json", "results/diag/tuning_ve/items_tune_ve.json", "results/pf/tuning/items_tune_pf.json"]
ARMS = [("n0", 0, 1.0), ("n250_q100", 250, 1.0), ("n1000_q100", 1000, 1.0), ("n1000_q085", 1000, 0.85)]


def sha(rel):
    return hashlib.sha256(open(os.path.join(REPO, rel), "rb").read()).hexdigest()


def main():
    note = sys.argv[1] if len(sys.argv) > 1 else ""
    prompts = {f: sha("exp/pf_prompts/" + f) for f in ("pf_critic.user.txt", "pf_standard.txt")}
    fz = {"_meta": {"rule": "docs/SAP.md 22.7 / 22.12", "tuning_items_sha256": sha("results/pf/tuning/items_tune_pf.json"), "verdicts_sha256": sha("exp/pf_data/burnin_verdicts.jsonl"),
                    "note": note}, "prompts": prompts}
    json.dump(fz, open(os.path.join(REPO, "exp/pf_prompts/FROZEN.json"), "w"), indent=1)
    for tag, n, q in ARMS:
        txt = ["# FROZEN by code/tools/pf_freeze.py (SAP 22.12). patent_fb, arm n=%d q=%s. Do not edit: a change is a new lock entry (SPEC 1.3 step 6)." % (n, q),
               "# Fixed patent loop P0 + identifier quoting + frozen M6 store = exp/configs/frozen/full_fixed.yaml; only the Critic input differs (pf).",
               "temperature: 0.0", "ddl: {quote_keywords: true}",
               "m6: {policy: ws_full, path: exp/frozen/corpus_burnin_39e2477c.sqlite, update: false}", "pf:", "  n: %d" % n, "  q: %s" % q, "  k: 4", "  seed: 20261128",
               "  verdicts: exp/pf_data/burnin_verdicts.jsonl", "  verdicts_sha256: %s" % fz["_meta"]["verdicts_sha256"], "  prompt_dir: exp/pf_prompts",
               "  prompt_sha256: {pf_critic.user.txt: %s, pf_standard.txt: %s}" % (prompts["pf_critic.user.txt"], prompts["pf_standard.txt"]),
               "  forbid_files: [%s]" % ", ".join(FORBID)]
        open(os.path.join(REPO, "exp/configs/pf/patent_fb_%s.yaml" % tag), "w").write("\n".join(txt) + "\n")
    print(json.dumps(fz, indent=1))


if __name__ == "__main__":
    main()
