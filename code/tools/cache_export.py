"""Archive / restore the LLM call cache (see nl2sql/cache_archive.py).

  python tools/cache_export.py export  RUN_DIR [RUN_DIR ...] [--cache DIR]   # -> RUN_DIR/llm_cache.jsonl.gz (+ manifest); merges with an existing one
  python tools/cache_export.py import  ARCHIVE_OR_DIR [...]   [--cache DIR]   # warm a (machine-local) cache; dirs are searched recursively
  python tools/cache_export.py verify  ARCHIVE_OR_DIR [...]                   # keys re-derived from requests, manifest hash
  python tools/cache_export.py check   [--cache DIR]                           # integrity-check every shard, quarantine malformed ones
RUN_DIR may also be a parent (e.g. $R/runs/E1): every sub-directory with a usage*.jsonl is exported.
Default cache = nl2sql.paths.CACHE (machine-local <NL2SQL_CACHE>, or $NL2SQL_CACHE). No network access, no LLM calls."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nl2sql import cache_archive as CA, paths, sqlcache  # noqa: E402
from nl2sql.llm import _Shards  # noqa: E402


def run_dirs(targets):
    out = []
    for t in targets:
        for root, _, files in os.walk(t):
            if any(f.startswith("usage") and f.endswith(".jsonl") for f in files):
                out.append(root)
    return sorted(set(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "import", "verify", "check"])
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--cache", default=None, help="call-cache dir (default: <paths.CACHE>/llm)")
    a = ap.parse_args(argv)
    cache = a.cache or str(paths.CACHE / "llm")
    if a.cmd == "export":
        store = _Shards(cache)
        rc = 0
        for d in run_dirs(a.targets):
            man = CA.export_run(d, store=store)
            print(d, json.dumps(man))
            rc |= int(man.get("n_missing_in_cache", 0) > 0)
        return rc
    if a.cmd == "import":
        print(json.dumps(CA.import_archives(a.targets, cache)))
        return 0
    if a.cmd == "verify":
        rc = 0
        for p in CA.find_archives(a.targets):
            r = CA.verify_archive(p)
            print(p, json.dumps(r))
            rc |= int(not r["ok"])
        return rc
    if a.cmd == "check":
        n_bad = 0
        for f in sorted(os.listdir(cache)) if os.path.isdir(cache) else []:
            if f.endswith(".sqlite"):
                ok, detail = sqlcache.check_file(os.path.join(cache, f))
                print(f, "ok" if ok else "MALFORMED: " + detail)
                if not ok:
                    sqlcache.quarantine(os.path.join(cache, f), reason=detail)
                    n_bad += 1
        return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
