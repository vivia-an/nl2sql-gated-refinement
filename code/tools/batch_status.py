"""Print status/errors of batches recorded in a state dir (never prints the key)."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nl2sql import paths  # noqa: E402
from nl2sql.llm import BASE_URL, LLMClient  # noqa: E402

llm = LLMClient("/tmp/_unused_cache", None, paths.KEY_FILE)
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "batch_*.json"))):
        st = json.load(open(f))
        j = llm._request("GET", BASE_URL + "/batches/" + st["batch_id"]).json()
        err = j.get("errors")
        print(os.path.basename(d), st["model"], j.get("status"), "counts", j.get("request_counts"), "errors", json.dumps(err)[:400] if err else None,
              "out", bool(j.get("output_file_id")), "err_file", bool(j.get("error_file_id")))
