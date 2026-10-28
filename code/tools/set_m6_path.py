"""After the M6 burn-in (python -m nl2sql.stream burnin ...) and copying the store to exp/frozen/, point every dev config that uses M6 at it:
  python tools/set_m6_path.py exp/frozen/corpus_burnin_<sha>.sqlite
Appends/replaces one line `m6: {policy: ws_full, path: <p>, update: false}` in the listed configs (comments kept). abl_m6 (raw POOL57) and the E4/E4c configs
are deliberately not touched. Run before `python -m nl2sql.lock add` (config sha256 enters the lock)."""
import os
import re
import sys

CONFIGS = ["full_fixed", "full_rl", "full_x", "full_verbal", "full_noevidence", "abl_m1", "abl_m2", "abl_m3", "abl_m5", "crit_v1", "crit_v2", "crit_v5", "e6_tier"]
root = os.path.join(os.path.dirname(__file__), "..", "..", "exp", "configs")
p = sys.argv[1]
line = "m6: {policy: ws_full, path: %s, update: false}\n" % p
for c in CONFIGS:
    f = os.path.join(root, c + ".yaml")
    s = open(f).read()
    if re.search(r"^m6:.*$", s, re.M):
        s = re.sub(r"^m6:.*$\n?", line, s, flags=re.M)
    else:
        s = s.rstrip("\n") + "\n" + line
    open(f, "w").write(s)
    print("patched", f)
