#!/usr/bin/env python3
"""Fetch the one vendored baseline file that is not redistributed (upstream MAC-SQL repository has no licence file).
    python code/tools/fetch_macsql_const.py        # writes code/nl2sql/systems/baselines/_vendor/macsql_const.py, checks the sha256
Upstream: https://github.com/wbbeyourself/MAC-SQL, file core/const.py at commit 31a9df5e0d520be4769be57a4b9022e5e34a14f4."""
import hashlib, os, sys, urllib.request
URL = "https://raw.githubusercontent.com/wbbeyourself/MAC-SQL/31a9df5e0d520be4769be57a4b9022e5e34a14f4/core/const.py"
SHA256 = "dcf385e3f5cb13c12c418e8fd92b7b13d9cf1b470348414a750113e4d4ebaeae"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nl2sql", "systems", "baselines", "_vendor", "macsql_const.py")
data = urllib.request.urlopen(URL, timeout=60).read()
h = hashlib.sha256(data).hexdigest()
if h != SHA256:
    sys.exit("sha256 mismatch: got %s, expected %s" % (h, SHA256))
open(OUT, "wb").write(data)
print("wrote", os.path.normpath(OUT))
