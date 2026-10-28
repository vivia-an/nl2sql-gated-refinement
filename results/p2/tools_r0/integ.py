import sqlite3, glob, sys, os, time
R = os.environ["R"]
files = sorted(glob.glob(R + "/cache/llm/*.sqlite")) + [R + "/cache/emb.sqlite" if False else R + "/cache/llm/emb.sqlite", R + "/cache/gold_exec.sqlite"] + sorted(glob.glob(R + "/cache/kstore/*.sqlite"))[:0]
seen = set()
for f in files:
    if f in seen or not os.path.exists(f): continue
    seen.add(f)
    t = time.time()
    try:
        c = sqlite3.connect("file:%s?mode=ro" % f, uri=True, timeout=60)
        r = c.execute("PRAGMA integrity_check").fetchall()
        n = [(t_, c.execute("select count(*) from %s" % t_).fetchone()[0]) for (t_,) in c.execute("select name from sqlite_master where type='table'").fetchall()]
        print(os.path.basename(f), r[:2], n, "%.1fs" % (time.time() - t), flush=True)
    except Exception as e:
        print(os.path.basename(f), "ERR", type(e).__name__, str(e)[:100], flush=True)
