"""When root and forced revise(T=0) disagree and exactly one is right: who wins, by root status (executable roots only) and by root score."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from execdata import *
Qs = load_trees(); annotate(Qs)
res = {}
def tab(filt):
    w = l = 0; dis = 0
    for q in Qs:
        r0 = q.first_child(0, "revise")
        if r0 is None or not filt(q): continue
        if r0["rk"] != q.root["rk"]:
            dis += 1
            if r0["ex"] != q.root["ex"]:
                w += int(r0["ex"] == 1); l += int(r0["ex"] == 0)
    return {"disagree": dis, "revision_right": w, "root_right": l, "revision_win_rate": w / max(1, w + l)}
exe = lambda q: q.root["status"] in ("ok_nonempty", "ok_empty")
res["executable_roots"] = tab(exe)
res["executable_roots_score<8.5"] = tab(lambda q: exe(q) and q.root["score"] < 8.5)
res["executable_roots_score>=8.5"] = tab(lambda q: exe(q) and q.root["score"] >= 8.5)
res["executable_roots_score==10"] = tab(lambda q: exe(q) and q.root["score"] >= 10)
res["root_nonempty"] = tab(lambda q: q.root["status"] == "ok_nonempty")
res["root_empty"] = tab(lambda q: q.root["status"] == "ok_empty")
res["exec_error_roots"] = tab(lambda q: not exe(q))
# same for escalate vs root
def tab_esc(filt):
    w = l = 0
    for q in Qs:
        e = q.first_child(0, "escalate")
        if e is None or not filt(q): continue
        if e["rk"] != q.root["rk"] and e["ex"] != q.root["ex"]:
            w += int(e["ex"] == 1); l += int(e["ex"] == 0)
    return {"escalate_right": w, "root_right": l, "win_rate": w / max(1, w + l)}
res["escalate_vs_root_executable"] = tab_esc(exe)
dump("q9b_pairs.json", res)
print(json.dumps(res, indent=1))
