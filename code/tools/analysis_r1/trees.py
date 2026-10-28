"""Tree helpers: structure, P0 replay, oracle sets. HR only (8 dbs, 1101 q). No gold information is used except node['ex'] labels."""
from common import *


class Q:
    """One question's tree."""
    def __init__(self, t):
        self.qid, self.db, self.fold = t["qid"], t["db_id"], t["fold"]
        self.nodes = t["nodes"]
        self.by_id = {n["id"]: n for n in self.nodes}
        self.kids = collections.defaultdict(list)
        for n in self.nodes:
            if n["parent"] is not None:
                self.kids[n["parent"]].append(n)
        self.root = self.nodes[0]

    def first_child(self, nid, action):
        """first-created child of that action = the T=0 child (T=0.7 child is created after it)."""
        for c in self.kids.get(nid, []):
            if c["action"] == action:
                return c
        return None

    def children(self, nid, action=None):
        return [c for c in self.kids.get(nid, []) if action is None or c["action"] == action]

    def depth1(self):
        return self.kids.get(0, [])

    def desc(self, nid):
        out = []
        st = [nid]
        while st:
            x = st.pop()
            for c in self.kids.get(x, []):
                out.append(c); st.append(c)
        return out

    # ---- P0 replay (policy of FullFixed._run_p0), parameters: theta_acc, t_max, theta_h
    def p0(self, theta_acc=8.5, t_max=3, theta_h=7.5, tie_prefer="low_k", route_human=True):
        cur = self.root
        nodes = [cur]
        while True:
            if cur["score"] >= theta_acc:
                break
            if cur["k"] >= t_max:
                break
            ch = self.first_child(cur["id"], "revise")
            if ch is None:
                break
            nodes.append(ch)
            if ch["sql"].strip() == cur["sql"].strip():
                break
            cur = ch
        if tie_prefer == "low_k":
            best = max(nodes, key=lambda n: (n["score"], -n["k"]))
        else:
            best = max(nodes, key=lambda n: (n["score"], n["k"]))
        ab = bool(route_human and best["score"] < theta_h)
        calls = sum(n["calls"] for n in nodes)
        toks = sum(n["prompt_tokens"] + n["completion_tokens"] for n in nodes)
        return {"final": best, "abstain": ab, "calls": calls, "tokens": toks, "nodes": nodes, "ex": 0 if ab else best["ex"]}

    def p0_chain_nodes(self, t_max=3):
        cur = self.root; out = [cur]
        while cur["k"] < t_max:
            ch = self.first_child(cur["id"], "revise")
            if ch is None:
                break
            out.append(ch); cur = ch
        return out

    def oracle(self, nodes):
        return int(any(n["ex"] == 1 for n in nodes))


def load_trees():
    return [Q(t) for t in jl(TREES)]
