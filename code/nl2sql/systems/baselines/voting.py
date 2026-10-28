"""Best-of-n with execution-result majority vote (bon_exec_vote) and self-consistency with a brief plan (self_consistency).

Selection rule (EXPERIMENT_SPEC section 4, `bon_vote`): n samples at T=0.7, drop non-executable, cluster by result set
(BIRD EX equivalence: set of row tuples), pick the largest cluster, tie -> lowest sample index.  Sample j of run r uses
seed 1000*r + j (EXPERIMENT_SPEC section 0); each sample is one LLM request, so `calls` = n.
Execution-based voting follows C3 / DAIL-SQL's self-consistency; the prompts are B1's own ("our adaptation").
"""
from collections import OrderedDict

from ._common import BaselineSystem, extract_sql, result_key, run_index
from .zero_shot import SYSTEM as ZS_SYSTEM, build_prompt

COT_SYSTEM = ("You write one SQLite query that answers the question. First write a brief plan (at most 4 short bullet points: "
              "tables, joins, filters, aggregation/ordering). Then give the final query in a single ```sql code block.")


def vote(ctx, sqls):
    """Return (index, info). Cluster executable samples by result set."""
    clusters = OrderedDict()
    n_exec = 0
    for j, s in enumerate(sqls):
        if not s:
            continue
        res = ctx.execute(s)
        k = result_key(res)
        if k is None:
            continue
        n_exec += 1
        clusters.setdefault(k, []).append(j)
    if not clusters:
        return 0, {"n_exec": 0, "n_clusters": 0, "top_cluster": 0}
    best = max(clusters.values(), key=lambda idx: (len(idx), -idx[0]))
    return best[0], {"n_exec": n_exec, "n_clusters": len(clusters), "top_cluster": len(best)}


class BoNExecVote(BaselineSystem):
    name = "bon_exec_vote"
    default_cfg = {"n": 5, "sample_temperature": 0.7, "max_tokens": 1024, "notes": False, "cot": False}
    SYSTEM = ZS_SYSTEM

    def solve(self, item, seed, ctx):
        n = int(self.cfg["n"])
        r = run_index(seed)
        prompt = build_prompt(item, self.cfg)
        mt = self.cfg["max_tokens"]

        def one(j):
            return ctx.user(prompt, system=self.SYSTEM, temperature=self.cfg["sample_temperature"], seed=1000 * r + j,
                            max_tokens=mt, tag=self.name + ".sample")

        texts = ctx.pmap(one, [(j,) for j in range(n)])
        sqls = [extract_sql(t, "last" if self.cfg.get("cot") else "first") for t in texts]
        idx, info = vote(ctx, sqls)
        ctx.act("sample x%d" % n)
        ctx.act("vote")
        err = "no_executable" if info["n_exec"] == 0 else None
        return sqls[idx], {"iterations": 1, "actions": ["sample x%d" % n, "vote"], "error_class": err,
                           "extra": {"vote": info, "n": n}}


class SelfConsistency(BoNExecVote):
    name = "self_consistency"
    default_cfg = dict(BoNExecVote.default_cfg, cot=True, max_tokens=1536)
    SYSTEM = COT_SYSTEM
