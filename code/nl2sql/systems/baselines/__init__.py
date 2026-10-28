"""B1 baselines.  `BASELINES = {name: class}`; X2 merges this dict into nl2sql.systems.REGISTRY.

Names follow the B1 brief; the keys of EXPERIMENT_SPEC section 4 are provided as aliases (same class, same defaults)."""
from .zero_shot import ZeroShot
from .voting import BoNExecVote, SelfConsistency
from .self_debug import SelfDebug
from .ac_sql import ACSQL, ACSQLCross
from .dail_sql import DailSQL
from .mac_sql import MACSQL
from .alpha_sql import AlphaSQL
from .memo_sql import MemoSQL

BASELINES = {
    "zero_shot": ZeroShot,
    "dail_sql": DailSQL,
    "self_debug": SelfDebug,
    "mac_sql": MACSQL,
    "ac_sql": ACSQL,
    "ac_sql_x": ACSQLCross,
    "alpha_sql": AlphaSQL,
    "memo_sql": MemoSQL,
    "bon_exec_vote": BoNExecVote,
    "self_consistency": SelfConsistency,
}

# EXPERIMENT_SPEC section 4 keys -> same classes
SPEC_ALIASES = {"zs": "zero_shot", "dail": "dail_sql", "sdebug": "self_debug", "macsql": "mac_sql", "acsql_s": "ac_sql",
                "acsql_x": "ac_sql_x", "memosql": "memo_sql", "alphasql": "alpha_sql", "bon_vote": "bon_exec_vote",
                "sc_cot": "self_consistency"}
for _a, _n in SPEC_ALIASES.items():
    if _n in BASELINES:
        BASELINES[_a] = BASELINES[_n]
