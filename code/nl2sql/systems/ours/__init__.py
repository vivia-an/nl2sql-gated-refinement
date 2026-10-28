from .patent_fb import PatentFB
from .pipeline import FullFixed, FullGated, FullGatedBlind, FullRL, FullSystem, FullX, ZeroShot
from .zs_ex import ZeroShotExamples

SYSTEMS = {
    "full": FullSystem,          # behaviour decided by the config (m7.controller / m5.variant ...)
    "full_fixed": FullFixed,     # M1-M6 + patent policy P0
    "full_gated": FullGated,     # X5: M1-M6 + execution-gated loop (m5.policy gated); routing / consistency / arbitration by config
    "full_gated_blind": FullGatedBlind,   # X6: full_gated with the error text withheld from the revision prompt (P5 control)
    "patent_fb": PatentFB,       # SAP Amendment 9 (section 22): P0 loop with a feedback-calibrated Critic (config key `pf`); exploratory-posthoc
    "full_rl": FullRL,           # M1-M7, needs m7.controller.path
    "full_x": FullX,             # critic = cross-vendor model (dsv41flash)
    "zs": ZeroShot,              # fallback; B1's zs overrides in the registry
    "zs_ours": ZeroShot,
    "zs_ex": ZeroShotExamples,   # R3 (SAP section 26): single call + the examples the root-only arm of the same backbone and seed retrieved
}
