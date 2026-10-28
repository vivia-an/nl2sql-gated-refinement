"""System registry. Ours: systems/ours; baselines (B1): systems/baselines (imported if present)."""
import importlib

REGISTRY = {}


def register(name, cls):
    REGISTRY[name] = cls


def _load():
    from .ours import SYSTEMS as OURS
    REGISTRY.update(OURS)
    try:
        mod = importlib.import_module(".baselines", __name__)
        for attr in ("SYSTEMS", "BASELINES", "REGISTRY"):
            REGISTRY.update(getattr(mod, attr, {}) or {})   # B1 wins on name clashes (e.g. zs)
    except ImportError as e:
        if "baselines" not in str(e):
            raise


def get_system(name):
    if not REGISTRY:
        _load()
    if name not in REGISTRY:
        raise KeyError("unknown system %s (known: %s)" % (name, sorted(REGISTRY)))
    return REGISTRY[name]
