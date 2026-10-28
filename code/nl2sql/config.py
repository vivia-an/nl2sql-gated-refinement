"""models.yaml / experiment-config loading, model resolution, hashing of artefacts for the lock."""
import hashlib
import subprocess
from pathlib import Path

import yaml

from . import paths
from .util import canon_json, sha256_file, sha256_text

FORBIDDEN_PREFIXES = ("qwen-plus", "qwen-max", "qwen-flash", "qwen-turbo", "qwen3-max")

_MODELS = None


def models_cfg():
    global _MODELS
    if _MODELS is None:
        with open(paths.MODELS_YAML) as f:
            _MODELS = yaml.safe_load(f)
    return _MODELS


def resolve_model(key_or_id: str) -> str:
    """models.yaml key (qwen37plus) or role (primary_actor) or raw id -> model id. Refuses forbidden names."""
    cfg = models_cfg()
    if key_or_id in cfg.get("roles", {}) and isinstance(cfg["roles"][key_or_id], str):
        key_or_id = cfg["roles"][key_or_id]
    for m in cfg["models"]:
        if m["key"] == key_or_id:
            key_or_id = m["id"]
            break
    check_model_allowed(key_or_id)
    return key_or_id


def model_key(model_id: str) -> str:
    for m in models_cfg()["models"]:
        if m["id"] == model_id or m["key"] == model_id:
            return m["key"]
    return model_id.replace(".", "").replace("-", "")


def check_model_allowed(model_id: str):
    low = model_id.lower()
    for p in FORBIDDEN_PREFIXES:
        if low.startswith(p):
            raise ValueError("forbidden model id (CONTRACT section 2): %s" % model_id)


def _abs(path):
    p = Path(str(path))
    return p if p.is_absolute() or p.exists() else paths.REPO / p


def load_config(path) -> dict:
    with open(_abs(path)) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def config_sha(path) -> str:
    return sha256_file(_abs(path))


def prompts_sha() -> str:
    h = hashlib.sha256()
    for p in sorted(paths.PROMPTS_DIR.glob("*.txt")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def models_sha() -> str:
    return sha256_file(paths.MODELS_YAML)


def git_hash() -> str:
    f = paths.CODE_DIR / "GIT_HASH.txt"
    if f.exists():
        return f.read_text().strip()
    try:
        return subprocess.check_output(["git", "-C", str(paths.REPO), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def deep_update(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
