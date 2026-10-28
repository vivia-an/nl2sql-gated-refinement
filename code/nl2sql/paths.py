"""Central path resolution. Layout on host and worker machine is identical relative to the repo root:
<repo>/code (this package's parent), <repo>/exp. Data/cache/runs live under NL2SQL_ROOT."""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
REPO = CODE_DIR.parent
EXP_DIR = REPO / "exp"
PROMPTS_DIR = CODE_DIR / "prompts"
ROOT = Path(os.environ.get("NL2SQL_ROOT", str(Path.home() / "nl2sql_work")))
DATA = Path(os.environ.get("NL2SQL_DATA", str(ROOT / "data")))
# Caches (sqlite call cache, embeddings, gold execution results, schema/kstore files) must be on a LOCAL disk: sqlite files on a shared network
# volume were corrupted by concurrent writers from several machines. Reproducibility/audit copies are per-run archives
# (<run dir>/llm_cache.jsonl.gz, nl2sql.cache_archive / tools/cache_export.py), not the live cache. Override with NL2SQL_CACHE.
DEFAULT_CACHE = str(Path.home() / ".cache" / "nl2sql")
CACHE = Path(os.environ.get("NL2SQL_CACHE", DEFAULT_CACHE))
RUNS = Path(os.environ.get("NL2SQL_RUNS", str(ROOT / "runs")))
KEY_FILE = os.environ.get("NL2SQL_KEY_FILE", str(Path.home() / ".nl2sql" / "qwen.env"))
SPLITS_JSON = EXP_DIR / "splits" / "train_holdout.json"
DEV_LOCK = EXP_DIR / "dev_lock.json"
DEV_ACCESS_LOG = EXP_DIR / "dev_access.log"
MODELS_YAML = EXP_DIR / "models.yaml"
