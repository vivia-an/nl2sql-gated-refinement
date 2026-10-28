#!/bin/bash
# Create a venv under $NL2SQL_ROOT (idempotent). Usage: bash scripts/setup_venv.sh
set -e
ROOT=${NL2SQL_ROOT:-$HOME/nl2sql_work}
VENV=$ROOT/venv
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/pip" install -q -r "$(dirname "$0")/../requirements.txt" || \
  "$VENV/bin/pip" install -q sqlglot func_timeout
"$VENV/bin/python" -c "import sqlglot, yaml, httpx, numpy, sklearn, networkx; print('venv ok', sqlglot.__version__)"
