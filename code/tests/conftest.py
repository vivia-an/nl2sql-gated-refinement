import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import pytest  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_bird: needs the BIRD data under $NL2SQL_DATA (not redistributed, see DATA_LICENCES.md)")


def pytest_collection_modifyitems(config, items):
    try:
        from nl2sql import paths
        have = (paths.DATA / "bird" / "dev" / "dev_20240627" / "dev.json").exists() and (paths.DATA / "bird" / "train" / "train" / "train.json").exists()
    except Exception:
        have = False
    if have:
        return
    skip = pytest.mark.skip(reason="BIRD data not present under $NL2SQL_DATA (not redistributed, see DATA_LICENCES.md)")
    for it in items:
        if "needs_bird" in it.keywords:
            it.add_marker(skip)
