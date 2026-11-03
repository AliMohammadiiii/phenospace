"""Package-level tests for the phenospace `src` package and repo hygiene."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _ensure_cwd_is_repo_root():
    import os

    os.chdir(REPO_ROOT)


def test_repo_root_constant():
    from phenospace.common.paths import REPO_ROOT

    assert REPO_ROOT.resolve() == REPO_ROOT


def test_paths_helpers_are_absolute_within_repo():
    from phenospace.common.paths import cfg, data, reports

    for p in (cfg(), data(), reports()):
        assert p.is_absolute()
        assert REPO_ROOT in p.parents


def test_manifest_files_exist():
    inv = reports_path("original_file_inventory.csv")
    reorg = reports_path("reorganization_manifest.csv")
    assert inv.exists()
    assert reorg.exists()


def reports_path(name: str) -> Path:
    return REPO_ROOT / "reports" / "repository_audit" / name


def test_no_absolute_dev_paths_in_source():
    """Guards the 'zero hardcoded absolute paths' rule for active code."""
    import re

    offenders = []
    pats = re.compile(r"/Users/|/root/|/home/|C:\\")
    import os

    for root_str, _, files in os.walk(REPO_ROOT / "src"):
        root = Path(root_str)
        for fn in files:
            if not fn.endswith(".py"):
                continue
            txt = (root / fn).read_text(errors="ignore")
            for i, line in enumerate(txt.splitlines(), 1):
                if pats.search(line) and "pathlib" not in line:
                    offenders.append(f"{root.name}/{fn}:{i}")
    assert offenders == [], "\n".join(offenders)


def test_untitled_scorecard_parses():
    p = REPO_ROOT / "reports" / "evaluation" / "phase1_metric_finetune_scorecard_untitled1.json"
    if not p.exists():
        pytest.skip("scorecard not present")
    d = json.loads(p.read_text())
    assert d["best_step"] == 500
    assert d["final"]["mean_parent_rank"] < d["baseline"]["mean_parent_rank"]


def test_data_sources_yaml_loads():
    import yaml

    p = REPO_ROOT / "data" / "manifests" / "data_sources.yaml"
    d = yaml.safe_load(p.read_text())
    ids = {r["id"] for r in d["registries"]}
    assert "tdpa_zenodo_10423537" in ids
    assert "hp_ontology" in ids
