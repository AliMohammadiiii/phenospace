"""Canonical path resolution for the phenospace repository.

Single source of truth for the canonical layout so that scripts, notebooks,
and tests never hardcode absolute paths:

    configs/
    scripts/
    src/phenospace/
    data/{raw,interim,processed,schemas,manifests}
    artifacts/
    experiments/
    notebooks/
    tests/
    docs/
    reports/
    papers/
    archive/

All functions derive paths from ``REPO_ROOT``, which is discovered relative to
this file (``src/phenospace/common/paths.py`` → `parents[2]` == repo root), so
the package keeps working no matter where the checkout lives.
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

REPO_ROOT = _REPO_ROOT


def _p(rel: str) -> Path:
    p = REPO_ROOT / rel
    return p


def cfg(*parts: str) -> Path:
    """Public config path under configs/."""
    return _p("configs").joinpath(*parts)


def data(*parts: str) -> Path:
    """Public data path (raw / interim / processed live under data/)."""
    return _p("data").joinpath(*parts)


def artifacts(*parts: str) -> Path:
    return _p("artifacts").joinpath(*parts)


def experiments(*parts: str) -> Path:
    return _p("experiments").joinpath(*parts)


def scripts(*parts: str) -> Path:
    return _p("scripts").joinpath(*parts)


def src_pkg(*parts: str) -> Path:
    return _p("src", "phenospace").joinpath(*parts)


def notebooks(*parts: str) -> Path:
    return _p("notebooks").joinpath(*parts)


def tests(*parts: str) -> Path:
    return _p("tests").joinpath(*parts)


def docs(*parts: str) -> Path:
    return _p("docs").joinpath(*parts)


def reports(*parts: str) -> Path:
    return _p("reports").joinpath(*parts)


def papers(*parts: str) -> Path:
    return _p("papers").joinpath(*parts)


def archive(*parts: str) -> Path:
    return _p("archive").joinpath(*parts)


# ---------------------------------------------------------------------------
# Commonly used concrete shortcuts
# ---------------------------------------------------------------------------

def notation(*parts: str) -> Path:
    """Archetype-stability / operator notes (unused outside audits)."""
    return _p("archive", "notes").joinpath(*parts)


def artf(*parts: str) -> Path:
    """Alias for artifacts()."""
    return artifacts(*parts)


def manifest_path(name: str) -> Path:
    """Location of a provenance manifest in the repository audit directory."""
    return reports("repository_audit", name)


__all__ = [
    "REPO_ROOT",
    "archive",
    "artf",
    "artifacts",
    "cfg",
    "data",
    "docs",
    "experiments",
    "manifest_path",
    "notebooks",
    "papers",
    "reports",
    "scripts",
    "src_pkg",
    "tests",
]
