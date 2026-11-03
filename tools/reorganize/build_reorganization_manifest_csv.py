#!/usr/bin/env python3
"""
build_final_manifest_csv.py — reconstruct the ground-truth migration manifest.

After a move, every original file's bytes live somewhere under the repo
(possibly transformed in path). Because moves were checksum-verified, we can
reconstruct the authoritative source → destination table by matching the
original inventory's sha256 against every file currently on disk.

Files that were intentionally edited *after* migration are reconciled through
reports/repository_audit/post_migration_edits.csv (old → new bytes); they get
status `superseded` with a pointer to their current bytes, otherwise `identical`.

Output: reports/repository_audit/reorganization_manifest.csv
"""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INV = ROOT / "reports" / "repository_audit" / "original_file_inventory.csv"
EDITS = ROOT / "reports" / "repository_audit" / "post_migration_edits.csv"
OUT = ROOT / "reports" / "repository_audit" / "reorganization_manifest.csv"
EXCLUDED = (".git", ".venv")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def all_files(start: Path):
    for root, _, files in os.walk(start):
        r = str(Path(root).relative_to(ROOT))
        if any(r == e or r.startswith(e + "/") for e in EXCLUDED):
            continue
        for fn in files:
            yield Path(root) / fn


def load_edits() -> dict[str, dict[str, str]]:
    """source_rel_path -> {current_rel_path, current_sha256}."""
    if not EDITS.exists():
        return {}
    from io import StringIO

    noncomment = [
        ln for ln in EDITS.read_text().splitlines() if not ln.lstrip().startswith("#")
    ]
    rows = list(csv.DictReader(StringIO("\n".join(noncomment) + "\n")))
    return {
        r["source_rel_path"]: {
            "current_rel_path": r["current_rel_path"],
            "current_sha256": r["current_sha256"],
        }
        for r in rows
    }


def main() -> None:
    with open(INV) as fh:
        inv_rows = list(csv.DictReader(fh))
    edits = load_edits()
    # index current tree by checksum
    index: dict[str, list[str]] = {}
    for p in all_files(ROOT):
        if p in (OUT, EDITS):
            continue
        h = sha256(p)
        index.setdefault(h, []).append(str(p.relative_to(ROOT)))

    rows: list[dict[str, Any]] = []
    unresolved: list[str] = []
    resolved_identical = 0
    resolved_superseded = 0
    for r in inv_rows:
        h = r["sha256"]
        dests = index.get(h, [])
        row: dict[str, Any] = {
            "source_rel_path": r["rel_path"],
            "size_bytes": r["size_bytes"],
            "sha256": h,
            "destination_rel_path": dests[0] if dests else "",
            "current_sha256": h,
            "status": "identical" if dests else "superseded",
            "n_restored_copies": max(len(dests) - 1, 0),
        }
        if dests:
            resolved_identical += 1
        else:
            edit = edits.get(r["rel_path"])
            if edit:
                row["destination_rel_path"] = edit["current_rel_path"]
                row["current_sha256"] = edit["current_sha256"]
                resolved_superseded += 1
            else:
                unresolved.append(r["rel_path"])
        rows.append(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["source_rel_path"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"manifest rows: {len(rows)}")
    print(f"resolved (byte-identical): {resolved_identical}")
    print(f"resolved (superseded by post-migration edit): {resolved_superseded}")
    print(f"unresolved (no copy found on disk): {len(unresolved)}")
    for m in unresolved:
        print("  ", m)
    dup = sum(r["n_restored_copies"] for r in rows)
    print(f"extra byte-identical copies preserved: {dup}")


if __name__ == "__main__":
    main()
