# tools/

Repository-maintained utilities.

| Tool | Purpose |
|---|---|
| `tools/reorganize/migrate.py` | Rule-based migration engine used for the Aug-2026 reorganization. Planner supports per-directory rules, canonical relocation, dedupe of planned sources/destinations, checksum-verified moves, and a `--dry-run` plan preview. |
| `tools/reorganize/build_reorganization_manifest_csv.py` | Rebuilds `reports/repository_audit/reorganization_manifest.csv` from the migrate plan + inventory; emits counts of resolved and byte-identical extra copies. |

Both operate read-only on the repo tree except for the move/rewrite steps
explicitly requested via CLI flags. They are kept for reproducibility of the
audit; they are not part of the runtime `phenospace` package.
