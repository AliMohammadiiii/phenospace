#!/usr/bin/env python3
"""
migrate.py — phenospace repository reorganization engine (Cytognosis, cytognosis.org).

Moves the legacy "phase-numbered / duplicate-bundle" tree into the canonical
layout while:
  * never deleting bytes (superseded originals are *moved* into archive/…),
  * checksum-verifying every move,
  * writing a full provenance manifest
    (reports/repository_audit/reorganization_manifest.csv).

Usage:
  python3 tools/reorganize/migrate.py --dry-run   # plan only, no changes
  python3 tools/reorganize/migrate.py --apply     # execute + verify
"""
from __future__ import annotations

import csv
import hashlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "reports" / "repository_audit" / "original_file_inventory.csv"
MANIFEST = ROOT / "reports" / "repository_audit" / "reorganization_manifest.csv"
ARCHIVE = ROOT / "archive"
EXCLUDED_PREFIXES = (".git", ".venv")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def is_excluded(p: Path) -> bool:
    r = rel(p)
    return any(r.startswith(e + "/") or r == e for e in EXCLUDED_PREFIXES)


def tree_files(start: Path):
    for root, _dirs, files in os.walk(start):
        for fn in files:
            yield Path(root) / fn


class Planner:
    """Accumulates unique source→destination ops; prevents double-planning."""

    def __init__(self):
        self.planned_sources: set[Path] = set()
        self.planned_dirs: set[Path] = set()
        self.ops: list[dict] = []

    def add(self, src: Path, dst: Path, kind: str, note: str = "") -> None:
        src = src.resolve()
        if src in self.planned_sources:
            return  # already extracted/moved; do not also archive
        if not src.exists():
            return
        if src.is_dir():
            if src in self.planned_dirs:
                return
            self.planned_dirs.add(src)
            for parent in src.parents:
                if parent in self.planned_dirs and parent != src:
                    return
            self.planned_sources.add(src)
            self.ops.append({"src": rel(src), "dst": rel(dst.resolve()),
                             "kind": kind, "note": note})
            return
        for parent in src.parents:
            if parent in self.planned_dirs:
                return
        self.planned_sources.add(src)
        self.ops.append({"src": rel(src), "dst": rel(dst.resolve()),
                         "kind": kind, "note": note})

    def archive_tree(self, tree: Path, sub: str) -> None:
        if not tree.exists():
            return
        for f in tree_files(tree):
            dst = ARCHIVE / "original_packages" / sub / f.relative_to(tree)
            if f.resolve() not in self.planned_sources:
                self.add(f, dst, "archive", "superseded original package")

    def move_tree(self, tree: Path, dst_dir: Path, sub: str) -> None:
        if not tree.exists():
            return
        for f in tree_files(tree):
            dst = dst_dir / sub / f.relative_to(tree)
            self.add(f, dst, "move", "canonical relocation")


def plan() -> list[dict]:
    P = Planner()

    # ----------------------------------------------------------------
    # 1) Root-level files
    # ----------------------------------------------------------------
    P.add(ROOT / "Research Roadmap.md", ROOT / "docs" / "research_roadmap.md",
          "move", "canonical doc")
    P.add(ROOT / "phase1_closeout.md",
          ROOT / "reports" / "phase_closeouts" / "phase1_closeout.md",
          "move", "canonical closeout report")
    P.add(ROOT / "Untitled-1.json",
          ROOT / "experiments" / "gemma_metric_tuning" / "untitled1_run.json",
          "move", "phase1 metric-tuning run (config+eval snapshot)")
    P.add(ROOT / "artifacts.zip", ARCHIVE / "original_packages" / "artifacts.zip",
          "archive", "original root-level archive package")

    # ----------------------------------------------------------------
    # 2) Archetype tree -> canonical runnable pipeline (kept intact)
    # ----------------------------------------------------------------
    a = ROOT / "Archetype"
    if a.exists():
        P.add(a / "supervisor_package_aanet_v0_1",
              ARCHIVE / "original_packages" / "supervisor_package_aanet_v0_1",
              "archive_dir", "superseded packaged deliverable (code lives in 11_anndata)")
        P.add(a / "package_aanet_v0_1.zip",
              ARCHIVE / "original_packages" / "package_aanet_v0_1.zip",
              "archive", "original supervisor package zip")
        P.add(a / "FINAL_MANIFEST.txt",
              ROOT / "reports" / "repository_audit" / "final_real_v0_manifest.txt",
              "move", "provenance manifest")
        P.add(a / "README_FINAL_REAL_V0.md",
              ROOT / "docs" / "real_data_pipeline_README.md",
              "move", "canonical pipeline readme")
        P.add(a / "08_notebooks" / "dataset.ipynb",
              ROOT / "notebooks" / "analysis" / "real_dataset_v0.ipynb",
              "move", "canonical analysis notebook")
        P.move_tree(a, ROOT / "experiments", "real_data_pipeline")

    # ----------------------------------------------------------------
    # 3) Dataset tree
    # ----------------------------------------------------------------
    d = ROOT / "Dataset"
    if d.exists():
        for f in tree_files(d / "Dateset"):
            P.add(f, ROOT / "data" / "raw" / "questionnaires" / f.name,
                  "move", "canonical raw questionnaire export")
        st = d / "temporal_assessment_outputs_v0" / "synthetic_temporal_assessments_raw_v0.csv"
        P.add(st, ROOT / "data" / "interim" / "synthetic_temporal_assessments_raw_v0.csv",
              "move", "interim synthetic temporal prepared data")
        fr = d / "cytognosis_neuroverse_final_real_v0"
        if fr.exists():
            viz = fr / "11_visualizations"
            if viz.exists():
                for f in tree_files(viz):
                    P.add(f, ROOT / "experiments" / "real_data_pipeline" / "11_visualizations" / f.name,  # noqa: E501
                          "merge", "unique visualization merged into canonical pipeline tree")
            P.archive_tree(fr, "cytognosis_neuroverse_final_real_v0")
        prep = d / "temporal_assessment_outputs_real_v0_prepared"
        if prep.exists():
            variants = [
                ("projection_outputs_v7d_config_patch_phq9_gad7_pss10_isi",
                 ROOT / "experiments" / "real_data_pipeline_variants" / "projection_config_patch_phq9_gad7_pss10_isi"),  # noqa: E501
                ("projection_outputs_v7d_score_fixed_phq9_gad7_pss10_isi",
                 ROOT / "experiments" / "real_data_pipeline_variants" / "projection_score_fixed_phq9_gad7_pss10_isi"),  # noqa: E501
            ]
            for sub, dst in variants:
                src = prep / sub
                if src.exists():
                    for f in tree_files(src):
                        P.add(f, dst / f.name, "move", "unique variant projection run")
            qc = prep / "qc"
            if qc.exists():
                for f in tree_files(qc):
                    if f.name in ("projection_run_v7d_report.json",
                                  "projection_run_v7d_fixed_report.json"):
                        P.add(f, ROOT / "reports" / "validation" / f.name,
                              "move", "unique QC/validation report")
            P.archive_tree(prep, "temporal_assessment_outputs_real_v0_prepared")
        bundle = d / "cytognosis_neuroverse_pipeline_v0_bundle"
        if bundle.exists():
            papers_src = bundle / "06_papers_and_project_context"
            if papers_src.exists():
                for f in tree_files(papers_src):
                    if f.name == "Neuroverse_macro.md":
                        P.add(f, ROOT / "docs" / "neuroverse_macro.md", "move",
                              "canonical project context doc")
                    elif f.name == "Research Roadmap.md":
                        P.add(f, ARCHIVE / "original_packages" / "bundle_06_research_roadmap.md",
                              "archive", "duplicate of docs/research_roadmap.md")
                    elif f.name == "phase1_closeout.md":
                        P.add(f, ARCHIVE / "original_packages" / "bundle_06_phase1_closeout.md",
                              "archive", "duplicate of reports/phase_closeouts/phase1_closeout.md")
                    elif f.suffix.lower() == ".pdf":
                        P.add(f, ROOT / "papers" / f.name, "move", "canonical paper pdf")
            dash = bundle / "05_dashboards_reports" / "archetype_dashboard_v0_outputs"
            if dash.exists():
                for f in tree_files(dash):
                    P.add(f, ROOT / "experiments" / "archetypal_analysis" / "dashboard_v0_outputs" / f.name,  # noqa: E501
                          "move", "canonical archetype dashboard outputs")
            arcmod = bundle / "03_archetype_models"
            if arcmod.exists():
                for f in tree_files(arcmod):
                    P.add(f, ROOT / "experiments" / "archetypal_analysis" / "models_v0_outputs" / f.name,  # noqa: E501
                          "move", "canonical archetype model outputs (synthetic + v0)")
            rdp = bundle / "04_real_data_pipeline"
            if rdp.exists():
                cfg = rdp / "real_data_config_v0.yaml"
                P.add(cfg, ROOT / "configs" / "questionnaire_projection" / "real_data_config_v0.yaml",  # noqa: E501
                      "move", "canonical v0 config")
                out = rdp / "real_data_projection_v0_outputs"
                if out.exists():
                    for f in tree_files(out):
                        P.add(f, ROOT / "experiments" / "questionnaire_projection" / "real_data_v0_outputs" / f.name,  # noqa: E501
                              "move", "canonical v0 real-data projection run")
            P.archive_tree(bundle, "cytosis_neuro_archetype_pipeline_v0_bundle")

    # ----------------------------------------------------------------
    # 4) Phase1  (literature → papers; scripts → scripts/)
    # ----------------------------------------------------------------
    p1 = ROOT / "Phase1"
    if p1.exists():
        dr = p1 / "Deep research"
        if dr.exists():
            for f in tree_files(dr):
                P.add(f, ROOT / "papers" / "literature_review" / f.name,
                      "move", "canonical literature review")
        for f, d2 in [
            (p1 / "next_stage_training_v3_phase1_fixed.py", ROOT / "scripts" / "gemma_metric_tuning"),  # noqa: E501
            (p1 / "transbox_geometry_diagnostic.py", ROOT / "scripts" / "ontology_embedding"),
            (p1 / "transbox_v2_train.py", ROOT / "scripts" / "ontology_embedding"),
        ]:
            P.add(f, d2 / f.name, "move", "canonical phase1 script")
        P.archive_tree(p1, "Phase1")

    # ----------------------------------------------------------------
    # 5) Phase1-5  (synthetic population canonical split)
    # ----------------------------------------------------------------
    p15 = ROOT / "Phase1-5"
    if p15.exists():
        cfgs = {"synthetic_coverage_balance_plan_v0.json",
                "synthetic_coverage_balanced_run_config_v0.json",
                "synthetic_population_manifest_v0.json",
                "synthetic_population_balanced_manifest_v0_1.json"}
        for f in tree_files(p15):
            name = f.name
            if f.suffix == ".py":
                P.add(f, ROOT / "scripts" / "synthetic_population" / name,
                      "move", "canonical synthetic population script")
            elif name in cfgs:
                P.add(f, ROOT / "configs" / "synthetic_population" / name,
                         "move", "canonical synthetic config/manifest")
            else:
                P.add(f, ROOT / "experiments" / "synthetic_population" / name,
                         "move", "canonical synthetic population outputs")

    # ----------------------------------------------------------------
    # 6) Phase2  (disease space)
    # ----------------------------------------------------------------
    p2 = ROOT / "Phase2"
    if p2.exists():
        P.add(p2 / "src" / "phase2_curate_disease_set.py",
              ROOT / "scripts" / "disease_mapping" / "phase2_curate_disease_set.py",
              "move", "canonical disease curation script")
        for name, dst in [
            ("hp.obo", ROOT / "data" / "raw" / "ontology" / "hp.obo"),
            ("phenotype.hpoa", ROOT / "data" / "raw" / "ontology" / "phenotype.hpoa"),
        ]:
            P.add(p2 / "data" / name, dst, "move", "canonical raw external ontology")
        cur = p2 / "phase2_curated_data"
        if cur.exists():
            for f in tree_files(cur):
                if f.name == "mondo_teacher_dist.npy":
                    P.add(f, ROOT / "artifacts" / "teacher_distances" / "mondo_teacher_dist.npy",
                          "move", "canonical teacher-distance matrix")
                elif f.name == "curation_report.md":
                    P.add(f, ROOT / "reports" / "validation" / "disease_curation_report.md",
                          "move", "canonical curation report")
                else:
                    P.add(f, ROOT / "data" / "processed" / "disease_mapping" / f.name,
                          "move", "canonical curated disease-space data")
        P.archive_tree(p2, "Phase2")

    # ----------------------------------------------------------------
    # 7) artifact/  (projection engine canonical split)
    # ----------------------------------------------------------------
    art = ROOT / "artifact"
    if art.exists():
        for f in tree_files(art):
            name = f.name
            if f.suffix == ".py":
                P.add(f, ROOT / "scripts" / "projection_engine" / name, "move",
                       "canonical projection engine script")
            elif name == "boxel_centers_aligned.npy":
                P.add(f, ROOT / "artifacts" / "ontology_embeddings" / name, "move",
                       "canonical ontology embedding artifact")
            elif name.startswith(("hpopp_", "phenotype_axes_", "test_registry_",
                                  "test_to_axis_", "scoring_rules_",
                                  "batch_phenotype_matrix", "vector_artifacts_")):
                P.add(f, ROOT / "data" / "processed" / "phenotype_axes" / name, "move",
                       "canonical curated phenotype-axes table")
            elif name.startswith(("sample_", "batch_user_projections")) or \
                    name in ("projection_engine_v0.validation_report.json",
                             "batch_projection_v0.validation_summary.json"):
                P.add(f, ROOT / "experiments" / "questionnaire_projection" / name, "move",
                       "canonical projection sample/validation artifact")
            else:  # logs, warnings
                P.add(f, ROOT / "reports" / "validation" / name, "move",
                       "canonical projection engine log")

    return P.ops


# ----------------------------------------------------------------------
# Collision handling + execution
# ----------------------------------------------------------------------

def dedupe_pairs(ops, dry: bool = False):
    by_dst: dict[str, list[dict]] = {}
    for op in ops:
        by_dst.setdefault(op["dst"], []).append(op)
    kept, archived = [], []
    for dst, group in by_dst.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        hashes: dict[str, list[dict]] = {}
        for op in group:
            p = ROOT / op["src"]
            h = sha256(p)
            hashes.setdefault(h, []).append(op)
        if len(hashes) == 1:
            kept.append(group[0])
            for op in group[1:]:
                archived.append(
                    {**op, "dst": str((ARCHIVE / "dedupe_refs" / h[:16] / op["src"]).resolve()),
                     "kind": "dedupe_archive",
                     "note": "byte-identical duplicate of " + group[0]["src"]})
        else:
            raise SystemExit(
                f"COLLISION on {dst} with differing content — manual review needed")
    return kept, archived


def execute(ops: list[dict]) -> None:
    moved, failed, identical = 0, [], 0
    for op in ops:
        src, dst = ROOT / op["src"], ROOT / op["dst"]
        if not src.exists():
            failed.append((op, "source missing"))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if os.path.isdir(src):
            if dst.exists():
                failed.append((op, "directory destination exists"))
                continue
            shutil.move(str(src), str(dst))
            moved += 1
            continue
        if dst.exists():
            if sha256(src) == sha256(dst):
                os.remove(src)
                identical += 1
                continue
            failed.append((op, "destination exists with different bytes"))
            continue
        h_before = sha256(src)
        shutil.move(str(src), str(dst))
        if sha256(dst) != h_before:
            failed.append((op, "checksum mismatch after move"))
            continue
        moved += 1
    if failed:
        print("FAILED:", len(failed))
        for op, why in failed:
            print("  ", op["src"], "->", op["dst"], "::", why)
        raise SystemExit(1)
    print(f"OK — {moved} moved, {identical} identical-removed, {len(ops)} total ops")


def verify() -> None:
    """Every file in the original inventory must be accounted for (moved/archived)."""
    with open(INVENTORY) as fh:
        rows = list(csv.DictReader(fh))
    stragglers = [r["rel_path"] for r in rows
                  if (ROOT / r["rel_path"]).exists()]
    print(f"verify: {len(rows)} inventoried; still at legacy path = {len(stragglers)}")
    for s in stragglers:
        print("  STILL-LEGACY:", s)


def write_manifest(ops: list[dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source_rel_path", "destination_rel_path", "action",
                    "sha256_source", "size_bytes", "note"])
        for op in sorted(ops, key=lambda o: o["src"]):
            p = ROOT / op["src"]
            try:
                h = sha256(p) if p.exists() else ""
                sz = os.path.getsize(p) if p.exists() else ""
            except OSError:
                h, sz = "", ""
            w.writerow([op["src"], op["dst"], op["kind"], h, sz, op["note"]])
    print("manifest written:", rel(MANIFEST))


def main() -> None:
    dry = "--dry-run" in sys.argv
    ops = plan()
    kept, archived = dedupe_pairs(ops)
    all_ops = kept + archived
    print(f"plan: {len(ops)} ops -> {len(kept)} moves, {len(archived)} dedupe-archives")
    if dry:
        for op in sorted(all_ops, key=lambda o: o["src"]):
            print(f"  {op['kind']:<8} {op['src']}  ->  {op['dst']}")
        return
    execute(all_ops)
    verify()
    write_manifest(all_ops)
    print("DONE")


if __name__ == "__main__":
    main()
