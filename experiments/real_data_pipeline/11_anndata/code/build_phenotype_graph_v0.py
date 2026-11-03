"""
build_phenotype_graph_v0.py
----------------------------
Task 5 for the Cytognosis / Neuroverse AnnData pipeline.

This script populates phenotype-to-phenotype graph slots in `adata.varp` and
writes an edge table for audit/debugging.

Inputs, by default:
  11_anndata/outputs/real_phenotype_data_v0.h5ad
  11_anndata/outputs/phenotype_var_table_v0.csv
  11_anndata/outputs/phenotype_embeddings_text_v0.csv or .npy, if available

Outputs, by default:
  11_anndata/outputs/real_phenotype_data_v0_2_graph.h5ad
  11_anndata/outputs/phenotype_graph_edges_v0.csv
  11_anndata/outputs/phenotype_graph_meta_v0.json

What is added:
  adata.varm['phenotype_embedding_text_v0']       # if loaded/generated
  adata.varm['phenotype_embedding_initial']       # initial semantic embedding

  adata.varp['phenotype_similarity_text_v0']
  adata.varp['phenotype_similarity_archetype_v0']
  adata.varp['phenotype_similarity_empirical_corr_pos_v0']
  adata.varp['phenotype_similarity_ontology_v0']
  adata.varp['phenotype_similarity_source_v0']
  adata.varp['phenotype_graph_combined_v0']       # main graph for PyG
  adata.varp['phenotype_graph_binary_v0']

Usage:
  python 11_anndata/code/build_phenotype_graph_v0.py

Optional:
  python build_phenotype_graph_v0.py \
    --h5ad 11_anndata/outputs/real_phenotype_data_v0.h5ad \
    --var-table 11_anndata/outputs/phenotype_var_table_v0.csv \
    --out-h5ad 11_anndata/outputs/real_phenotype_data_v0_2_graph.h5ad \
    --k 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: anndata. Install in your project venv, e.g. `pip install anndata`."
    ) from e

try:
    from scipy import sparse
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: scipy. Install in your project venv, e.g. `pip install scipy`."
    ) from e


# ---------------------------------------------------------------------
# Defaults follow the same repo layout as your previous scripts.
# ---------------------------------------------------------------------
def infer_repo_root() -> Path:
    """Infer repo root for scripts located under 11_anndata/code/."""
    here = Path(__file__).resolve()
    if here.parent.name == "code" and here.parent.parent.name == "11_anndata":
        return here.parents[2]
    return Path.cwd()


REPO_ROOT = infer_repo_root()
OUT_DIR = REPO_ROOT / "11_anndata" / "outputs"

DEFAULT_H5AD = OUT_DIR / "real_phenotype_data_v0.h5ad"
DEFAULT_VAR_TABLE = OUT_DIR / "phenotype_var_table_v0.csv"
DEFAULT_TEXT_CSV = OUT_DIR / "phenotype_embeddings_text_v0.csv"
DEFAULT_TEXT_NPY = OUT_DIR / "phenotype_embeddings_text_v0.npy"
DEFAULT_TEXT_META = OUT_DIR / "phenotype_embeddings_meta_v0.json"
DEFAULT_OUT_H5AD = OUT_DIR / "real_phenotype_data_v0_2_graph.h5ad"
DEFAULT_EDGE_CSV = OUT_DIR / "phenotype_graph_edges_v0.csv"
DEFAULT_META_JSON = OUT_DIR / "phenotype_graph_meta_v0.json"

GRAPH_KEY = "phenotype_graph_combined_v0"


# ---------------------------------------------------------------------
# Logging / parsing helpers
# ---------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[build_phenotype_graph_v0] {msg}", flush=True)


def split_terms(value: object) -> set[str]:
    """Split semicolon/comma/pipe-delimited ontology/test fields into normalized terms."""
    if value is None:
        return set()
    if isinstance(value, float) and math.isnan(value):
        return set()
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return set()
    raw = re.split(r"[;,|]", s)
    out: set[str] = set()
    for x in raw:
        x = re.sub(r"\s+", " ", x).strip().lower()
        if x and x not in {"nan", "none", "null"}:
            out.add(x)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def minmax01(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    M = np.where(np.isfinite(M), M, 0.0)
    mn = float(np.min(M))
    mx = float(np.max(M))
    if mx <= mn:
        return np.zeros_like(M, dtype=np.float32)
    return ((M - mn) / (mx - mn)).astype(np.float32)


def row_l2_normalize(E: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    E = np.asarray(E, dtype=np.float32)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.maximum(norms, eps)


def cosine01(E: np.ndarray) -> np.ndarray:
    """Cosine similarity clipped to [0, 1], diagonal set to 0."""
    En = row_l2_normalize(E)
    S = En @ En.T
    S = np.clip(S, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(S, 0.0)
    return S


# ---------------------------------------------------------------------
# Deterministic fallback text embedding
# ---------------------------------------------------------------------
def axis_text(row: pd.Series) -> str:
    parts = [
        str(row.get("phenotype_axis_name") or ""),
        str(row.get("description") or ""),
        f"aliases: {row.get('aliases') or ''}",
        f"hpo: {row.get('hpo_label') or ''} | parents: {row.get('hpo_parents') or ''}",
        f"rdoc: {row.get('rdoc_domain') or ''} / {row.get('rdoc_construct') or ''}",
        f"source_test: {row.get('source_test') or ''}",
        f"source_item: {row.get('source_item') or ''}",
    ]
    text = " . ".join(p for p in parts if p and p.lower() not in {"nan", "none"})
    return re.sub(r"\s+", " ", text).strip()


def hashed_features(text: str, dim: int = 256) -> np.ndarray:
    """Stable md5 hashing-trick embedding used only if no text embedding file exists."""
    v = np.zeros(dim, dtype=np.float32)
    text_l = text.lower()
    feats = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text_l)
    padded = f" {text_l} "
    for n in (2, 3, 4):
        feats.extend(padded[i : i + n] for i in range(max(0, len(padded) - n + 1)))
    for f in feats:
        h = hashlib.md5(f.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if (h[4] & 1) else -1.0
        v[idx] += sign
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v /= norm
    return v


# ---------------------------------------------------------------------
# Loading metadata / embeddings
# ---------------------------------------------------------------------
def load_var_table(path: Path, adata_var: pd.DataFrame) -> pd.DataFrame:
    """Load enriched phenotype var table; fallback to adata.var if unavailable."""
    axis_ids = list(map(str, adata_var.index))
    if path.exists():
        log(f"loading enriched var table: {path}")
        var = pd.read_csv(path)
        if "phenotype_axis_id" in var.columns:
            var = var.set_index("phenotype_axis_id")
        var.index = var.index.astype(str)
        missing = [a for a in axis_ids if a not in var.index]
        if missing:
            raise RuntimeError(f"var table is missing {len(missing)} axes: {missing[:5]}")
        return var.reindex(axis_ids)

    log(f"var table not found at {path}; using adata.var only")
    var = adata_var.copy()
    var.index = var.index.astype(str)
    return var.reindex(axis_ids)


def load_text_embedding(
    axis_ids: list[str],
    var: pd.DataFrame,
    text_csv: Path,
    text_npy: Path,
    meta_json: Path,
    adata,
) -> tuple[np.ndarray, str]:
    """Load text embedding if available; otherwise use h5ad varm; otherwise hash text."""
    n = len(axis_ids)

    if text_csv.exists():
        log(f"loading text embeddings CSV: {text_csv}")
        emb_df = pd.read_csv(text_csv, index_col=0)
        emb_df.index = emb_df.index.astype(str)
        missing = [a for a in axis_ids if a not in emb_df.index]
        if missing:
            raise RuntimeError(f"text embedding CSV missing axes: {missing[:5]}")
        E = emb_df.reindex(axis_ids).to_numpy(dtype=np.float32)
        return row_l2_normalize(E), "phenotype_embeddings_text_v0.csv"

    if text_npy.exists():
        log(f"loading text embeddings NPY: {text_npy}")
        E = np.load(text_npy).astype(np.float32)
        if E.shape[0] != n:
            raise RuntimeError(f"text NPY row count {E.shape[0]} != n_axes {n}")
        if meta_json.exists():
            try:
                meta = json.loads(meta_json.read_text())
                meta_axis = list(map(str, meta.get("axis_ids", [])))
                if meta_axis and meta_axis != axis_ids:
                    log("warning: meta axis_ids differ from h5ad var order; assuming NPY already aligned")
            except Exception:
                pass
        return row_l2_normalize(E), "phenotype_embeddings_text_v0.npy"

    if "phenotype_embedding_initial" in adata.varm:
        log("using existing adata.varm['phenotype_embedding_initial']")
        E = np.asarray(adata.varm["phenotype_embedding_initial"], dtype=np.float32)
        return row_l2_normalize(E), "adata.varm['phenotype_embedding_initial']"

    log("no text embedding file found; generating deterministic hashed text embeddings")
    texts = [axis_text(var.loc[a]) for a in axis_ids]
    E = np.stack([hashed_features(t, dim=256) for t in texts], axis=0)
    return row_l2_normalize(E), "hashed_axis_text_dim256"


# ---------------------------------------------------------------------
# Similarity matrices
# ---------------------------------------------------------------------
def pairwise_jaccard_matrix(var: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    n = len(var)
    term_sets: list[set[str]] = []
    for _, row in var.iterrows():
        terms: set[str] = set()
        for c in columns:
            if c in row.index:
                terms |= split_terms(row.get(c))
        term_sets.append(terms)

    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            val = jaccard(term_sets[i], term_sets[j])
            M[i, j] = M[j, i] = val
    np.fill_diagonal(M, 0.0)
    return M


def rdoc_similarity_matrix(var: pd.DataFrame) -> np.ndarray:
    n = len(var)
    M = np.zeros((n, n), dtype=np.float32)
    domains = var.get("rdoc_domain", pd.Series([""] * n, index=var.index)).fillna("").astype(str).str.lower()
    constructs = var.get("rdoc_construct", pd.Series([""] * n, index=var.index)).fillna("").astype(str).str.lower()
    for i in range(n):
        for j in range(i + 1, n):
            score = 0.0
            if domains.iloc[i] and domains.iloc[i] == domains.iloc[j]:
                score += 0.5
            if constructs.iloc[i] and constructs.iloc[i] == constructs.iloc[j]:
                score += 0.5
            M[i, j] = M[j, i] = score
    return M


def empirical_corr_pos(adata, layer: str = "imputed") -> np.ndarray:
    if layer in adata.layers:
        X = np.asarray(adata.layers[layer], dtype=np.float32)
    else:
        X = np.asarray(adata.X, dtype=np.float32)
        # column-mean impute locally
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        rows, cols = np.where(np.isnan(X))
        X = X.copy()
        X[rows, cols] = col_mean[cols]

    C = np.corrcoef(X, rowvar=False).astype(np.float32)
    C = np.where(np.isfinite(C), C, 0.0)
    C = np.clip(C, 0.0, 1.0)
    np.fill_diagonal(C, 0.0)
    return C


def knn_symmetric(S: np.ndarray, k: int = 6, min_weight: float = 0.01) -> sparse.csr_matrix:
    """Keep top-k outgoing edges per node, then symmetrize by max weight."""
    S = np.asarray(S, dtype=np.float32).copy()
    np.fill_diagonal(S, 0.0)
    n = S.shape[0]
    A = np.zeros_like(S, dtype=np.float32)
    for i in range(n):
        row = S[i].copy()
        row[row < min_weight] = 0.0
        if np.count_nonzero(row) == 0:
            # Fallback: keep best non-self neighbor even if below threshold.
            j = int(np.argmax(S[i]))
            if j != i and S[i, j] > 0:
                A[i, j] = S[i, j]
            continue
        idx = np.argsort(row)[::-1][:k]
        for j in idx:
            if j != i and row[j] > 0:
                A[i, j] = row[j]
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return sparse.csr_matrix(A)


def edge_type_from_scores(text: float, ont: float, source: float, arch: float, corr: float) -> str:
    tags = []
    if text >= 0.50:
        tags.append("semantic_similarity")
    if ont >= 0.05:
        tags.append("ontology_overlap")
    if source >= 0.05:
        tags.append("same_source_test_or_item")
    if arch >= 0.50:
        tags.append("archetype_loading_similarity")
    if corr >= 0.20:
        tags.append("empirical_positive_correlation")
    return ";".join(tags) if tags else "combined_low_evidence"


def make_edge_table(
    A: sparse.csr_matrix,
    axis_ids: list[str],
    var: pd.DataFrame,
    S_text: np.ndarray,
    S_ont: np.ndarray,
    S_source: np.ndarray,
    S_arch: np.ndarray,
    S_corr: np.ndarray,
) -> pd.DataFrame:
    labels = var.get("phenotype_axis_name", pd.Series(axis_ids, index=axis_ids)).astype(str)
    coo = A.tocoo()
    rows = []
    seen = set()
    for i, j, w in zip(coo.row, coo.col, coo.data):
        if i == j:
            continue
        a, b = sorted((int(i), int(j)))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append(
            {
                "source_phenotype_axis_id": axis_ids[a],
                "target_phenotype_axis_id": axis_ids[b],
                "source_phenotype_axis_name": labels.iloc[a],
                "target_phenotype_axis_name": labels.iloc[b],
                "edge_weight": float(A[a, b]),
                "edge_type": edge_type_from_scores(
                    float(S_text[a, b]),
                    float(S_ont[a, b]),
                    float(S_source[a, b]),
                    float(S_arch[a, b]),
                    float(S_corr[a, b]),
                ),
                "evidence_text_similarity": float(S_text[a, b]),
                "evidence_ontology_overlap": float(S_ont[a, b]),
                "evidence_source_overlap": float(S_source[a, b]),
                "evidence_archetype_similarity": float(S_arch[a, b]),
                "evidence_empirical_corr_pos": float(S_corr[a, b]),
            }
        )
    out = pd.DataFrame(rows).sort_values("edge_weight", ascending=False)
    return out


# ---------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------
def build(args: argparse.Namespace) -> None:
    log(f"reading h5ad: {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    axis_ids = list(map(str, adata.var.index))
    n = len(axis_ids)
    log(f"AnnData shape: {adata.n_obs} obs × {adata.n_vars} vars")

    var = load_var_table(args.var_table, adata.var)

    # Load/generate text embedding and write it into varm.
    E_text, text_source = load_text_embedding(
        axis_ids=axis_ids,
        var=var,
        text_csv=args.text_csv,
        text_npy=args.text_npy,
        meta_json=args.text_meta,
        adata=adata,
    )
    if E_text.shape[0] != n:
        raise RuntimeError(f"text embedding rows {E_text.shape[0]} != n_vars {n}")
    adata.varm["phenotype_embedding_text_v0"] = E_text.astype(np.float32)
    adata.varm["phenotype_embedding_initial"] = E_text.astype(np.float32)

    # Build individual evidence matrices.
    S_text = cosine01(E_text)

    if "archetype_loadings_k13" in adata.varm:
        S_arch = cosine01(np.asarray(adata.varm["archetype_loadings_k13"], dtype=np.float32))
    else:
        S_arch = np.zeros((n, n), dtype=np.float32)

    S_corr = empirical_corr_pos(adata, layer=args.empirical_layer)

    S_ont = pairwise_jaccard_matrix(
        var,
        columns=["hpo_id", "hpo_label", "hpo_parents", "ontology_neighbors", "aliases"],
    )
    # HPO/RDoC signals are complementary; keep them in the same ontology-ish block.
    S_ont = np.maximum(S_ont, rdoc_similarity_matrix(var))
    S_ont = np.clip(S_ont, 0.0, 1.0)

    S_source = pairwise_jaccard_matrix(var, columns=["source_test", "source_item", "source_subscale"])
    S_source = np.clip(S_source, 0.0, 1.0)

    # Combined evidence. All pieces are 0-1 and diagonal-free.
    weights = {
        "text": float(args.w_text),
        "ontology": float(args.w_ontology),
        "source": float(args.w_source),
        "archetype": float(args.w_archetype),
        "empirical_corr_pos": float(args.w_empirical),
    }
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("at least one graph weight must be > 0")
    weights = {k: v / total_w for k, v in weights.items()}

    S_combined = (
        weights["text"] * S_text
        + weights["ontology"] * S_ont
        + weights["source"] * S_source
        + weights["archetype"] * S_arch
        + weights["empirical_corr_pos"] * S_corr
    ).astype(np.float32)
    np.fill_diagonal(S_combined, 0.0)

    # Store dense evidence matrices as sparse CSR matrices.
    adata.varp["phenotype_similarity_text_v0"] = sparse.csr_matrix(S_text)
    adata.varp["phenotype_similarity_archetype_v0"] = sparse.csr_matrix(S_arch)
    adata.varp["phenotype_similarity_empirical_corr_pos_v0"] = sparse.csr_matrix(S_corr)
    adata.varp["phenotype_similarity_ontology_v0"] = sparse.csr_matrix(S_ont)
    adata.varp["phenotype_similarity_source_v0"] = sparse.csr_matrix(S_source)

    A = knn_symmetric(S_combined, k=args.k, min_weight=args.min_weight)
    A_binary = A.copy()
    A_binary.data = np.ones_like(A_binary.data, dtype=np.float32)
    adata.varp[GRAPH_KEY] = A.astype(np.float32)
    adata.varp["phenotype_graph_binary_v0"] = A_binary.astype(np.float32)

    edge_df = make_edge_table(A, axis_ids, var, S_text, S_ont, S_source, S_arch, S_corr)
    args.edge_csv.parent.mkdir(parents=True, exist_ok=True)
    edge_df.to_csv(args.edge_csv, index=False)
    log(f"wrote edge table: {args.edge_csv} ({len(edge_df)} undirected edges)")

    # Keep graph metadata in uns.
    degrees = np.asarray((A > 0).sum(axis=1)).ravel().astype(int)
    meta = {
        "schema_version": "0.2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "graph_key": GRAPH_KEY,
        "binary_graph_key": "phenotype_graph_binary_v0",
        "n_nodes": int(n),
        "n_undirected_edges": int(len(edge_df)),
        "knn_k": int(args.k),
        "min_weight": float(args.min_weight),
        "weights": weights,
        "text_embedding_source": text_source,
        "evidence_matrices": [
            "phenotype_similarity_text_v0",
            "phenotype_similarity_archetype_v0",
            "phenotype_similarity_empirical_corr_pos_v0",
            "phenotype_similarity_ontology_v0",
            "phenotype_similarity_source_v0",
        ],
        "degree_summary": {
            "min": int(degrees.min()),
            "max": int(degrees.max()),
            "mean": float(degrees.mean()),
            "median": float(np.median(degrees)),
        },
        "edge_csv": str(args.edge_csv),
        "notes": (
            "Phenotype graph built from semantic text similarity, ontology/RDoC overlap, "
            "source-test/item overlap, archetype-loading similarity, and positive empirical "
            "axis correlation over the imputed real-data matrix. The main downstream graph "
            "slot is adata.varp['phenotype_graph_combined_v0']."
        ),
    }
    adata.uns["phenotype_graph_v0"] = meta
    adata.uns["schema_version"] = "0.2.0"
    adata.uns["dataset_version"] = "real_phenotype_anndata_v0_2_graph"

    args.meta_json.parent.mkdir(parents=True, exist_ok=True)
    args.meta_json.write_text(json.dumps(meta, indent=2))
    log(f"wrote graph metadata: {args.meta_json}")

    args.out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    log(f"writing graph-enriched h5ad: {args.out_h5ad}")
    adata.write_h5ad(args.out_h5ad, compression="gzip", compression_opts=4)
    log("DONE")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    p.add_argument("--var-table", type=Path, default=DEFAULT_VAR_TABLE)
    p.add_argument("--text-csv", type=Path, default=DEFAULT_TEXT_CSV)
    p.add_argument("--text-npy", type=Path, default=DEFAULT_TEXT_NPY)
    p.add_argument("--text-meta", type=Path, default=DEFAULT_TEXT_META)
    p.add_argument("--out-h5ad", type=Path, default=DEFAULT_OUT_H5AD)
    p.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    p.add_argument("--meta-json", type=Path, default=DEFAULT_META_JSON)
    p.add_argument("--empirical-layer", type=str, default="imputed")
    p.add_argument("--k", type=int, default=6, help="top-k neighbors kept per phenotype node before symmetrization")
    p.add_argument("--min-weight", type=float, default=0.01)
    p.add_argument("--w-text", type=float, default=0.45)
    p.add_argument("--w-ontology", type=float, default=0.20)
    p.add_argument("--w-source", type=float, default=0.10)
    p.add_argument("--w-archetype", type=float, default=0.15)
    p.add_argument("--w-empirical", type=float, default=0.10)
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
