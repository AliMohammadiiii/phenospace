"""
anndata_to_pyg_v0.py
--------------------
Task 6 for the Cytognosis / Neuroverse AnnData pipeline.

Convert the graph-enriched AnnData object into PyTorch Geometric `Data` objects.
Each participant/session becomes one phenotype graph:

  nodes      = phenotype axes, aligned with adata.var
  edges      = adata.varp['phenotype_graph_combined_v0']
  node x     = phenotype semantic embedding + participant measurement + masks/time
  target y   = participant measurement vector, NaNs converted to 0
  masks      = observed_mask for loss masking

Default input:
  11_anndata/outputs/real_phenotype_data_v0_2_graph.h5ad

Usage smoke test:
  python 11_anndata/code/anndata_to_pyg_v0.py \
    --h5ad 11_anndata/outputs/real_phenotype_data_v0_2_graph.h5ad \
    --max-samples 8

Optional output:
  11_anndata/outputs/pyg_smoke_samples_v0.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import anndata as ad
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: anndata. Install with `pip install anndata`.") from e

try:
    from scipy import sparse
except Exception as e:  # pragma: no cover
    raise RuntimeError("Missing dependency: scipy. Install with `pip install scipy`.") from e

try:
    import torch
    from torch.utils.data import Dataset
    from torch_geometric.data import Data
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: torch and/or torch_geometric. Install PyTorch first, then PyG. "
        "Example: `pip install torch torch-geometric`."
    ) from e


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
def infer_repo_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "code" and here.parent.parent.name == "11_anndata":
        return here.parents[2]
    return Path.cwd()


REPO_ROOT = infer_repo_root()
OUT_DIR = REPO_ROOT / "11_anndata" / "outputs"
DEFAULT_H5AD = OUT_DIR / "real_phenotype_data_v0_2_graph.h5ad"
DEFAULT_OUT_PT = OUT_DIR / "pyg_smoke_samples_v0.pt"
DEFAULT_META_JSON = OUT_DIR / "pyg_dataset_meta_v0.json"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[anndata_to_pyg_v0] {msg}", flush=True)


def as_dense_array(M) -> np.ndarray:
    if sparse.issparse(M):
        return M.toarray()
    return np.asarray(M)


def graph_to_edge_index_and_attr(graph) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert adata.varp graph into PyG edge_index and edge_attr."""
    if not sparse.issparse(graph):
        graph = sparse.csr_matrix(np.asarray(graph, dtype=np.float32))
    coo = graph.tocoo()
    edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    edge_attr = torch.tensor(coo.data[:, None], dtype=torch.float32)
    return edge_index, edge_attr


def robust_scale_raw_time(raw_time: np.ndarray) -> np.ndarray:
    """Log1p + column robust scaling for RT values; NaNs become 0."""
    T = np.asarray(raw_time, dtype=np.float32).copy()
    T = np.log1p(T)
    med = np.nanmedian(T, axis=0)
    q25 = np.nanpercentile(T, 25, axis=0)
    q75 = np.nanpercentile(T, 75, axis=0)
    iqr = q75 - q25
    iqr = np.where(np.isfinite(iqr) & (iqr > 1e-6), iqr, 1.0)
    T = (T - med) / iqr
    T = np.where(np.isfinite(T), T, 0.0)
    return T.astype(np.float32)


def load_matrix_layer(adata, layer: str) -> np.ndarray:
    if layer == "X":
        return as_dense_array(adata.X).astype(np.float32)
    if layer not in adata.layers:
        raise KeyError(f"layer {layer!r} not found. Available layers: {list(adata.layers.keys())}")
    return as_dense_array(adata.layers[layer]).astype(np.float32)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class AnnDataPhenotypeGraphDataset(Dataset):
    """Participant/session-level PyG dataset from graph-enriched AnnData.

    The graph topology is shared across all participants. Each sample overlays
    that participant/session's phenotype-axis measurements onto the phenotype
    graph nodes.
    """

    def __init__(
        self,
        h5ad_path: str | Path,
        graph_key: str = "phenotype_graph_combined_v0",
        embedding_key: str = "phenotype_embedding_initial",
        measurement_layer: str = "X",
        mask_layer: str = "mask",
        include_observed_mask_feature: bool = True,
        include_raw_time_feature: bool = True,
        raw_time_layer: str = "raw_time",
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.h5ad_path = Path(h5ad_path)
        self.adata = ad.read_h5ad(self.h5ad_path)

        if graph_key not in self.adata.varp:
            raise KeyError(f"graph_key {graph_key!r} not found in adata.varp: {list(self.adata.varp.keys())}")
        if embedding_key not in self.adata.varm:
            raise KeyError(f"embedding_key {embedding_key!r} not found in adata.varm: {list(self.adata.varm.keys())}")
        if mask_layer not in self.adata.layers:
            raise KeyError(f"mask_layer {mask_layer!r} not found in adata.layers: {list(self.adata.layers.keys())}")

        self.graph_key = graph_key
        self.embedding_key = embedding_key
        self.measurement_layer = measurement_layer
        self.mask_layer = mask_layer
        self.include_observed_mask_feature = include_observed_mask_feature
        self.include_raw_time_feature = include_raw_time_feature
        self.raw_time_layer = raw_time_layer

        self.axis_ids = list(map(str, self.adata.var.index))
        self.obs_ids = list(map(str, self.adata.obs.index))

        self.edge_index, self.edge_attr = graph_to_edge_index_and_attr(self.adata.varp[graph_key])
        self.embedding = torch.tensor(np.asarray(self.adata.varm[embedding_key], dtype=np.float32), dtype=torch.float32)
        self.embedding_dim = int(self.embedding.shape[1])

        self.measurement = load_matrix_layer(self.adata, measurement_layer)
        self.target = np.where(np.isfinite(self.measurement), self.measurement, 0.0).astype(np.float32)
        self.observed_mask = load_matrix_layer(self.adata, mask_layer).astype(bool)

        if include_raw_time_feature:
            if raw_time_layer not in self.adata.layers:
                raise KeyError(f"raw_time_layer {raw_time_layer!r} not found in adata.layers")
            self.raw_time = robust_scale_raw_time(load_matrix_layer(self.adata, raw_time_layer))
        else:
            self.raw_time = None

        self.n_total = self.adata.n_obs
        self.n = min(self.n_total, int(max_samples)) if max_samples is not None else self.n_total

        # Position of the participant measurement scalar inside Data.x.
        # x = [embedding..., measurement, observed_mask?, raw_time?]
        self.measurement_col = self.embedding_dim
        self.num_node_features = self.embedding_dim + 1
        if include_observed_mask_feature:
            self.num_node_features += 1
        if include_raw_time_feature:
            self.num_node_features += 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Data:
        if idx < 0 or idx >= self.n:
            raise IndexError(idx)

        meas = self.measurement[idx].astype(np.float32)
        y = self.target[idx].astype(np.float32)
        obs_mask = self.observed_mask[idx].astype(bool)
        meas_visible = np.where(np.isfinite(meas), meas, 0.0).astype(np.float32)

        pieces = [self.embedding]
        pieces.append(torch.tensor(meas_visible[:, None], dtype=torch.float32))
        if self.include_observed_mask_feature:
            pieces.append(torch.tensor(obs_mask.astype(np.float32)[:, None], dtype=torch.float32))
        if self.include_raw_time_feature and self.raw_time is not None:
            pieces.append(torch.tensor(self.raw_time[idx][:, None], dtype=torch.float32))

        x = torch.cat(pieces, dim=1)
        data = Data(
            x=x,
            edge_index=self.edge_index.clone(),
            edge_attr=self.edge_attr.clone(),
            y=torch.tensor(y, dtype=torch.float32),
            observed_mask=torch.tensor(obs_mask, dtype=torch.bool),
            obs_idx=torch.tensor([idx], dtype=torch.long),
        )
        return data

    def metadata(self) -> dict:
        return {
            "h5ad_path": str(self.h5ad_path),
            "n_samples_available": int(self.n_total),
            "n_samples_exposed": int(self.n),
            "n_nodes": int(len(self.axis_ids)),
            "n_edges_directed": int(self.edge_index.shape[1]),
            "graph_key": self.graph_key,
            "embedding_key": self.embedding_key,
            "embedding_dim": int(self.embedding_dim),
            "measurement_layer": self.measurement_layer,
            "measurement_col_in_x": int(self.measurement_col),
            "include_observed_mask_feature": bool(self.include_observed_mask_feature),
            "include_raw_time_feature": bool(self.include_raw_time_feature),
            "axis_ids": self.axis_ids,
        }


# ---------------------------------------------------------------------
# Masking helper for the next training step
# ---------------------------------------------------------------------
def make_masked_copy(data: Data, measurement_col: int, mask_fraction: float = 0.25) -> Data:
    """Return a copy with a random subset of observed node measurements hidden.

    Adds:
      training_mask : bool per node, True where the model should predict
      x[:, measurement_col] is set to 0 where training_mask is True

    Works on a single Data object or a PyG Batch, because observed_mask and x
    are still node-aligned after batching.
    """
    if not (0.0 < mask_fraction <= 1.0):
        raise ValueError("mask_fraction must be in (0, 1]")

    out = data.clone()
    observed_idx = torch.where(out.observed_mask)[0]
    training_mask = torch.zeros_like(out.observed_mask, dtype=torch.bool)

    if observed_idx.numel() > 0:
        n_mask = max(1, int(round(float(observed_idx.numel()) * mask_fraction)))
        perm = observed_idx[torch.randperm(observed_idx.numel(), device=observed_idx.device)[:n_mask]]
        training_mask[perm] = True
        out.x[training_mask, measurement_col] = 0.0

    out.training_mask = training_mask
    return out


# ---------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    p.add_argument("--graph-key", type=str, default="phenotype_graph_combined_v0")
    p.add_argument("--embedding-key", type=str, default="phenotype_embedding_initial")
    p.add_argument("--measurement-layer", type=str, default="X")
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--out-pt", type=Path, default=DEFAULT_OUT_PT)
    p.add_argument("--meta-json", type=Path, default=DEFAULT_META_JSON)
    p.add_argument("--no-raw-time", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ds = AnnDataPhenotypeGraphDataset(
        h5ad_path=args.h5ad,
        graph_key=args.graph_key,
        embedding_key=args.embedding_key,
        measurement_layer=args.measurement_layer,
        include_raw_time_feature=not args.no_raw_time,
        max_samples=args.max_samples,
    )
    log(json.dumps(ds.metadata(), indent=2)[:2000])

    samples = [ds[i] for i in range(len(ds))]
    for i, d in enumerate(samples[:3]):
        log(
            f"sample {i}: x={tuple(d.x.shape)} edge_index={tuple(d.edge_index.shape)} "
            f"edge_attr={tuple(d.edge_attr.shape)} y={tuple(d.y.shape)} "
            f"observed={int(d.observed_mask.sum())}/{d.num_nodes}"
        )

    args.out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(samples, args.out_pt)
    args.meta_json.write_text(json.dumps(ds.metadata(), indent=2))
    log(f"wrote smoke samples: {args.out_pt}")
    log(f"wrote metadata: {args.meta_json}")


if __name__ == "__main__":
    main()
