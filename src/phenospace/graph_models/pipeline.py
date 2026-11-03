"""Inductive masked reconstruction, adapted from the historical v0.1 pipeline.

Historical sources: experiments/real_data_pipeline/11_anndata/code/
{anndata_to_pyg_v0,aanet_bottleneck_gnn_v0_1,build_phenotype_graph_v0}.py.
This maintained path deliberately changes the batch-hull inference contract.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool


def grouped_split(ids, groups, *, seed=42, test_fraction=0.25):
    """Split whole groups before fitting; callers choose the appropriate unit."""
    ids, groups = list(ids), list(groups)
    if not ids or len(ids) != len(groups) or len(set(ids)) != len(ids):
        raise ValueError("IDs and groups must be nonempty, aligned and IDs unique")
    if not 0 < test_fraction < 1:
        raise ValueError("test fraction must be between zero and one")
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("at least two groups are required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_test = min(len(unique) - 1, max(1, round(len(unique) * test_fraction)))
    test_groups = {unique[i] for i in order[:n_test]}
    return (
        [x for x, g in zip(ids, groups, strict=True) if g not in test_groups],
        [x for x, g in zip(ids, groups, strict=True) if g in test_groups],
    )


def fit_preprocessing(adata, train_ids):
    """Fit axis mean, scale and correlation graph on named training rows only."""
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("observation and axis IDs must be unique")
    if adata.n_obs == 0 or adata.n_vars < 2:
        raise ValueError("need observations and at least two axes")
    ids = list(train_ids)
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(adata.obs_names):
        raise ValueError("training IDs must be unique and present")
    x = adata[ids].X
    x = x.toarray() if sparse.issparse(x) else np.asarray(x)
    x = np.asarray(x, dtype=float)
    if np.isinf(x).any() or np.any(np.all(np.isnan(x), axis=0)):
        raise ValueError("training axes must contain finite observations")
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale[scale < 1e-8] = 1.0
    z = np.where(np.isnan(x), 0.0, (x - mean) / scale)
    corr = np.nan_to_num(np.corrcoef(z, rowvar=False), nan=0.0)
    np.fill_diagonal(corr, 0)
    # Nonnegative correlation weights; isolated axes remain legal due to GCN self-loops.
    weights = np.maximum(corr, 0)
    src, dst = np.nonzero(weights > 0.1)
    return {
        "axis_ids": list(adata.var_names),
        "train_ids": ids,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "edge_index": [src.tolist(), dst.tolist()],
        "edge_weight": weights[src, dst].tolist(),
    }


class AnnDataPhenotypeGraphDataset:
    """One graph per row; NaNs are missing, with explicit held-out node masks."""

    def __init__(self, adata, fitted, hidden):
        if not adata.obs_names.is_unique or not adata.var_names.is_unique:
            raise ValueError("observation and axis IDs must be unique")
        if list(adata.var_names) != fitted["axis_ids"]:
            raise ValueError("axis order differs from fitted preprocessing")
        self.x = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        self.x = np.asarray(self.x, dtype=float)
        self.hidden = np.asarray(hidden, dtype=bool)
        if not self.x.size or self.hidden.shape != self.x.shape or np.isinf(self.x).any():
            raise ValueError("invalid matrix or hidden mask")
        self.observed = np.isfinite(self.x)
        if np.any(self.hidden & ~self.observed):
            raise ValueError("cannot hide an unobserved target")
        self.mean = np.asarray(fitted["mean"])
        self.scale = np.asarray(fitted["scale"])
        if (
            self.mean.shape != (adata.n_vars,)
            or self.scale.shape != (adata.n_vars,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.scale).all()
            or np.any(self.scale <= 0)
        ):
            raise ValueError("invalid fitted scaling dimensions or values")
        self.edges = torch.as_tensor(fitted["edge_index"], dtype=torch.long).reshape(2, -1)
        self.weights = torch.as_tensor(fitted["edge_weight"], dtype=torch.float32)
        if (
            self.edges.shape[1] != len(self.weights)
            or (self.edges.numel() and (self.edges.min() < 0 or self.edges.max() >= adata.n_vars))
            or not torch.isfinite(self.weights).all()
            or (self.weights < 0).any()
        ):
            raise ValueError("invalid graph endpoints or weights")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        obs = self.observed[index]
        hidden = self.hidden[index]
        values = np.where(obs, (self.x[index] - self.mean) / self.scale, 0)
        visible = obs & ~hidden
        features = np.column_stack([np.where(visible, values, 0), visible.astype(float)])
        return Data(
            x=torch.tensor(features, dtype=torch.float32),
            y=torch.tensor(values, dtype=torch.float32),
            target_mask=torch.tensor(hidden),
            edge_index=self.edges,
            edge_weight=self.weights,
        )


def masked_mse(pred, y, mask):
    if not mask.any():
        raise ValueError("mask selects no observed targets")
    return F.mse_loss(pred[mask], y[mask])


class AANetStyleGraphBottleneckGNN(nn.Module):
    """GCN plus convex archetypes over fixed train-derived graph references.

    Edge weights are passed to GCNConv; no SAGE mode is offered. Fixed reference
    embeddings are refreshed during training and persisted for independent inference.
    """

    def __init__(self, hidden_dim=16, n_archetypes=3, use_graph=True, use_bottleneck=True):
        super().__init__()
        self.use_graph = use_graph
        self.use_bottleneck = use_bottleneck
        self.conv = GCNConv(2, hidden_dim)
        self.alpha = nn.Linear(hidden_dim, n_archetypes)
        self.queries = nn.Parameter(torch.randn(n_archetypes, hidden_dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.register_buffer("references", torch.empty(0, hidden_dim))

    def encode(self, data):
        edges = data.edge_index if self.use_graph else data.edge_index[:, :0]
        weights = data.edge_weight if self.use_graph else data.edge_weight[:0]
        nodes = F.relu(self.conv(data.x, edges, weights))
        return nodes, global_mean_pool(nodes, data.batch)

    def set_references(self, graphs):
        with torch.no_grad():
            self.references = self.encode(graphs)[1].detach().clone()

    def forward(self, data):
        nodes, graph = self.encode(data)
        if not len(self.references):
            raise ValueError("training references must be set before inference")
        beta = torch.softmax(
            self.queries @ self.references.T / self.references.shape[1] ** 0.5, dim=1
        )
        archetypes = beta @ self.references
        context = torch.softmax(self.alpha(graph), dim=1) @ archetypes
        if not self.use_bottleneck:
            context = graph
        return self.decoder(torch.cat((nodes, context[data.batch]), dim=1)).squeeze(-1)
