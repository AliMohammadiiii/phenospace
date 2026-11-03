"""
train_subgraph_masking_gnn_v0.py
--------------------------------
Task 7-8 baseline: train a PyTorch Geometric GNN to reconstruct masked
phenotype-axis observations.

This is intentionally a baseline, not the final Cytognosis model:
  - graph topology comes from adata.varp['phenotype_graph_combined_v0']
  - node features come from AnnDataPhenotypeGraphDataset
  - a random subset of observed node measurements is hidden per batch
  - loss is MSE only on those hidden, originally-observed nodes

Usage:
  python 11_anndata/code/train_subgraph_masking_gnn_v0.py \
    --h5ad 11_anndata/outputs/real_phenotype_data_v0_2_graph.h5ad \
    --epochs 20 \
    --batch-size 128

Outputs:
  11_anndata/outputs/subgraph_masking_gnn_v0.pt
  11_anndata/outputs/subgraph_masking_gnn_metrics_v0.csv
  11_anndata/outputs/subgraph_masking_gnn_config_v0.json
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GCNConv, SAGEConv
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: torch and/or torch_geometric. Install PyTorch first, then PyG."
    ) from e

from anndata_to_pyg_v0 import AnnDataPhenotypeGraphDataset, make_masked_copy


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
DEFAULT_CKPT = OUT_DIR / "subgraph_masking_gnn_v0.pt"
DEFAULT_METRICS = OUT_DIR / "subgraph_masking_gnn_metrics_v0.csv"
DEFAULT_CONFIG = OUT_DIR / "subgraph_masking_gnn_config_v0.json"


def log(msg: str) -> None:
    print(f"[train_subgraph_masking_gnn_v0] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class NodeRegressionGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.10,
        conv_type: str = "gcn",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = float(dropout)
        self.conv_type = conv_type

        conv_cls = GCNConv if conv_type == "gcn" else SAGEConv
        self.convs = nn.ModuleList()
        self.convs.append(conv_cls(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(conv_cls(hidden_dim, hidden_dim))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for conv in self.convs:
            if self.conv_type == "gcn":
                h = conv(h, edge_index, edge_weight=edge_weight)
            else:
                h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------
def edge_weight_from_batch(batch) -> torch.Tensor | None:
    if getattr(batch, "edge_attr", None) is None:
        return None
    if batch.edge_attr.numel() == 0:
        return None
    return batch.edge_attr.view(-1).float()


def run_epoch(model, loader, optimizer, device, measurement_col: int, mask_fraction: float, train: bool) -> dict:
    model.train(train)
    total_loss = 0.0
    total_nodes = 0
    total_masked = 0

    for batch in loader:
        batch = batch.to(device)
        batch = make_masked_copy(batch, measurement_col=measurement_col, mask_fraction=mask_fraction)
        edge_weight = edge_weight_from_batch(batch)

        with torch.set_grad_enabled(train):
            pred = model(batch.x, batch.edge_index, edge_weight=edge_weight)
            m = batch.training_mask
            if m.sum() == 0:
                continue
            loss = F.mse_loss(pred[m], batch.y[m])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        n_masked = int(m.sum().item())
        total_loss += float(loss.item()) * n_masked
        total_nodes += int(batch.num_nodes)
        total_masked += n_masked

    return {
        "loss_mse": total_loss / max(1, total_masked),
        "masked_nodes": total_masked,
        "total_nodes": total_nodes,
    }


@dataclass
class TrainConfig:
    h5ad: str
    graph_key: str
    embedding_key: str
    measurement_layer: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    hidden_dim: int
    num_layers: int
    dropout: float
    conv_type: str
    mask_fraction: float
    val_fraction: float
    max_samples: int | None
    seed: int
    device: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    p.add_argument("--graph-key", type=str, default="phenotype_graph_combined_v0")
    p.add_argument("--embedding-key", type=str, default="phenotype_embedding_initial")
    p.add_argument("--measurement-layer", type=str, default="X")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--conv-type", type=str, choices=["gcn", "sage"], default="gcn")
    p.add_argument("--mask-fraction", type=float, default=0.25)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--config-json", type=Path, default=DEFAULT_CONFIG)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    log(f"device={device}")

    ds = AnnDataPhenotypeGraphDataset(
        h5ad_path=args.h5ad,
        graph_key=args.graph_key,
        embedding_key=args.embedding_key,
        measurement_layer=args.measurement_layer,
        max_samples=args.max_samples,
    )
    log(f"dataset: n={len(ds)} nodes={len(ds.axis_ids)} features={ds.num_node_features}")

    n = len(ds)
    indices = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(n * args.val_fraction)))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()

    train_subset = torch.utils.data.Subset(ds, train_idx)
    val_subset = torch.utils.data.Subset(ds, val_idx)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    model = NodeRegressionGNN(
        in_dim=ds.num_node_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        conv_type=args.conv_type,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cfg = TrainConfig(
        h5ad=str(args.h5ad),
        graph_key=args.graph_key,
        embedding_key=args.embedding_key,
        measurement_layer=args.measurement_layer,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        conv_type=args.conv_type,
        mask_fraction=args.mask_fraction,
        val_fraction=args.val_fraction,
        max_samples=args.max_samples,
        seed=args.seed,
        device=str(device),
    )

    rows = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, ds.measurement_col, args.mask_fraction, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer, device, ds.measurement_col, args.mask_fraction, train=False
        )
        row = {
            "epoch": epoch,
            "train_loss_mse": train_metrics["loss_mse"],
            "val_loss_mse": val_metrics["loss_mse"],
            "train_masked_nodes": train_metrics["masked_nodes"],
            "val_masked_nodes": val_metrics["masked_nodes"],
        }
        rows.append(row)
        log(
            f"epoch={epoch:03d} train_mse={row['train_loss_mse']:.6f} "
            f"val_mse={row['val_loss_mse']:.6f}"
        )
        if row["val_loss_mse"] < best_val:
            best_val = row["val_loss_mse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "config": asdict(cfg),
        "dataset_metadata": ds.metadata(),
        "best_val_loss_mse": best_val,
    }
    torch.save(ckpt, args.checkpoint)
    pd.DataFrame(rows).to_csv(args.metrics_csv, index=False)
    args.config_json.write_text(json.dumps(asdict(cfg), indent=2))

    log(f"wrote checkpoint: {args.checkpoint}")
    log(f"wrote metrics: {args.metrics_csv}")
    log(f"wrote config: {args.config_json}")


if __name__ == "__main__":
    main()
