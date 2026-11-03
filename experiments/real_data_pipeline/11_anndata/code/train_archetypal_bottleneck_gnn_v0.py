"""
train_archetypal_bottleneck_gnn_v0.py
-------------------------------------
Task 9 baseline: train a masked-observation GNN with a participant/session-level
archetypal bottleneck.

Input:
  11_anndata/outputs/real_phenotype_data_v0_2_graph.h5ad

Requires the previous file in the same folder:
  anndata_to_pyg_v0.py

Outputs:
  11_anndata/outputs/archetypal_bottleneck_gnn_v0.pt
  11_anndata/outputs/archetypal_bottleneck_gnn_metrics_v0.csv
  11_anndata/outputs/archetypal_bottleneck_gnn_config_v0.json
  11_anndata/outputs/archetypal_bottleneck_user_profiles_v0.csv
  11_anndata/outputs/archetypal_bottleneck_codes_v0.csv

Usage:
  python 11_anndata/code/train_archetypal_bottleneck_gnn_v0.py \
    --h5ad 11_anndata/outputs/real_phenotype_data_v0_2_graph.h5ad \
    --epochs 30 \
    --batch-size 128 \
    --n-archetypes 9
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from anndata_to_pyg_v0 import AnnDataPhenotypeGraphDataset, make_masked_copy
from archetypal_bottleneck_gnn_v0 import (
    ArchetypalBottleneckGNN,
    archetype_diversity_loss,
    archetype_entropy_loss,
)


def infer_repo_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "code" and here.parent.parent.name == "11_anndata":
        return here.parents[2]
    return Path.cwd()


REPO_ROOT = infer_repo_root()
OUT_DIR = REPO_ROOT / "11_anndata" / "outputs"
DEFAULT_H5AD = OUT_DIR / "real_phenotype_data_v0_2_graph.h5ad"
DEFAULT_CKPT = OUT_DIR / "archetypal_bottleneck_gnn_v0.pt"
DEFAULT_METRICS = OUT_DIR / "archetypal_bottleneck_gnn_metrics_v0.csv"
DEFAULT_CONFIG = OUT_DIR / "archetypal_bottleneck_gnn_config_v0.json"
DEFAULT_PROFILES = OUT_DIR / "archetypal_bottleneck_user_profiles_v0.csv"
DEFAULT_CODES = OUT_DIR / "archetypal_bottleneck_codes_v0.csv"


def log(msg: str) -> None:
    print(f"[train_archetypal_bottleneck_gnn_v0] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def edge_weight_from_batch(batch) -> torch.Tensor | None:
    if getattr(batch, "edge_attr", None) is None:
        return None
    if batch.edge_attr.numel() == 0:
        return None
    return batch.edge_attr.view(-1).float()


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
    n_archetypes: int
    mask_fraction: float
    entropy_weight: float
    diversity_weight: float
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
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--conv-type", type=str, choices=["gcn", "sage"], default="gcn")
    p.add_argument("--n-archetypes", type=int, default=9)
    p.add_argument("--mask-fraction", type=float, default=0.25)
    p.add_argument("--entropy-weight", type=float, default=1e-3)
    p.add_argument("--diversity-weight", type=float, default=1e-2)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--config-json", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--profiles-csv", type=Path, default=DEFAULT_PROFILES)
    p.add_argument("--codes-csv", type=Path, default=DEFAULT_CODES)
    return p.parse_args()


def run_epoch(
    model: ArchetypalBottleneckGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    measurement_col: int,
    mask_fraction: float,
    entropy_weight: float,
    diversity_weight: float,
    train: bool,
) -> dict:
    model.train(train)
    total_recon = 0.0
    total_loss = 0.0
    total_entropy = 0.0
    total_diversity = 0.0
    total_masked = 0
    batches = 0

    for batch in loader:
        batch = batch.to(device)
        batch = make_masked_copy(batch, measurement_col=measurement_col, mask_fraction=mask_fraction)
        edge_weight = edge_weight_from_batch(batch)

        with torch.set_grad_enabled(train):
            out = model(
                batch.x,
                batch.edge_index,
                batch=batch.batch,
                edge_weight=edge_weight,
                return_latents=True,
            )
            m = batch.training_mask
            if m.sum() == 0:
                continue

            recon = F.mse_loss(out.pred[m], batch.y[m])
            entropy = archetype_entropy_loss(out.archetype_weights)
            diversity = archetype_diversity_loss(model.archetype_codes)
            loss = recon + entropy_weight * entropy + diversity_weight * diversity

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        n_masked = int(m.sum().item())
        total_recon += float(recon.item()) * n_masked
        total_loss += float(loss.item()) * n_masked
        total_entropy += float(entropy.item())
        total_diversity += float(diversity.item())
        total_masked += n_masked
        batches += 1

    return {
        "loss_total": total_loss / max(1, total_masked),
        "loss_recon_mse": total_recon / max(1, total_masked),
        "entropy": total_entropy / max(1, batches),
        "diversity": total_diversity / max(1, batches),
        "masked_nodes": total_masked,
    }


@torch.no_grad()
def export_profiles(
    model: ArchetypalBottleneckGNN,
    ds: AnnDataPhenotypeGraphDataset,
    device: torch.device,
    batch_size: int,
    profiles_csv: Path,
    codes_csv: Path,
) -> None:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    rows: list[dict] = []

    for batch in loader:
        batch = batch.to(device)
        edge_weight = edge_weight_from_batch(batch)
        out = model(
            batch.x,
            batch.edge_index,
            batch=batch.batch,
            edge_weight=edge_weight,
            return_latents=True,
        )
        W = out.archetype_weights.detach().cpu().numpy()
        obs_idx = batch.obs_idx.detach().cpu().numpy().reshape(-1)
        entropy = -(W * np.log(np.clip(W, 1e-8, 1.0))).sum(axis=1)
        dominant = W.argmax(axis=1)
        dominant_weight = W.max(axis=1)

        for r, idx in enumerate(obs_idx):
            row = {
                "obs_idx": int(idx),
                "obs_id": ds.obs_ids[int(idx)] if int(idx) < len(ds.obs_ids) else str(idx),
                "dominant_archetype_id": f"GNN_ARC_{int(dominant[r]):02d}",
                "dominant_archetype_weight": float(dominant_weight[r]),
                "archetype_entropy": float(entropy[r]),
            }
            for k in range(W.shape[1]):
                row[f"GNN_ARC_{k:02d}_weight"] = float(W[r, k])
            rows.append(row)

    profiles_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(profiles_csv, index=False)

    codes = model.archetype_codes.detach().cpu().numpy()
    codes_df = pd.DataFrame(
        codes,
        index=[f"GNN_ARC_{k:02d}" for k in range(codes.shape[0])],
        columns=[f"hidden_{j:03d}" for j in range(codes.shape[1])],
    )
    codes_df.index.name = "archetype_id"
    codes_df.to_csv(codes_csv)

    log(f"wrote user archetype profiles: {profiles_csv}")
    log(f"wrote learned archetype codes: {codes_csv}")


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

    indices = np.arange(len(ds))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(len(ds) * args.val_fraction)))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()

    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False)

    model = ArchetypalBottleneckGNN(
        in_dim=ds.num_node_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_archetypes=args.n_archetypes,
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
        n_archetypes=args.n_archetypes,
        mask_fraction=args.mask_fraction,
        entropy_weight=args.entropy_weight,
        diversity_weight=args.diversity_weight,
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
            model,
            train_loader,
            optimizer,
            device,
            ds.measurement_col,
            args.mask_fraction,
            args.entropy_weight,
            args.diversity_weight,
            train=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            measurement_col=ds.measurement_col,
            mask_fraction=args.mask_fraction,
            entropy_weight=args.entropy_weight,
            diversity_weight=args.diversity_weight,
            train=False,
        )

        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["loss_total"],
            "train_recon_mse": train_metrics["loss_recon_mse"],
            "train_entropy": train_metrics["entropy"],
            "train_diversity": train_metrics["diversity"],
            "train_masked_nodes": train_metrics["masked_nodes"],
            "val_total_loss": val_metrics["loss_total"],
            "val_recon_mse": val_metrics["loss_recon_mse"],
            "val_entropy": val_metrics["entropy"],
            "val_diversity": val_metrics["diversity"],
            "val_masked_nodes": val_metrics["masked_nodes"],
        }
        rows.append(row)
        log(
            f"epoch={epoch:03d} "
            f"train_recon={row['train_recon_mse']:.6f} "
            f"val_recon={row['val_recon_mse']:.6f} "
            f"entropy={row['val_entropy']:.4f} "
            f"diversity={row['val_diversity']:.4f}"
        )

        if row["val_recon_mse"] < best_val:
            best_val = row["val_recon_mse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "config": asdict(cfg),
            "dataset_metadata": ds.metadata(),
            "best_val_recon_mse": best_val,
        },
        args.checkpoint,
    )
    pd.DataFrame(rows).to_csv(args.metrics_csv, index=False)
    args.config_json.write_text(json.dumps(asdict(cfg), indent=2))

    log(f"wrote checkpoint: {args.checkpoint}")
    log(f"wrote metrics: {args.metrics_csv}")
    log(f"wrote config: {args.config_json}")

    export_profiles(
        model=model,
        ds=ds,
        device=device,
        batch_size=args.batch_size,
        profiles_csv=args.profiles_csv,
        codes_csv=args.codes_csv,
    )


if __name__ == "__main__":
    main()
