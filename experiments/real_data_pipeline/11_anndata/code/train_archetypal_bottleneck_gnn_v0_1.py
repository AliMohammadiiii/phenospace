"""
train_archetypal_bottleneck_gnn_v0_1.py
---------------------------------------
Train a collapse-resistant AA bottleneck GNN.

This is the corrected follow-up to train_archetypal_bottleneck_gnn_v0.py.
The v0 run reconstructed masked phenotype nodes successfully, but collapsed
all users to one archetype. v0_1 adds population usage balancing, latent AA
reconstruction, and stronger archetype separation diagnostics.

Input:
  11_anndata/outputs/real_phenotype_data_v0_2_graph.h5ad

Requires in the same folder:
  anndata_to_pyg_v0.py
  archetypal_bottleneck_gnn_v0_1.py

Outputs:
  11_anndata/outputs/archetypal_bottleneck_gnn_v0_1.pt
  11_anndata/outputs/archetypal_bottleneck_gnn_metrics_v0_1.csv
  11_anndata/outputs/archetypal_bottleneck_gnn_config_v0_1.json
  11_anndata/outputs/archetypal_bottleneck_user_profiles_v0_1.csv
  11_anndata/outputs/archetypal_bottleneck_codes_v0_1.csv

Recommended first run:
  python 11_anndata/code/train_archetypal_bottleneck_gnn_v0_1.py \
    --h5ad 11_anndata/outputs/real_phenotype_data_v0_2_graph.h5ad \
    --epochs 30 --batch-size 128 --n-archetypes 9 \
    --usage-balance-weight 0.25 --latent-aa-weight 0.05 \
    --sparsity-weight 0.0 --temperature 1.5
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
from archetypal_bottleneck_gnn_v0_1 import (
    ArchetypalBottleneckGNNAA,
    active_archetype_count,
    archetype_diversity_loss,
    latent_aa_reconstruction_loss,
    simplex_entropy,
    usage_balance_loss,
    usage_entropy_normalized,
)


def infer_repo_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "code" and here.parent.parent.name == "11_anndata":
        return here.parents[2]
    return Path.cwd()


REPO_ROOT = infer_repo_root()
OUT_DIR = REPO_ROOT / "11_anndata" / "outputs"
DEFAULT_H5AD = OUT_DIR / "real_phenotype_data_v0_2_graph.h5ad"
DEFAULT_CKPT = OUT_DIR / "archetypal_bottleneck_gnn_v0_1.pt"
DEFAULT_METRICS = OUT_DIR / "archetypal_bottleneck_gnn_metrics_v0_1.csv"
DEFAULT_CONFIG = OUT_DIR / "archetypal_bottleneck_gnn_config_v0_1.json"
DEFAULT_PROFILES = OUT_DIR / "archetypal_bottleneck_user_profiles_v0_1.csv"
DEFAULT_CODES = OUT_DIR / "archetypal_bottleneck_codes_v0_1.csv"


def log(msg: str) -> None:
    print(f"[train_archetypal_bottleneck_gnn_v0_1] {msg}", flush=True)


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
    temperature: float
    usage_balance_weight: float
    latent_aa_weight: float
    diversity_weight: float
    sparsity_weight: float
    detach_latent_target: bool
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
    p.add_argument("--temperature", type=float, default=1.50)

    # AA anti-collapse controls. These defaults intentionally prioritize
    # non-collapse over a tiny reconstruction gain.
    p.add_argument("--usage-balance-weight", type=float, default=0.25)
    p.add_argument("--latent-aa-weight", type=float, default=0.05)
    p.add_argument("--diversity-weight", type=float, default=1e-2)
    p.add_argument("--sparsity-weight", type=float, default=0.0)
    p.add_argument("--detach-latent-target", action="store_true")

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
    model: ArchetypalBottleneckGNNAA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    measurement_col: int,
    mask_fraction: float,
    usage_balance_weight: float,
    latent_aa_weight: float,
    diversity_weight: float,
    sparsity_weight: float,
    detach_latent_target: bool,
    train: bool,
) -> dict:
    model.train(train)
    total_recon = 0.0
    total_loss = 0.0
    total_usage_balance = 0.0
    total_latent_aa = 0.0
    total_diversity = 0.0
    total_sparsity = 0.0
    total_usage_entropy = 0.0
    total_active_k = 0.0
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
            usage = usage_balance_loss(out.archetype_weights)
            latent_aa = latent_aa_reconstruction_loss(
                out.graph_embeddings,
                out.bottleneck_embeddings,
                detach_target=detach_latent_target,
            )
            diversity = archetype_diversity_loss(model.archetype_codes)
            sparsity = simplex_entropy(out.archetype_weights)

            loss = (
                recon
                + usage_balance_weight * usage
                + latent_aa_weight * latent_aa
                + diversity_weight * diversity
                + sparsity_weight * sparsity
            )

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        n_masked = int(m.sum().item())
        total_recon += float(recon.item()) * n_masked
        total_loss += float(loss.item()) * n_masked
        total_usage_balance += float(usage.item())
        total_latent_aa += float(latent_aa.item())
        total_diversity += float(diversity.item())
        total_sparsity += float(sparsity.item())
        total_usage_entropy += float(usage_entropy_normalized(out.archetype_weights).item())
        total_active_k += float(active_archetype_count(out.archetype_weights).item())
        total_masked += n_masked
        batches += 1

    return {
        "loss_total": total_loss / max(1, total_masked),
        "loss_recon_mse": total_recon / max(1, total_masked),
        "usage_balance": total_usage_balance / max(1, batches),
        "latent_aa": total_latent_aa / max(1, batches),
        "diversity": total_diversity / max(1, batches),
        "sparsity_entropy": total_sparsity / max(1, batches),
        "usage_entropy_norm": total_usage_entropy / max(1, batches),
        "active_k": total_active_k / max(1, batches),
        "masked_nodes": total_masked,
    }


@torch.no_grad()
def export_profiles(
    model: ArchetypalBottleneckGNNAA,
    ds: AnnDataPhenotypeGraphDataset,
    device: torch.device,
    batch_size: int,
    profiles_csv: Path,
    codes_csv: Path,
) -> None:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    rows: list[dict] = []

    all_w = []
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
        all_w.append(W)
        obs_idx = batch.obs_idx.detach().cpu().numpy().reshape(-1)
        entropy = -(W * np.log(np.clip(W, 1e-8, 1.0))).sum(axis=1)
        dominant = W.argmax(axis=1)
        dominant_weight = W.max(axis=1)

        for r, idx in enumerate(obs_idx):
            row = {
                "obs_idx": int(idx),
                "obs_id": ds.obs_ids[int(idx)] if int(idx) < len(ds.obs_ids) else str(idx),
                "dominant_archetype_id": f"GNN_AA_ARC_{int(dominant[r]):02d}",
                "dominant_archetype_weight": float(dominant_weight[r]),
                "archetype_entropy": float(entropy[r]),
            }
            for k in range(W.shape[1]):
                row[f"GNN_AA_ARC_{k:02d}_weight"] = float(W[r, k])
            rows.append(row)

    profiles_csv.parent.mkdir(parents=True, exist_ok=True)
    profiles = pd.DataFrame(rows)
    profiles.to_csv(profiles_csv, index=False)

    codes = model.archetype_codes.detach().cpu().numpy()
    codes_df = pd.DataFrame(
        codes,
        index=[f"GNN_AA_ARC_{k:02d}" for k in range(codes.shape[0])],
        columns=[f"hidden_{j:03d}" for j in range(codes.shape[1])],
    )
    codes_df.index.name = "archetype_id"
    codes_df.to_csv(codes_csv)

    if all_w:
        W_all = np.vstack(all_w)
        mean_usage = W_all.mean(axis=0)
        log("mean archetype usage: " + json.dumps({f"GNN_AA_ARC_{i:02d}": float(v) for i, v in enumerate(mean_usage)}, indent=2))
        log(f"active archetypes @ usage>0.02: {int((mean_usage > 0.02).sum())}/{W_all.shape[1]}")

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

    model = ArchetypalBottleneckGNNAA(
        in_dim=ds.num_node_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_archetypes=args.n_archetypes,
        dropout=args.dropout,
        conv_type=args.conv_type,
        temperature=args.temperature,
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
        temperature=args.temperature,
        usage_balance_weight=args.usage_balance_weight,
        latent_aa_weight=args.latent_aa_weight,
        diversity_weight=args.diversity_weight,
        sparsity_weight=args.sparsity_weight,
        detach_latent_target=bool(args.detach_latent_target),
        val_fraction=args.val_fraction,
        max_samples=args.max_samples,
        seed=args.seed,
        device=str(device),
    )

    rows = []
    best_score = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            ds.measurement_col,
            args.mask_fraction,
            args.usage_balance_weight,
            args.latent_aa_weight,
            args.diversity_weight,
            args.sparsity_weight,
            args.detach_latent_target,
            train=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            measurement_col=ds.measurement_col,
            mask_fraction=args.mask_fraction,
            usage_balance_weight=args.usage_balance_weight,
            latent_aa_weight=args.latent_aa_weight,
            diversity_weight=args.diversity_weight,
            sparsity_weight=args.sparsity_weight,
            detach_latent_target=args.detach_latent_target,
            train=False,
        )

        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["loss_total"],
            "train_recon_mse": train_metrics["loss_recon_mse"],
            "train_usage_balance": train_metrics["usage_balance"],
            "train_latent_aa": train_metrics["latent_aa"],
            "train_diversity": train_metrics["diversity"],
            "train_sparsity_entropy": train_metrics["sparsity_entropy"],
            "train_usage_entropy_norm": train_metrics["usage_entropy_norm"],
            "train_active_k": train_metrics["active_k"],
            "train_masked_nodes": train_metrics["masked_nodes"],
            "val_total_loss": val_metrics["loss_total"],
            "val_recon_mse": val_metrics["loss_recon_mse"],
            "val_usage_balance": val_metrics["usage_balance"],
            "val_latent_aa": val_metrics["latent_aa"],
            "val_diversity": val_metrics["diversity"],
            "val_sparsity_entropy": val_metrics["sparsity_entropy"],
            "val_usage_entropy_norm": val_metrics["usage_entropy_norm"],
            "val_active_k": val_metrics["active_k"],
            "val_masked_nodes": val_metrics["masked_nodes"],
        }
        rows.append(row)

        log(
            f"epoch={epoch:03d} "
            f"train_recon={row['train_recon_mse']:.6f} "
            f"val_recon={row['val_recon_mse']:.6f} "
            f"usage={row['val_usage_balance']:.5f} "
            f"latentAA={row['val_latent_aa']:.4f} "
            f"activeK={row['val_active_k']:.1f}/{args.n_archetypes} "
            f"usageH={row['val_usage_entropy_norm']:.3f}"
        )

        # Selection score: prioritize reconstruction, but reject collapse by
        # including a mild non-collapse penalty.
        score = row["val_recon_mse"] + 0.05 * row["val_usage_balance"] + 0.01 * row["val_latent_aa"]
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state if best_state is not None else model.state_dict(),
            "config": asdict(cfg),
            "dataset_metadata": ds.metadata(),
            "best_selection_score": best_score,
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
