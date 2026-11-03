"""
train_aanet_bottleneck_gnn_v0_1.py
-----------------------------------
Train an AAnet-style archetypal bottleneck GNN for the Cytognosis / Neuroverse
phenotype graph pipeline.

This replaces the collapsed v0 bottleneck. It keeps the same masked-observation
prediction task, but the participant/session graph embedding is compressed with
an AA-style two-simplex bottleneck:

    archetypes = Beta @ graph_embeddings
    graph_hat  = Alpha @ archetypes

Outputs:
  11_anndata/outputs/aanet_bottleneck_gnn_v0_1.pt
  11_anndata/outputs/aanet_bottleneck_gnn_metrics_v0_1.csv
  11_anndata/outputs/aanet_bottleneck_gnn_config_v0_1.json
  11_anndata/outputs/aanet_bottleneck_user_profiles_v0_1.csv
  11_anndata/outputs/aanet_bottleneck_codes_v0_1.csv

Usage:
  python 11_anndata/code/train_aanet_bottleneck_gnn_v0_1.py \
    --h5ad 11_anndata/outputs/real_phenotype_data_v0_2_graph.h5ad \
    --epochs 30 \
    --batch-size 128 \
    --n-archetypes 9
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from anndata_to_pyg_v0 import AnnDataPhenotypeGraphDataset, make_masked_copy
from aanet_bottleneck_gnn_v0_1 import (
    AANetStyleGraphBottleneckGNN,
    aa_graph_reconstruction_loss,
    archetype_entropy,
    archetype_separation_loss,
    archetype_usage_balance_loss,
    beta_diversity_loss,
    effective_archetype_count,
    entropy_floor_loss,
)


def infer_repo_root() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "code" and here.parent.parent.name == "11_anndata":
        return here.parents[2]
    return Path.cwd()


REPO_ROOT = infer_repo_root()
OUT_DIR = REPO_ROOT / "11_anndata" / "outputs"
DEFAULT_H5AD = OUT_DIR / "real_phenotype_data_v0_2_graph.h5ad"
DEFAULT_CKPT = OUT_DIR / "aanet_bottleneck_gnn_v0_1.pt"
DEFAULT_METRICS = OUT_DIR / "aanet_bottleneck_gnn_metrics_v0_1.csv"
DEFAULT_CONFIG = OUT_DIR / "aanet_bottleneck_gnn_config_v0_1.json"
DEFAULT_PROFILES = OUT_DIR / "aanet_bottleneck_user_profiles_v0_1.csv"
DEFAULT_CODES = OUT_DIR / "aanet_bottleneck_codes_v0_1.csv"


def log(msg: str) -> None:
    print(f"[train_aanet_bottleneck_gnn_v0_1] {msg}", flush=True)


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
    temperature: float
    hull_mode: str
    mask_fraction: float
    graph_recon_weight: float
    usage_balance_weight: float
    entropy_floor_weight: float
    entropy_floor_frac: float
    archetype_separation_weight: float
    beta_diversity_weight: float
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
    p.add_argument("--temperature", type=float, default=1.50)
    p.add_argument("--hull-mode", type=str, choices=["batch", "free"], default="batch")
    p.add_argument("--mask-fraction", type=float, default=0.25)

    # AA-specific regularization. Defaults are intentionally stronger than v0
    # because the previous run collapsed all users into one archetype.
    p.add_argument("--graph-recon-weight", type=float, default=0.10)
    p.add_argument("--usage-balance-weight", type=float, default=1.00)
    p.add_argument("--entropy-floor-weight", type=float, default=0.05)
    p.add_argument("--entropy-floor-frac", type=float, default=0.25,
                   help="minimum per-sample entropy as fraction of log(K)")
    p.add_argument("--archetype-separation-weight", type=float, default=0.05)
    p.add_argument("--beta-diversity-weight", type=float, default=0.05)

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
    model: AANetStyleGraphBottleneckGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    measurement_col: int,
    mask_fraction: float,
    graph_recon_weight: float,
    usage_balance_weight: float,
    entropy_floor_weight: float,
    entropy_floor_frac: float,
    archetype_separation_weight: float,
    beta_diversity_weight: float,
    train: bool,
) -> dict:
    model.train(train)
    total_recon = 0.0
    total_loss = 0.0
    total_graph_recon = 0.0
    total_usage = 0.0
    total_entropy_floor = 0.0
    total_sep = 0.0
    total_beta = 0.0
    total_alpha_entropy = 0.0
    total_eff_k = 0.0
    total_masked = 0
    batches = 0

    min_entropy = entropy_floor_frac * math.log(float(model.n_archetypes))

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

            node_recon = F.mse_loss(out.pred[m], batch.y[m])
            graph_recon = aa_graph_reconstruction_loss(out)
            usage = archetype_usage_balance_loss(out.alpha_weights)
            ent_floor = entropy_floor_loss(out.alpha_weights, min_entropy=min_entropy)
            sep = archetype_separation_loss(out.archetype_codes)
            beta_div = beta_diversity_loss(out.beta_weights)
            if beta_div.device != node_recon.device:
                beta_div = beta_div.to(node_recon.device)

            loss = (
                node_recon
                + graph_recon_weight * graph_recon
                + usage_balance_weight * usage
                + entropy_floor_weight * ent_floor
                + archetype_separation_weight * sep
                + beta_diversity_weight * beta_div
            )

            if train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        n_masked = int(m.sum().item())
        total_recon += float(node_recon.item()) * n_masked
        total_loss += float(loss.item()) * n_masked
        total_graph_recon += float(graph_recon.item())
        total_usage += float(usage.item())
        total_entropy_floor += float(ent_floor.item())
        total_sep += float(sep.item())
        total_beta += float(beta_div.item())
        total_alpha_entropy += float(archetype_entropy(out.alpha_weights).item())
        total_eff_k += float(effective_archetype_count(out.alpha_weights).item())
        total_masked += n_masked
        batches += 1

    return {
        "loss_total": total_loss / max(1, total_masked),
        "loss_recon_mse": total_recon / max(1, total_masked),
        "graph_recon": total_graph_recon / max(1, batches),
        "usage_balance": total_usage / max(1, batches),
        "entropy_floor": total_entropy_floor / max(1, batches),
        "archetype_separation": total_sep / max(1, batches),
        "beta_diversity": total_beta / max(1, batches),
        "alpha_entropy": total_alpha_entropy / max(1, batches),
        "effective_k": total_eff_k / max(1, batches),
        "masked_nodes": total_masked,
    }


@torch.no_grad()
def collect_graph_embeddings(
    model: AANetStyleGraphBottleneckGNN,
    ds: AnnDataPhenotypeGraphDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Collect graph embeddings and Alpha weights for the whole dataset."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    graph_h_all: list[np.ndarray] = []
    alpha_all: list[np.ndarray] = []
    obs_ids: list[str] = []

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
        graph_h_all.append(out.graph_embeddings.detach().cpu().numpy())
        alpha_all.append(out.alpha_weights.detach().cpu().numpy())
        obs_idx = batch.obs_idx.detach().cpu().numpy().reshape(-1)
        obs_ids.extend(ds.obs_ids[int(i)] if int(i) < len(ds.obs_ids) else str(int(i)) for i in obs_idx)

    return np.vstack(graph_h_all), np.vstack(alpha_all), obs_ids


@torch.no_grad()
def export_profiles_and_codes(
    model: AANetStyleGraphBottleneckGNN,
    ds: AnnDataPhenotypeGraphDataset,
    device: torch.device,
    batch_size: int,
    profiles_csv: Path,
    codes_csv: Path,
) -> None:
    """Export Alpha profiles and global archetype codes.

    In batch-hull mode, archetype codes are batch-dependent during training.
    For inspection we compute a global set of archetypes by applying the learned
    archetype queries over all graph embeddings collected from the dataset.
    """
    graph_h_np, alpha_np, obs_ids = collect_graph_embeddings(model, ds, device, batch_size)
    graph_h = torch.as_tensor(graph_h_np, dtype=torch.float32, device=device)
    codes, beta = model.compute_archetypes_from_graph_h(graph_h)
    codes_np = codes.detach().cpu().numpy()

    entropy = -(alpha_np * np.log(np.clip(alpha_np, 1e-8, 1.0))).sum(axis=1)
    dominant = alpha_np.argmax(axis=1)
    dominant_weight = alpha_np.max(axis=1)

    rows: list[dict] = []
    for i in range(alpha_np.shape[0]):
        row = {
            "obs_idx": i,
            "obs_id": obs_ids[i],
            "dominant_archetype_id": f"AANET_ARC_{int(dominant[i]):02d}",
            "dominant_archetype_weight": float(dominant_weight[i]),
            "archetype_entropy": float(entropy[i]),
        }
        for k in range(alpha_np.shape[1]):
            row[f"AANET_ARC_{k:02d}_weight"] = float(alpha_np[i, k])
        rows.append(row)

    profiles_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(profiles_csv, index=False)

    codes_df = pd.DataFrame(
        codes_np,
        index=[f"AANET_ARC_{k:02d}" for k in range(codes_np.shape[0])],
        columns=[f"hidden_{j:03d}" for j in range(codes_np.shape[1])],
    )
    codes_df.index.name = "archetype_id"
    codes_df.to_csv(codes_csv)

    # Add beta diagnostics if available. This is useful to inspect which samples
    # define each archetype in latent space.
    if beta is not None:
        beta_np = beta.detach().cpu().numpy()
        beta_csv = codes_csv.with_name(codes_csv.stem + "_beta_over_samples.csv")
        beta_df = pd.DataFrame(
            beta_np,
            index=[f"AANET_ARC_{k:02d}" for k in range(beta_np.shape[0])],
            columns=[f"sample_{i:05d}" for i in range(beta_np.shape[1])],
        )
        beta_df.index.name = "archetype_id"
        beta_df.to_csv(beta_csv)
        log(f"wrote global beta-over-samples: {beta_csv}")

    log(f"wrote user archetype profiles: {profiles_csv}")
    log(f"wrote global archetype codes: {codes_csv}")


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

    model = AANetStyleGraphBottleneckGNN(
        in_dim=ds.num_node_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_archetypes=args.n_archetypes,
        dropout=args.dropout,
        conv_type=args.conv_type,
        temperature=args.temperature,
        hull_mode=args.hull_mode,
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
        temperature=args.temperature,
        hull_mode=args.hull_mode,
        mask_fraction=args.mask_fraction,
        graph_recon_weight=args.graph_recon_weight,
        usage_balance_weight=args.usage_balance_weight,
        entropy_floor_weight=args.entropy_floor_weight,
        entropy_floor_frac=args.entropy_floor_frac,
        archetype_separation_weight=args.archetype_separation_weight,
        beta_diversity_weight=args.beta_diversity_weight,
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
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            measurement_col=ds.measurement_col,
            mask_fraction=args.mask_fraction,
            graph_recon_weight=args.graph_recon_weight,
            usage_balance_weight=args.usage_balance_weight,
            entropy_floor_weight=args.entropy_floor_weight,
            entropy_floor_frac=args.entropy_floor_frac,
            archetype_separation_weight=args.archetype_separation_weight,
            beta_diversity_weight=args.beta_diversity_weight,
            train=True,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            measurement_col=ds.measurement_col,
            mask_fraction=args.mask_fraction,
            graph_recon_weight=args.graph_recon_weight,
            usage_balance_weight=args.usage_balance_weight,
            entropy_floor_weight=args.entropy_floor_weight,
            entropy_floor_frac=args.entropy_floor_frac,
            archetype_separation_weight=args.archetype_separation_weight,
            beta_diversity_weight=args.beta_diversity_weight,
            train=False,
        )

        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["loss_total"],
            "train_recon_mse": train_metrics["loss_recon_mse"],
            "train_graph_recon": train_metrics["graph_recon"],
            "train_usage_balance": train_metrics["usage_balance"],
            "train_entropy_floor": train_metrics["entropy_floor"],
            "train_archetype_separation": train_metrics["archetype_separation"],
            "train_beta_diversity": train_metrics["beta_diversity"],
            "train_alpha_entropy": train_metrics["alpha_entropy"],
            "train_effective_k": train_metrics["effective_k"],
            "train_masked_nodes": train_metrics["masked_nodes"],
            "val_total_loss": val_metrics["loss_total"],
            "val_recon_mse": val_metrics["loss_recon_mse"],
            "val_graph_recon": val_metrics["graph_recon"],
            "val_usage_balance": val_metrics["usage_balance"],
            "val_entropy_floor": val_metrics["entropy_floor"],
            "val_archetype_separation": val_metrics["archetype_separation"],
            "val_beta_diversity": val_metrics["beta_diversity"],
            "val_alpha_entropy": val_metrics["alpha_entropy"],
            "val_effective_k": val_metrics["effective_k"],
            "val_masked_nodes": val_metrics["masked_nodes"],
        }
        rows.append(row)
        log(
            f"epoch={epoch:03d} "
            f"train_recon={row['train_recon_mse']:.6f} "
            f"val_recon={row['val_recon_mse']:.6f} "
            f"val_eff_k={row['val_effective_k']:.2f} "
            f"val_alpha_entropy={row['val_alpha_entropy']:.4f} "
            f"usage={row['val_usage_balance']:.5f}"
        )

        # Select by reconstruction, but reject fully collapsed checkpoints unless
        # every checkpoint is collapsed. This simple gate prevents saving the
        # exact failure mode seen in v0.
        if row["val_recon_mse"] < best_val and row["val_effective_k"] >= max(2.0, 0.35 * args.n_archetypes):
            best_val = row["val_recon_mse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Fallback: if every checkpoint violated effective-k gate, save final state
    # but make the config/metrics show the issue clearly.
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        log("WARNING: no checkpoint passed the effective-k anti-collapse gate; saving final state")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
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

    export_profiles_and_codes(
        model=model,
        ds=ds,
        device=device,
        batch_size=args.batch_size,
        profiles_csv=args.profiles_csv,
        codes_csv=args.codes_csv,
    )


if __name__ == "__main__":
    main()
