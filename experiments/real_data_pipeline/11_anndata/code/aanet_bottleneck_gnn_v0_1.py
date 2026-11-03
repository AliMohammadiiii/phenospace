"""
aanet_bottleneck_gnn_v0_1.py
--------------------------------
AAnet-style archetypal bottleneck for the Cytognosis / Neuroverse phenotype
GNN pipeline.

Why this replaces the previous v0 bottleneck
===========================================
The first archetypal_bottleneck_gnn_v0.py used free learned prototype codes plus
softmax weights. That can reconstruct masked phenotype values, but it is not a
proper Archetypal Analysis / AAnet bottleneck and can collapse to one archetype.

This v0.1 module uses an AA-style two-simplex factorization over graph-level
latent embeddings:

    H        = participant/session graph embeddings, shape [B, D]
    Beta     = archetype-to-sample convex coefficients, shape [K, B]
    A        = Beta @ H, archetype codes inside the batch latent convex hull
    Alpha    = sample-to-archetype convex coefficients, shape [B, K]
    H_hat    = Alpha @ A

So each archetype is a convex combination of latent samples, and each sample is
a convex combination of archetypes. This is much closer to classical AA and to
AAnet-style latent archetypal learning than a free prototype bottleneck.

The model still performs the existing masked node reconstruction task:

    node embeddings + graph AA bottleneck -> per-node phenotype value prediction

but now the bottleneck is regularized to avoid one-archetype collapse:

    - graph-level AA reconstruction loss: H_hat should reconstruct H
    - archetype usage balance: mean Alpha should not use only one archetype
    - optional per-sample entropy floor: prevents premature one-hot collapse
    - archetype separation: archetype codes should not be duplicates
    - beta diversity: different archetypes should not attend to the same samples

This implementation is self-contained and depends only on PyTorch + PyG.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool


@dataclass
class AANetForwardOutput:
    """Full forward-pass outputs used by the training script."""

    pred: torch.Tensor
    node_embeddings: torch.Tensor
    graph_embeddings: torch.Tensor
    alpha_logits: torch.Tensor
    alpha_weights: torch.Tensor
    beta_weights: torch.Tensor | None
    archetype_codes: torch.Tensor
    bottleneck_embeddings: torch.Tensor
    graph_reconstruction: torch.Tensor


class AANetStyleGraphBottleneckGNN(nn.Module):
    """Graph GNN with an AAnet-style archetypal bottleneck.

    Parameters
    ----------
    in_dim:
        Number of node input features from AnnDataPhenotypeGraphDataset.
    hidden_dim:
        Hidden size for GNN node embeddings and graph latent embeddings.
    num_layers:
        Number of message-passing layers.
    n_archetypes:
        Number of archetypes K.
    dropout:
        Dropout probability.
    conv_type:
        "gcn" or "sage".
    temperature:
        Softmax temperature for Alpha. Higher values make assignments less
        sharp and reduce early collapse.
    hull_mode:
        "batch" gives the strict AA-style batch convex-hull archetypes:
            A = softmax(Q @ H.T) @ H
        "free" uses learned free archetype codes. Keep only for ablations;
        batch mode is recommended.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        n_archetypes: int = 9,
        dropout: float = 0.10,
        conv_type: str = "gcn",
        temperature: float = 1.50,
        hull_mode: str = "batch",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if n_archetypes < 2:
            raise ValueError("n_archetypes must be >= 2")
        if conv_type not in {"gcn", "sage"}:
            raise ValueError("conv_type must be 'gcn' or 'sage'")
        if hull_mode not in {"batch", "free"}:
            raise ValueError("hull_mode must be 'batch' or 'free'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.n_archetypes = int(n_archetypes)
        self.dropout = float(dropout)
        self.conv_type = conv_type
        self.temperature = float(temperature)
        self.hull_mode = hull_mode

        conv_cls = GCNConv if conv_type == "gcn" else SAGEConv
        self.convs = nn.ModuleList()
        self.convs.append(conv_cls(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(conv_cls(hidden_dim, hidden_dim))

        # Context encoder: graph embedding -> Alpha logits.
        # This mirrors the ContextualizedML pattern: a context encoder maps a
        # sample context into sample-specific latent parameters.
        self.graph_to_alpha_logits = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_archetypes),
        )

        # Queries used to form archetypes from the convex hull of batch graph
        # embeddings. Beta = softmax(Q @ H.T), A = Beta @ H.
        self.archetype_queries = nn.Parameter(torch.randn(n_archetypes, hidden_dim) * 0.02)

        # Free codes are retained only for ablation. Recommended hull_mode is batch.
        self.free_archetype_codes = nn.Parameter(torch.randn(n_archetypes, hidden_dim) * 0.02)

        # Decode each phenotype node conditioned on the graph-level AA bottleneck.
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = x
        for conv in self.convs:
            if self.conv_type == "gcn":
                h = conv(h, edge_index, edge_weight=edge_weight)
            else:
                h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def compute_alpha(self, graph_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.graph_to_alpha_logits(graph_h)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return logits, weights

    def compute_archetypes_from_graph_h(
        self,
        graph_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (archetype_codes, beta_weights).

        In batch-hull mode, each archetype is a convex combination of the graph
        embeddings in the current batch:

            beta[k, :] >= 0, beta[k, :].sum() = 1
            archetype[k] = sum_i beta[k, i] * graph_h[i]

        This is the AA-specific part that prevents archetypes from being free,
        unconstrained vectors.
        """
        if self.hull_mode == "free":
            return self.free_archetype_codes, None

        # graph_h: [B, D], queries: [K, D], scores: [K, B]
        scale = math.sqrt(float(graph_h.shape[-1]))
        scores = (self.archetype_queries @ graph_h.T) / max(scale, 1.0)
        beta = torch.softmax(scores, dim=-1)
        archetype_codes = beta @ graph_h
        return archetype_codes, beta

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        return_latents: bool = False,
    ) -> torch.Tensor | AANetForwardOutput:
        node_h = self.encode_nodes(x, edge_index, edge_weight=edge_weight)
        graph_h = global_mean_pool(node_h, batch)

        alpha_logits, alpha = self.compute_alpha(graph_h)
        archetype_codes, beta = self.compute_archetypes_from_graph_h(graph_h)

        graph_hat = alpha @ archetype_codes
        node_context = graph_hat[batch]
        pred = self.node_decoder(torch.cat([node_h, node_context], dim=-1)).squeeze(-1)

        if return_latents:
            return AANetForwardOutput(
                pred=pred,
                node_embeddings=node_h,
                graph_embeddings=graph_h,
                alpha_logits=alpha_logits,
                alpha_weights=alpha,
                beta_weights=beta,
                archetype_codes=archetype_codes,
                bottleneck_embeddings=graph_hat,
                graph_reconstruction=graph_hat,
            )
        return pred


# ---------------------------------------------------------------------------
# Losses / diagnostics
# ---------------------------------------------------------------------------

def aa_graph_reconstruction_loss(out: AANetForwardOutput) -> torch.Tensor:
    """MSE between graph latent embedding H and AA reconstruction H_hat."""
    return F.mse_loss(out.graph_reconstruction, out.graph_embeddings)


def archetype_usage_balance_loss(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL-like penalty that prevents all samples using one archetype.

    This acts on the batch mean usage, not on individual samples. It therefore
    still allows individual samples to be sharp/vertex-like, while preventing
    global collapse.
    """
    k = alpha.shape[-1]
    mean_usage = torch.clamp(alpha.mean(dim=0), min=eps)
    uniform = torch.full_like(mean_usage, 1.0 / k)
    return F.kl_div(torch.log(mean_usage), uniform, reduction="batchmean")


def entropy_floor_loss(alpha: torch.Tensor, min_entropy: float) -> torch.Tensor:
    """Penalize only samples whose entropy is below a minimum floor.

    This is safer than minimizing entropy. The previous v0 implementation
    minimized entropy and therefore encouraged universal one-hot assignments.
    """
    if min_entropy <= 0:
        return torch.zeros((), device=alpha.device)
    w = torch.clamp(alpha, min=1e-8)
    entropy = -(w * torch.log(w)).sum(dim=-1)
    return F.relu(min_entropy - entropy).pow(2).mean()


def archetype_separation_loss(archetype_codes: torch.Tensor) -> torch.Tensor:
    """Penalize duplicate archetype directions via off-diagonal cosine^2."""
    k = archetype_codes.shape[0]
    if k <= 1:
        return torch.zeros((), device=archetype_codes.device)
    z = F.normalize(archetype_codes, p=2, dim=-1)
    sim = z @ z.T
    eye = torch.eye(k, device=archetype_codes.device, dtype=torch.bool)
    off_diag = sim.masked_select(~eye)
    if off_diag.numel() == 0:
        return torch.zeros((), device=archetype_codes.device)
    return off_diag.pow(2).mean()


def beta_diversity_loss(beta: torch.Tensor | None) -> torch.Tensor:
    """Prevent different archetypes from attending to the same batch samples.

    beta rows are distributions over batch samples. We penalize pairwise cosine
    similarity between rows. No-op for free archetypes.
    """
    if beta is None:
        return torch.zeros(())
    k = beta.shape[0]
    if k <= 1:
        return torch.zeros((), device=beta.device)
    z = F.normalize(beta, p=2, dim=-1)
    sim = z @ z.T
    eye = torch.eye(k, device=beta.device, dtype=torch.bool)
    off_diag = sim.masked_select(~eye)
    return off_diag.pow(2).mean()


def archetype_entropy(alpha: torch.Tensor) -> torch.Tensor:
    w = torch.clamp(alpha, min=1e-8)
    return -(w * torch.log(w)).sum(dim=-1).mean()


def effective_archetype_count(alpha: torch.Tensor) -> torch.Tensor:
    """exp(entropy(mean usage)); equals K when usage is uniform."""
    usage = torch.clamp(alpha.mean(dim=0), min=1e-8)
    ent = -(usage * torch.log(usage)).sum()
    return torch.exp(ent)
