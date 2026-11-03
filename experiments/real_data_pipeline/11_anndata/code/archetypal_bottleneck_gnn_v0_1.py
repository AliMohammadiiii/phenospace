"""
archetypal_bottleneck_gnn_v0_1.py
----------------------------------
Collapse-resistant archetypal bottleneck for the Cytognosis / Neuroverse
phenotype-graph pipeline.

Why v0_1 exists
---------------
The first v0 implementation trained, but collapsed all participant/session
profiles to one archetype. That happened because the objective encouraged
sharp individual mixtures but did not force population-level archetype usage,
and the decoder could solve masked-node reconstruction with a single shared
bottleneck.

This version keeps the useful GNN masked-observation setup, but adds the
minimum AA-style constraints needed for a usable archetypal layer:

  1. Participant/session mixture weights live on the simplex.
  2. The bottleneck embedding is a convex combination of learned archetype codes.
  3. The bottleneck is trained to reconstruct the graph-level latent state.
  4. Batch/population usage is regularized toward non-collapse.
  5. Archetype codes are separated in latent space.

The implementation is closer to AAnet / Contextualized-style neural AA than to
plain clustering: the GNN is the context encoder, the simplex vector is the
archetypal profile, and archetype codes define the convex latent geometry.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool


@dataclass
class ArchetypalForwardOutput:
    pred: torch.Tensor
    node_embeddings: torch.Tensor
    graph_embeddings: torch.Tensor
    archetype_logits: torch.Tensor
    archetype_weights: torch.Tensor
    bottleneck_embeddings: torch.Tensor


class ArchetypalBottleneckGNNAA(nn.Module):
    """GNN with a collapse-resistant participant-level AA bottleneck.

    Parameters
    ----------
    in_dim:
        Number of node input features from AnnDataPhenotypeGraphDataset.
    hidden_dim:
        Hidden size for GNN node embeddings and archetype codes.
    num_layers:
        Number of message-passing layers.
    n_archetypes:
        Number of learned archetypes in the bottleneck.
    dropout:
        Dropout probability.
    conv_type:
        ``"gcn"`` or ``"sage"``.
    temperature:
        Softmax temperature for archetype mixtures. Larger values soften
        mixtures; smaller values sharpen them. Keep >= 1.0 while debugging
        anti-collapse behavior.
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
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if n_archetypes < 2:
            raise ValueError("n_archetypes must be >= 2")
        if conv_type not in {"gcn", "sage"}:
            raise ValueError("conv_type must be 'gcn' or 'sage'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.n_archetypes = int(n_archetypes)
        self.dropout = float(dropout)
        self.conv_type = conv_type
        self.temperature = float(temperature)

        conv_cls = GCNConv if conv_type == "gcn" else SAGEConv
        self.convs = nn.ModuleList()
        self.convs.append(conv_cls(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(conv_cls(hidden_dim, hidden_dim))

        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.graph_to_archetype_logits = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_archetypes),
        )

        # Learned archetype vertices/codes in graph latent space.
        self.archetype_codes = nn.Parameter(torch.randn(n_archetypes, hidden_dim) * 0.05)

        # Decode each node conditioned on its participant/session-level AA latent.
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        return_latents: bool = False,
    ) -> torch.Tensor | ArchetypalForwardOutput:
        node_h = self.encode_nodes(x, edge_index, edge_weight=edge_weight)
        graph_h = global_mean_pool(node_h, batch)
        graph_h = self.graph_norm(graph_h)

        logits = self.graph_to_archetype_logits(graph_h)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        bottleneck = weights @ self.archetype_codes

        node_context = bottleneck[batch]
        pred = self.node_decoder(torch.cat([node_h, node_context], dim=-1)).squeeze(-1)

        if return_latents:
            return ArchetypalForwardOutput(
                pred=pred,
                node_embeddings=node_h,
                graph_embeddings=graph_h,
                archetype_logits=logits,
                archetype_weights=weights,
                bottleneck_embeddings=bottleneck,
            )
        return pred


# ---------------------------------------------------------------------
# AA-style regularizers / diagnostics
# ---------------------------------------------------------------------
def simplex_entropy(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean per-sample entropy H(w_i)."""
    w = torch.clamp(weights, min=eps)
    return -(w * torch.log(w)).sum(dim=-1).mean()


def usage_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """Prevent population-level collapse by balancing average archetype use.

    This minimizes the squared distance between the batch mean archetype usage
    and the uniform distribution. Unlike per-sample entropy, this does not force
    every individual to be uniform; it only forces the population/minibatch to
    use more than one archetype.
    """
    k = weights.shape[-1]
    mean_w = weights.mean(dim=0)
    target = torch.full_like(mean_w, 1.0 / float(k))
    return F.mse_loss(mean_w, target)


def archetype_diversity_loss(archetype_codes: torch.Tensor) -> torch.Tensor:
    """Penalize cosine overlap between archetype codes."""
    k = archetype_codes.shape[0]
    z = F.normalize(archetype_codes, p=2, dim=-1)
    sim = z @ z.T
    eye = torch.eye(k, device=archetype_codes.device, dtype=torch.bool)
    off_diag = sim.masked_select(~eye)
    if off_diag.numel() == 0:
        return torch.zeros((), device=archetype_codes.device)
    return off_diag.pow(2).mean()


def latent_aa_reconstruction_loss(
    graph_embeddings: torch.Tensor,
    bottleneck_embeddings: torch.Tensor,
    detach_target: bool = False,
) -> torch.Tensor:
    """AA latent reconstruction: graph_h should be explainable by archetypes.

    Cosine loss is scale-stable. This keeps the bottleneck meaningful even when
    the node-level decoder could otherwise ignore archetype identity.
    """
    target = graph_embeddings.detach() if detach_target else graph_embeddings
    target = F.normalize(target, p=2, dim=-1)
    pred = F.normalize(bottleneck_embeddings, p=2, dim=-1)
    return (1.0 - (target * pred).sum(dim=-1)).mean()


def active_archetype_count(weights: torch.Tensor, threshold: float = 0.02) -> torch.Tensor:
    """Number of archetypes whose mean usage exceeds threshold."""
    mean_w = weights.mean(dim=0)
    return (mean_w > threshold).sum()


def usage_entropy_normalized(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Entropy of mean archetype usage, normalized to [0, 1]."""
    k = weights.shape[-1]
    mean_w = torch.clamp(weights.mean(dim=0), min=eps)
    ent = -(mean_w * torch.log(mean_w)).sum()
    return ent / torch.log(torch.tensor(float(k), device=weights.device))
