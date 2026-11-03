"""
archetypal_bottleneck_gnn_v0.py
--------------------------------
Task 9 for the Cytognosis / Neuroverse graph pipeline.

This module adds the custom archetypal bottleneck layer on top of the baseline
PyG node-regression GNN. It is designed for the existing AnnData -> PyG dataset:

  nodes      = phenotype axes
  edges      = adata.varp['phenotype_graph_combined_v0']
  node x     = phenotype embedding + participant measurement + mask/time features
  target y   = participant phenotype-axis values

The bottleneck works at the participant/session graph level:

  node embeddings -> graph embedding -> simplex archetype weights
  archetype weights @ learned archetype codes -> bottleneck embedding
  concat(node embedding, graph bottleneck) -> per-node value prediction

The simplex weights are directly interpretable as each participant/session's
learned archetype mixture.
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


class ArchetypalBottleneckGNN(nn.Module):
    """GNN with a participant-level archetypal bottleneck.

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
        "gcn" or "sage".
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        n_archetypes: int = 9,
        dropout: float = 0.10,
        conv_type: str = "gcn",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if n_archetypes < 2:
            raise ValueError("n_archetypes must be >= 2")
        if conv_type not in {"gcn", "sage"}:
            raise ValueError("conv_type must be 'gcn' or 'sage'")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.n_archetypes = int(n_archetypes)
        self.dropout = float(dropout)
        self.conv_type = conv_type

        conv_cls = GCNConv if conv_type == "gcn" else SAGEConv
        self.convs = nn.ModuleList()
        self.convs.append(conv_cls(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(conv_cls(hidden_dim, hidden_dim))

        self.graph_to_archetype_logits = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_archetypes),
        )

        # Learned archetype prototype/codes in hidden space.
        self.archetype_codes = nn.Parameter(torch.randn(n_archetypes, hidden_dim) * 0.02)

        # Decode each node conditioned on its participant/session-level bottleneck.
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
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

        logits = self.graph_to_archetype_logits(graph_h)
        weights = torch.softmax(logits, dim=-1)
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


def archetype_entropy_loss(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean entropy of simplex weights. Minimize gently to encourage sharper mixtures."""
    w = torch.clamp(weights, min=eps)
    return -(w * torch.log(w)).sum(dim=-1).mean()


def archetype_diversity_loss(archetype_codes: torch.Tensor) -> torch.Tensor:
    """Penalize cosine overlap between learned archetype codes.

    Lower value means prototypes are more separated. This is useful because
    otherwise multiple archetypes can collapse into the same hidden direction.
    """
    k = archetype_codes.shape[0]
    z = F.normalize(archetype_codes, p=2, dim=-1)
    sim = z @ z.T
    eye = torch.eye(k, device=archetype_codes.device, dtype=torch.bool)
    off_diag = sim.masked_select(~eye)
    if off_diag.numel() == 0:
        return torch.zeros((), device=archetype_codes.device)
    return off_diag.pow(2).mean()
