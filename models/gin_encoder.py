

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class GINEncoder(nn.Module):
    """GIN encoder — accepts PyG Batch or separate (x, edge_index, batch) tensors."""

    def __init__(
        self,
        in_channels: int = 74,
        hidden_dim: int = 256,
        out_dim: int = 64,
        num_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_dim = out_dim

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_d = in_channels if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_d, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        x_or_batch: Union[Tensor, Batch, Data],
        edge_index: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        if isinstance(x_or_batch, (Batch, Data)):
            pyg = x_or_batch
            x = pyg.x.float()
            edge_index = pyg.edge_index
            batch = getattr(pyg, 'batch', None)
        else:
            x = x_or_batch.float()

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x_graph = torch.cat([x_mean, x_add], dim=-1)

        return self.proj(x_graph)


class GINEncoderWithProjection(nn.Module):
    """GIN encoder with separate projection head for contrastive learning."""

    def __init__(
        self,
        in_channels: int = 74,
        hidden_dim: int = 256,
        out_dim: int = 64,
        proj_dim: int = 128,
        num_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GINEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.out_dim = out_dim

        self.projection_head = nn.Sequential(
            nn.Linear(out_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(
        self,
        x_or_batch: Union[Tensor, Batch, Data],
        edge_index: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Returns (embedding, projection)."""
        embedding = self.encoder(x_or_batch, edge_index, batch)
        projection = self.projection_head(embedding)
        return embedding, projection

    def encode(
        self,
        x_or_batch: Union[Tensor, Batch, Data],
        edge_index: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode only - returns embedding without projection head."""
        return self.encoder(x_or_batch, edge_index, batch)


def build_gin_encoder(cfg, in_channels=None):
    """Build GINEncoderWithProjection from config."""
    g = cfg.gin
    if in_channels is None:
        try:
            in_channels = cfg.molecular_graph.atom_feature_dim
        except AttributeError:
            in_channels = getattr(g, 'atom_feature_dim', 80)

    return GINEncoderWithProjection(
        in_channels=in_channels,
        hidden_dim=getattr(g, 'hidden_dim', 256),
        out_dim=getattr(g, 'embedding_dim', 64),
        proj_dim=getattr(cfg.contrastive, 'projection_dim', 128),
        num_layers=getattr(g, 'num_layers', 5),
        dropout=getattr(g, 'dropout', 0.1),
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
