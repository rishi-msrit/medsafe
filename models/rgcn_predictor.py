
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import RGCNConv


# ── R-GCN Model ───────────────────────────────────────────────────────────────

class RGCNDDIPredictor(nn.Module):
    """R-GCN DDI predictor: binary + severity + type + FAERS multi-task heads."""

    def __init__(
        self,
        drug_feature_dim: int = 116,
        target_feature_dim: int = 64,
        hidden_dim: int = 128,
        num_relations: int = 4,
        num_layers: int = 2,
        num_bases: int = 16,
        dropout: float = 0.2,
        num_severity_classes: int = 4,
        num_interaction_types: int = 86,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_severity_classes = num_severity_classes
        self.num_interaction_types = num_interaction_types

        # Project drug and target features into the same hidden dim
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.target_proj = nn.Sequential(
            nn.Linear(target_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # R-GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            # RGCNConv (not Fast) does scatter per relation type — O(num_nodes)
            # FastRGCNConv materializes [num_edges, in, out] weight → OOM on 4GB GPU
            self.convs.append(RGCNConv(hidden_dim, hidden_dim,
                                       num_relations=num_relations,
                                       num_bases=num_bases,
                                       aggr='mean'))
            self.bns.append(nn.LayerNorm(hidden_dim))

        # Prediction head inputs: concat(h_i, h_j) → pair_dim
        pair_dim = hidden_dim * 2

        self.binary_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_severity_classes),
        )
        self.type_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_interaction_types),
        )
        self.faers_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._num_drugs: Optional[int] = None  # set during get_drug_embeddings

    def _encode(
        self,
        drug_x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        target_x: Optional[Tensor],
    ) -> Tensor:
        """R-GCN forward on combined drug+target graph. Returns all node embeddings."""
        num_drugs = drug_x.size(0)
        self._num_drugs = num_drugs

        # Project drug features
        h_drug = self.drug_proj(drug_x)

        # Concatenate target features if available
        if target_x is not None:
            # Handle dim mismatch via target_proj
            try:
                h_target = self.target_proj(target_x)
            except RuntimeError:
                # If target feature dim doesn't match, use zero-pad/slice
                t_dim = target_x.size(1)
                p_dim = self.target_proj[0].in_features
                if t_dim < p_dim:
                    pad = torch.zeros(target_x.size(0), p_dim - t_dim,
                                      device=target_x.device, dtype=target_x.dtype)
                    target_x = torch.cat([target_x, pad], dim=1)
                else:
                    target_x = target_x[:, :p_dim]
                h_target = self.target_proj(target_x)
            h = torch.cat([h_drug, h_target], dim=0)  # [N_drug + N_target, hidden]
        else:
            # If no target_x but edge_index references target nodes (idx >= num_drugs),
            # RGCNConv scatter will crash with dim_size error.
            # Auto-pad with zeros so the operation succeeds.
            if edge_index.numel() > 0:
                max_node_idx = int(edge_index.max().item())
                num_extra = max(0, max_node_idx + 1 - num_drugs)
            else:
                num_extra = 0
            if num_extra > 0:
                t_in = self.target_proj[0].in_features
                target_pad = torch.zeros(num_extra, t_in,
                                         device=drug_x.device, dtype=drug_x.dtype)
                h_target = self.target_proj(target_pad)
                h = torch.cat([h_drug, h_target], dim=0)
            else:
                h = h_drug

        # R-GCN layers
        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, edge_index, edge_type)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h_new

        return h

    def get_drug_embeddings(
        self,
        drug_x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        target_x: Optional[Tensor] = None,
    ) -> Tensor:
        """Full R-GCN pass → drug-only embeddings [num_drugs, hidden_dim]."""
        h = self._encode(drug_x, edge_index, edge_type, target_x)
        return h[:drug_x.size(0)]  # return only drug node embeddings

    def prediction_head(
        self,
        h_i: Tensor,
        h_j: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Predict binary/severity/type/FAERS for drug pairs (h_i, h_j)."""
        pair = torch.cat([h_i, h_j], dim=-1)
        return (
            self.binary_head(pair),
            self.severity_head(pair),
            self.type_head(pair),
            self.faers_head(pair),
        )

    def forward(
        self,
        drug_x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        src: Tensor,
        dst: Tensor,
        target_x: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full forward pass — used in evaluate()."""
        drug_emb = self.get_drug_embeddings(drug_x, edge_index, edge_type, target_x)
        h_i = drug_emb[src]
        h_j = drug_emb[dst]
        return self.prediction_head(h_i, h_j)

    @torch.no_grad()
    def predict_pair(
        self,
        drug_x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        idx_a: int,
        idx_b: int,
        mc_samples: int = 1,
        target_x: Optional[Tensor] = None,
    ) -> dict:
        """Predict a single drug pair. Returns interaction_prob and predicted_severity."""
        was_training = self.training
        self.eval()

        drug_emb = self.get_drug_embeddings(drug_x, edge_index, edge_type, target_x)

        src = torch.tensor([idx_a], dtype=torch.long, device=drug_x.device)
        dst = torch.tensor([idx_b], dtype=torch.long, device=drug_x.device)
        h_i = drug_emb[src]
        h_j = drug_emb[dst]

        bin_logit, sev_logits, _, _ = self.prediction_head(h_i, h_j)
        interaction_prob = float(torch.sigmoid(bin_logit).squeeze())
        predicted_severity = int(sev_logits.argmax(dim=-1).squeeze())

        if was_training:
            self.train()

        return {
            "interaction_prob": interaction_prob,
            "predicted_severity": predicted_severity,
        }


# ── Multi-Task Loss ────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss: binary + severity + type + FAERS regression."""

    def __init__(
        self,
        lambda_binary: float = 1.0,
        lambda_severity: float = 0.8,
        lambda_type: float = 0.5,
        lambda_faers: float = 0.3,
        num_interaction_types: int = 86,
        pos_weight: float = 3.0,
    ):
        super().__init__()
        self.lambda_binary = lambda_binary
        self.lambda_severity = lambda_severity
        self.lambda_type = lambda_type
        self.lambda_faers = lambda_faers
        self.num_interaction_types = num_interaction_types
        self.pos_weight_val = pos_weight

    def forward(
        self,
        binary_logit: Tensor,
        severity_logits: Tensor,
        type_logits: Tensor,
        faers_pred: Tensor,
        binary_target: Tensor,
        severity_target: Tensor,
        type_target: Tensor,
        faers_target: Tensor,
    ) -> tuple[Tensor, dict]:
        device = binary_logit.device

        # Binary interaction loss
        pw = torch.tensor([self.pos_weight_val], device=device)
        binary_loss = F.binary_cross_entropy_with_logits(
            binary_logit.squeeze(-1),
            binary_target.float(),
            pos_weight=pw,
        )

        # Severity loss — only on positive (interacting) pairs
        pos_mask = binary_target > 0.5
        if pos_mask.sum() > 0:
            severity_loss = F.cross_entropy(
                severity_logits[pos_mask],
                severity_target[pos_mask].long().clamp(0, 3),
            )
        else:
            severity_loss = torch.tensor(0.0, device=device)

        # Interaction type loss — only on positive pairs with valid type label
        valid_type_mask = pos_mask & (type_target >= 0)
        if valid_type_mask.sum() > 0:
            type_loss = F.cross_entropy(
                type_logits[valid_type_mask],
                type_target[valid_type_mask].long().clamp(0, self.num_interaction_types - 1),
            )
        else:
            type_loss = torch.tensor(0.0, device=device)

        # FAERS signal regression loss
        faers_loss = F.mse_loss(
            faers_pred.squeeze(-1),
            faers_target.float(),
        )

        total = (
            self.lambda_binary   * binary_loss +
            self.lambda_severity * severity_loss +
            self.lambda_type     * type_loss +
            self.lambda_faers    * faers_loss
        )

        return total, {
            "total":    total.item(),
            "binary":   binary_loss.item(),
            "severity": severity_loss.item(),
            "type":     type_loss.item(),
            "faers":    faers_loss.item(),
        }


# ── Builder ────────────────────────────────────────────────────────────────────

def build_rgcn_predictor(cfg, drug_feature_dim: int = 116) -> RGCNDDIPredictor:
    """Build RGCNDDIPredictor from config and actual drug feature dimension."""
    r = cfg.rgcn
    return RGCNDDIPredictor(
        drug_feature_dim=drug_feature_dim,
        target_feature_dim=getattr(r, 'target_feature_dim', 64),
        hidden_dim=getattr(r, 'hidden_dim', 128),
        num_relations=4,
        num_layers=getattr(r, 'num_layers', 2),
        num_bases=getattr(r, 'num_bases', 16),
        dropout=getattr(r, 'dropout', 0.2),
        num_severity_classes=getattr(r, 'num_severity_levels', 4),
        num_interaction_types=getattr(r, 'num_interaction_types', 86),
    )
