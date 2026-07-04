
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


# ── NT-Xent Loss (SimCLR) ────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss (SimCLR)."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature

        mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))

        labels = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N, device=z.device),
        ])
        return F.cross_entropy(sim, labels)


# ── Augmentation helpers ──────────────────────────────────────────────────────

def _drop_edges(data: Data, drop_ratio: float) -> Data:
    """Randomly drop edges from a molecular graph."""
    if data.edge_index.size(1) == 0 or drop_ratio <= 0:
        return data
    new = data.clone()
    num_edges = data.edge_index.size(1)
    keep = torch.rand(num_edges) > drop_ratio
    if keep.sum() == 0:
        keep[0] = True
    new.edge_index = data.edge_index[:, keep]
    if data.edge_attr is not None:
        new.edge_attr = data.edge_attr[keep]
    return new


def _mask_nodes(data: Data, mask_ratio: float) -> Data:
    """Randomly zero-out node features."""
    if data.x is None or mask_ratio <= 0:
        return data
    new = data.clone()
    mask = torch.rand(data.x.size(0)) < mask_ratio
    new.x = data.x.clone().float()
    new.x[mask] = 0.0
    return new


def _add_noise(data: Data, sigma: float) -> Data:
    """Add Gaussian noise to continuous node features."""
    if data.x is None or sigma <= 0:
        return data
    new = data.clone()
    new.x = data.x.clone().float() + torch.randn_like(data.x.float()) * sigma
    return new


# ── ContrastiveBatch — standalone augmentor ───────────────────────────────────

class ContrastiveBatch:
    """Applies two different augmentations to each graph, returns (view1, view2) Batch objects."""

    def __init__(
        self,
        mask_ratio_1: float = 0.10,
        mask_ratio_2: float = 0.15,
        drop_ratio: float = 0.15,
        noise_sigma: float = 0.01,
    ):
        self.mask_ratio_1 = mask_ratio_1
        self.mask_ratio_2 = mask_ratio_2
        self.drop_ratio = drop_ratio
        self.noise_sigma = noise_sigma

    def _augment_one(self, data: Data, mask_ratio: float) -> Data:
        data = _drop_edges(data, self.drop_ratio)
        data = _mask_nodes(data, mask_ratio)
        data = _add_noise(data, self.noise_sigma)
        return data

    def __call__(self, batch_list):
        """Apply two augmentations, return collated (view1, view2)."""
        view1 = [self._augment_one(d, self.mask_ratio_1) for d in batch_list]
        view2 = [self._augment_one(d, self.mask_ratio_2) for d in batch_list]
        return Batch.from_data_list(view1), Batch.from_data_list(view2)


# ── MolecularGraphDataset ─────────────────────────────────────────────────────

class MolecularGraphDataset(Dataset):
    """Loads pre-built molecular PyG graphs from disk. Filters by drug_id if provided."""

    def __init__(
        self,
        graph_dir: Path,
        drug_ids=None,
        augment: bool = False,   # augmentation done by ContrastiveBatch augmentor
        drop_edge_prob: float = 0.0,
        mask_node_prob: float = 0.0,
    ):
        super().__init__()
        self.graph_dir = Path(graph_dir)

        # Normalise drug_ids → set of strings or None
        id_set: Optional[set] = None
        if drug_ids is not None:
            try:
                import pandas as pd
                if isinstance(drug_ids, pd.DataFrame):
                    col = "drug_id" if "drug_id" in drug_ids.columns else drug_ids.columns[0]
                    id_set = set(drug_ids[col].astype(str))
                else:
                    id_set = set(str(d) for d in drug_ids)
            except Exception:
                id_set = set(str(d) for d in drug_ids)

        all_files = sorted(self.graph_dir.glob("*.pt"))
        if id_set is not None:
            all_files = [f for f in all_files if f.stem in id_set]

        self.files = all_files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        return torch.load(self.files[idx], weights_only=False)
