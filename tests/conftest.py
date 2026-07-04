"""
tests/conftest.py
=================
Shared pytest fixtures for MedSafe test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def cfg():
    """Load MedSafe configuration."""
    from configs.loader import load_config
    return load_config()


@pytest.fixture(scope="session")
def sample_drugs():
    """A realistic sample of drug names for testing."""
    return [
        "Warfarin", "Aspirin", "Metformin", "Lisinopril",
        "Atorvastatin", "Ibuprofen", "Amoxicillin", "Omeprazole",
    ]


@pytest.fixture(scope="session")
def sample_drug_ids():
    """Corresponding DrugBank-style IDs for sample drugs."""
    return {
        "warfarin": "DB00682",
        "aspirin": "DB00945",
        "metformin": "DB00331",
        "lisinopril": "DB00722",
        "atorvastatin": "DB01076",
        "ibuprofen": "DB01050",
        "amoxicillin": "DB01060",
        "omeprazole": "DB00338",
    }


@pytest.fixture(scope="session")
def device():
    """Return best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def gin_model(cfg, device):
    """Build a GIN encoder model for testing (without loading checkpoint)."""
    from models.gin_encoder import build_gin_encoder
    model = build_gin_encoder(cfg).to(device)
    model.eval()
    return model


@pytest.fixture(scope="session")
def rgcn_model(cfg):
    """Build an R-GCN predictor model for testing."""
    from models.rgcn_predictor import build_rgcn_predictor
    # Drug feature dim: 116 (see build_ddi_graph.py)
    model = build_rgcn_predictor(cfg, drug_feature_dim=116)
    model.eval()
    return model


@pytest.fixture(scope="session")
def small_molecular_graph():
    """Create a small test molecular graph (ethanol: CC O)."""
    from torch_geometric.data import Data
    from pipeline.build_molecular_graphs import ATOM_FEATURE_DIM

    # Ethanol: C-C-O (3 atoms, 2 bonds = 4 directed edges)
    x = torch.randn(3, ATOM_FEATURE_DIM)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    mol_descriptors = torch.tensor([[0.046, 0.0, 0.204, 0.1, 0.1, 0.1]])
    return Data(x=x, edge_index=edge_index, mol_descriptors=mol_descriptors)


@pytest.fixture(scope="session")
def small_ddi_graph(cfg):
    """Create a minimal DDI graph for testing (10 drugs, some interactions)."""
    from torch_geometric.data import HeteroData

    num_drugs = 10
    num_targets = 5
    drug_feature_dim = 116  # Must match actual feature dim

    data = HeteroData()
    data["drug"].x = torch.randn(num_drugs, drug_feature_dim)
    data["drug"].num_nodes = num_drugs
    data["target"].x = torch.randn(num_targets, 16)
    data["target"].num_nodes = num_targets

    # DDI edges: first 5 drugs interact with last 5
    ddi_src = list(range(5)) * 2
    ddi_dst = list(range(5, 10)) * 2
    data["drug", "interacts_with", "drug"].edge_index = torch.tensor(
        [ddi_src + ddi_dst, ddi_dst + ddi_src], dtype=torch.long
    )
    data["drug", "interacts_with", "drug"].severity = torch.randint(0, 4, (20,))
    data["drug", "interacts_with", "drug"].interaction_type = torch.randint(0, 86, (20,))
    data["drug", "interacts_with", "drug"].faers_score = torch.rand(20)
    data["drug", "interacts_with", "drug"].support_count = torch.randint(1, 10, (20,))

    # CYP edges
    data["drug", "shares_cyp_enzyme", "drug"].edge_index = torch.tensor(
        [[0, 1, 2], [1, 0, 3]], dtype=torch.long
    )
    data["drug", "shares_cyp_enzyme", "drug"].enzyme_id = torch.tensor([4, 4, 2])  # CYP3A4 = idx 4

    # Target edges
    data["drug", "has_target", "target"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
    )
    data["target", "targeted_by", "drug"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
    )

    # Metadata
    drug_ids = [f"DB{i:05d}" for i in range(num_drugs)]
    drug_names = [
        "Warfarin", "Aspirin", "Ibuprofen", "Metformin", "Lisinopril",
        "Atorvastatin", "Amoxicillin", "Omeprazole", "Metoprolol", "Amlodipine",
    ]
    data.drug_to_idx = {did: i for i, did in enumerate(drug_ids)}
    data.idx_to_drug = {i: name for i, name in enumerate(drug_names)}
    data.drug_names = drug_names

    return data
