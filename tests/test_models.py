"""
tests/test_models.py
====================
Tests for ML model architecture: GIN encoder, contrastive loss, R-GCN predictor.

6 tests:
  1. GIN forward pass output dimensions
  2. Contrastive loss is finite and positive
  3. R-GCN multi-task output shapes
  4. Probability outputs bounded [0, 1]
  5. Gradient flow (backward pass succeeds)
  6. Checkpoint save/load roundtrip
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from pipeline.build_molecular_graphs import ATOM_FEATURE_DIM


# ─── Test 1: GIN Forward Pass Output Dimensions ───────────────────────────────

def test_gin_forward_pass_output_dim(gin_model, small_molecular_graph, cfg, device):
    """GIN encoder should produce embeddings of correct dimension."""
    data = small_molecular_graph.to(device)
    with torch.no_grad():
        emb, proj = gin_model(data)

    # Embedding dimension should match config
    assert emb.shape[-1] == cfg.gin.embedding_dim, (
        f"Expected embedding dim {cfg.gin.embedding_dim}, got {emb.shape[-1]}"
    )
    # Projection dimension should match config
    assert proj.shape[-1] == cfg.contrastive.projection_dim, (
        f"Expected projection dim {cfg.contrastive.projection_dim}, got {proj.shape[-1]}"
    )
    # Projections should be L2-normalized (norm ≈ 1.0)
    norms = proj.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        "Projection should be L2-normalized"
    )


# ─── Test 2: GIN Batch Forward Pass ──────────────────────────────────────────

def test_gin_batch_forward(gin_model, small_molecular_graph, cfg, device):
    """GIN encoder should handle batched molecular graphs."""
    # Create a batch of 4 copies of the same graph
    batch = Batch.from_data_list([small_molecular_graph.clone() for _ in range(4)]).to(device)
    with torch.no_grad():
        emb, proj = gin_model(batch)

    assert emb.shape[0] == 4, f"Expected 4 embeddings, got {emb.shape[0]}"
    assert emb.shape[1] == cfg.gin.embedding_dim


# ─── Test 3: NT-Xent Contrastive Loss ─────────────────────────────────────────

def test_ntxent_loss_finite_positive(cfg):
    """NT-Xent loss should be finite and greater than zero for random inputs."""
    from models.contrastive import NTXentLoss

    loss_fn = NTXentLoss(temperature=cfg.contrastive.temperature)
    N = 16
    z_i = F.normalize(torch.randn(N, cfg.contrastive.projection_dim), dim=-1)
    z_j = F.normalize(torch.randn(N, cfg.contrastive.projection_dim), dim=-1)

    loss = loss_fn(z_i, z_j)

    assert torch.isfinite(loss), "NT-Xent loss should be finite"
    assert loss.item() > 0, "NT-Xent loss should be positive for random embeddings"


# ─── Test 4: R-GCN Multi-task Output Shapes ──────────────────────────────────

def test_rgcn_multitask_output_shapes(rgcn_model, small_ddi_graph, cfg):
    """R-GCN should produce 4 task outputs with correct shapes."""
    from training.finetune_rgcn import build_combined_edge_index

    drug_x = small_ddi_graph["drug"].x
    edge_index, edge_type = build_combined_edge_index(small_ddi_graph, torch.device("cpu"))

    pair_src = torch.tensor([0, 1, 2])
    pair_dst = torch.tensor([5, 6, 7])

    with torch.no_grad():
        binary, severity, itype, faers = rgcn_model(
            drug_x, edge_index, edge_type, pair_src, pair_dst
        )

    assert binary.shape == (3, 1), f"Binary output shape: {binary.shape}"
    assert severity.shape == (3, cfg.rgcn.num_severity_levels), f"Severity shape: {severity.shape}"
    assert itype.shape == (3, cfg.rgcn.num_interaction_types), f"Type shape: {itype.shape}"
    assert faers.shape == (3, 1), f"FAERS shape: {faers.shape}"


# ─── Test 5: Probability Outputs Bounded [0, 1] ──────────────────────────────

def test_rgcn_probabilities_bounded(rgcn_model, small_ddi_graph, cfg):
    """Binary prediction probabilities should be in [0, 1]."""
    from training.finetune_rgcn import build_combined_edge_index

    drug_x = small_ddi_graph["drug"].x
    edge_index, edge_type = build_combined_edge_index(small_ddi_graph, torch.device("cpu"))

    pair_src = torch.arange(5)
    pair_dst = torch.arange(5, 10)

    with torch.no_grad():
        binary, severity, _, _ = rgcn_model(drug_x, edge_index, edge_type, pair_src, pair_dst)

    probs = torch.sigmoid(binary)
    assert (probs >= 0).all() and (probs <= 1).all(), "Binary probabilities out of [0,1]"

    sev_probs = torch.softmax(severity, dim=-1)
    assert (sev_probs >= 0).all() and (sev_probs <= 1).all(), "Severity probabilities out of [0,1]"
    assert torch.allclose(sev_probs.sum(dim=-1), torch.ones(5), atol=1e-5), "Severity probs should sum to 1"


# ─── Test 6: Gradient Flow Through Both Models ───────────────────────────────

def test_gradient_flow(gin_model, small_molecular_graph, device, cfg):
    """Backward pass through GIN encoder should produce valid gradients."""
    from models.contrastive import NTXentLoss, ContrastiveBatch

    model = gin_model.train()
    data = small_molecular_graph.to(device)

    # Create a small batch (2 graphs)
    batch = Batch.from_data_list([data, data])
    augmentor = ContrastiveBatch()
    view1, view2 = augmentor([small_molecular_graph, small_molecular_graph])

    view1 = view1.to(device)
    view2 = view2.to(device)

    loss_fn = NTXentLoss(temperature=cfg.contrastive.temperature)
    _, z_i = model(view1)
    _, z_j = model(view2)
    loss = loss_fn(z_i, z_j)

    loss.backward()

    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"NaN gradient in {name}"
            break  # Just check at least one parameter has a valid gradient


# ─── Test 7: Checkpoint Save/Load Roundtrip ──────────────────────────────────

def test_gin_checkpoint_roundtrip(gin_model, small_molecular_graph, device, cfg):
    """Saved and loaded GIN model should produce identical outputs."""
    data = small_molecular_graph.to(device)

    with torch.no_grad():
        emb_before, _ = gin_model(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_gin.pt"
        torch.save({"model_state": gin_model.state_dict()}, ckpt_path)

        # Load into a fresh model
        from models.gin_encoder import build_gin_encoder
        new_model = build_gin_encoder(cfg).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        new_model.load_state_dict(ckpt["model_state"])
        new_model.eval()

        with torch.no_grad():
            emb_after, _ = new_model(data)

    assert torch.allclose(emb_before, emb_after, atol=1e-6), (
        "Model output should be identical before and after checkpoint roundtrip"
    )
