---
title: MedSafe API
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# MedSafe

Patient-level polypharmacy drug interaction safety analyzer powered by Graph Neural Networks.


---

## What this does

MedSafe takes a patient's full medication list and predicts pairwise drug interaction risk using a GIN encoder for drug structure and an R-GCN layer for typed interaction edges (pharmacokinetic, pharmacodynamic, metabolic). Instead of a binary yes/no per drug pair, it scores the whole regimen: each interaction gets a severity level, a confidence score, and a Shapley-based breakdown showing which drugs/pathways are driving the risk. Output includes plain-English explanations and safer alternative suggestions. Backend is FastAPI, frontend is React, deployed via Docker.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange?style=flat-square&logo=pytorch)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.6-red?style=flat-square)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=flat-square&logo=typescript)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?style=flat-square&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)

---

## Architecture

```
DrugBank XML + TWOSIDES CSV + OGBL-DDI + FAERS
              │
              ▼
    pipeline/run_pipeline.py
    ├── parse_drugbank.py       → drugs.parquet, interactions.parquet
    ├── build_molecular_graphs.py → per-drug PyG .pt files
    ├── load_twosides.py        → multi-drug side effects
    ├── load_faers.py           → adverse event signals
    └── build_ddi_graph.py      → ddi_hetero_graph.pt
              │
              ▼
    training/pretrain_gin.py
    GIN Encoder (5-layer, 256-dim)
    Self-supervised SimCLR on molecular graphs
    → 64-dim drug embeddings
              │
              ▼
    training/finetune_rgcn.py
    R-GCN (2-layer, 128-dim, 4 relation types)
    Multi-task: binary + severity + type + FAERS
    → checkpoints/rgcn_finetune/rgcn_best.pt
              │
              ▼
    serving/api.py  (FastAPI, port 8000)
              │
              ▼
    frontend/  (React + TypeScript + Vite, port 5173)
```

---

## Key ML Components

- **Self-supervised molecular pretraining** — GIN trained with NT-Xent (SimCLR) loss on augmented molecular graphs (edge drop, node masking, Gaussian noise). No labels required.
- **R-GCN multi-task prediction** — Relational GCN over a heterogeneous drug-target-interaction graph. Simultaneously predicts: binary interaction, severity (4 classes), mechanism type (86 classes), FAERS harm score.
- **GNNExplainer attribution** — Identifies which graph edges drove a specific interaction prediction. Shows the model's reasoning, not just its output.
- **Monte Carlo Dropout uncertainty** — Runs N stochastic forward passes to produce confidence intervals on every prediction. Flags low-confidence results with a data-quality warning.
- **Shapley value risk scoring** — Computes each drug's marginal contribution to the overall risk score via permutation-based Shapley attribution. Identifies the "risk culprit" drug in a polypharmacy regimen.
- **Special rule detection** — Hardcoded clinical safety rules for QT prolongation, CNS depression, NSAID + anticoagulant combinations, and Warfarin interactions — applied on top of GNN predictions.

---

## Dataset Sources

| Dataset | Size | Access / Download Link |
|---|---|---|
| **DrugBank Full Database** | 14,000+ drugs, 2.5M+ interactions | [Download via DrugBank Portal](https://go.drugbank.com/releases/latest) (Requires free academic license registration. Place the raw `drugbank_full_database.xml` file in `data/raw/`.) |
| **TWOSIDES** | 4.6M multi-drug side effect pairs | [Download from Tatonetti Lab](https://tatonettilab.org/resources/twosides/) (Place the raw `twosides.csv` in `data/raw/` or run `scripts/download_all.py` to fetch via PyTDC.) |
| **OGBL-DDI** | 1.3M interaction pairs, 4,267 drugs | Auto-downloaded by `scripts/download_all.py` from Stanford OGB |
| **FDA FAERS** | Adverse event safety report metrics | Auto-processed via FAERS reporting utility scripts (fetches quarterly FDA files from `https://fis.fda.gov/content/Exports`) |

---

## Setup

### 1. Clone and install

This repository uses **Git LFS (Large File Storage)** to version model weights (`.pt`) and processed datasets (`.parquet`). Ensure Git LFS is installed on your system before cloning:

```bash
# Install Git LFS
git lfs install

# Clone the repository (LFS will automatically download binary assets)
git clone https://github.com/rishi-msrit/medsafe.git
cd medsafe
```

Install PyG extensions first (pre-built wheels, must match your torch+CUDA version):

```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

Install everything else:

```bash
pip install -r requirements.txt
```

### 2. Download data

Place `drugbank_full_database.xml` in `data/raw/` (requires free DrugBank academic registration).

Download all other datasets automatically:

```bash
python scripts/download_all.py
```

### 3. Run the data pipeline

```bash
python pipeline/run_pipeline.py
```

This processes DrugBank, TWOSIDES, FAERS, builds molecular graphs, and constructs the DDI knowledge graph. Takes ~20–40 minutes on first run.

### 4. Train

```bash
# Full training (RTX 3050 4GB VRAM, ~3–5 hours)
python train.py

# Demo mode — 20 epochs, ~10–15 minutes, verifies the pipeline works
python train.py --demo

# Skip GIN pretraining (use existing checkpoint)
python train.py --skip-pretrain

# Larger GPU (8GB+ VRAM) — more layers, bigger hidden dims
python train.py --full
```

### 5. Launch API

```bash
python -m uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

### 6. Launch frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

---

## Demo Mode

```bash
python train.py --demo
```

Runs 20 epochs on a subset of data. Completes in ~10–15 minutes. Prints final metrics and confirms the full pipeline works end-to-end. No DrugBank license required for demo — uses OGBL-DDI only.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Docker

```bash
docker-compose up --build
```

Starts the FastAPI backend on port 8000 and the React frontend on port 3000. Model weights must be present at `checkpoints/rgcn_finetune/rgcn_best.pt` before building.

---

## Model Performance

Results on OGBL-DDI test split after full training:

| Metric | Score |
|---|---|
| Hits@10 | 0.74 |
| Hits@20 | 0.81 |
| Hits@50 | 0.89 |
| AUROC | 0.91 |
| AUPR | 0.88 |
| Severity Accuracy | 0.72 |
| Severity F1 (macro) | 0.68 |

> Run `python evaluation/benchmark.py` after training to reproduce these numbers on your checkpoint.

---

## Project Structure

```
medsafe/
├── configs/
│   ├── config.yaml              # all hyperparameters
│   └── loader.py                # typed config dataclasses
├── pipeline/
│   ├── parse_drugbank.py        # DrugBank XML → parquet
│   ├── build_molecular_graphs.py# SMILES → PyG .pt files
│   ├── build_ddi_graph.py       # heterogeneous DDI graph
│   ├── load_twosides.py         # TWOSIDES side effects
│   ├── load_faers.py            # FDA FAERS signals
│   └── run_pipeline.py          # orchestrates all steps
├── models/
│   ├── gin_encoder.py           # GIN + contrastive projection head
│   ├── contrastive.py           # NT-Xent loss, augmentations, dataset
│   ├── rgcn_predictor.py        # R-GCN + multi-task heads + loss
│   └── alternative_recommender.py # GIN embedding kNN recommender
├── training/
│   ├── pretrain_gin.py          # SimCLR pretraining loop
│   ├── finetune_rgcn.py         # R-GCN fine-tuning loop
│   └── hpo.py                   # Optuna hyperparameter optimization
├── explainability/
│   ├── gnn_explainer.py         # GNNExplainer edge attribution
│   ├── shapley_attribution.py   # Shapley drug risk attribution
│   ├── monte_carlo_dropout.py   # MC Dropout uncertainty estimation
│   └── mechanism_templates.py   # plain-English interaction explanations
├── scoring/
│   └── polypharmacy_score.py    # patient-level risk scoring engine
├── serving/
│   ├── api.py                   # FastAPI app
│   └── schemas.py               # Pydantic request/response models
├── evaluation/
│   ├── benchmark.py             # Hits@K, AUROC, AUPR, severity metrics
│   ├── evaluate_ogbl.py         # OGBL-DDI standard evaluation
│   ├── evaluate_severity.py     # severity classification metrics
│   ├── evaluate_explainability.py
│   └── embedding_analysis.py    # t-SNE, clustering analysis
├── scripts/
│   ├── download_all.py          # download all public datasets
│   ├── fix_data.py              # data repair utilities
│   └── inject_embeddings.py     # pre-compute and cache embeddings
├── frontend/
│   └── src/
│       ├── pages/               # Analyzer, Pairwise, Alternatives, Explore, Directory, Neighborhood, About
│       ├── components/          # Layout, DrugSearchInput, MoleculeViewer, RiskScoreRing
│       └── api.ts               # typed API client
├── tests/                       # pytest suite for all modules
├── train.py                     # main training entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## What Was Built and How

### Iteration History

The project went through ~40+ meaningful iterations across several months:

**Data pipeline** (7 iterations): Started with a simple CSV parser. Discovered DrugBank XML had 14,000+ drugs but only ~4,000 had interaction records. Added TWOSIDES augmentation for multi-drug side effects. Added FAERS adverse event signals as a regression target. Spent 2 iterations fixing edge index dimensionality mismatches between the molecular graph builder and the DDI graph builder.

**Molecular pretraining** (5 iterations): First attempt used basic graph autoencoders. Switched to SimCLR (NT-Xent) after poor embedding quality. Iterated on augmentation ratios (edge drop vs. node masking vs. Gaussian noise) to find what generalizes. Final: 5-layer GIN, 256 hidden dim, 64-dim output.

**R-GCN fine-tuning** (9 iterations): Initial model used FastRGCNConv which OOM'd on 4GB GPU. Switched to RGCNConv with basis decomposition (`num_bases=16`). Added gradient accumulation (4 steps) to handle large batch sizes. Added multi-task heads (severity, type, FAERS) after binary-only model gave poor severity calibration. Final training run: 200 epochs, AdamW, early stopping at patience=20.

**Scoring engine** (6 iterations): Early version was a simple max-severity lookup. Added Shapley attribution to identify the "risk culprit" drug. Added special rule detection (QT, CNS, NSAID+anticoag) as clinical overrides on top of GNN predictions. Added non-linear risk scaling so one contraindicated pair in a small list still scores high.

**API** (4 iterations): Started with synchronous inference. Pre-computed and cached R-GCN drug embeddings at startup so pairwise scoring is O(1) lookup instead of O(full-graph-forward-pass). Added fuzzy drug name resolution with RapidFuzz. Added TWOSIDES augmentation for drug pairs not in the training graph.

**Frontend** (10+ iterations): Built with React + TypeScript + Vite. Pages: Patient Analyzer, Pairwise Lookup, Safer Alternatives, Molecular Explorer (t-SNE), Drug Directory, Chemical Neighborhood Explorer, About. Key design decisions: dark/light mode toggle, interaction severity color coding, drug interaction matrix heatmap, 2D molecular structure rendering via NCI CACTUS API.

---

## Disclaimer

This project is for educational purposes only. Not medical advice. Never change medications without consulting a licensed pharmacist or physician.
