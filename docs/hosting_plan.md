# MedSafe — Complete Backup & Hosting Plan

This plan is designed to ensure that if you delete this project from your laptop, all critical resources (including trained weights and processed datasets) are preserved on GitHub, allowing you to restore the project or deploy it from scratch at any time.

---

## 1. GitHub Upload & Backup Strategy

Since you plan to delete this project from your laptop, you need to store the essential files in Git. 

### File Size Analysis & Git Recommendations

| File / Folder | Estimated Size | Git Strategy | Rationale |
|---|---|---|---|
| **Source Code & Configs** | < 2 MB | **Commit to Git** | Core logic (Vite frontend, FastAPI, models) |
| **Model Weights** (`checkpoints/`) | ~15 MB | **Commit to Git** | Files are small enough (GIN: 1.5MB, R-GCN: 13MB) to be pushed directly to GitHub (which has a 100MB limit). No need to host external assets. |
| **Processed Parquet Files** (`data/processed/`) | ~5 MB | **Commit to Git** | Keeps `drugs.parquet`, `smiles_cache.parquet`, and `faers_harm_signals.parquet` so you don't have to rebuild the dataset pipeline. |
| **DDI Heterogeneous Graph** (`data/graphs/ddi_hetero_graph.pt`) | ~18 MB | **Commit to Git** | Pushing this pre-built GNN graph allows you to run evaluations or re-train immediately without parsing raw datasets. |
| **Raw Datasets** (`data/raw/`) | > 700 MB | **Exclude** | The raw DrugBank XML (~120MB) and TWOSIDES raw CSV (~670MB) are too large and subject to academic/proprietary license terms. |

### `.gitignore` configuration

Update your `.gitignore` to allow committing model checkpoints and processed indices, but exclude large raw datasets, MLflow logs, and local dependencies:

```gitignore
# Exclude raw large datasets
data/raw/*

# Keep processed graphs and caches for recovery, but ignore local scratchpads
data/processed/*.log
data/graphs/*.log

# Allow checkpoints in Git since they are <20MB
!checkpoints/
!checkpoints/**/*.pt

# Ignore local runtimes, logs, and build outputs
mlruns/
node_modules/
frontend/dist/
frontend/.vite/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.env
*.env.local
.DS_Store
Thumbs.db
.vscode/
.idea/
```

---

## 2. API Hosting — Hugging Face Spaces (FastAPI Backend)

Hugging Face Spaces is the recommended option for hosting the backend on the free tier. It provides 16 GB of CPU RAM (more than enough for the R-GCN model), is always-on (or sleeps when inactive, waking up automatically within ~20 seconds), and is free.

### Step-by-step Deployment Steps

1. **Create Space**: Go to [huggingface.co/new-space](https://huggingface.co/new-space). Select **Docker** as the SDK, and choose the free CPU Basic tier.
2. **Add Configuration**: Create a `README.md` at the root of the Space repository with the following YAML header:
   ```yaml
   ---
   title: MedSafe API
   emoji: 💊
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 8000
   ---
   ```
3. **Dockerfile**: Create a `Dockerfile` in the root of the project to bundle the server:
   ```dockerfile
   FROM python:3.11-slim

   WORKDIR /app

   # Install compiler tools
   RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

   # Install PyTorch CPU
   RUN pip install torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu
   RUN pip install torch-geometric

   # Install pip requirements
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   # Copy source files
   COPY . .

   EXPOSE 8000
   CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
4. **Push Code**: Sync your GitHub repository to Hugging Face or push directly using git:
   ```bash
   git remote add space https://huggingface.co/spaces/yourusername/medsafe-api
   git push space main --force
   ```

---

## 3. Frontend Hosting — Vercel

Vercel provides free, high-performance static hosting for Single Page Applications (SPAs) like Vite + React.

### Step-by-step Deployment Steps

1. **Create Vercel Project**: Go to [vercel.com](https://vercel.com) and click **Add New** → **Project**.
2. **Import Repository**: Connect your GitHub account and import your `medsafe` repository.
3. **Configure Settings**:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. **Add Environment Variable**: Add an environment variable to link the frontend to your backend API:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://yourusername-medsafe-api.hf.space` (use your actual Hugging Face Space URL)
5. **Deploy**: Click **Deploy**. Vercel will build the frontend and provide a live `medsafe.vercel.app` link.

