# MedSafe — LinkedIn Post

---

## Section 1: The Post

Built something I'm genuinely proud of.

MedSafe is a polypharmacy drug interaction safety analyzer. You give it a patient's full medication list, and it tells you which combinations are dangerous, why, how confident it is, and which safer alternatives exist. It's the kind of tool that should exist in every pharmacy system but doesn't.

What makes it different from every academic DDI paper: existing research is almost all pairwise binary classifiers — "do drug A and drug B interact? yes/no." That's not how real patients work. Real patients take 5–12 medications simultaneously. MedSafe operates at the patient level with severity-weighted risk scoring, interaction matrix analysis, Shapley value attribution (which drug is the risk culprit?), plain-English mechanism explanations, and GNN embedding-based safer alternative recommendations.

The ML stack, in plain terms:

- **Self-supervised GIN pretraining** — trained the graph encoder on molecular structure alone, no labels, using SimCLR contrastive learning. The model learns drug chemistry before seeing any interaction data.
- **R-GCN multi-task prediction** — relational graph neural network over a drug-target-interaction knowledge graph, jointly predicting interaction probability, severity, mechanism type, and FDA adverse event signal in one pass.
- **GNNExplainer** — shows which graph edges drove each specific prediction. Not a black box.
- **Monte Carlo Dropout** — runs multiple stochastic passes to give confidence intervals, not just point estimates.
- **Shapley values** — identifies which drug in a regimen contributes most to the overall risk score.

Trained on DrugBank (14,000+ drugs), TWOSIDES (4.6M multi-drug side effect pairs), and the OGBL-DDI benchmark. Full stack: PyTorch Geometric, FastAPI, React + TypeScript, Docker, MLflow, GitHub Actions.

Built solo over ~4 months, mostly on weekends and late nights between semester exams and placement prep. Went through 40+ iterations — the R-GCN alone had 9 major rewrites. The first version OOM'd on my 4GB GPU every time. The scoring engine started as a simple max-severity lookup and ended up with Shapley attribution and non-linear risk amplification.

It's open source. If you're working on clinical AI, drug safety, or GNN applications in healthcare, I'd like to hear what you think.

[GitHub link]

#MachineLearning #GraphNeuralNetworks #HealthcareAI #PyTorch #OpenSource #MLEngineering #StudentProject

---

## Section 2: Screenshot Guide

Take these screenshots before posting. Each one serves a specific purpose.

---

### Screenshot 1 — Patient Analyzer: High-Risk Combination

**Page:** `http://localhost:5173/` (Patient Analyzer)

**State:** Enter these drugs: `Warfarin`, `Aspirin`, `Ibuprofen`, `Diazepam`, `Amiodarone`

**Show:**
- The red risk score ring at 85+/100 with "Critical" tier label
- The interaction matrix heatmap showing the danger zones in red/orange
- At least one expanded interaction card with mechanism explanation, CYP enzymes, and confidence score
- The "Risk Culprit" badge showing Warfarin or Amiodarone

**Why this matters:** This is the hero shot. Shows the core value in one frame — not just "interaction detected" but a full scored breakdown with reasoning.

---

### Screenshot 2 — Pairwise Lookup: Warfarin + Aspirin

**Page:** `http://localhost:5173/pairwise`

**State:** Enter `Warfarin` and `Aspirin`

**Show:**
- The mechanism explanation card ("Aspirin inhibits platelet aggregation while Warfarin...")
- The confidence bar and severity badge (Major / Contraindicated)
- CYP enzyme involvement list
- The FAERS adverse event score

**Why this matters:** Shows the explainability — it's not just a risk number, it's a clinical reason.

---

### Screenshot 3 — Molecular Explorer (t-SNE)

**Page:** `http://localhost:5173/molecular`

**State:** Select a drug (e.g., Warfarin), let it highlight with neighbor lines

**Show:**
- The t-SNE scatter plot with colored ATC drug clusters
- One drug highlighted with nearest neighbor connection lines visible
- The drug info panel on the right showing SMILES, molecular weight, ATC class

**Why this matters:** Visual proof of the GNN embedding quality — structurally similar drugs cluster together.

---

### Screenshot 4 — Drug Directory: Drug Profile

**Page:** `http://localhost:5173/directory`

**State:** Search and select `Atorvastatin`

**Show:**
- Left panel: drug metadata, mechanism of action, molecular weight, ATC class
- Right panel: 2D molecular structure rendered from SMILES
- Bottom: Chemical Neighborhood section with 5 similar drugs and similarity percentages

**Why this matters:** Shows the full drug profiler feature — both the structure and the GNN neighborhood.

---

### Screenshot 5 — Chemical Neighborhood Explorer

**Page:** `http://localhost:5173/neighborhood`

**State:** Search `Warfarin`, wait for results

**Show:**
- Center node (Warfarin) with the 5 neighbor cards arranged below
- Similarity bar chart on the right showing GNN cosine similarity scores
- The "How this works" explanation text visible

**Why this matters:** Makes the GNN latent space tangible for a non-ML audience.

---

### Screenshot 6 — Safer Alternatives

**Page:** `http://localhost:5173/alternatives`

**State:** Drug to replace: `Ibuprofen`, current drugs: `Warfarin`, `Aspirin`

**Show:**
- The alternatives list with risk reduction percentages
- Similarity scores and ATC class match badges
- The mechanism explanation for why each alternative is safer

**Why this matters:** Shows the actionability — it doesn't just warn you, it helps you do something about it.

---

### Screenshot 7 — MLflow Training Dashboard

**Command:** `mlflow ui` (in project root)

**Show:**
- The training run with loss curves (binary loss, severity loss, total loss dropping over epochs)
- Metric comparison across runs if multiple exist
- Final Hits@20 and AUROC metrics logged

**Why this matters:** Shows this is a real trained model, not a demo with hardcoded outputs.

---

### Screenshot 8 — Mobile Responsiveness

**How:** Open Chrome DevTools → toggle device toolbar → set to iPhone 14 Pro (390px width)

**Page:** Patient Analyzer with the Warfarin combo loaded

**Show:**
- The nav collapses to hamburger menu
- Risk score ring renders correctly at mobile size
- Interaction cards stack vertically and are readable

**Why this matters:** Production-quality frontend, not a research prototype.

---

### Screenshot 9 — Demo Mode Terminal

**Command:**
```bash
python train.py --demo
```

**Show:**
- The Rich console output with the training panel
- Stage 1 (GIN) and Stage 2 (R-GCN) progress bars completing
- Final metrics printed at the bottom
- Total training time shown

**Why this matters:** Proves the training pipeline works end-to-end in under 15 minutes on any machine.
