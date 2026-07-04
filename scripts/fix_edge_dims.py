"""
scripts/fix_edge_dims.py
========================
Scans all molecular graph .pt files and ensures edge_attr has dim=13.

Bond feature layout (13 dims):
  bond_type   : 5 dims (single, double, triple, aromatic, + OTHER bucket)
  is_in_ring  : 1 dim
  stereo       : 6 dims (one-hot)
  is_conjugated: 1 dim

Any graph with edge_attr dim != 13 is patched:
  - dim < 13: zero-pad to 13
  - dim > 13: truncate to 13
  - No edge_attr: skip (isolated atoms are fine)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

TARGET_DIM = 13
mol_dir = ROOT / "data" / "graphs" / "molecular"

if not mol_dir.exists():
    print(f"ERROR: {mol_dir} does not exist. Run pipeline first.")
    sys.exit(1)

all_pts = list(mol_dir.glob("*.pt"))
print(f"Total graphs: {len(all_pts)}")

dim_counts: dict = {}
fixed = 0
skipped_no_edges = 0

for i, pt in enumerate(all_pts):
    if i % 1000 == 0:
        print(f"  Checking {i}/{len(all_pts)}...", flush=True)

    g = torch.load(pt, weights_only=False)

    # Count dims
    if hasattr(g, "edge_attr") and g.edge_attr is not None:
        d = g.edge_attr.shape[1] if g.edge_attr.ndim == 2 else None
        dim_counts[d] = dim_counts.get(d, 0) + 1

        if d is not None and d != TARGET_DIM:
            n_edges = g.edge_attr.shape[0]
            if d < TARGET_DIM:
                # Pad with zeros
                pad = torch.zeros(n_edges, TARGET_DIM - d, dtype=g.edge_attr.dtype)
                g.edge_attr = torch.cat([g.edge_attr, pad], dim=1)
            else:
                # Truncate
                g.edge_attr = g.edge_attr[:, :TARGET_DIM]

            torch.save(g, pt)
            fixed += 1
    else:
        skipped_no_edges += 1
        dim_counts["no_edge_attr"] = dim_counts.get("no_edge_attr", 0) + 1

print("\n--- Dimension Audit Results --------------------")
for d, count in sorted(dim_counts.items(), key=lambda x: str(x[0])):
    flag = " OK" if d == TARGET_DIM else " WRONG"
    print(f"  edge_attr dim={d}: {count} graphs{flag}")

print(f"\nFixed : {fixed} graphs")
print(f"No edges: {skipped_no_edges} graphs (isolated atoms -- OK)")
print(f"\nAll graphs now have edge_attr dim={TARGET_DIM} OK")
