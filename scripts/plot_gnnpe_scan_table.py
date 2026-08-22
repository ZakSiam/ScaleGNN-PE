#!/usr/bin/env python
"""GNN-PE scratch/finetune x (C.1)/(C.2)/both table for QM9 and ZINC.

Isolates the GNN-PE (PhasedGnnUCB) method and reports, for each training mode and each
domain-scan apply target, the three quantities that trade off against each other:

    * cumulative regret at T = 300 (lower is better; normalize_rewards=True so comparable)
    * wall-clock time (min)
    * peak CUDA allocation (GB)

The (C.1)/(C.2)/both axis is `domain_scan_apply_to` and is only defined in finetune mode --
scan reductions are never applied when training from scratch. (C.1) = max-variance arm
selection, (C.2) = maximizer-set update. Reduced finetune rows use the masked (mask_revisit)
variant, matching scripts/plot_regret_tables.py. Peak RSS is omitted: it is ~1 GB for every
row because it is dominated by holding the 10^3 Graph objects, i.e. a domain property.
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# row tint by training/scan block
TINT_MODE = "#E8F0F7"     # scratch / finetune full
TINT_SCAN = "#FDF7EE"     # finetune + scan reduction


def agg(ds, filt):
    reg, dur, cuda, seeds = [], [], [], []
    for f in glob.glob(f"results/{ds}_phasedgp_T300_N1000_neuron1024*/**/*.json", recursive=True):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        p = j["params"]
        if p.get("num_actions") != 1000 or p.get("neuron_per_layer") != 1024:
            continue
        if not filt(p, f):
            continue
        reg.append(j["exp_results"]["regrets"][-1])
        dur.append(j.get("duration_total"))
        cuda.append((j.get("peak_cuda_alloc_mb") or 0) / 1024.0)
        seeds.append(p.get("seed"))
    if not reg:
        return None
    a = np.array(reg)
    return len(a), a.mean(), a.std(), float(np.mean(dur)), float(np.mean(cuda))


def rows_for(ds):
    def pe(scan, apply_to, mode="finetune"):
        return agg(ds, lambda p, f:
                   p.get("train_mode") == mode
                   and p.get("domain_scan_method") == scan
                   and p.get("domain_scan_apply_to") == apply_to
                   and (scan == "full" or "_masked" in f))
    spec = [
        ("scratch  ·  full",            pe("full", "both", "scratch"),  TINT_MODE),
        ("finetune ·  full",            pe("full", "both", "finetune"), TINT_MODE),
    ]
    for scan, nice in (("fft", "FFT"), ("pivchol", "pivoted-Chol."), ("graph_kmeans", "graph k-means")):
        for ap in ("c1", "c2", "both"):
            lab = f"finetune ·  {nice} ·  {ap.upper() if ap!='both' else 'both'}"
            spec.append((lab, pe(scan, ap), TINT_SCAN))
    out = []
    for name, v, tint in spec:
        if v is None:
            continue
        n, m, s, t, c = v
        out.append((name, n, m, s, t, c, tint))
    return out


fig, axes = plt.subplots(2, 1, figsize=(9.6, 10.4))
for ax, ds in zip(axes, ("qm9", "zinc")):
    rows = rows_for(ds)
    best_reg = min(r[2] for r in rows)
    fast = min(r[4] for r in rows)
    lomem = min(r[5] for r in rows)
    cells = [[n, str(k), f"{m:.1f} ± {s:.1f}", f"{t:.2f}", f"{c:.2f}"]
             for n, k, m, s, t, c, _ in rows]
    tbl = ax.table(cellText=cells,
                   colLabels=["GNN-PE variant", "seeds", "cumulative regret", "time (min)", "peak CUDA (GB)"],
                   colWidths=[0.42, 0.10, 0.22, 0.13, 0.16], cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0)
        if r == 0:
            cell.set_facecolor("#2F2F2F")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            name, _, mean, _, t, mem, tint = rows[r - 1]
            cell.set_facecolor(tint)
            if (c in (0, 2) and mean == best_reg) or (c == 3 and t == fast) or (c == 4 and mem == lomem):
                cell.set_text_props(fontweight="bold")
        if c > 0:
            cell.set_text_props(ha="center")
    ax.axis("off")
    ax.set_title(f"{ds.upper()}    |G| = 1000,  T = 300", fontsize=12.5, pad=12, fontweight="bold")

fig.suptitle("GNN-PE: scratch/finetune x (C.1)/(C.2)/both  —  cumulative regret, time, memory",
             fontsize=12.5, y=0.995)
fig.text(0.5, 0.012,
         "(C.1) = max-variance arm selection, (C.2) = maximizer-set update; apply target = domain_scan_apply_to "
         "(finetune only).  Reduced rows use the masked variant.\n"
         "regret mean ± s.d. over seeds (lower better); time & CUDA are means.  Bold = best regret / fastest / "
         "lowest memory.  Peak RSS ~1 GB for all rows (domain-dominated).",
         ha="center", fontsize=8, color="#666666")
fig.tight_layout(rect=[0, 0.03, 1, 0.965])

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/gnnpe_scan_table_qm9_zinc.png", dpi=220, bbox_inches="tight", facecolor="white")
print("wrote figures/gnnpe_scan_table_qm9_zinc.png")
