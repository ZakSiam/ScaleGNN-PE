#!/usr/bin/env python
"""Render the 3CLPro / rtcb docking results as clean PNG tables.

Columns: seeds, distinct top-1% hits, enrichment factor, runtime, peak CUDA allocation.

Peak CUDA *allocation* is reported rather than reserved: reserved is PyTorch caching-allocator
behaviour (GP reserves 78.6 GB while allocating 9.9 GB, and chunking the same math moves
reserved by 41x while allocation barely changes), so it does not measure what the algorithm
needs. Peak RSS is omitted entirely -- it is ~4.8-5.8 GB for every method because it is
dominated by holding 10^6 Graph objects, i.e. a property of the domain, not the method.

Runtimes are as-implemented and NOT optimisation-matched: the GP path spends ~915 of its
~920 s in a cuSOLVER trsm pathology (chunking the identical math gives 15.0 s with bit-identical
actions), and the GNN path recomputes cacheable node aggregations every forward call. The
runtime column is therefore flagged, not presented as a property of the algorithms.
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

GROUP = {"GNN-PE": "#E8F0F7", "GNN-SS": "#FDF0E4", "GP": "#F3EEF6", "random": "#F2F2F2"}

SRC = [
    ("random",                      "docking_{t}_dock_norm_timed/random"),
    ("GP-UCB",                      "docking_{t}_dock_norm_timed/gpucb"),
    ("GP-EI",                       "docking_{t}_dock_norm_timed/gpei"),
    ("GP-TS",                       "docking_{t}_dock_norm_timed/gpts"),
    ("GNN-PE pivchol",              "docking_{t}_dock_norm_timed/fullntk"),
    ("GNN-PE pivchol + LoRA",       "docking_{t}_dock_norm_timed/peft_lora"),
    ("GNN-PE pivchol + last_layer", "docking_{t}_dock_norm_timed/peft_lastlayer"),
    ("GNN-SS FJ",                   "docking_{t}_dock_norm_gnnss_fjl_B"),
]


def family(name):
    for k in ("GNN-PE", "GNN-SS", "GP"):
        if name.startswith(k):
            return k
    return "random"


def rows_for(target):
    y = pq.read_table(f"data/docking/{target}_processed.parquet", columns=["dock"]).column("dock").to_numpy()
    istop = (y >= y.max())
    N, K = len(y), int(istop.sum())
    out = []
    for label, pat in SRC:
        rec = []
        for f in glob.glob(f"results/{pat.format(t=target)}/**/*.json", recursive=True):
            d = json.load(open(f))
            a = np.asarray(d["exp_results"]["actions"], np.int64)
            u = int(istop[np.unique(a)].sum())
            rec.append((u, (u / len(a)) / (K / N),
                        d["duration_total"] * 60,
                        (d.get("peak_cuda_alloc_mb") or 0) / 1024))
        if not rec:
            continue
        r = np.array(rec)
        out.append((label, len(rec), r[:, 0], r[:, 1], r[:, 2].mean(), r[:, 3].mean()))
    return sorted(out, key=lambda x: -x[2].mean()), N, K


fig, axes = plt.subplots(2, 1, figsize=(11.6, 8.2))
for ax, target in zip(axes, ("3CLPro", "rtcb")):
    rows, N, K = rows_for(target)
    best_hits = max(r[2].mean() for r in rows)
    best_mem = min(r[5] for r in rows if r[5] > 0.02)     # exclude random's unused GP matrix
    cells = [[n, str(s), f"{h.mean():.1f} ± {h.std():.1f}", f"{e.mean():.2f} ± {e.std():.2f}",
              f"{t:.0f} s", f"{m:.3f} GB"] for n, s, h, e, t, m in rows]
    tbl = ax.table(cellText=cells,
                   colLabels=["method", "seeds", "hits (top-1%)", "EF", "runtime", "peak CUDA alloc"],
                   colWidths=[0.30, 0.09, 0.17, 0.16, 0.13, 0.17],
                   cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0)
        if r == 0:
            cell.set_facecolor("#2F2F2F")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            name, _, h, _, _, m = rows[r - 1]
            cell.set_facecolor(GROUP[family(name)])
            if (c in (0, 2, 3) and h.mean() == best_hits) or (c in (0, 5) and m == best_mem):
                cell.set_text_props(fontweight="bold")
        if c > 0:
            cell.set_text_props(ha="center")
    ax.axis("off")
    ax.set_title(f"{target}     N = {N:,},   top-1% = {K:,},   T = 300",
                 fontsize=12, pad=10, fontweight="bold")

fig.suptitle("Docking results: distinct top-1% hits, enrichment factor, runtime, peak CUDA allocation",
             fontsize=12.5, y=0.985)
fig.text(0.5, 0.012,
         "hits = distinct top-1% molecules acquired;  EF = (hits/300) / (top-1%/N), random = 1.0;  "
         "mean ± s.d. over seeds.  Bold = best hits / lowest memory.\n"
         "Runtime is as-implemented and not optimisation-matched: the GP path is dominated by a "
         "cuSOLVER trsm pathology (identical math, chunked = 15.0 s).",
         ha="center", fontsize=8, color="#666666")
fig.tight_layout(rect=[0, 0.045, 1, 0.955])

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/docking_table_3clpro_rtcb.png", dpi=220, bbox_inches="tight", facecolor="white")
print("wrote figures/docking_table_3clpro_rtcb.png")
