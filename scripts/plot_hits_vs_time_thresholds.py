#!/usr/bin/env python
"""Distinct top-X% hits vs wall-clock time, 0-300 s window, on 3CLPro / rtcb / 6T2W.

Generalises plot_hits_vs_time_300s.py over the active-set threshold X in {1,5,10}%:
the curve rises by one whenever an acquired molecule is a NEW member of the top-X% set,
holds flat while a method is still computing, and is truncated at the 300 s right edge
for methods whose 300 rounds run past the budget.

Active set: top-1% uses the parquet's clipped max plateau (the 10,000 arms == max, matching
the existing hits@1% figures); top-5%/10% use the (1-frac) empirical quantile threshold.

Hits are counted DISTINCT: re-acquiring a molecule already found does not increment.
Outputs: figures/hits_vs_time_<target>_top<pct>pct_300s.{png,pdf}
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

BUDGET = 300.0
GRID = np.linspace(0.0, BUDGET, 601)          # 0.5 s resolution
TARGETS = ("3CLPro", "rtcb", "6T2W")
THRESHOLDS = (0.01, 0.05)                     # active-set fractions

# (label, results-path template, colour, linestyle)  -- fixed order, never cycled
METHODS = [
    ("Random",                     "docking_{t}_dock_norm_timed/random",         "#8C8C8C", "-"),
    ("GP-UCB",                     "docking_{t}_dock_norm_timed/gpucb",          "#4C78A8", "-"),
    ("GP-EI",                      "docking_{t}_dock_norm_timed/gpei",           "#72B7B2", "-"),
    ("GP-TS",                      "docking_{t}_dock_norm_timed/gpts",           "#B279A2", "-"),
    ("GNN-SS",                     "docking_{t}_dock_norm_gnnss_fjl_B",          "#F58518", "--"),
    ("ScaleGNN-PE",                "docking_{t}_dock_norm_timed/fullntk",        "#E45756", "-"),
    ("ScaleGNN-PE + Last-Layer",   "docking_{t}_dock_norm_timed/peft_lastlayer", "#E45756", "--"),
    ("ScaleGNN-PE + LoRA",         "docking_{t}_dock_norm_timed/peft_lora",      "#E45756", ":"),
]


def active_mask(target, frac):
    y = pq.read_table(f"data/docking/{target}_processed.parquet", columns=["dock"]).column("dock").to_numpy()
    if abs(frac - 0.01) < 1e-9:
        mask = (y >= y.max())                 # clipped top-1% plateau (matches existing figures)
    else:
        mask = (y >= np.quantile(y, 1.0 - frac))
    return mask, int(mask.sum())


def resolve(pattern, target):
    """Prefer GPU-timed GP results when they exist for this target.

    The 3CLPro and rtcb GP baselines in the archived *_timed dirs were run on GPU (~15 min,
    9.89 GB CUDA), but 6T2W's were forced to CPU by scripts/run_docking_timed.py (~43 min,
    0 GB), which
    would make 6T2W's GP curves ~2.9x slower than the other targets' for no algorithmic
    reason. results/docking_6T2W_dock_norm_timed_gpu/ holds the GPU re-run (10 seeds,
    per-seed regret bit-identical to the CPU runs), so the comparison is like-for-like.
    Only the GP baselines have a _timed_gpu counterpart; everything else falls through.
    """
    p = pattern.format(t=target)
    gpu = p.replace("_dock_norm_timed/", "_dock_norm_timed_gpu/")
    if gpu != p and glob.glob(f"results/{gpu}/**/*.json", recursive=True):
        return gpu
    return p


def curves(pattern, target, active):
    """One step-curve per seed, resampled onto GRID. Holds final value past a run's last round."""
    out = []
    for f in glob.glob(f"results/{resolve(pattern, target)}/**/*.json", recursive=True):
        e = json.load(open(f))["exp_results"]
        el = np.asarray(e.get("elapsed", e.get("elapsed_sec", [])), dtype=float)
        a = np.asarray(e["actions"], dtype=np.int64)
        if len(el) != len(a) or len(el) == 0:
            continue
        seen, hits, run = set(), np.empty(len(a), dtype=float), 0
        for i, act in enumerate(a):
            if active[act] and act not in seen:
                run += 1
            seen.add(act)
            hits[i] = run
        idx = np.searchsorted(el, GRID, side="right") - 1
        y = np.where(idx >= 0, hits[np.clip(idx, 0, len(hits) - 1)], 0.0)
        out.append(y)
    return np.array(out) if out else None


os.makedirs("figures", exist_ok=True)
for target in TARGETS:
    for frac in THRESHOLDS:
        pct = int(round(frac * 100))
        active, nact = active_mask(target, frac)
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        for label, pat, colour, ls in METHODS:
            C = curves(pat, target, active)
            if C is None:
                continue
            ax.plot(GRID, C.mean(axis=0), color=colour, lw=2.0, ls=ls, label=label, zorder=3)
        ax.set_xlabel("wall-clock time (s)")
        ax.set_ylabel(f"distinct top-{pct}% molecules found")
        ax.set_xlim(0, BUDGET)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#EDEDED", lw=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        fig.tight_layout()
        stem = f"figures/hits_vs_time_{target}_top{pct:02d}pct_300s"
        for ext in ("png", "pdf"):
            fig.savefig(f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {stem}.png  (top-{pct}% active set = {nact})")
