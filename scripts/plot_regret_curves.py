#!/usr/bin/env python
"""Two-panel per-round cumulative-regret curves for QM9 and ZINC (|G|=1000, T=300).

x = round t, y = cumulative regret (mean over seeds). Method->dir mapping mirrors
scripts/plot_regret_tables.py (the authoritative source). All runners use
normalize_rewards=True on these datasets, so curves are comparable within a panel.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_curves(pat, filt=None):
    """Return list of full regret trajectories (one per seed, num_actions==1000)."""
    by_seed = {}
    for f in glob.glob(f"results/{pat}/**/*.json", recursive=True):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        p = j.get("params", {})
        if p.get("num_actions") != 1000:
            continue
        if filt and not filt(p, f):
            continue
        try:
            by_seed[p.get("seed")] = np.asarray(j["exp_results"]["regrets"], dtype=float)
        except Exception:
            continue
    return [by_seed[s] for s in sorted(by_seed)]


def mean_std_curve(curves):
    if not curves:
        return None, None
    L = min(len(c) for c in curves)
    A = np.stack([c[:L] for c in curves], axis=0)
    return A.mean(axis=0), A.std(axis=0)


def pe(ds, scan, mode="finetune"):
    """Curves for the headline GNN-PE configuration at the given scan method.

    The directory is named EXACTLY, not globbed. A `_neuron1024*` wildcard also matches
    the ablation directories that share the prefix -- the beta sweep
    (..._pivchol_both_masked_beta{0.5,2,4}x), the representative-count sweep
    (..._pivchol_both_masked_m{100,500,1000,2000}), and the full-scan mask ablations
    (..._finetune_full{,_nomask}). Since load_curves() keys by seed, those extra runs
    silently overwrite the intended ones with whichever file glob order happened to
    return last, mixing several configurations into a single curve.
    """
    exact = (f"{ds}_phasedgp_T300_N1000_neuron1024_finetune_full_masked" if scan == "full"
             else f"{ds}_phasedgp_T300_N1000_neuron1024_pivchol_both_masked")
    return load_curves(
        exact,
        lambda p, f: p.get("train_mode") == mode and p.get("neuron_per_layer") == 1024
        and p.get("domain_scan_method") == scan and p.get("domain_scan_apply_to") == "both")


# (label, color, linestyle) — Okabe-Ito hues; ScaleGNN-PE family shares vermillion,
# GP family shares purple, distinguished by line style.
BLUE, ORANGE, GREEN, VERM, PURPLE, GRAY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#7F7F7F")
STYLE = {
    "GNN-UCB":            (BLUE,   "-"),
    "GNN-SS":             (ORANGE, "-"),
    "GNN-PE":             (GREEN,  "-"),
    "ScaleGNN-PE (full)": (VERM,   "-"),
    "ScaleGNN-PE (LoRA)": (VERM,   "--"),
    "ScaleGNN-PE (Last)": (VERM,   ":"),
    "GP-EI":              (PURPLE, "-"),
    "GP-TS":              (PURPLE, "--"),
    "GP-UCB":             (PURPLE, ":"),
    "Random":             (GRAY,   "-"),
}


def rows(ds):
    return [
        ("GNN-UCB", load_curves(f"{ds}_gnnucb_T300_N1000_matched",
                                lambda p, f: p.get("train_mode") == "finetune")),
        ("GNN-SS", load_curves(f"{ds}_gnnss_T300_N1000_K300")),
        ("GNN-PE", pe(ds, "full", "finetune")),
        ("ScaleGNN-PE (full)", pe(ds, "pivchol")),
        ("ScaleGNN-PE (LoRA)", load_curves(f"{ds}_peft_T300_N1000_neuron1024_lora_pivchol_both_masked")),
        ("ScaleGNN-PE (Last)", load_curves(f"{ds}_peft_T300_N1000_neuron1024_lastlayer_pivchol_both_masked")),
        ("GP-EI", load_curves(f"{ds}_gp_baselines_T300_N1000/gpei")),
        ("GP-TS", load_curves(f"{ds}_gp_baselines_T300_N1000/gpts")),
        ("GP-UCB", load_curves(f"{ds}_gp_baselines_T300_N1000/gpucb")),
        ("Random", load_curves(f"{ds}_gp_baselines_T300_N1000/random")),
    ]


os.makedirs("figures", exist_ok=True)
for ds in ("qm9", "zinc"):
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for name, curves in rows(ds):
        m, sd = mean_std_curve(curves)
        if m is None:
            print(f"[warn] no data: {ds} {name}")
            continue
        c, ls = STYLE[name]
        t = np.arange(1, len(m) + 1)
        ax.fill_between(t, m - sd, m + sd, color=c, alpha=0.12, linewidth=0)
        ax.plot(t, m, label=name, color=c, linestyle=ls, linewidth=1.9)
    ax.set_xlabel("round $t$", fontsize=11)
    ax.set_ylabel("cumulative regret", fontsize=11)
    ax.set_xlim(1, 300)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=1)
    fig.tight_layout()
    out = f"figures/regret_curves_{ds}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)
