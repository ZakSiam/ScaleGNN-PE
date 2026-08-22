#!/usr/bin/env python
"""Render the QM9 / ZINC |G|=1000 cumulative-regret tables as clean PNGs.

Both datasets use normalize_rewards=True in every runner (GP and GNN alike), so the regret
values are unit-consistent and directly comparable down the column. This is NOT true of the
docking tables, where the GP runner uses raw rewards and the GNN runners z-score them.
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GROUP = {  # method -> family, used only for row tinting
    "GNN-PE": "#E8F0F7", "GNN-UCB": "#EAF4EE", "GNN-SS": "#FDF0E4",
    "GP": "#F3EEF6", "random": "#F2F2F2",
}


def family(name):
    for k in ("GNN-PE", "GNN-UCB", "GNN-SS", "GP"):
        if name.startswith(k):
            return k
    return "random"


def load(pat, filt=None):
    out = {}
    for f in glob.glob(f"results/{pat}/**/*.json", recursive=True):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        p = j["params"]
        if p.get("num_actions") != 1000:
            continue
        if filt and not filt(p, f):
            continue
        out[p.get("seed")] = j["exp_results"]["regrets"][-1]
    return out


def rows_for(ds):
    def pe(scan, mode="finetune"):
        return load(f"{ds}_phasedgp_T300_N1000_neuron1024*",
                    lambda p, f: p.get("train_mode") == mode
                    and p.get("neuron_per_layer") == 1024
                    and p.get("domain_scan_method") == scan
                    and p.get("domain_scan_apply_to") == "both"
                    and (scan == "full" or "_masked" in f))
    spec = [
        ("random",                      load(f"{ds}_gp_baselines_T300_N1000/random")),
        ("GP-EI",                       load(f"{ds}_gp_baselines_T300_N1000/gpei")),
        ("GP-TS",                       load(f"{ds}_gp_baselines_T300_N1000/gpts")),
        ("GP-UCB",                      load(f"{ds}_gp_baselines_T300_N1000/gpucb")),
        ("GNN-UCB (scratch)",           load(f"{ds}_gnnucb_T300_N1000_matched",
                                             lambda p, f: p.get("train_mode") == "scratch")),
        ("GNN-UCB (finetune)",          load(f"{ds}_gnnucb_T300_N1000_matched",
                                             lambda p, f: p.get("train_mode") == "finetune")),
        ("GNN-SS FJ",          load(f"{ds}_gnnss_T300_N1000_K300")),
        ("GNN-PE full (scratch)",       pe("full", "scratch")),
        ("GNN-PE full (finetune)",      pe("full", "finetune")),
        ("GNN-PE graph_kmeans",         pe("graph_kmeans")),
        ("GNN-PE pivchol",              pe("pivchol")),
        ("GNN-PE fft",                  pe("fft")),
        ("GNN-PE pivchol + LoRA",       load(f"{ds}_peft_T300_N1000_neuron1024_lora_pivchol_both_masked")),
        ("GNN-PE pivchol + last_layer", load(f"{ds}_peft_T300_N1000_neuron1024_lastlayer_pivchol_both_masked")),
    ]
    out = []
    for name, v in spec:
        if not v:
            continue
        a = np.array([v[s] for s in sorted(v)])
        out.append((name, len(a), a.mean(), a.std()))
    return sorted(out, key=lambda r: r[2])


fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
for ax, ds in zip(axes, ("qm9", "zinc")):
    rows = rows_for(ds)
    best = min(r[2] for r in rows)
    cells = [[n, str(k), f"{m:.1f} ± {s:.1f}"] for n, k, m, s in rows]
    tbl = ax.table(cellText=cells,
                   colLabels=["method", "seeds", "cumulative regret"],
                   colWidths=[0.54, 0.14, 0.32], cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0)
        if r == 0:
            cell.set_facecolor("#2F2F2F")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            name, _, mean, _ = rows[r - 1]
            cell.set_facecolor(GROUP[family(name)])
            if mean == best:
                cell.set_text_props(fontweight="bold")
        if c > 0:
            cell.set_text_props(ha="center")
    ax.axis("off")
    ax.set_title(f"{ds.upper()}   |G| = 1000,  T = 300", fontsize=12.5, pad=14, fontweight="bold")

fig.suptitle("Cumulative regret (lower is better), mean ± s.d. over seeds", fontsize=13, y=0.98)
fig.text(0.5, 0.015,
         "All runners use normalize_rewards=True on these datasets, so values are directly "
         "comparable within a column.  Bold = best.",
         ha="center", fontsize=8.5, color="#666666")
fig.tight_layout(rect=[0, 0.035, 1, 0.95])

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/regret_table_qm9_zinc.png", dpi=220, bbox_inches="tight",
            facecolor="white")
print("wrote figures/regret_table_qm9_zinc.png")
