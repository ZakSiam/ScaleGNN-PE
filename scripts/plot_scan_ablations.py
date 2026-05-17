#!/usr/bin/env python3
"""Plot the 8-way GNN-PE scan ablation comparison from result JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from result_utils import load_group, mean_and_se, print_runtime_table, require_subdir, write_runtime_csv


ABLATION_SPECS = [
    ("GNN-PE scratch (full scan)", "qm9_phasedgp_T300_N1000/scratch"),
    ("GNN-PE finetune (full scan)", "qm9_phasedgp_T300_N1000/finetune"),
    ("FFT (Eq. 1 + Eq. 2)", "qm9_phasedgp_T300_N1000_fft/finetune"),
    ("FFT (Eq. 1 only)", "qm9_phasedgp_T300_N1000_fft_C1only/finetune"),
    ("FFT (Eq. 2 only)", "qm9_phasedgp_T300_N1000_fft_C2only/finetune"),
    ("Graph k-means (Eq. 1 + Eq. 2)", "qm9_phasedgp_T300_N1000_kmeans/finetune"),
    ("Graph k-means (Eq. 1 only)", "qm9_phasedgp_T300_N1000_kmeans_C1only/finetune"),
    ("Graph k-means (Eq. 2 only)", "qm9_phasedgp_T300_N1000_kmeans_C2only/finetune"),
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
            "axes.linewidth": 1.1,
            "lines.linewidth": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the 8-way GNN-PE scan ablation comparison.")
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--out_dir", type=Path, default=Path("figures/scan_ablations"))
    args = parser.parse_args()

    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = [load_group(label, require_subdir(args.results_root, relative)) for label, relative in ABLATION_SPECS]

    fig, ax = plt.subplots(figsize=(14.0, 8.0))
    for group in groups:
        x, mean, se = mean_and_se(group)
        ax.plot(x, mean, label=f"{group.label} (n={group.n})")
        ax.fill_between(x, mean - se, mean + se, alpha=0.16)
    ax.set_xlabel("Rounds (t)")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("QM9: GNN-PE scan ablations (mean ± SE over seeds)", pad=18)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(args.out_dir / "scan_ablation_8way.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "scan_ablation_8way.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    write_runtime_csv(groups, args.out_dir / "scan_ablation_runtime_summary.csv")
    print_runtime_table(groups)
    print("\nWrote:")
    print(f"  {args.out_dir / 'scan_ablation_8way.pdf'}")
    print(f"  {args.out_dir / 'scan_ablation_runtime_summary.csv'}")


if __name__ == "__main__":
    main()
