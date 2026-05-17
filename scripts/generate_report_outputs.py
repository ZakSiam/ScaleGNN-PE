#!/usr/bin/env python3
"""Generate the main QM9 report figures and runtime summary from result JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from result_utils import load_group, mean_and_se, print_runtime_table, require_subdir, write_runtime_csv


MAIN_COMPARISON = [
    ("GP-UCB", "qm9_gp_baselines_T300_N1000/gpucb"),
    ("GP-EI", "qm9_gp_baselines_T300_N1000/gpei"),
    ("GP-TS", "qm9_gp_baselines_T300_N1000/gpts"),
    ("Random", "qm9_gp_baselines_T300_N1000/random"),
    ("GNN-PE (finetune)", "qm9_phasedgp_T300_N1000/finetune"),
    ("GNN-PE (scratch)", "qm9_phasedgp_T300_N1000/scratch"),
    ("GNN-UCB (finetune)", "qm9_gnnucb_T300_N1000/finetune"),
    ("GNN-UCB (scratch)", "qm9_gnnucb_T300_N1000/scratch"),
]

GNNPE_SCAN_COMPARISON = [
    ("GNN-PE scratch", "qm9_phasedgp_T300_N1000/scratch"),
    ("GNN-PE finetune", "qm9_phasedgp_T300_N1000/finetune"),
    ("Finetune + FFT scan", "qm9_phasedgp_T300_N1000_fft/finetune"),
    ("Finetune + Graph k-means scan", "qm9_phasedgp_T300_N1000_kmeans/finetune"),
]

RUNTIME_ORDER = [
    ("GNN-UCB fine-tune", "qm9_gnnucb_T300_N1000/finetune"),
    ("GNN-PE fine-tune", "qm9_phasedgp_T300_N1000/finetune"),
    ("GNN-PE scratch", "qm9_phasedgp_T300_N1000/scratch"),
    ("GNN-UCB scratch", "qm9_gnnucb_T300_N1000/scratch"),
    ("GNN-PE fine-tune + FFT scan (Eq. 1 and Eq. 2)", "qm9_phasedgp_T300_N1000_fft/finetune"),
    ("GNN-PE fine-tune + graph k-means scan (Eq. 1 and Eq. 2)", "qm9_phasedgp_T300_N1000_kmeans/finetune"),
    ("GNN-PE fine-tune + FFT scan (Eq. 1 only)", "qm9_phasedgp_T300_N1000_fft_C1only/finetune"),
    ("GNN-PE fine-tune + graph k-means scan (Eq. 1 only)", "qm9_phasedgp_T300_N1000_kmeans_C1only/finetune"),
    ("GNN-PE fine-tune + FFT scan (Eq. 2 only)", "qm9_phasedgp_T300_N1000_fft_C2only/finetune"),
    ("GNN-PE fine-tune + graph k-means scan (Eq. 2 only)", "qm9_phasedgp_T300_N1000_kmeans_C2only/finetune"),
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 13,
            "axes.linewidth": 1.1,
            "lines.linewidth": 2.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_required(root: Path, specs: list[tuple[str, str]]):
    return [load_group(label, require_subdir(root, relative)) for label, relative in specs]


def plot_figure_1a(groups, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    for group in groups:
        x, mean, se = mean_and_se(group)
        ax.plot(x, mean, label=group.label)
        ax.fill_between(x, mean - se, mean + se, alpha=0.18)
    ax.set_xlabel("Round (t)")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("QM9 |G|=1000, T=300: Cumulative regret (mean ± SE over 10 seeds)", pad=16)
    ax.legend(loc="upper left", frameon=True)
    ax.tick_params(length=6, width=1.1)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_1a_gnnpe_scans.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "figure_1a_gnnpe_scans.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_figure_1b(groups, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    for group in groups:
        x, mean, se = mean_and_se(group)
        ax.plot(x, mean, label=f"{group.label} (n={group.n})")
        ax.fill_between(x, mean - se, mean + se, alpha=0.16)
    ax.set_xlabel("Rounds (t)")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("QM9: cumulative regret (mean ± SE over seeds)", pad=18)
    ax.grid(True, alpha=0.22)
    ax.tick_params(length=6, width=1.1)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=2,
        frameon=True,
        columnspacing=1.6,
        handlelength=2.2,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "figure_1b_main_comparison.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "figure_1b_main_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _load_available_runtime_groups(root: Path):
    groups = []
    for label, relative in RUNTIME_ORDER:
        directory = root / relative
        if directory.is_dir():
            groups.append(load_group(label, directory))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the two main report figures and runtime summary.")
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument("--out_dir", type=Path, default=Path("figures/report"))
    args = parser.parse_args()

    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    figure_1a_groups = _load_required(args.results_root, GNNPE_SCAN_COMPARISON)
    figure_1b_groups = _load_required(args.results_root, MAIN_COMPARISON)
    runtime_groups = _load_available_runtime_groups(args.results_root)

    plot_figure_1a(figure_1a_groups, args.out_dir)
    plot_figure_1b(figure_1b_groups, args.out_dir)
    write_runtime_csv(runtime_groups, args.out_dir / "runtime_summary.csv")
    print_runtime_table(runtime_groups)

    print("\nWrote:")
    for filename in (
        "figure_1a_gnnpe_scans.pdf",
        "figure_1b_main_comparison.pdf",
        "runtime_summary.csv",
    ):
        print(f"  {args.out_dir / filename}")


if __name__ == "__main__":
    main()
