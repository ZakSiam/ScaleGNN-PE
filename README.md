# Improved GNN-PE for QM9 graph bandits

This repository is a cleaned, QM9-focused extension of the public [GNNBO](https://github.com/lasgroup/GNNBO) codebase for the paper *Graph Neural Network Bandits*. The project studies how to make GNN Phased Elimination (GNN-PE) more practical for graph bandit optimization over molecular candidates:

- fine-tuned GNN-PE versus retraining from scratch,
- GNN-UCB scratch and fine-tune baselines,
- graph-space scan acceleration for the two expensive GNN-PE scan steps using farthest-first traversal (FFT) and WL-kernel graph k-means,
- separate scan ablations for Eq. (1) only, Eq. (2) only, and both equations,
- GP-UCB, GP-EI, GP-TS, and Random baselines.

## Repository layout

```text
.
├── algorithms.py                  # GNN-UCB and scan-aware GNN-PE
├── graph_scan_accel.py            # WL kernel, FFT, and graph k-means scan indexers
├── qm9_bandit_env.py              # QM9 finite-action bandit environment
├── run_qm9_phasedgp.py            # GNN-PE scratch/fine-tune runner
├── run_qm9_gnnucb.py               # GNN-UCB scratch/fine-tune runner
├── run_qm9_gp_baselines.py         # GP-UCB / GP-EI / GP-TS / Random runner
├── scripts/
│   ├── generate_report_outputs.py # Figure 1a, Figure 1b, runtime summary
│   └── plot_scan_ablations.py      # 8-way scan-ablation figure + runtime table
└── requirements.txt
```

`data/`, `results/`, and `figures/` are intentionally ignored by Git. A fresh run downloads QM9 into `data/qm9/` automatically through PyTorch Geometric.

## Setup

The code was cleaned around a lean Python environment. A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the platform needs a specific PyTorch / PyTorch Geometric wheel combination, one must install the compatible PyTorch stack first, then install the remaining requirements.

## Reproduce the experiments

The experiment uses QM9 domains with `|G|=1000`, `T=300`, and seeds `0..9`. All runner defaults are aligned with those values; the commands below spell out the result folders so the plotting scripts can consume them directly.

### 1. GNN-PE scratch and fine-tune

```bash
for seed in {0..9}; do
  python run_qm9_phasedgp.py \
    --seed "$seed" \
    --train_modes scratch,finetune \
    --exp_result_folder results/qm9_phasedgp_T300_N1000
done
```

### 2. GNN-UCB scratch and fine-tune

```bash
for seed in {0..9}; do
  python run_qm9_gnnucb.py \
    --seed "$seed" \
    --train_modes scratch,finetune \
    --exp_result_folder results/qm9_gnnucb_T300_N1000
done
```

### 3. GP and Random baselines

```bash
for baseline in gpucb gpei gpts random; do
  for seed in {0..9}; do
    python run_qm9_gp_baselines.py \
      --baseline "$baseline" \
      --seed "$seed" \
      --exp_result_folder results/qm9_gp_baselines_T300_N1000
  done
done
```

### 4. Fine-tuned GNN-PE scan accelerations

The GNN-PE runner exposes:

- `--domain_scan_method full|fft|graph_kmeans`
- `--domain_scan_apply_to both|c1|c2`

where `c1` means Eq. (C.1) only and `c2` means Eq. (C.2) only.

```bash
# Approximate Eq. (1) and Eq. (2)
for seed in {0..9}; do
  python run_qm9_phasedgp.py \
    --seed "$seed" \
    --train_modes finetune \
    --domain_scan_method fft \
    --domain_scan_apply_to both \
    --exp_result_folder results/qm9_phasedgp_T300_N1000_fft

  python run_qm9_phasedgp.py \
    --seed "$seed" \
    --train_modes finetune \
    --domain_scan_method graph_kmeans \
    --domain_scan_apply_to both \
    --exp_result_folder results/qm9_phasedgp_T300_N1000_kmeans
done
```

```bash
# Eq. (1)-only ablations
for seed in {0..9}; do
  python run_qm9_phasedgp.py \
    --seed "$seed" \
    --train_modes finetune \
    --domain_scan_method fft \
    --domain_scan_apply_to c1 \
    --exp_result_folder results/qm9_phasedgp_T300_N1000_fft_C1only

  python run_qm9_phasedgp.py \
    --seed "$seed" \
    --train_modes finetune \
    --domain_scan_method graph_kmeans \
    --domain_scan_apply_to c1 \
    --exp_result_folder results/qm9_phasedgp_T300_N1000_kmeans_C1only
done
```

```bash
# Eq. (2)-only ablations
for seed in {0..9}; do
  python run_qm9_phasedgp.py \
    --seed "$seed" \
    --train_modes finetune \
    --domain_scan_method fft \
    --domain_scan_apply_to c2 \
    --exp_result_folder results/qm9_phasedgp_T300_N1000_fft_C2only

  python run_qm9_phasedgp.py \
    --seed "$seed" \
    --train_modes finetune \
    --domain_scan_method graph_kmeans \
    --domain_scan_apply_to c2 \
    --exp_result_folder results/qm9_phasedgp_T300_N1000_kmeans_C2only
done
```

## Regenerate figures and runtime tables

After the result folders above exist:

```bash
python scripts/generate_report_outputs.py \
  --results_root results \
  --out_dir figures/report
```

This writes:

- `figures/report/figure_1a_gnnpe_scans.pdf`
- `figures/report/figure_1b_main_comparison.pdf`
- `figures/report/runtime_summary.csv`

To regenerate the full 8-way scan-ablation comparison:

```bash
python scripts/plot_scan_ablations.py \
  --results_root results \
  --out_dir figures/scan_ablations
```

This writes:

- `figures/scan_ablations/scan_ablation_8way.pdf`
- `figures/scan_ablations/scan_ablation_runtime_summary.csv`

## Result-file format

All runners emit JSON files with a shared structure:

- `exp_results.regrets`: cumulative regret over rounds,
- `params`: the full run configuration, including seed and train mode,
- `duration_total`: runtime in minutes.

The plotting scripts consume only that shared contract, so additional compatible runs can be added without changing the reporting layer.

## Provenance

This codebase keeps the original project license and builds on the public GNNBO implementation associated with:

> P. Kassraie, A. Krause, and I. Bogunovic. *Graph Neural Network Bandits*. NeurIPS, 2022.

The scan-acceleration additions in this repository are project-specific extensions for the QM9 study described above.
