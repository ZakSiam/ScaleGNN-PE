# ScaleGNN-PE

Reference implementation for **"Scaling Graph Neural Bandits to Million-Molecule Virtual
Screening"** (KDD 2027, AI for Science track).

GNN phased elimination (GNN-PE), introduced together with GNN-UCB by Kassraie, Krause, and
Bogunovic [[1](#references)], is theoretically grounded but retrains its network every
episode and rescans the entire candidate pool twice per round — once for uncertainty-based
exploration, once for confidence-based elimination. **ScaleGNN-PE** keeps that
phased structure and removes the redundant computation with two orthogonal compressions:

* **Parameter-space compression** — each episode warm-starts from the previous episode's
  weights (`--train_modes finetune`) and adapts only a subnetwork
  (`--peft {lora,last_layer}`), shrinking both the training cost and the dimension of the
  diagonal-NTK uncertainty model from `p` to `s`.
* **Graph-domain compression** — normalized sparse Weisfeiler–Lehman features `X` define an
  implicit kernel `K = XXᵀ`; **matrix-free pivoted Cholesky** picks `M` representatives
  without ever forming the `N×N` kernel, and every molecule inherits the posterior
  `(μ̂, σ̂)` of its nearest representative (`--domain_scan_method pivchol`).

The two axes compose multiplicatively, which is what makes phased elimination run at
`n = 10⁶` where GNN-PE and GNN-UCB go out of memory.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyarrow tqdm     # needed for the docking (Parquet) pipeline
```

Install a PyTorch / PyTorch Geometric wheel combination matching your CUDA version first if
the platform needs one. All reported experiments ran on NVIDIA A100 80 GB GPUs.

## Data

**QM9 and ZINC** download automatically through PyTorch Geometric into `data/qm9/` and
`data/zinc/` on first run. Each run draws a candidate pool of `|G| = 1000` graphs.

**Docking libraries** come from the DrugImprover ZINC15 subsets (1 M molecules per target).
`data/docking/<TARGET>_processed.parquet` ships with the repo for `3CLPro`, `rtcb`, and
`6T2W`; convert each once to PyG shards:

```bash
python SMILES_to_graph_converter/smiles_parquet_to_pyg.py \
  --input data/docking/3CLPro_processed.parquet --target 3CLPro --output-dir data/graphs/3CLPro
```

Raw docking scores are normalized so that higher reward = more favorable predicted binding
(`--objective dock_norm`).

## Acknowledgements

ScaleGNN-PE builds directly on **GNN-PE** and **GNN-UCB**, the algorithms introduced by
Kassraie, Krause, and Bogunovic [[1](#references)]. Their phased-elimination scheme and its
NTK-based confidence bounds are the starting point for this work — the contributions here
are the parameter-space and graph-domain compressions layered on top — and both algorithms
serve as baselines throughout our experiments. The bandit and graph-environment scaffolding
in `algorithms.py` and `graph_env/` follows their formulation. We thank the authors for
making the method and its analysis available.

## References

[1] P. Kassraie, A. Krause, and I. Bogunovic. **Graph Neural Network Bandits**. *Advances in
Neural Information Processing Systems (NeurIPS)*, 2022.

```bibtex
@inproceedings{kassraie2022graph,
  title     = {Graph Neural Network Bandits},
  author    = {Kassraie, Parnian and Krause, Andreas and Bogunovic, Ilija},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}
```
