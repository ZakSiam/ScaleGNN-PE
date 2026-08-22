# docking_bandit_env.py
"""
Bandit domain built from the docking Parquet datasets.

The graphs are produced upstream by SMILES_to_graph_converter/smiles_parquet_to_pyg.py,
which writes `graphs_*.pt` shards of PyG `Data` objects (one directory per target).
This module only reads those shards, picks a reward, and adapts each `Data` into the
`Graph` container the nets/algorithms expect.

Return contract matches load_qm9_domain / load_zinc_domain: (graphs, rewards, feat_dim).

Rewards (all higher = better):
  dock_norm | solubility | sa | qed | similarity_to_topK
      -> taken straight from `y`, already normalized to [0, 1] by the converter.
         NOTE: dock_norm saturates at the 1st/99th percentiles of dock_raw, so the
         best ~1% of molecules are tied at exactly 1.0.
  dock
      -> -dock_raw (raw docking energy; lower energy = better binding).
         Only this objective carries docking-failure sentinels, which are dropped:
         dock_raw > 1e30 (float32-max) on any target, and dock_raw == 0.0 except on
         3CLPro, where 0.0 is a genuine best score.
"""

import glob
import os
from multiprocessing import Pool

import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_dense_adj

from graph_env.graph_generator import Graph

TARGETS = ("3CLPro", "6T2W", "WRN", "rtcb")
SENTINEL = 1e30

# Converter label order: y = [dock, solubility, sa, qed, similarity_to_topK]
Y_INDEX = {
    "dock_norm": 0,
    "solubility": 1,
    "sa": 2,
    "qed": 3,
    "similarity_to_topK": 4,
}
OBJECTIVES = tuple(Y_INDEX) + ("dock",)


class _ShardDS(InMemoryDataset):
    """Load one `graphs_*.pt` shard written by InMemoryDataset.save()."""

    def __init__(self, path):
        super().__init__(None)
        self.load(path)


def _keep(target, dock_raw, objective):
    """Docking-failure cleaning. Only the raw-energy objective needs it."""
    if objective != "dock":
        return True
    if dock_raw > SENTINEL:
        return False
    if target != "3CLPro" and dock_raw == 0.0:
        return False
    return True


def _reward(data, objective):
    if objective == "dock":
        return -float(data.dock_raw)
    return float(data.y.view(-1)[Y_INDEX[objective]])


def _to_graph(data):
    """PyG Data -> Graph (dense adjacency with self-loops), as in the QM9/ZINC domains."""
    num_nodes = int(data.num_nodes)
    adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes)[0].numpy()
    np.fill_diagonal(adj, 1.0)
    x = data.x.float().numpy()
    return Graph(dim_feats=x.shape[1], num_nodes=num_nodes, adj_mat=adj, feat_mat=x)


def _process_shard(job):
    """Load one shard and return (mats, rewards, row_ids, n_dropped) for its valid molecules
    in on-disk order. `mats` are (num_nodes, adj, feat) numpy tuples, NOT Graph objects:
    Graph instances don't pickle back cleanly across processes, so we ship plain arrays and
    wrap them into Graph in the parent (a trivial, cheap step). Module-level => picklable.
    Pool.imap preserves input order, so the assembled domain is identical to the serial build."""
    shard, target, objective = job
    ds = _ShardDS(shard)
    mats, rs, ids = [], [], []
    n_drop = 0
    for i in range(len(ds)):
        data = ds[i]
        if not _keep(target, float(data.dock_raw), objective):
            n_drop += 1
            continue
        num_nodes = int(data.num_nodes)
        adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes)[0].numpy()
        np.fill_diagonal(adj, 1.0)
        x = data.x.float().numpy()
        mats.append((num_nodes, adj, x))
        rs.append(_reward(data, objective))
        ids.append(int(data.row_id))
    return mats, rs, ids, n_drop


def _mats_to_graphs(mats):
    """Wrap (num_nodes, adj, feat) tuples into Graph objects (parent-side, cheap)."""
    return [Graph(dim_feats=x.shape[1], num_nodes=nn, adj_mat=adj, feat_mat=x)
            for (nn, adj, x) in mats]


def load_docking_domain(
    graph_dir: str,
    target: str,
    objective: str = "dock_norm",
    num_actions: int = None,
    seed: int = 0,
    normalize_rewards: bool = True,
    max_shards: int = None,
    n_workers: int = None,
    verbose: bool = True,
):
    """
    Build a finite-action bandit domain from the converter shards under `graph_dir`.

    graph_dir   : directory of graphs_*.pt shards for ONE target (e.g. data/graphs/3CLPro)
    target      : one of TARGETS (only affects the dock_raw == 0 cleaning rule)
    objective   : see module docstring
    num_actions : deterministic subsample size; None => the full domain
    max_shards  : read only the first N shards (smoke tests)
    n_workers   : parallel shard-decoding processes; None => auto (min(#shards, 16)).
                  Pool.imap preserves order, so the domain is identical to n_workers=1.
                  Runs before any CUDA use, so there is no fork-after-CUDA hazard.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")

    shards = sorted(glob.glob(os.path.join(graph_dir, "graphs_*.pt")))
    if not shards:
        raise FileNotFoundError(
            f"No graph shards in {graph_dir}. Run "
            f"SMILES_to_graph_converter/smiles_parquet_to_pyg.py for this target first."
        )
    if max_shards is not None:
        shards = shards[:max_shards]

    if n_workers is None:
        n_workers = min(len(shards), max(1, (os.cpu_count() or 8) - 2), 16)

    jobs = [(shard, target, objective) for shard in shards]
    graphs = []
    rewards = []
    row_ids = []
    n_dropped = 0

    if n_workers > 1:
        # imap keeps results in input (shard) order -> byte-identical to the serial path.
        with Pool(processes=int(n_workers)) as pool:
            for mats, rs, ids, nd in pool.imap(_process_shard, jobs):
                graphs.extend(_mats_to_graphs(mats))
                rewards.extend(rs)
                row_ids.extend(ids)
                n_dropped += nd
    else:
        for job in jobs:
            mats, rs, ids, nd = _process_shard(job)
            graphs.extend(_mats_to_graphs(mats))
            rewards.extend(rs)
            row_ids.extend(ids)
            n_dropped += nd

    n_valid = len(graphs)
    if n_valid == 0:
        raise RuntimeError(f"No valid molecules in {graph_dir} for objective={objective}")

    rewards = np.asarray(rewards, dtype=np.float64)
    row_ids = np.asarray(row_ids, dtype=np.int64)

    # Deterministic subsample (None => full domain)
    if num_actions is not None and num_actions < n_valid:
        rng = np.random.RandomState(seed)
        sel = np.sort(rng.choice(n_valid, size=int(num_actions), replace=False))
        graphs = [graphs[i] for i in sel]
        rewards = rewards[sel]
        row_ids = row_ids[sel]

    feat_dim = int(graphs[0].dim_feats)
    rewards_raw = rewards.copy()

    rewards = rewards.astype(np.float32)
    if normalize_rewards:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    if verbose:
        print(f"[docking:{target}] |G|={len(graphs)} of {n_valid} valid "
              f"({n_dropped} dropped) feat_dim={feat_dim} objective={objective} "
              f"raw_reward=[{rewards_raw.min():.3f},{rewards_raw.max():.3f}] "
              f"normalized={normalize_rewards}")

    # Provenance: map action indices back to parquet rows for the metrics layer.
    load_docking_domain.last_row_ids = row_ids
    load_docking_domain.last_rewards_raw = rewards_raw

    return graphs, rewards.tolist(), feat_dim
