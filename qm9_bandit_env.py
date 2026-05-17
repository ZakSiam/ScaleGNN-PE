# qm9_bandit_env.py
import os
import numpy as np

from torch_geometric.datasets import QM9
from torch_geometric.utils import to_dense_adj

from graph_env.graph_generator import Graph


def load_qm9_domain(
    root: str = "data/qm9",
    num_actions: int = 500,
    target_idx: int = 4,
    seed: int = 0,
    normalize_rewards: bool = True,
):
    """
    Load a bandit domain from the QM9 dataset.

    - Each action = one molecule (Graph object).
    - Reward = one scalar property y[target_idx], e.g. HOMO–LUMO gap (idx=4).
    """
    rng = np.random.RandomState(seed)

    # PyG will download QM9 on first use into `root`
    dataset = QM9(root=root)

    if num_actions > len(dataset):
        raise ValueError(
            f"Requested num_actions={num_actions} > dataset size={len(dataset)}"
        )

    # Sample a subset of molecules as our finite bandit action set
    indices = rng.choice(len(dataset), size=num_actions, replace=False)

    graphs = []
    rewards = []

    # Node feature dimension is fixed across QM9
    feat_dim = dataset.num_node_features  # typically 11

    for idx in indices:
        data = dataset[idx]
        num_nodes = int(data.num_nodes)

        # Dense adjacency matrix [num_nodes, num_nodes]
        # Add self-loops to match how synthetic graphs are constructed
        adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes)[0].numpy()
        np.fill_diagonal(adj, 1.0)

        # Node features: [num_nodes, feat_dim]
        x = data.x.numpy()

        # Wrap into their Graph class
        g = Graph(
            dim_feats=feat_dim,
            num_nodes=num_nodes,
            adj_mat=adj,
            feat_mat=x,
        )
        graphs.append(g)

        # Reward: single scalar target from QM9 (e.g., HOMO-LUMO gap)
        # y = float(data.y[target_idx].item())
        y_flat = data.y.view(-1)     # now shape [19]
        y = float(y_flat[target_idx].item())
        rewards.append(y)

    rewards = np.asarray(rewards, dtype=np.float32)

    if normalize_rewards:
        # Simple z-score normalization for numerical stability
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        rewards = (rewards - mean) / std

    return graphs, rewards.tolist(), feat_dim
