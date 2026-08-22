import numpy as np
import torch

from torch_geometric.datasets import ZINC
from torch_geometric.utils import to_dense_adj

from graph_env.graph_generator import Graph


# ZINC ships 28 atom types in the full version (21 in the 12k subset). Fixed so
# that feat_dim does not depend on which molecules a seed happens to sample.
NUM_ATOM_TYPES = 28


def _load_splits(root: str, subset: bool, split: str):
    if split == "all":
        return [ZINC(root=root, subset=subset, split=s) for s in ("train", "val", "test")]
    return [ZINC(root=root, subset=subset, split=split)]


def load_zinc_domain(
    root: str = "data/zinc",
    num_actions: int = 500,
    seed: int = 0,
    normalize_rewards: bool = True,
    split: str = "all",
    subset: bool = False,
    num_atom_types: int = NUM_ATOM_TYPES,
):
    """
    Load a bandit domain from the ZINC 250k dataset.

    - Each action = one molecule (Graph object).
    - Reward = penalized logP (ZINC's `y`, i.e. constrained solubility
      logP - SA - #cycles), which we maximize.

    split="all" concatenates train/val/test for the full 249,456-molecule domain.
    Node features are one-hot atom types, so `dim_feats == num_atom_types`.
    """
    rng = np.random.RandomState(seed)

    datasets = _load_splits(root, subset, split)
    sizes = [len(ds) for ds in datasets]
    total = int(sum(sizes))

    if num_actions > total:
        raise ValueError(
            f"Requested num_actions={num_actions} > dataset size={total}"
        )

    indices = rng.choice(total, size=num_actions, replace=False)
    offsets = np.cumsum([0] + sizes)

    graphs = []
    rewards = []

    for idx in indices:
        ds_i = int(np.searchsorted(offsets, idx, side="right") - 1)
        data = datasets[ds_i][int(idx - offsets[ds_i])]
        num_nodes = int(data.num_nodes)

        # Dense adjacency [num_nodes, num_nodes] with self-loops, matching
        # how the QM9 domain and the synthetic graphs are constructed.
        adj = to_dense_adj(data.edge_index, max_num_nodes=num_nodes)[0].numpy()
        np.fill_diagonal(adj, 1.0)

        # ZINC stores atom types as integer indices [num_nodes, 1]; one-hot them.
        atom_types = data.x.view(-1).long()
        if int(atom_types.max()) >= num_atom_types:
            raise ValueError(
                f"Atom type {int(atom_types.max())} exceeds num_atom_types={num_atom_types}"
            )
        x = torch.nn.functional.one_hot(
            atom_types, num_classes=num_atom_types
        ).float().numpy()

        g = Graph(
            dim_feats=num_atom_types,
            num_nodes=num_nodes,
            adj_mat=adj,
            feat_mat=x,
        )
        graphs.append(g)

        rewards.append(float(data.y.view(-1)[0].item()))

    rewards = np.asarray(rewards, dtype=np.float32)

    if normalize_rewards:
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        rewards = (rewards - mean) / std

    return graphs, rewards.tolist(), num_atom_types
