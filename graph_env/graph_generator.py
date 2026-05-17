"""Minimal graph container used by the QM9 bandit environment."""

from __future__ import annotations

from typing import Optional

import numpy as np

class Graph:
    def __init__(self, num_nodes: int, dim_feats: int, sampling_mode: str = 'binomial',
                 adj_mat: Optional[np.ndarray] = None, edge_prob: Optional[float] = None, num_edges: Optional[int] = None,
                 feat_mat: Optional[np.ndarray] = None,
                 random_state=None, **kwargs):
        self.num_nodes = num_nodes
        self.dim_feats = dim_feats
        self._rds = np.random if random_state is None else random_state

        if adj_mat is None:
            if sampling_mode == 'binomial':
                adj_mat = self.sample_binomial_graph(edge_prob)
            elif sampling_mode == 'uniform':
                adj_mat = self.sample_uniform_graph(num_edges)
        self.adj_mat = adj_mat

        if feat_mat is None:
            feat_mat = self.sample_features()
        self.feat_mat = feat_mat


    #@cached_property
    def neighbors(self):
        neighbors = [[] for i in range(self.num_nodes)]
        for node in range(self.num_nodes):
            neighbors[node] = list(np.where(self.adj_mat[node, :] > 0)[0])
        return neighbors

    #@cached_property
    def feat_mat_aggr_normed(self): #matrix of \bar h_u
        normed_feats = np.zeros((self.num_nodes, self.dim_feats))
        neighbors = self.neighbors()
        for node in range(self.num_nodes):
            sum_feats = np.sum(self.feat_mat[neighbors[node]], axis=0)
            normed_feats[node] = sum_feats / np.linalg.norm(sum_feats)
        return normed_feats

    #@cached_property
    def feat_mat_normed(self): # matrix of \bar h_u assuming there are no edges
        norms = np.expand_dims(np.linalg.norm(self.feat_mat, axis=1), axis=1)
        return self.feat_mat / norms

    #@cached_property
    def node_scale_coefs(self):
        # this gives a vector containing the c_u coefs from the GNTK paper. Not used in our implementation
        node_scale = np.zeros(self.num_nodes)
        neighbors = self.neighbors()
        for node in range(self.num_nodes):
            sum_feats = np.sum(self.feat_mat[neighbors[node]], axis=0)
            node_scale[node] = 1/np.linalg.norm(sum_feats)
        return node_scale

    def sample_binomial_graph(self, prob: float):
        if prob is None:
            raise ValueError("edge_prob is required for binomial graph sampling.")
        adj_mat = self._rds.uniform(0, 1, (self.num_nodes, self.num_nodes))
        adj_mat = (adj_mat < prob).astype(int)
        adj_mat = np.tril(adj_mat) + np.triu(adj_mat.T, 1)
        np.fill_diagonal(adj_mat, 1)
        return adj_mat

    def sample_uniform_graph(self, num_edges: int):
        if num_edges is None:
            raise ValueError("num_edges is required for uniform graph sampling.")
        adj_mat = np.zeros((self.num_nodes, self.num_nodes))
        lower_i, lower_j = np.tril_indices(self.num_nodes, k=-1)
        choices = self._rds.choice(len(lower_i), size=num_edges, replace=False)
        adj_mat[lower_i[choices], lower_j[choices]] = 1
        adj_mat[lower_j[choices], lower_i[choices]] = 1
        np.fill_diagonal(adj_mat, 1)
        return adj_mat

    def sample_features(self):
        return self._rds.normal(size=(self.num_nodes, self.dim_feats))
