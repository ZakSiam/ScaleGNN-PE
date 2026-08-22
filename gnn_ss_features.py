"""Absolute graph-Laplacian positional encodings, as specified in GNN-SS Appendix C.

    "Graph Laplacian positional encodings are eigenvectors of the symmetrically normalized
     graph Laplacian D = I - D^{-1/2} A D^{-1/2} ... From the matrix of eigenvectors U, we take
     k < N eigenvectors, which are concatenated directly to node features as X <- concat(X, U_k)
     ... To address the sign ambiguity of the eigenvectors we enforce that each element is
     non-negative (Dwivedi et al., 2023)."

Feature dim goes from d to d + k, so the net's input_dim must follow.
"""

import hashlib
import os

import numpy as np

from graph_env.graph_generator import Graph


def laplacian_pe(adj_mat: np.ndarray, k: int) -> np.ndarray:
    """First k non-trivial eigenvectors of the symmetric normalized Laplacian, |.| applied.

    adj_mat carries self-loops (the domain builders set the diagonal to 1); they are removed
    here so D and A are the graph's own adjacency, as the definition intends.
    Returns (num_nodes, k); zero-padded when the graph has fewer than k+1 nodes.
    """
    A = np.array(adj_mat, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    if k <= 0:
        return np.zeros((n, 0), dtype=np.float32)

    deg = A.sum(axis=1)
    dinv = np.zeros_like(deg)
    nz = deg > 0
    dinv[nz] = 1.0 / np.sqrt(deg[nz])
    L = np.eye(n) - (dinv[:, None] * A * dinv[None, :])
    L = 0.5 * (L + L.T)  # symmetrise against round-off so eigh is well posed

    try:
        w, U = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.zeros((n, k), dtype=np.float32)

    order = np.argsort(w)
    U = U[:, order][:, 1:k + 1]          # drop the trivial constant eigenvector
    U = np.abs(U)                        # sign-ambiguity convention (Dwivedi et al. 2023)
    if U.shape[1] < k:                   # tiny graphs: pad
        U = np.hstack([U, np.zeros((n, k - U.shape[1]))])
    return U.astype(np.float32)


def _cache_key(graphs, k):
    h = hashlib.sha1()
    h.update(f"lappe_k{k}_n{len(graphs)}".encode())
    for i in np.linspace(0, len(graphs) - 1, min(64, len(graphs))).astype(int):
        g = graphs[i]
        h.update(np.ascontiguousarray(g.adj_mat, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def augment_with_laplacian_pe(graphs, k: int, cache_dir: str = None, verbose: bool = True):
    """Return graphs whose node features are concat(X, |U_k|). k <= 0 returns them unchanged.

    Deterministic in the graphs alone (independent of the bandit seed), so the encodings are
    cached once per domain and shared across seeds -- same contract as the WL scan cache.
    """
    if k <= 0:
        return graphs, int(graphs[0].dim_feats)

    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"lappe_k{k}_{_cache_key(graphs, k)}.npy")
        if os.path.exists(path):
            flat = np.load(path)
            if verbose:
                print(f"[lappe] loaded cache {path}", flush=True)
            out, o = [], 0
            for g in graphs:
                n = int(g.num_nodes)
                X = np.hstack([g.feat_mat, flat[o:o + n]]).astype(np.float32)
                o += n
                out.append(Graph(dim_feats=X.shape[1], num_nodes=n,
                                 adj_mat=g.adj_mat, feat_mat=X))
            return out, int(out[0].dim_feats)

    out, blocks = [], []
    for i, g in enumerate(graphs):
        pe = laplacian_pe(g.adj_mat, k)
        blocks.append(pe)
        X = np.hstack([g.feat_mat, pe]).astype(np.float32)
        out.append(Graph(dim_feats=X.shape[1], num_nodes=int(g.num_nodes),
                         adj_mat=g.adj_mat, feat_mat=X))
        if verbose and (i + 1) % 200000 == 0:
            print(f"[lappe] {i+1}/{len(graphs)}", flush=True)
    if path:
        np.save(path, np.vstack(blocks).astype(np.float32))
        if verbose:
            print(f"[lappe] wrote cache {path}", flush=True)
    return out, int(out[0].dim_feats)
