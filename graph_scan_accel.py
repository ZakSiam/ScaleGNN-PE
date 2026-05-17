
"""
Graph-space scanning accelerators for GNN-PE (PhasedGNN-UCB) on finite graph domains.

We provide two lightweight approximations intended for *efficiency ablations*:
  1) Farthest-First Traversal (FFT) to select a representative subset S of graphs.
  2) Graph "k-means" via kernel k-means over a Weisfeiler–Lehman (WL) subtree graph kernel,
     with cluster medoids used as representatives.

Both methods:
  - operate on the graph *objects* (adjacency + node features) and a graph kernel computed from them,
  - avoid evaluating the GNN (mean/variance) on every graph each round,
  - return a mapping rep_of[i] that assigns every graph i to one representative r in S,
    so we can approximate UCB/LCB/variance for all i using only the GNN evaluated on r.

Dependencies: numpy only. If scipy is available, we use scipy.sparse CSR for speed; otherwise we fall back to an exact (but slower) pure-Python kernel computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Optional: SciPy speeds up WL-kernel computation via sparse CSR matrices.
# We do NOT require SciPy; if it is missing we compute the kernel exactly via dict dot-products.
try:
    from scipy.sparse import csr_matrix  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    csr_matrix = None  # type: ignore
    _HAS_SCIPY = False



def _node_base_labels_from_features(feat_mat: np.ndarray) -> List[str]:
    """
    Robust labeling for QM9 features without assuming a specific encoding.
    - If features look like (almost) one-hot, use argmax index.
    - Otherwise, quantize to a short tuple string.
    """
    x = np.asarray(feat_mat)
    # Heuristic: one-hot-ish if entries are near {0,1} and row-sum near 1
    near01 = np.mean((np.abs(x - np.round(x)) < 1e-6))
    row_sum = np.mean(np.abs(x.sum(axis=1) - 1.0) < 1e-6) if x.ndim == 2 else 0.0
    if near01 > 0.95 and row_sum > 0.5:
        return [f"onehot:{int(np.argmax(row))}" for row in x]
    # Otherwise quantize
    q = np.round(x, 2)
    return ["q:" + ",".join(map(str, row.tolist())) for row in q]


def wl_subtree_feature_dicts(graphs: Sequence, h: int = 2) -> List[Dict[str, int]]:
    """
    Compute WL subtree features (counts of node labels) for each graph.

    graphs: sequence of Graph objects with attributes:
        - num_nodes
        - adj_mat : (n,n) numpy array with 0/1 entries
        - feat_mat : (n,d) numpy array

    Returns: list of dicts, one per graph, mapping WL-label tokens to counts.
    """
    feats: List[Dict[str, int]] = []
    for g in graphs:
        n = int(g.num_nodes)
        adj = np.asarray(g.adj_mat)
        # neighbors: include self if adj has diagonal 1; if not, include self explicitly
        neigh = [list(np.where(adj[i] > 0)[0]) for i in range(n)]
        if not np.all(np.diag(adj) > 0):
            for i in range(n):
                if i not in neigh[i]:
                    neigh[i].append(i)

        labels = _node_base_labels_from_features(np.asarray(g.feat_mat))
        # Count base labels
        feat_dict: Dict[str, int] = {}
        for lb in labels:
            feat_dict[f"0:{lb}"] = feat_dict.get(f"0:{lb}", 0) + 1

        # WL iterations
        cur = labels
        for it in range(1, h + 1):
            new = []
            for i in range(n):
                multiset = sorted(cur[j] for j in neigh[i])
                token = f"{cur[i]}|" + "|".join(multiset)
                # Use a stable hash-like token (string) – DictVectorizer will map to columns
                new.append(token)
            cur = new
            for lb in cur:
                key = f"{it}:{lb}"
                feat_dict[key] = feat_dict.get(key, 0) + 1

        feats.append(feat_dict)
    return feats


def wl_kernel_matrix(graphs: Sequence, h: int = 2, normalize: bool = True, dtype=np.float32) -> np.ndarray:
    """
    WL subtree kernel K where K[i,j] = <phi(G_i), phi(G_j)> in WL-count feature space.

    Implementation detail:
      * If SciPy is available, we build a CSR sparse matrix X (rows = graphs, cols = WL tokens),
        then compute K = X X^T efficiently.
      * If SciPy is not available, we compute K exactly via sparse dict dot-products.

    This removes any scikit-learn dependency while producing the same kernel values that a
    DictVectorizer-based pipeline would yield.
    """
    feat_dicts = wl_subtree_feature_dicts(graphs, h=h)
    n = len(feat_dicts)

    if _HAS_SCIPY:
        # Build a deterministic vocabulary (only used to construct CSR).
        # Sorting matches DictVectorizer(sort=True) default behavior, but ordering does not
        # affect K = X X^T.
        vocab = {}
        for d in feat_dicts:
            for k in d.keys():
                if k not in vocab:
                    vocab[k] = None
        feature_names = sorted(vocab.keys())
        vocab = {k: i for i, k in enumerate(feature_names)}
        p = len(feature_names)

        # Build CSR arrays: indices/data per row + indptr offsets.
        indptr = np.zeros(n + 1, dtype=np.int64)
        nnz = 0
        row_items: List[List[Tuple[int, float]]] = []
        for i, d in enumerate(feat_dicts):
            items = [(vocab[k], float(v)) for k, v in d.items() if k in vocab]
            items.sort(key=lambda t: t[0])
            row_items.append(items)
            nnz += len(items)
            indptr[i + 1] = nnz

        indices = np.empty(nnz, dtype=np.int64)
        data = np.empty(nnz, dtype=np.float64)
        pos = 0
        for items in row_items:
            for j, v in items:
                indices[pos] = j
                data[pos] = v
                pos += 1

        X = csr_matrix((data, indices, indptr), shape=(n, p), dtype=np.float64)
        K = (X @ X.T).astype(dtype)
        K = K.toarray() if hasattr(K, "toarray") else np.asarray(K, dtype=dtype)
    else:
        # Exact fallback: K[i,j] = sum_k feat_i[k] * feat_j[k]
        K = np.zeros((n, n), dtype=np.float64)
        diag = np.zeros(n, dtype=np.float64)
        for i, d in enumerate(feat_dicts):
            diag[i] = sum(float(v) * float(v) for v in d.values())
            K[i, i] = diag[i]
        for i in range(n):
            di = feat_dicts[i]
            for j in range(i + 1, n):
                dj = feat_dicts[j]
                # iterate over the smaller dict for speed
                if len(di) <= len(dj):
                    dot = sum(float(v) * float(dj.get(k, 0.0)) for k, v in di.items())
                else:
                    dot = sum(float(v) * float(di.get(k, 0.0)) for k, v in dj.items())
                K[i, j] = dot
                K[j, i] = dot
        K = K.astype(dtype, copy=False)

    if normalize:
        diag = np.sqrt(np.maximum(np.diag(K), 1e-12))
        K = K / (diag[:, None] * diag[None, :])
        np.fill_diagonal(K, 1.0)
    return K


def _dist2_from_kernel(K: np.ndarray, i: int, j: int) -> float:
    # For normalized kernels, this is in [0,2]
    return float(K[i, i] + K[j, j] - 2.0 * K[i, j])


def farthest_first_traversal(K: np.ndarray, m: int, init: int = 0) -> List[int]:
    """
    Gonzalez-style farthest-first traversal for k-center approximation, using kernel-induced distance.
    Complexity: O(m * n).
    """
    n = K.shape[0]
    m = int(min(max(m, 1), n))
    diag = np.diag(K)
    S = [int(init % n)]
    min_d2 = diag + diag[S[0]] - 2.0 * K[:, S[0]]
    min_d2[S[0]] = 0.0
    for _ in range(1, m):
        nxt = int(np.argmax(min_d2))
        S.append(nxt)
        d2_new = diag + diag[nxt] - 2.0 * K[:, nxt]
        min_d2 = np.minimum(min_d2, d2_new)
        min_d2[nxt] = 0.0
    return S


def assign_to_reps(K: np.ndarray, reps: Sequence[int]) -> np.ndarray:
    """
    Map each item i to the nearest representative in reps, by kernel-distance.
    Returns rep_of[i] = rep_index (actual domain index).
    """
    reps = list(map(int, reps))
    n = K.shape[0]
    diag = np.diag(K)
    rep_of = np.empty(n, dtype=np.int32)
    best_d2 = np.full(n, np.inf, dtype=np.float64)
    for r in reps:
        d2 = diag + diag[r] - 2.0 * K[:, r]
        mask = d2 < best_d2
        best_d2[mask] = d2[mask]
        rep_of[mask] = r
    # ensure reps map to themselves
    for r in reps:
        rep_of[r] = r
    return rep_of


def kernel_kmeans(K: np.ndarray, k: int, n_iter: int = 10, random_state: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Kernel k-means clustering using precomputed kernel matrix K (assumed symmetric, PSD-ish).
    Returns cluster assignments (n,).
    """
    n = K.shape[0]
    k = int(min(max(k, 1), n))
    rng = np.random.RandomState(0) if random_state is None else random_state
    # init: random labels
    labels = rng.randint(0, k, size=n).astype(np.int32)

    diagK = np.diag(K).astype(np.float64)
    for _ in range(int(n_iter)):
        # Build indicator matrix H (n x k)
        H = np.zeros((n, k), dtype=np.float64)
        H[np.arange(n), labels] = 1.0
        sizes = H.sum(axis=0)  # (k,)
        # avoid empty clusters by re-seeding (guarantee sizes>0)
        empty = np.where(sizes < 1)[0]
        if len(empty) > 0:
            # Fill empty clusters by moving points from the currently largest non-singleton clusters.
            # This avoids the (rare but possible) situation where re-seeding picks the same index
            # multiple times and leaves some clusters empty, which would cause divide-by-zero below.
            for c in empty:
                donors = np.where(sizes > 1)[0]
                if donors.size == 0:
                    # Fallback: force k distinct labels on k distinct points.
                    perm = rng.permutation(n)
                    labels[perm[:k]] = np.arange(k, dtype=np.int32)
                    break
                d = donors[int(np.argmax(sizes[donors]))]
                donor_members = np.where(labels == d)[0]
                idx = int(donor_members[rng.randint(0, len(donor_members))])
                labels[idx] = int(c)
                sizes[d] -= 1.0
                sizes[c] += 1.0

            H[:] = 0.0
            H[np.arange(n), labels] = 1.0
            sizes = H.sum(axis=0)
            if np.any(sizes < 1):
                # Final guard (shouldn't trigger): deterministic repair.
                perm = rng.permutation(n)
                labels[perm[:k]] = np.arange(k, dtype=np.int32)
                H[:] = 0.0
                H[np.arange(n), labels] = 1.0
                sizes = H.sum(axis=0)

        KH = K @ H  # (n,k) where KH[x,c] = sum_{y in C_c} K[x,y]
        # cluster_self[c] = sum_{y,z in C_c} K[y,z] = diag(H^T K H)
        cluster_self = np.sum(H * KH, axis=0)  # (k,)

        # dist2[x,c] = Kxx - 2/|C| sum_{y in C} Kxy + 1/|C|^2 sum_{y,z in C} Kyz
        inv_sizes = 1.0 / sizes
        term2 = 2.0 * KH * inv_sizes  # broadcasting (n,k)
        term3 = cluster_self * (inv_sizes ** 2)  # (k,)
        dist2 = diagK[:, None] - term2 + term3[None, :]
        dist2 = np.nan_to_num(dist2, nan=np.inf, posinf=np.inf, neginf=np.inf)
        new_labels = np.argmin(dist2, axis=1).astype(np.int32)
        if np.all(new_labels == labels):
            break
        labels = new_labels
    return labels



def cluster_medoids_from_kernel(K: np.ndarray, labels: np.ndarray, k: int) -> List[int]:
    """
    Pick one medoid per cluster: argmax_i sum_{j in cluster} K[i,j].

    Returns a list of length k (one medoid per cluster id 0..k-1). If a cluster is empty,
    we fall back to a deterministic choice.
    """
    n = K.shape[0]
    medoids: List[int] = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            medoids.append(int(c % n))
            continue
        sims = K[np.ix_(members, members)].sum(axis=1)
        medoids.append(int(members[int(np.argmax(sims))]))
    return medoids


@dataclass
class ScanIndexer:
    """
    Represents a scan-approximation mapping for a finite domain.
      reps: list of representative indices
      rep_of: array (n,) mapping each i to a representative index
    """
    method: str
    reps: List[int]
    rep_of: np.ndarray
    meta: Dict


def build_scan_indexer(
    graphs: Sequence,
    method: str,
    wl_h: int,
    fft_m: int,
    kmeans_k: int,
    kmeans_iter: int,
    random_state: Optional[np.random.RandomState] = None,
) -> ScanIndexer:
    """
    Build ScanIndexer for a given graph domain.
    """
    method = str(method).lower()
    K = wl_kernel_matrix(graphs, h=int(wl_h), normalize=True, dtype=np.float32)

    if method == "fft":
        reps = farthest_first_traversal(K, m=int(fft_m), init=0)
        rep_of = assign_to_reps(K, reps)
        meta = {"wl_h": int(wl_h), "fft_m": int(fft_m)}
        return ScanIndexer(method="fft", reps=reps, rep_of=rep_of, meta=meta)

    if method in {"graph_kmeans", "kmeans"}:
        rng = np.random.RandomState(0) if random_state is None else random_state
        labels = kernel_kmeans(K, k=int(kmeans_k), n_iter=int(kmeans_iter), random_state=rng)
        k_eff = int(np.max(labels)) + 1
        medoids = cluster_medoids_from_kernel(K, labels, k=k_eff)
        rep_of = np.array([medoids[int(labels[i])] for i in range(K.shape[0])], dtype=np.int32)
        # ensure medoids map to themselves
        for m in medoids:
            rep_of[m] = m
        meta = {"wl_h": int(wl_h), "kmeans_k": int(kmeans_k), "kmeans_iter": int(kmeans_iter)}
        return ScanIndexer(method="graph_kmeans", reps=medoids, rep_of=rep_of, meta=meta)

    raise ValueError(f"Unknown scan method: {method}. Use 'fft' or 'graph_kmeans'.")
