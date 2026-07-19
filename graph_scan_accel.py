
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
    from scipy.sparse import csr_matrix, diags as _sp_diags  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    csr_matrix = None  # type: ignore
    _sp_diags = None  # type: ignore
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


def wl_subtree_feature_dicts(graphs: Sequence, h: int = 2) -> List[Dict[int, int]]:
    """
    Compute WL subtree features (counts of node labels) for each graph, using integer
    label hashing instead of ever-growing string tokens.

    This is mathematically identical to the classic string-token WL kernel — the kernel
    K = <phi_i, phi_j> is invariant to any *consistent* relabeling of tokens — but it avoids
    the geometric blow-up in token length as h grows: each WL label is compressed to a small
    integer id shared across all graphs, so cost stays linear in the true WL work rather than
    in the (exponentially growing) string length.

    graphs: sequence of Graph objects with attributes:
        - num_nodes
        - adj_mat : (n,n) numpy array with 0/1 entries
        - feat_mat : (n,d) numpy array

    Returns: list of dicts, one per graph, mapping integer WL-token ids to counts.
    """
    n_graphs = len(graphs)

    # --- Precompute neighbor lists and compress base (h=0) labels to integer ids. ---
    base_dict: Dict[str, int] = {}
    graph_neigh: List[List[List[int]]] = []
    cur_labels: List[List[int]] = []
    for g in graphs:
        n = int(g.num_nodes)
        adj = np.asarray(g.adj_mat)
        # neighbors: include self if adj has diagonal 1; if not, include self explicitly
        neigh = [list(np.where(adj[i] > 0)[0]) for i in range(n)]
        if not np.all(np.diag(adj) > 0):
            for i in range(n):
                if i not in neigh[i]:
                    neigh[i].append(i)
        graph_neigh.append(neigh)
        base = _node_base_labels_from_features(np.asarray(g.feat_mat))
        cur_labels.append([base_dict.setdefault(lb, len(base_dict)) for lb in base])

    # Feature-token ids are namespaced by WL iteration via token_dict[(it, label_id)] so that
    # a label reused numerically in different rounds never collides in feature space. This
    # reproduces the original per-round "it:" string prefix exactly.
    feats: List[Dict[int, int]] = [dict() for _ in range(n_graphs)]
    token_dict: Dict[Tuple[int, int], int] = {}

    def _accumulate(it: int) -> None:
        for gi in range(n_graphs):
            fd = feats[gi]
            for lb in cur_labels[gi]:
                tok = token_dict.setdefault((it, lb), len(token_dict))
                fd[tok] = fd.get(tok, 0) + 1

    # h = 0: base labels
    _accumulate(0)

    # WL relabeling iterations. `compress` is fresh per iteration (standard WL): two nodes get
    # the same new id iff they share (center label, sorted multiset of neighbor labels), which
    # by induction holds iff they shared the same grown string label in the original code.
    for it in range(1, h + 1):
        compress: Dict[Tuple[int, Tuple[int, ...]], int] = {}
        new_labels: List[List[int]] = []
        for gi in range(n_graphs):
            cur = cur_labels[gi]
            neigh = graph_neigh[gi]
            nl = [0] * len(cur)
            for i in range(len(cur)):
                key = (cur[i], tuple(sorted(cur[j] for j in neigh[i])))
                nl[i] = compress.setdefault(key, len(compress))
            new_labels.append(nl)
        cur_labels = new_labels
        _accumulate(it)

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


def pivoted_cholesky(K: np.ndarray, m: int, tol: float = 1e-12) -> List[int]:
    """
    Greedy pivoted (partial) Cholesky selection on a PSD kernel K.

    At each step we hold a residual diagonal d, where d[i] is the leftover variance of
    point i NOT yet explained by the already-picked pivots (equivalently, the posterior
    variance of i given the current pivot set). We greedily pick the point with the
    largest residual (the least-explained / most-informative one), append it as a
    representative, then downdate every point's residual by the new Cholesky column.

    This greedily minimizes trace(K - K_hat), i.e. the kernel reconstruction error, in
    O(N*m) time touching only m columns of K (never the full N^2 factorization).

    Returns the list of pivot indices (the representatives), in selection order.
    """
    n = K.shape[0]
    m = int(min(max(m, 1), n))
    d = np.array(np.diag(K), dtype=np.float64, copy=True)  # residual variances
    L = np.zeros((n, m), dtype=np.float64)                 # partial Cholesky columns
    pivots: List[int] = []
    for t in range(m):
        p = int(np.argmax(d))
        if d[p] <= tol:
            break  # numerically rank-deficient: remaining points are already explained
        pivots.append(p)
        col = K[:, p].astype(np.float64)
        if t > 0:
            col = col - L[:, :t] @ L[p, :t]   # subtract off what earlier pivots explain
        ell = col / np.sqrt(d[p])
        L[:, t] = ell
        d = np.clip(d - ell * ell, 0.0, None)  # every point's residual shrinks
        d[p] = 0.0                             # picked point is fully explained
    return pivots


# ---------------------------------------------------------------------------
# Matrix-free path: never materialize the dense N x N kernel.
#
# The WL kernel is K = X X^T for a sparse WL-count feature matrix X (rows = graphs,
# cols = WL tokens). Pivoted Cholesky and nearest-rep assignment only ever touch the
# diagonal of K and a handful of its COLUMNS, and each column is one sparse matvec
# K[:, p] = X @ X[p]^T. So for large N we keep X sparse (~O(nnz), hundreds of MB at 1M)
# and compute the O(m) needed columns on demand, instead of a dense N x N array (4 TB at 1M).
# These routines select the SAME pivots as pivoted_cholesky / assign_to_reps, but not
# bit-for-bit: the dense path stores K in float32 while the matrix-free path works in
# float64, so intermediate values differ in the last few ulps. Selections agree because
# both paths share the unit-diagonal convention below (see `unit_diag`).
#
# That convention matters. wl_kernel_matrix(normalize=True) ends with fill_diagonal(K, 1.0),
# so the dense residual diagonal starts at EXACTLY all-ones. Row-normalizing X instead leaves
# ~1e-16 of rounding noise on diag(K) = ||X_i||^2. Since the first pivot is argmax over that
# diagonal, the noise — not the data — would decide it, and the two paths would pick different
# first pivots and diverge from step 0. Passing unit_diag=True snaps the matrix-free diagonal
# (and the K[p,p] entry of each fetched column) to exactly 1.0, matching the dense path.
# ---------------------------------------------------------------------------


def wl_feature_matrix(graphs: Sequence, h: int = 2, normalize: bool = True, dtype=np.float64):
    """
    Build the sparse WL-count feature matrix X (n x vocab) as a scipy CSR matrix, WITHOUT
    forming K. If normalize=True, rows are scaled to unit L2 norm so that K = X X^T has a
    unit diagonal — matching wl_kernel_matrix(normalize=True).
    """
    if not _HAS_SCIPY:
        raise RuntimeError("wl_feature_matrix requires scipy (matrix-free scan path).")
    feat_dicts = wl_subtree_feature_dicts(graphs, h=int(h))
    n = len(feat_dicts)

    vocab: Dict[int, int] = {}
    for d in feat_dicts:
        for k in d.keys():
            if k not in vocab:
                vocab[k] = len(vocab)
    p = len(vocab)

    nnz = sum(len(d) for d in feat_dicts)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)
    pos = 0
    for i, d in enumerate(feat_dicts):
        for k, v in d.items():
            indices[pos] = vocab[k]
            data[pos] = float(v)
            pos += 1
        indptr[i + 1] = pos

    X = csr_matrix((data, indices, indptr), shape=(n, p), dtype=np.float64)
    if normalize:
        sq = np.asarray(X.multiply(X).sum(axis=1)).ravel()
        inv = 1.0 / np.sqrt(np.maximum(sq, 1e-12))
        X = csr_matrix(_sp_diags(inv) @ X)
    return X.astype(dtype)


def _wl_kernel_column(X, p: int, unit_diag: bool = False) -> np.ndarray:
    """Dense column p of K = X X^T, i.e. K[:, p] = X @ X[p]^T, via one sparse matvec."""
    col = X @ X.getrow(int(p)).transpose()      # (n, 1) sparse
    col = np.asarray(col.todense(), dtype=np.float64).ravel()
    if unit_diag:
        col[int(p)] = 1.0   # dense path does fill_diagonal(K, 1.0); match it exactly
    return col


def _matfree_diag(X, unit_diag: bool = False) -> np.ndarray:
    """diag(K) = ||X_i||^2, snapped to exactly 1.0 under the normalized convention."""
    n = X.shape[0]
    if unit_diag:
        return np.ones(n, dtype=np.float64)
    return np.asarray(X.multiply(X).sum(axis=1)).ravel().astype(np.float64)


def pivoted_cholesky_matfree(X, m: int, tol: float = 1e-12, unit_diag: bool = False) -> List[int]:
    """
    Matrix-free greedy pivoted Cholesky on K = X X^T. Selects the same pivots as
    pivoted_cholesky(K, m) but computes each of the m needed kernel columns on demand from
    the sparse X, so it never allocates the N x N kernel. Memory: sparse X + one (n x m) L.

    Set unit_diag=True when X was built with normalize=True, so diag(K) is treated as
    exactly 1.0 — matching wl_kernel_matrix's fill_diagonal and keeping the first pivot
    deterministic instead of decided by float noise.
    """
    n = X.shape[0]
    m = int(min(max(m, 1), n))
    d = _matfree_diag(X, unit_diag=unit_diag)   # residual variances = diag(K)
    L = np.zeros((n, m), dtype=np.float64)
    pivots: List[int] = []
    for t in range(m):
        p = int(np.argmax(d))
        if d[p] <= tol:
            break
        pivots.append(p)
        col = _wl_kernel_column(X, p, unit_diag=unit_diag)
        if t > 0:
            col = col - L[:, :t] @ L[p, :t]
        ell = col / np.sqrt(d[p])
        L[:, t] = ell
        d = np.clip(d - ell * ell, 0.0, None)
        d[p] = 0.0
    return pivots


def assign_to_reps_matfree(X, reps: Sequence[int], unit_diag: bool = False) -> np.ndarray:
    """Matrix-free version of assign_to_reps: nearest rep by kernel distance, columns on demand."""
    reps = list(map(int, reps))
    n = X.shape[0]
    diag = _matfree_diag(X, unit_diag=unit_diag)
    rep_of = np.empty(n, dtype=np.int32)
    best_d2 = np.full(n, np.inf, dtype=np.float64)
    for r in reps:
        kr = _wl_kernel_column(X, r, unit_diag=unit_diag)
        d2 = diag + diag[r] - 2.0 * kr
        mask = d2 < best_d2
        best_d2[mask] = d2[mask]
        rep_of[mask] = r
    for r in reps:
        rep_of[r] = r
    return rep_of


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
    pivchol_m: int = 300,
    matfree: Optional[bool] = None,
    matfree_threshold: int = 20000,
    random_state: Optional[np.random.RandomState] = None,
) -> ScanIndexer:
    """
    Build ScanIndexer for a given graph domain.

    For method="pivchol" on large domains we use the matrix-free path (no dense N x N kernel).
    `matfree=None` auto-enables it when len(graphs) > matfree_threshold; pass True/False to force.
    Only pivchol supports matrix-free — fft and graph_kmeans need the full kernel.
    """
    method = str(method).lower()

    # ---- pivoted Cholesky: matrix-free for large N (never build dense K) ----
    if method == "pivchol":
        n = len(graphs)
        use_matfree = (n > int(matfree_threshold)) if matfree is None else bool(matfree)
        if use_matfree:
            if not _HAS_SCIPY:
                raise RuntimeError("Matrix-free pivchol requires scipy; install scipy or pass matfree=False.")
            X = wl_feature_matrix(graphs, h=int(wl_h), normalize=True)
            # normalize=True -> unit diagonal convention, same as the dense path.
            reps = pivoted_cholesky_matfree(X, m=int(pivchol_m), unit_diag=True)
            rep_of = assign_to_reps_matfree(X, reps, unit_diag=True)
            meta = {"wl_h": int(wl_h), "pivchol_m": len(reps), "matfree": True}
        else:
            # Dense path (small N).
            K = wl_kernel_matrix(graphs, h=int(wl_h), normalize=True, dtype=np.float32)
            reps = pivoted_cholesky(K, m=int(pivchol_m))
            rep_of = assign_to_reps(K, reps)
            meta = {"wl_h": int(wl_h), "pivchol_m": len(reps), "matfree": False}
        return ScanIndexer(method="pivchol", reps=reps, rep_of=rep_of, meta=meta)

    # ---- fft / graph_kmeans: require the dense kernel ----
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

    raise ValueError(f"Unknown scan method: {method}. Use 'fft', 'graph_kmeans', or 'pivchol'.")
