"""
Fixed (frozen) GNN embedding + GP baselines on ZINC 250k (penalized logP).

Baselines:
  - gpucb : GP-UCB (discrete candidates)
  - gpei  : GP-EI (Expected Improvement; maximization)
  - gpts  : GP Thompson Sampling (diagonal posterior approximation)
  - random: random action

Key properties:
  - The GNN embedder is NEVER trained (frozen). It only maps each graph -> vector.
  - GP uses an RBF (squared-exponential) kernel on standardized embeddings.
  - Rewards are standardized internally for numerical stability WITHOUT re-fitting:
        posterior mean is computed using the standardized y,
        but the final reported mean is unstandardized back to the raw scale.
"""

import os
import json
import time
import math
import argparse
import sys
import resource

import numpy as np
import torch
# Runners live in experiments/; put the repo root on sys.path so the shared
# modules (algorithms, *_bandit_env, nets, ...) import when this file is run
# directly: python experiments/run_<dataset>_<algo>.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import device

from zinc_bandit_env import load_zinc_domain
from nets import GNNEmbedder, normalize_init
from utils_exp import NumpyArrayEncoder, stable_hash_dict


def str2bool(v):
    """Robust bool parser for argparse (avoids the type=bool trap)."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t"}:
        return True
    if s in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _normal_pdf(z):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z * z)


def _normal_cdf(z):
    # Vectorized CDF via math.erf (no SciPy dependency).
    # Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))


def expected_improvement(mu, sigma, best, xi=0.0):
    """EI for maximization."""
    sigma = np.maximum(sigma, 1e-12)
    imp = mu - best - xi
    z = imp / sigma
    ei = imp * _normal_cdf(z) + sigma * _normal_pdf(z)
    ei[sigma < 1e-10] = 0.0
    return ei


def evaluate(idx, reward_list, noisy, _rds, noise_var):
    y = float(reward_list[idx])
    if noisy:
        y = y + float(_rds.normal(0.0, noise_var))
    return y


def build_fixed_embeddings(graphs, feat_dim, args):
    """Compute fixed (frozen) GNN embeddings for the whole finite action set."""
    # Seed embedder init independently of bandit seed for consistency.
    torch.manual_seed(args.embedder_seed)
    np.random.seed(args.embedder_seed)

    embedder = GNNEmbedder(
        input_dim=feat_dim,
        depth=args.embed_depth,
        width=args.embed_dim,
        aggr_feats=args.nn_aggr_feat,
    )
    embedder = normalize_init(embedder)
    # Ensure weights live on the same device as the tensors created in forward()
    embedder = embedder.to(device)
    embedder.eval()

    # Freeze: no training for baselines
    for p in embedder.parameters():
        p.requires_grad = False

    embs = []
    with torch.no_grad():
        for g in graphs:
            z = embedder(g)  # [embed_dim]
            embs.append(z.detach().cpu().numpy())

    Z = np.stack(embs, axis=0).astype(np.float64)  # (N, D)

    # Standardize embedding dimensions for RBF kernel stability
    Z_mean = Z.mean(axis=0, keepdims=True)
    Z_std = Z.std(axis=0, keepdims=True) + 1e-8
    Z = (Z - Z_mean) / Z_std
    return Z


class IncrementalRBF_GP:
    """
    A small GP regressor for RBF kernel with incremental Cholesky updates.
    No external dependencies (no sklearn/scipy).

    Kernel:
        k(x,x') = sigma_f^2 * exp(-||x-x'||^2 / (2 * ell^2))

    We keep Cholesky factor L of (K + sigma_n^2 I).

    Reward standardization:
        We do not re-fit to standardized y each time.
        Instead, we solve for:
            alpha_y = (K+σ_n^2 I)^(-1) y_raw
            alpha_1 = (K+σ_n^2 I)^(-1) 1
        Then for current y_mean, y_std:
            alpha_std = (alpha_y - y_mean * alpha_1) / y_std
        Predict standardized mean:
            m_std(x*) = k_*^T alpha_std
        Convert back:
            m_raw = m_std * y_std + y_mean
    Variance does not depend on y-scaling.
    """

    def __init__(self, length_scale: float, noise_var: float, output_scale: float = 1.0, jitter: float = 1e-8):
        self.ell = float(length_scale)
        self.noise_var = float(noise_var)
        self.sigma_f2 = float(output_scale) ** 2
        self.jitter = float(jitter)

        self.X = None  # (n, d)
        self.y = None  # (n,)
        self.L = None  # (n, n) lower-triangular Cholesky of K + noise_var I

    def _rbf(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # Efficient squared distance: ||a-b||^2 = a^2 + b^2 - 2ab
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        A2 = np.sum(A * A, axis=1, keepdims=True)  # (n,1)
        B2 = np.sum(B * B, axis=1, keepdims=True).T  # (1,m)
        d2 = A2 + B2 - 2.0 * (A @ B.T)
        d2 = np.maximum(d2, 0.0)
        return self.sigma_f2 * np.exp(-0.5 * d2 / (self.ell ** 2))

    def _kxx(self) -> float:
        return self.sigma_f2  # RBF self-kernel with same point, before noise

    def add_observation(self, x_new: np.ndarray, y_new: float):
        x_new = np.asarray(x_new, dtype=np.float64).reshape(1, -1)
        y_new = float(y_new)

        if self.X is None:
            # n=1
            self.X = x_new
            self.y = np.array([y_new], dtype=np.float64)
            k11 = self._kxx() + self.noise_var + self.jitter
            self.L = np.array([[math.sqrt(k11)]], dtype=np.float64)
            return

        # Append X, y
        X_old = self.X
        n = X_old.shape[0]
        k = self._rbf(X_old, x_new).reshape(n)  # (n,)
        kappa = self._kxx() + self.noise_var + self.jitter

        # Solve v = L^{-1} k
        v = np.linalg.solve(self.L, k)

        diag_sq = kappa - float(v.T @ v)
        # Numerical guard
        diag_sq = max(diag_sq, 1e-12)
        diag = math.sqrt(diag_sq)

        # Build new L
        L_new = np.zeros((n + 1, n + 1), dtype=np.float64)
        L_new[:n, :n] = self.L
        L_new[n, :n] = v
        L_new[n, n] = diag

        self.L = L_new
        self.X = np.vstack([self.X, x_new])
        self.y = np.append(self.y, y_new)

    def _cho_solve(self, b: np.ndarray) -> np.ndarray:
        # Solve (L L^T) x = b
        # First solve L u = b, then L^T x = u
        u = np.linalg.solve(self.L, b)
        x = np.linalg.solve(self.L.T, u)
        return x

    def predict(self, Z: np.ndarray):
        """
        Predict on candidate set Z (N,d).
        Returns: mean_raw (N,), std (N,)
        """
        Z = np.asarray(Z, dtype=np.float64)
        N = Z.shape[0]

        if self.X is None:
            return np.zeros(N, dtype=np.float64), np.sqrt(np.full(N, self._kxx(), dtype=np.float64))

        n = self.X.shape[0]

        K_star = self._rbf(self.X, Z)  # (n,N)
        # v = L^{-1} K_star
        v = np.linalg.solve(self.L, K_star)  # (n,N)

        # posterior variance of latent f
        k_diag = np.full(N, self._kxx(), dtype=np.float64)
        var = k_diag - np.sum(v * v, axis=0)
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var)

        # Standardize rewards based on observed y only
        y_mean = float(np.mean(self.y))
        y_std = float(np.std(self.y) + 1e-8)

        alpha_y = self._cho_solve(self.y)                  # (n,)
        alpha_1 = self._cho_solve(np.ones(n, dtype=np.float64))  # (n,)
        alpha_std = (alpha_y - y_mean * alpha_1) / y_std

        mean_std = K_star.T @ alpha_std   # (N,)
        mean_raw = mean_std * y_std + y_mean
        return mean_raw, std


def run_one(args):
    # Load domain: keep raw rewards (no z-score here); GP baseline standardizes internally.
    graphs, graph_rewards, feat_dim = load_zinc_domain(
        root=args.zinc_root,
        num_actions=args.num_actions,
        seed=args.seed,
        normalize_rewards=False,
        split=args.zinc_split,
        subset=args.zinc_subset,
    )

    env_rds = np.random.RandomState(args.seed)
    ts_rds = np.random.RandomState(args.seed + 12345)

    rewards_arr = np.asarray(graph_rewards, dtype=float)
    max_reward = float(np.max(rewards_arr))

    Z = build_fixed_embeddings(graphs, feat_dim, args)  # (N, D)
    N = Z.shape[0]

    gp = IncrementalRBF_GP(
        length_scale=args.rbf_length_scale,
        noise_var=args.noise_var,
        output_scale=1.0,
        jitter=1e-8,
    )

    actions_all = []
    rewards_all = []
    regrets = []
    cumulative_regret = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()

    for t in range(args.T):
        if args.baseline == "random" or t < args.init_random_steps:
            action_t = int(env_rds.randint(N))
        else:
            mu, std = gp.predict(Z)  # arrays length N

            if args.baseline == "gpucb":
                beta_t = args.ucb_beta * np.sqrt(np.log(t + 1.0))
                score = mu + beta_t * std
            elif args.baseline == "gpei":
                best = float(np.max(np.asarray(rewards_all, dtype=float)))
                score = expected_improvement(mu, std, best=best, xi=args.ei_xi)
            elif args.baseline == "gpts":
                # Diagonal posterior TS (fast). Joint TS over N=1000 would be too expensive here.
                score = mu + std * ts_rds.normal(size=mu.shape)
            else:
                raise ValueError(f"Unknown baseline: {args.baseline}")

            action_t = int(np.argmax(score))

        y_t = evaluate(
            idx=action_t,
            reward_list=graph_rewards,
            noisy=args.noisy_reward,
            _rds=env_rds,
            noise_var=args.noise_var,
        )

        actions_all.append(action_t)
        rewards_all.append(float(y_t))

        # Regret w.r.t. best noiseless reward in the finite action set
        cumulative_regret += (max_reward - float(graph_rewards[action_t]))
        regrets.append(float(cumulative_regret))

        # Update GP with the (embedding, noisy reward)
        gp.add_observation(Z[action_t], float(y_t))

        if (t % args.print_every == 0) and args.runner_verbose:
            print(f"[{args.baseline}] Step {t+1}: Action {action_t}, cum_regret {cumulative_regret:.3f}")

    duration_min = (time.time() - t0) / 60.0
    # --- Memory metrics (peak RSS + peak CUDA) ---
    ru = resource.getrusage(resource.RUSAGE_SELF)
    ru_maxrss = float(getattr(ru, "ru_maxrss", 0.0))
    # Linux: KB; macOS: bytes
    if sys.platform == "darwin":
        peak_rss_mb = ru_maxrss / (1024.0 * 1024.0)
        ru_maxrss_unit = "bytes"
    else:
        peak_rss_mb = ru_maxrss / 1024.0
        ru_maxrss_unit = "KB"

    peak_cuda_alloc_mb = 0.0
    peak_cuda_reserved_mb = 0.0
    if torch.cuda.is_available():
        try:
            peak_cuda_alloc_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
            peak_cuda_reserved_mb = float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0))
        except Exception:
            pass

    if args.runner_verbose:
        print(f"{args.baseline} with T={args.T} steps took {duration_min:.2f} min.")

    run_settings = vars(args).copy()

    exp_results = {
        "actions": actions_all,
        "rewards": rewards_all,
        "regrets": regrets,
    }

    results_dict = {
        "exp_results": exp_results,
        "params": run_settings,
        "duration_total": duration_min,
        "peak_rss_mb": peak_rss_mb,
        "ru_maxrss_raw": ru_maxrss,
        "ru_maxrss_unit": ru_maxrss_unit,
        "peak_cuda_alloc_mb": peak_cuda_alloc_mb,
        "peak_cuda_reserved_mb": peak_cuda_reserved_mb,
        "algorithm": args.baseline,
        "baseline": args.baseline,
    }
    return results_dict


def main(args):
    results_dict = run_one(args)

    if args.exp_result_folder is None:
        from pprint import pprint
        pprint(results_dict)
        return

    # Save under exp_result_folder/<baseline>/
    result_dir = os.path.join(args.exp_result_folder, args.baseline)
    os.makedirs(result_dir, exist_ok=True)

    exp_hash = stable_hash_dict(results_dict["params"])
    exp_result_file = os.path.join(result_dir, f"{exp_hash}.json")
    with open(exp_result_file, "w") as f:
        json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)

    print(f"Saved results to {exp_result_file}")
    print(f"Duration ({args.baseline}): {results_dict['duration_total']:.2f} min.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed-GNN-embedding GP baselines on ZINC 250k (penalized logP) (no sklearn)")

    parser.add_argument("--zinc_root", type=str, default="data/zinc")
    parser.add_argument("--zinc_split", type=str, default="all",
                        help="train | val | test | all (all = full 249,456-molecule domain)")
    parser.add_argument("--zinc_subset", type=str2bool, default=False,
                        help="Use the 12k ZINC subset instead of the full 250k set.")
    parser.add_argument("--num_actions", type=int, default=1000)
    parser.add_argument("--noise_var", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--noisy_reward", type=str2bool, default=True)

    parser.add_argument("--baseline", type=str, required=True,
                        choices=["gpucb", "gpei", "gpts", "random"],
                        help="Which baseline to run.")

    # Fixed (frozen) GNN embedder settings
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--embed_depth", type=int, default=2)
    parser.add_argument("--nn_aggr_feat", type=str2bool, default=True)
    parser.add_argument("--embedder_seed", type=int, default=0,
                        help="Seed used ONLY to initialize the fixed embedder weights.")

    # GP kernel / acquisition settings
    parser.add_argument("--rbf_length_scale", type=float, default=1.0)

    parser.add_argument("--init_random_steps", type=int, default=5,
                        help="Number of initial random observations before using the acquisition.")

    # UCB
    parser.add_argument("--ucb_beta", type=float, default=2.0)

    # EI
    parser.add_argument("--ei_xi", type=float, default=0.0)

    # output/logging
    parser.add_argument("--exp_result_folder", type=str, default="results/zinc_gp_baselines_T300_N1000")
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--runner_verbose", type=str2bool, default=True)

    args = parser.parse_args()
    main(args)
