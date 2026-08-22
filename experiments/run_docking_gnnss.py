import os
import json
import time
import argparse
import sys
import platform
import resource

import numpy as np
import torch

# Runners live in experiments/; put the repo root on sys.path so the shared
# modules (algorithms, *_bandit_env, nets, ...) import when this file is run
# directly: python experiments/run_<dataset>_<algo>.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import PhasedGnnUCB
from gnn_ss import GnnSS
from gnn_ss_features import augment_with_laplacian_pe
from docking_bandit_env import load_docking_domain
from utils_exp import NumpyArrayEncoder, stable_hash_dict

from graph_scan_accel import build_scan_indexer


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


def evaluate(idx_list, reward_list, noisy, _rds, noise_var):
    rew = np.array([reward_list[idx] for idx in idx_list], dtype=float)
    if noisy:
        rew = rew + _rds.normal(0.0, noise_var, size=rew.shape)
    return list(rew)


def _resolve_train_modes(args):
    """Backwards compatible:
    - if --train_modes is provided, use it (comma-separated)
    - else fall back to --train_from_scratch
    """
    if args.train_modes is None:
        return ["scratch"] if args.train_from_scratch else ["finetune"]
    modes = [m.strip().lower() for m in args.train_modes.split(",") if m.strip()]
    for m in modes:
        if m not in {"scratch", "finetune"}:
            raise ValueError(f"Unknown train mode: {m}. Use scratch or finetune.")
    return modes


def _mode_result_dir(base_dir: str, mode: str, multi: bool, mode_subdir: bool):
    if base_dir is None:
        return None
    if multi and mode_subdir:
        return os.path.join(base_dir, mode)
    return base_dir


def run_one(args, train_mode: str):
    # Make each mode fully reproducible and comparable (same seed, same domain sampling).
    env_rds = np.random.RandomState(args.seed)
    algo_rds = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    graph_data, graph_rewards, feat_dim = load_docking_domain(
        graph_dir=args.graph_dir,
        target=args.target,
        objective=args.objective,
        num_actions=(None if args.num_actions <= 0 else args.num_actions),
        max_shards=args.max_shards,
        seed=args.seed,
        normalize_rewards=True,
    )
    # Appendix C: k absolute graph-Laplacian positional encodings concatenated to node
    # features (k=4 with m=512 for the Diag/Sub/FJL comparison; k=0 reproduces the
    # MolPAL-comparable "Diag, NoLap" row). Cached per domain, seed-independent.
    if int(getattr(args, "lap_pe_k", 0)) > 0:
        graph_data, feat_dim = augment_with_laplacian_pe(
            graph_data, k=int(args.lap_pe_k),
            cache_dir=getattr(args, "scan_cache_dir", None), verbose=True)

    # For docking the loader decides the domain size (num_actions <= 0 => full).
    args.num_actions = len(graph_data)

    max_reward = float(np.max(graph_rewards))
    max_graph = int(np.argmax(graph_rewards))

    assert args.num_actions == len(graph_data)
    assert len(graph_data) == len(graph_rewards)

    if train_mode == "scratch":
        train_from_scratch = True
        max_train_steps = int(args.max_train_steps_scratch)
    elif train_mode == "finetune":
        train_from_scratch = False
        max_train_steps = int(args.max_train_steps_finetune)
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    # --- Scan acceleration ablations (apply ONLY to finetune mode) ---
    scan_indexer = None
    if (train_mode == "finetune") and (getattr(args, "domain_scan_method", "full") != "full"):
        # The scan is seed-independent, so cache it once per (target, method, params) and
        # reuse across seeds. Cache is written by the current pivchol code.
        scan_cache_path = None
        if getattr(args, "scan_cache_dir", None):
            key = (f"scan_{args.target}_{args.domain_scan_method}"
                   f"_wl{args.scan_wl_h}_m{args.scan_pivchol_m}"
                   f"_fft{args.scan_fft_m}_km{args.scan_kmeans_k}.npz")
            scan_cache_path = os.path.join(args.scan_cache_dir, key)
        scan_indexer = build_scan_indexer(
            graphs=graph_data,
            method=args.domain_scan_method,
            wl_h=args.scan_wl_h,
            fft_m=args.scan_fft_m,
            kmeans_k=args.scan_kmeans_k,
            kmeans_iter=args.scan_kmeans_iter,
            pivchol_m=args.scan_pivchol_m,
            matfree=(None if args.scan_matfree == "auto" else (args.scan_matfree == "true")),
            random_state=algo_rds,
            cache_path=scan_cache_path,
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    scan_indexer = None  # GNN-SS sub-samples uniformly at random; no WL scan index needed
    learner = GnnSS(
        net="GNN",
        feat_dim=feat_dim,
        num_nodes=args.num_nodes,
        num_actions=args.num_actions,
        action_domain=graph_data,
        verbose=args.runner_verbose,
        alg_lambda=args.alg_lambda,
        exploration_coef=args.exploration_coef,
        t_intersect=args.t_intersect,
        train_from_scratch=train_from_scratch,
        nn_aggr_feat=args.nn_aggr_feat,
        num_mlp_layers=args.num_mlp_layers_alg,
        neuron_per_layer=args.neuron_per_layer,
        lr=args.lr,
        random_state=algo_rds,
        # New knobs to compare scratch vs fine-tune fairly
        max_train_steps=max_train_steps,
        min_train_steps=args.min_train_steps,
        early_stop_loss=args.early_stop_loss,
        early_stop_rel_impr=args.early_stop_rel_impr,
        reuse_optimizer=args.reuse_optimizer,
        lazy_grads=args.lazy_grads,
        # --- GNN-SS (Algorithm 1) ---
        subsample_K=args.subsample_K,
        batch_size=args.ss_batch,
        explore_rounds=args.explore_rounds,
        acquisition=args.acquisition,
        uncertainty=args.uncertainty,
        proj_dim=args.proj_dim,
        cov_scaling=args.cov_scaling,
        ss_train_batch=args.ss_train_batch,
        paper_init=args.paper_init,
    )

    t0 = time.time()

    regrets = []
    regrets_bp = []
    cumulative_regret = 0.0
    cumulative_regret_bp = 0.0

    new_indices = []
    new_rewards = []
    actions_all = []
    elapsed_all = []   # wall-clock seconds since t0, one entry per round
    avg_vars = []
    pick_vars_all = []
    pick_rewards_all = []

    run_settings = dict(vars(args))
    run_settings.update({
        "train_mode": train_mode,
        "train_from_scratch_effective": train_from_scratch,
        "max_train_steps_effective": max_train_steps,
    })
    if scan_indexer is not None:
        run_settings.update({
            "domain_scan_method": scan_indexer.method,
            "domain_scan_meta": scan_indexer.meta,
            "domain_scan_num_reps": len(scan_indexer.reps),
            "scan_tie_break": getattr(args, "scan_tie_break", "first"),
        })
    else:
        run_settings.update({
            "domain_scan_method": getattr(args, "domain_scan_method", "full"),
            "domain_scan_meta": None,
            "domain_scan_num_reps": None,
        })
    if args.runner_verbose:
        print("Run settings:", run_settings)
    else:
        print(f"[{train_mode}] seed={args.seed} scan={getattr(args, 'domain_scan_method', 'full')} T={args.T} num_actions={args.num_actions}")

    for t in range(args.T):
        # Algorithm 1 queries the top-b of the sub-sample each round; b=1 is the sequential
        # case and reproduces the original loop exactly.
        if t > args.pretrain_steps:
            _batch = learner.select_batch() if int(args.ss_batch) > 1 else [learner.select()]
        else:
            _batch = [learner.explore() for _ in range(max(1, int(args.ss_batch)))]
        action_t = _batch[0]
        for _extra in _batch[1:]:
            _y = evaluate(idx_list=[_extra], noisy=args.noisy_reward, reward_list=graph_rewards,
                          noise_var=args.noise_var, _rds=env_rds)[0]
            learner.add_data([_extra], [_y])
            actions_all.append(_extra)

        actions_all.append(action_t)
        elapsed_all.append(time.time() - t0)

        observed_reward_t = evaluate(
            idx_list=[action_t],
            noisy=args.noisy_reward,
            reward_list=graph_rewards,
            noise_var=args.noise_var,
            _rds=env_rds,
        )[0]
        pick_rewards_all.append(observed_reward_t)

        regret_t = max_reward - graph_rewards[action_t]
        cumulative_regret += regret_t

        best_action_t = learner.exploit()
        regret_t_bp = max_reward - graph_rewards[best_action_t]
        cumulative_regret_bp += regret_t_bp

        if t < args.T0:
            learner.add_data([action_t], [observed_reward_t])
            if t > args.pretrain_steps:
                _ = learner.train()
        else:
            if len(new_rewards) > 0:
                new_rewards.append(observed_reward_t)
                new_indices.append(action_t)
            else:
                new_rewards = [observed_reward_t]
                new_indices = [action_t]

            if t % args.batch_size == 0:
                learner.add_data(new_indices, new_rewards)
                if t > args.pretrain_steps:
                    _ = learner.train()
                new_indices, new_rewards = [], []

        regrets.append(float(cumulative_regret))
        regrets_bp.append(float(cumulative_regret_bp))
        pick_vars_all.append(float(learner.get_post_var(action_t)))

        if (t + 1) % args.progress_every == 0:
            elapsed_min = (time.time() - t0) / 60.0
            # Minimal progress print (cheap)
            try:
                max_len = len(learner.maximizers)
            except Exception:
                max_len = -1
            print(f"[{train_mode}|{getattr(args, 'domain_scan_method', 'full')}] t={t+1}/{args.T} cum_regret={cumulative_regret:.3f} |maximizers|={max_len} elapsed_min={elapsed_min:.2f}")

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
        print(f"{learner.name} [{train_mode}] with T={args.T} steps took {duration_min:.2f} min.")

    exp_results = {
        "actions": actions_all,
        "elapsed": elapsed_all,
        "rewards": pick_rewards_all,
        "regrets": regrets,
        "regrets_bp": regrets_bp,
        "pick_vars_all": pick_vars_all,
        "avg_vars": avg_vars,
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
        "algorithm": "us",  # as in original phased code
    }

    return results_dict


def main(args):
    train_modes = _resolve_train_modes(args)
    multi = len(train_modes) > 1

    for mode in train_modes:
        results_dict = run_one(args, mode)

        if args.exp_result_folder is None:
            from pprint import pprint
            pprint(results_dict)
            continue

        result_dir = _mode_result_dir(args.exp_result_folder, mode, multi=multi, mode_subdir=args.mode_subdir)
        os.makedirs(result_dir, exist_ok=True)

        # Hash includes mode-specific params (train_mode, max_train_steps, etc.)
        exp_hash = stable_hash_dict(results_dict["params"])
        exp_result_file = os.path.join(result_dir, f"{exp_hash}.json")
        with open(exp_result_file, "w") as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)

        print(f"Saved results to {exp_result_file}")
        print(f"Duration ({mode}): {results_dict['duration_total']:.2f} min.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhasedGNN-UCB (GNN-PE) on docking (Parquet-derived molecules)")

    parser.add_argument("--graph_dir", type=str, default="data/graphs/3CLPro",
                        help="Directory of graphs_*.pt shards for ONE target "
                             "(output of SMILES_to_graph_converter/smiles_parquet_to_pyg.py).")
    parser.add_argument("--target", type=str, default="3CLPro",
                        choices=["3CLPro", "6T2W", "WRN", "rtcb"],
                        help="Docking target (controls the dock_raw == 0 cleaning rule).")
    parser.add_argument("--objective", type=str, default="dock_norm",
                        choices=["dock_norm", "solubility", "sa", "qed",
                                 "similarity_to_topK", "dock"],
                        help="Reward. dock_norm = normalized docking in [0,1] (higher=better); "
                             "dock = -dock_raw.")
    parser.add_argument("--max_shards", type=int, default=None,
                        help="Read only the first N shards (smoke tests).")
    parser.add_argument("--num_actions", type=int, default=-1,
                        help="Domain size. <= 0 means use the FULL domain (~1M).")
    parser.add_argument("--noise_var", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--net", type=str, default="GNN")
    parser.add_argument("--noisy_reward", type=str2bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_mlp_layers_alg", type=int, default=2)

    # Backwards-compatible single-flag mode
    parser.add_argument("--train_from_scratch", type=str2bool, default=True)

    # New: run both modes from the same script
    parser.add_argument(
        "--train_modes",
        type=str,
        default=None,
        help="Comma-separated list of train modes to run: scratch,finetune. If omitted, uses --train_from_scratch.",
    )
    parser.add_argument("--mode_subdir", type=str2bool, default=True,
                        help="If running multiple modes, save results under exp_result_folder/<mode>/")

    parser.add_argument("--pretrain_steps", type=int, default=40)
    parser.add_argument("--neuron_per_layer", type=int, default=2048)
    parser.add_argument("--exploration_coef", type=float, default=0.001341321712103193)
    parser.add_argument("--alg_lambda", type=float, default=1.156e-3)
    parser.add_argument("--t_intersect", type=int, default=80)
    parser.add_argument("--nn_aggr_feat", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--T0", type=int, default=100)
    parser.add_argument("--num_nodes", type=int, default=5)

    # New knobs for scratch vs fine-tune runtime/performance comparisons
    parser.add_argument("--max_train_steps_scratch", type=int, default=1000)
    parser.add_argument("--max_train_steps_finetune", type=int, default=20)
    parser.add_argument("--min_train_steps", type=int, default=0)
    parser.add_argument("--early_stop_loss", type=float, default=1e-4)
    parser.add_argument("--early_stop_rel_impr", type=float, default=1e-3)
    parser.add_argument("--reuse_optimizer", type=str2bool, default=True)

    # --- GNN-SS (Wang-Henderson et al., NeurIPS 2023 workshop) ---
    parser.add_argument("--subsample_K", type=int, default=4000,
                        help="|G_t|: uniform sub-sample scored each round (paper uses 4000)")
    parser.add_argument("--ss_batch", type=int, default=1,
                        help="b: arms queried per round from the sub-sample")
    parser.add_argument("--explore_rounds", type=int, default=0,
                        help="T0 warm-start rounds of uniform random querying (paper uses 500)")
    parser.add_argument("--acquisition", type=str, default="ucb", choices=["ucb", "greedy", "ts"])
    parser.add_argument("--uncertainty", type=str, default="sub", choices=["sub", "diag", "fjl"],
                        help="sub = JL random-projection NTK covariance (Eq. 2); diag = the "
                             "diagonal approximation used by GNN-UCB / GNN-PE")
    parser.add_argument("--cov_scaling", type=str, default="alg1", choices=["alg1", "eq2"],
                        help="alg1: K_t = lambda I + sum g g^T (Algorithm 1 / Appendix B). "
                             "eq2: K_t = lambda I + (1/t) sum g g^T (Eq. 1 and 2). The paper "
                             "states both; this selects which one to use.")
    parser.add_argument("--lap_pe_k", type=int, default=0,
                        help="k absolute graph-Laplacian positional encodings (Appendix C uses "
                             "k=4 with m=512; k=0 is the MolPAL-comparable NoLap setting)")
    parser.add_argument("--ss_train_batch", type=int, default=32,
                        help="minibatch size b drawn from H_t inside TrainGNN (Algorithm 2)")
    parser.add_argument("--paper_init", type=str2bool, default=False,
                        help="theta_0 ~ N(0,I) as written in Appendix A. Only valid under NTK "
                             "parameterisation; nets.py is standard-parameterised, so this blows "
                             "the output scale up by ~m^(L/2). Leave False.")
    parser.add_argument("--proj_dim", type=int, default=2048, help="d: JL projection dim")
    parser.add_argument("--exp_result_folder", type=str, default="results/docking_3CLPro_dock_norm_gnnss_T300")
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--runner_verbose", type=str2bool, default=False)
    parser.add_argument("--progress_every", type=int, default=50)

    # Scan acceleration ablations (apply to finetune only)
    parser.add_argument("--domain_scan_method", type=str, default="full",
                        help="Scan reduction for (C.1)/(C.2) in finetune mode: full|fft|graph_kmeans|pivchol")
    parser.add_argument("--domain_scan_apply_to", type=str, default="both",
                        help="Where to apply scan reduction in finetune mode: both|c1|c2 (C.1=max-var selection, C.2=maximizer update).")
    parser.add_argument("--scan_wl_h", type=int, default=2, help="WL subtree kernel iterations for scan mapping")
    parser.add_argument("--scan_fft_m", type=int, default=300, help="FFT shortlist size (representatives)")
    parser.add_argument("--scan_kmeans_k", type=int, default=300, help="Graph k-means clusters (kernel k-means)")
    parser.add_argument("--scan_kmeans_iter", type=int, default=8, help="Kernel k-means iterations")
    parser.add_argument("--scan_pivchol_m", type=int, default=300, help="Pivoted-Cholesky representative count (reps)")
    parser.add_argument("--scan_matfree", type=str, default="auto", choices=["auto", "true", "false"],
                        help="Matrix-free pivchol (no dense NxN kernel): auto=enable when N>20k, true/false to force.")
    parser.add_argument("--scan_cache_dir", type=str, default=None,
                        help="Directory to cache/reuse the scan indexer across seeds (seed-independent). "
                             "Built by the current pivchol code; do not reuse caches from an older version.")
    parser.add_argument("--scan_tie_break", type=str, default="first", choices=["first", "random"],
                        help="Tie policy for the (C.1) argmax. Scan approximations give whole clusters "
                             "identical variance, so ties are common: first=lowest-indexed tied arm "
                             "(original behaviour), random=uniform among tied arms.")
    parser.add_argument("--mask_revisit", type=str2bool, default=True,
                        help="Mask already-pulled arms out of the (C.1) argmax so each pull is a distinct "
                             "arm (default). Set False to allow re-pulling (pre-masking behavior) for ablation.")
    parser.add_argument("--lazy_grads", type=str2bool, default=True,
                        help="Compute f0 NTK gradients lazily (only for queried reps/arms) instead of all N up front.")

    args = parser.parse_args()
    main(args)
