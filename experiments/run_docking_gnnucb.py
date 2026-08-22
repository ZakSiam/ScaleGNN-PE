import os
import json
import time
import argparse
import sys
import resource

import numpy as np
import torch

# Runners live in experiments/; put the repo root on sys.path so the shared
# modules (algorithms, *_bandit_env, nets, ...) import when this file is run
# directly: python experiments/run_<dataset>_<algo>.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms import GnnUCB
from docking_bandit_env import load_docking_domain
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
    # Make each mode fully reproducible and comparable.
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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    learner = GnnUCB(
        net="GNN",
        feat_dim=feat_dim,
        num_nodes=args.num_nodes,
        num_actions=args.num_actions,
        action_domain=graph_data,
        verbose=args.runner_verbose,
        alg_lambda=args.alg_lambda,
        exploration_coef=args.exploration_coef,
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
    )

    t0 = time.time()

    regrets = []
    regrets_bp = []
    cumulative_regret = 0.0
    cumulative_regret_bp = 0.0

    actions_all = []
    avg_vars = []
    pick_vars_all = []
    pick_rewards_all = []

    run_settings = dict(vars(args))
    run_settings.update({
        "train_mode": train_mode,
        "train_from_scratch_effective": train_from_scratch,
        "max_train_steps_effective": max_train_steps,
    })
    print("Run settings:", run_settings)

    for t in range(args.T):
        action_t = learner.select()
        actions_all.append(action_t)

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

        learner.add_data([action_t], [observed_reward_t])
        if t > args.pretrain_steps:
            _ = learner.train()

        regrets.append(float(cumulative_regret))
        regrets_bp.append(float(cumulative_regret_bp))
        pick_vars_all.append(float(learner.get_post_var(action_t)))

        if t % args.print_every == 0 and args.runner_verbose:
            print(f"Step {t+1}: Action {action_t}, cum_regret {cumulative_regret:.3f}")

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
        "algorithm": "ucb",
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

        exp_hash = stable_hash_dict(results_dict["params"])
        exp_result_file = os.path.join(result_dir, f"{exp_hash}.json")
        with open(exp_result_file, "w") as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)

        print(f"Saved results to {exp_result_file}")
        print(f"Duration ({mode}): {results_dict['duration_total']:.2f} min.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-UCB on docking (Parquet-derived molecules)")

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
    parser.add_argument("--exploration_coef", type=float, default=0.0002616770633854424)
    parser.add_argument("--alg_lambda", type=float, default=0.0032548253715214955)
    parser.add_argument("--nn_aggr_feat", type=str2bool, default=True)
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--num_nodes", type=int, default=5)

    # New knobs for scratch vs fine-tune runtime/performance comparisons
    parser.add_argument("--max_train_steps_scratch", type=int, default=1000)
    parser.add_argument("--max_train_steps_finetune", type=int, default=20)
    parser.add_argument("--min_train_steps", type=int, default=0)
    parser.add_argument("--early_stop_loss", type=float, default=1e-4)
    parser.add_argument("--early_stop_rel_impr", type=float, default=1e-3)
    parser.add_argument("--reuse_optimizer", type=str2bool, default=True)

    parser.add_argument("--exp_result_folder", type=str, default="results/docking_3CLPro_dock_norm_gnnucb_T300")
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--runner_verbose", type=str2bool, default=True)

    args = parser.parse_args()
    main(args)
