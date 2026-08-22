# PhasedGNN-UCB (GNN-PE) on QM9 with two acceleration axes:
#   * graph-domain compression : --domain_scan_method {full,fft,graph_kmeans,pivchol} (+ apply_to)
#   * parameter-space (PEFT)    : --peft {none,lora,last_layer} (+ --lora_rank/--lora_alpha)
#
# This is a superset of run_qm9_phasedgp.py: with --peft none and no scan flags it reproduces the
# original phased runner. It additionally logs everything the drug-discovery / efficiency metrics
# need: per-round wallclock, train-vs-select runtime breakdown, uncertainty-model memory (|S| and
# gradient-cache MB), and an end-of-run dump of predicted means over all arms (for Model EF/recall).
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
from qm9_bandit_env import load_qm9_domain
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


def _grad_cache_mb(learner) -> float:
    """Footprint of the cached NTK feature vectors g (float32): #cached * |S| * 4 bytes."""
    dim = int(getattr(learner, "num_net_params", 0))
    if getattr(learner, "init_grad_list", None):
        n_cached = len(learner.init_grad_list)
    else:
        n_cached = len(getattr(learner, "_grad_cache", {}) or {})
    return float(n_cached * dim * 4) / (1024.0 * 1024.0)


def run_one(args, train_mode: str):
    env_rds = np.random.RandomState(args.seed)
    algo_rds = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    graph_data, graph_rewards, feat_dim = load_qm9_domain(
        root=args.qm9_root,
        num_actions=args.num_actions,
        target_idx=args.target_idx,
        seed=args.seed,
        normalize_rewards=True,
    )

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
        )

    # PEFT applies only in finetune mode (scratch resets weights each round => no adapter warm start).
    peft = args.peft if train_mode == "finetune" else "none"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    learner = PhasedGnnUCB(
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
        max_train_steps=max_train_steps,
        min_train_steps=args.min_train_steps,
        early_stop_loss=args.early_stop_loss,
        early_stop_rel_impr=args.early_stop_rel_impr,
        reuse_optimizer=args.reuse_optimizer,
        scan_indexer=scan_indexer,
        scan_apply_to=getattr(args, 'domain_scan_apply_to', 'both'),
        mask_revisit=getattr(args, 'mask_revisit', True),
        lazy_grads=args.lazy_grads,
        # --- PEFT (parameter-space compression) ---
        peft=peft,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        peft_uncertainty=args.peft_uncertainty,
    )

    # NTK / uncertainty dimension |S| (this is what PEFT shrinks): the dim of g and U.
    uncertainty_dim = int(getattr(learner, "num_net_params", 0))

    t0 = time.time()
    t_train = 0.0
    t_select = 0.0

    regrets = []
    regrets_bp = []
    cumulative_regret = 0.0
    cumulative_regret_bp = 0.0

    new_indices = []
    new_rewards = []
    actions_all = []
    avg_vars = []
    pick_vars_all = []
    pick_rewards_all = []
    elapsed_sec = []  # cumulative wallclock after each round -> best-reward-vs-time

    run_settings = dict(vars(args))
    run_settings.update({
        "train_mode": train_mode,
        "train_from_scratch_effective": train_from_scratch,
        "max_train_steps_effective": max_train_steps,
        "peft_effective": peft,
    })
    if scan_indexer is not None:
        run_settings.update({
            "domain_scan_method": scan_indexer.method,
            "domain_scan_meta": scan_indexer.meta,
            "domain_scan_num_reps": len(scan_indexer.reps),
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
        print(f"[{train_mode}] seed={args.seed} peft={peft} scan={getattr(args, 'domain_scan_method', 'full')} "
              f"T={args.T} num_actions={args.num_actions} |S|={uncertainty_dim}")

    for t in range(args.T):
        _ts = time.time()
        if t > args.pretrain_steps:
            action_t = learner.select()
        else:
            action_t = learner.explore()
        t_select += (time.time() - _ts)

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

        if t < args.T0:
            learner.add_data([action_t], [observed_reward_t])
            if t > args.pretrain_steps:
                _tt = time.time()
                _ = learner.train()
                t_train += (time.time() - _tt)
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
                    _tt = time.time()
                    _ = learner.train()
                    t_train += (time.time() - _tt)
                new_indices, new_rewards = [], []

        regrets.append(float(cumulative_regret))
        regrets_bp.append(float(cumulative_regret_bp))
        pick_vars_all.append(float(learner.get_post_var(action_t)))
        elapsed_sec.append(float(time.time() - t0))

        if (t + 1) % args.progress_every == 0:
            elapsed_min = (time.time() - t0) / 60.0
            try:
                max_len = len(learner.maximizers)
            except Exception:
                max_len = -1
            print(f"[{train_mode}|peft={peft}|{getattr(args, 'domain_scan_method', 'full')}] "
                  f"t={t+1}/{args.T} cum_regret={cumulative_regret:.3f} |maximizers|={max_len} elapsed_min={elapsed_min:.2f}")

    duration_min = (time.time() - t0) / 60.0

    # End-of-run surrogate quality: predicted mean over ALL arms (for Model EF / recall@k).
    pred_means_all = [float(learner.get_post_mean(i)) for i in range(args.num_actions)]
    grad_cache_mb = _grad_cache_mb(learner)

    # --- Memory metrics (peak RSS + peak CUDA) ---
    ru = resource.getrusage(resource.RUSAGE_SELF)
    ru_maxrss = float(getattr(ru, "ru_maxrss", 0.0))
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
        "elapsed_sec": elapsed_sec,
        "pred_means_all": pred_means_all,
        "true_rewards": list(map(float, graph_rewards)),
    }

    results_dict = {
        "exp_results": exp_results,
        "params": run_settings,
        "duration_total": duration_min,
        "runtime_train_sec": t_train,
        "runtime_select_sec": t_select,
        "uncertainty_dim": uncertainty_dim,
        "grad_cache_mb": grad_cache_mb,
        "peak_rss_mb": peak_rss_mb,
        "ru_maxrss_raw": ru_maxrss,
        "ru_maxrss_unit": ru_maxrss_unit,
        "peak_cuda_alloc_mb": peak_cuda_alloc_mb,
        "peak_cuda_reserved_mb": peak_cuda_reserved_mb,
        "algorithm": "us",
    }

    return results_dict


def main(args):
    train_modes = _resolve_train_modes(args)
    multi = len(train_modes) > 1

    for mode in train_modes:
        results_dict = run_one(args, mode)

        if args.exp_result_folder is None:
            from pprint import pprint
            pprint({k: v for k, v in results_dict.items() if k != "exp_results"})
            continue

        result_dir = _mode_result_dir(args.exp_result_folder, mode, multi=multi, mode_subdir=args.mode_subdir)
        os.makedirs(result_dir, exist_ok=True)

        exp_hash = stable_hash_dict(results_dict["params"])
        exp_result_file = os.path.join(result_dir, f"{exp_hash}.json")
        with open(exp_result_file, "w") as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)

        print(f"Saved results to {exp_result_file}")
        print(f"Duration ({mode}): {results_dict['duration_total']:.2f} min. "
              f"train={results_dict['runtime_train_sec']:.1f}s select={results_dict['runtime_select_sec']:.1f}s "
              f"|S|={results_dict['uncertainty_dim']} grad_cache={results_dict['grad_cache_mb']:.1f}MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhasedGNN-UCB (GNN-PE) on QM9 with PEFT + scan acceleration")

    parser.add_argument("--qm9_root", type=str, default="data/qm9")
    parser.add_argument("--target_idx", type=int, default=4)
    parser.add_argument("--num_actions", type=int, default=1000)
    parser.add_argument("--noise_var", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--net", type=str, default="GNN")
    parser.add_argument("--noisy_reward", type=str2bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_mlp_layers_alg", type=int, default=2)

    parser.add_argument("--train_from_scratch", type=str2bool, default=False)
    parser.add_argument(
        "--train_modes",
        type=str,
        default="finetune",
        help="Comma-separated list of train modes: scratch,finetune. PEFT applies to finetune only.",
    )
    parser.add_argument("--mode_subdir", type=str2bool, default=True)

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

    parser.add_argument("--max_train_steps_scratch", type=int, default=1000)
    parser.add_argument("--max_train_steps_finetune", type=int, default=20)
    parser.add_argument("--min_train_steps", type=int, default=0)
    parser.add_argument("--early_stop_loss", type=float, default=1e-4)
    parser.add_argument("--early_stop_rel_impr", type=float, default=1e-3)
    parser.add_argument("--reuse_optimizer", type=str2bool, default=True)

    parser.add_argument("--exp_result_folder", type=str, default="results/qm9_peft_T300_N1000")
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--runner_verbose", type=str2bool, default=False)
    parser.add_argument("--progress_every", type=int, default=50)

    # Scan acceleration ablations (apply to finetune only)
    parser.add_argument("--domain_scan_method", type=str, default="full",
                        help="Scan reduction for (C.1)/(C.2) in finetune mode: full|fft|graph_kmeans|pivchol")
    parser.add_argument("--domain_scan_apply_to", type=str, default="both",
                        help="Where to apply scan reduction: both|c1|c2 (C.1=max-var selection, C.2=maximizer update).")
    parser.add_argument("--scan_wl_h", type=int, default=2)
    parser.add_argument("--scan_fft_m", type=int, default=300)
    parser.add_argument("--scan_kmeans_k", type=int, default=300)
    parser.add_argument("--scan_kmeans_iter", type=int, default=8)
    parser.add_argument("--scan_pivchol_m", type=int, default=300)
    parser.add_argument("--scan_matfree", type=str, default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--lazy_grads", type=str2bool, default=True)
    parser.add_argument("--mask_revisit", type=str2bool, default=True,
                        help="Mask already-visited arms from (C.1) selection so each arm is pulled at most once. "
                             "false=arms may be re-pulled (matches pre-masking behaviour).")

    # PEFT (parameter-space compression) ablations (apply to finetune only)
    parser.add_argument("--peft", type=str, default="none", choices=["none", "lora", "last_layer"],
                        help="Parameter-efficient fine-tuning: none (full FT) | lora | last_layer.")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank r.")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA scaling alpha (s=alpha/r).")
    parser.add_argument("--peft_uncertainty", type=str2bool, default=True,
                        help="If true, NTK gradients/U are computed only over PEFT params (compressed uncertainty). "
                             "If false, train PEFT params but compute uncertainty over ALL params (ablation).")

    args = parser.parse_args()
    main(args)
