"""
Post-hoc drug-discovery + efficiency metrics for GNN-PE bandit runs.

Reads result JSONs, groups them by the immediate
result sub-folder (one config per folder, seeds within), and reports, aggregated as mean +/- SE
across seeds:

  Acquisition quality:
    * best_reward@{1,5,10}min : best TRUE reward among molecules acquired within a wallclock budget
    * acq_hits@k%             : # acquired molecules that are in the true top-k%
    * acq_EF@k%               : enrichment factor of the acquired set over random

  Surrogate quality:
    * model_recall@k% / model_EF@k% : rank ALL arms by the final predicted mean, compare top-k% to
                                      the true top-k%

  Regret + efficiency:
    * cum_regret (final) , runtime (min) + train/select breakdown, peak RSS, |S| and grad-cache MB

Usage:
  python scripts/compute_dd_metrics.py --results_root results --time_budgets 1 5 10 --topk_frac 0.01
  python scripts/compute_dd_metrics.py --results_root results --out results/dd_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _is_junk(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name.startswith("._") or path.name == ".DS_Store"


_DOMAIN_CACHE: Dict[tuple, np.ndarray] = {}


def _true_rewards(payload: dict) -> Optional[np.ndarray]:
    """Return the per-arm true (normalized) rewards for a run, from JSON or by reloading QM9."""
    exp = payload.get("exp_results", {})
    if isinstance(exp, dict) and exp.get("true_rewards"):
        return np.asarray(exp["true_rewards"], dtype=float)

    params = payload.get("params", {})
    seed = params.get("seed")
    num_actions = params.get("num_actions")
    target_idx = params.get("target_idx", 4)
    qm9_root = params.get("qm9_root", "data/qm9")
    if seed is None or num_actions is None:
        return None
    key = (str(qm9_root), int(num_actions), int(target_idx), int(seed))
    if key not in _DOMAIN_CACHE:
        try:
            from qm9_bandit_env import load_qm9_domain
            _, rewards, _ = load_qm9_domain(
                root=qm9_root, num_actions=int(num_actions),
                target_idx=int(target_idx), seed=int(seed), normalize_rewards=True,
            )
            _DOMAIN_CACHE[key] = np.asarray(rewards, dtype=float)
        except Exception as exc:  # pragma: no cover
            print(f"  [warn] could not rebuild domain for seed={seed}: {exc}")
            return None
    return _DOMAIN_CACHE[key]


def _elapsed_sec(payload: dict, T: int) -> tuple[np.ndarray, bool]:
    """Cumulative wallclock per round; (values, is_approx)."""
    exp = payload.get("exp_results", {})
    if isinstance(exp, dict) and exp.get("elapsed_sec"):
        return np.asarray(exp["elapsed_sec"], dtype=float), False
    dur_min = payload.get("duration_total")
    if isinstance(dur_min, (int, float)) and T > 0:
        total = float(dur_min) * 60.0
        return (np.arange(1, T + 1) * (total / T)), True
    return np.array([]), True


def _topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, min(k, len(values))))
    return np.argpartition(-values, k - 1)[:k]


def run_metrics(payload: dict, time_budgets_min: List[float], topk_frac: float) -> Optional[dict]:
    exp = payload.get("exp_results", {})
    actions = exp.get("actions")
    if not actions:
        return None
    actions = np.asarray(actions, dtype=int)
    T = len(actions)

    true_r = _true_rewards(payload)
    if true_r is None:
        return None
    N = len(true_r)
    k = int(max(1, math.ceil(topk_frac * N)))
    true_top = set(_topk_indices(true_r, k).tolist())

    elapsed, approx = _elapsed_sec(payload, T)

    out: dict = {}

    # --- Best-reward-vs-time ---
    acq_true = true_r[actions]
    for b in time_budgets_min:
        if elapsed.size == T and T > 0:
            mask = elapsed <= (b * 60.0)
            out[f"best_reward@{b:g}min"] = float(np.max(acq_true[mask])) if np.any(mask) else float("nan")
        else:
            out[f"best_reward@{b:g}min"] = float("nan")
    out["best_reward_time_approx"] = approx

    # --- Acquisition quality (unique acquired molecules) ---
    acquired = set(actions.tolist())
    n_acq = len(acquired)
    hits = len(acquired & true_top)
    out["acq_hits@k"] = int(hits)
    out["acq_n_unique"] = int(n_acq)
    # Enrichment factor vs random screen: (active-rate in acquired) / (active-rate in domain).
    out["acq_EF@k"] = float((hits / n_acq) / (k / N)) if n_acq > 0 else float("nan")

    # --- Surrogate quality (needs predicted means) ---
    pred = exp.get("pred_means_all")
    if pred:
        pred = np.asarray(pred, dtype=float)
        if len(pred) == N:
            model_top = set(_topk_indices(pred, k).tolist())
            inter = len(model_top & true_top)
            out["model_recall@k"] = float(inter / k)
            out["model_EF@k"] = float((inter / k) / (k / N))

    # --- Regret + efficiency ---
    regrets = exp.get("regrets")
    out["cum_regret"] = float(regrets[-1]) if regrets else float("nan")
    out["duration_min"] = float(payload.get("duration_total", float("nan")))
    out["runtime_train_sec"] = float(payload.get("runtime_train_sec", float("nan")))
    out["runtime_select_sec"] = float(payload.get("runtime_select_sec", float("nan")))
    out["uncertainty_dim"] = float(payload.get("uncertainty_dim", float("nan")))
    out["grad_cache_mb"] = float(payload.get("grad_cache_mb", float("nan")))
    out["peak_rss_mb"] = float(payload.get("peak_rss_mb", float("nan")))
    out["k_topk"] = k
    out["N"] = N
    return out


def _agg(values: List[float]) -> tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    mean = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0
    return mean, se


def main():
    ap = argparse.ArgumentParser(description="Drug-discovery + efficiency metrics for GNN-PE runs.")
    ap.add_argument("--results_root", type=str, required=True,
                    help="Root dir; each immediate sub-folder = one config (seeds inside).")
    ap.add_argument("--time_budgets", type=float, nargs="+", default=[1, 5, 10],
                    help="Wallclock budgets (minutes) for best-reward-vs-time.")
    ap.add_argument("--topk_frac", type=float, default=0.01, help="Top fraction defining 'actives' (e.g. 0.01 = top 1%).")
    ap.add_argument("--out", type=str, default=None, help="Optional CSV output path.")
    args = ap.parse_args()

    root = Path(args.results_root)
    if not root.is_dir():
        raise SystemExit(f"results_root not found: {root}")

    # Group JSONs by their immediate parent directory (one config per folder).
    groups: Dict[str, List[Path]] = {}
    for path in sorted(root.rglob("*.json")):
        if _is_junk(path):
            continue
        label = str(path.parent.relative_to(root)) or root.name
        groups.setdefault(label, []).append(path)

    if not groups:
        raise SystemExit(f"No result JSONs found under {root}")

    budget_keys = [f"best_reward@{b:g}min" for b in args.time_budgets]
    metric_order = (budget_keys +
                    ["acq_hits@k", "acq_EF@k", "model_recall@k", "model_EF@k",
                     "cum_regret", "duration_min", "runtime_train_sec", "runtime_select_sec",
                     "uncertainty_dim", "grad_cache_mb", "peak_rss_mb"])

    rows = []
    for label in sorted(groups):
        per_run: Dict[str, List[float]] = {m: [] for m in metric_order}
        n_ok = 0
        approx_any = False
        for path in groups[label]:
            with path.open() as fh:
                payload = json.load(fh)
            m = run_metrics(payload, args.time_budgets, args.topk_frac)
            if m is None:
                continue
            n_ok += 1
            approx_any = approx_any or m.get("best_reward_time_approx", False)
            for key in metric_order:
                if key in m:
                    per_run[key].append(m[key])
        if n_ok == 0:
            continue
        row = {"config": label, "n_seeds": n_ok, "time_approx": approx_any}
        for key in metric_order:
            mean, se = _agg(per_run[key])
            row[f"{key}_mean"] = mean
            row[f"{key}_se"] = se
        rows.append(row)

    
    print(f"\nDrug-discovery + efficiency metrics  (top-{args.topk_frac*100:g}%, mean +/- SE across seeds)\n")
    key_cols = budget_keys + ["acq_hits@k", "acq_EF@k", "model_recall@k", "cum_regret",
                              "duration_min", "uncertainty_dim", "grad_cache_mb", "peak_rss_mb"]
    header = f"{'config':30s} {'n':>2s} " + " ".join(f"{c:>18s}" for c in key_cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for c in key_cols:
            mean = row[f"{c}_mean"]
            se = row[f"{c}_se"]
            cells.append("n/a".rjust(18) if not np.isfinite(mean) else f"{mean:9.3f}+/-{se:6.3f}")
        flag = " *approx-time" if row["time_approx"] else ""
        print(f"{row['config']:30s} {row['n_seeds']:2d} " + " ".join(cells) + flag)
    print("\n(acq_hits@k = # acquired molecules in true top-k; acq_EF@k = enrichment vs random; "
          "grad_cache_mb & uncertainty_dim = parameter-space memory.)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["config", "n_seeds", "time_approx"] + [f"{m}_{s}" for m in metric_order for s in ("mean", "se")]
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
