#!/usr/bin/env python
"""Single entry point that dispatches to the per-dataset/per-algorithm runners.

The runners live in experiments/run_<dataset>_<algo>.py and can also be invoked
directly (from the repo root, so the data/ and results/ defaults resolve).

    python main.py --dataset zinc --algo phasedgp --T 300 --num_actions 1000
    python main.py --dataset qm9  --algo gp --baseline gpucb

Every flag other than --dataset/--algo is forwarded verbatim to the target
script, so the runners stay the source of truth for their own arguments
(`python main.py --dataset zinc --algo peft --help` shows them).
"""

import argparse
import os
import subprocess
import sys


DATASETS = ["qm9", "zinc", "docking"]

# algo -> run_<dataset>_<suffix>.py
ALGOS = {
    "gnnucb": "gnnucb",
    "phasedgp": "phasedgp",
    "gp_baselines": "gp_baselines",
    "peft": "peft",
    "gnnss": "gnnss",
}

# Convenience spellings for the same runners.
ALIASES = {
    "ucb": "gnnucb",
    "gnn-ucb": "gnnucb",
    "pe": "phasedgp",
    "gnnpe": "phasedgp",
    "gnn-pe": "phasedgp",
    "phased": "phasedgp",
    "gp": "gp_baselines",
    "baselines": "gp_baselines",
    "ss": "gnnss",
    "gnn-ss": "gnnss",
}

# The docking runners that log per-round wall-clock carry a _timed suffix.
SUFFIX_OVERRIDES = {
    ("docking", "peft"): "peft_timed",
    ("docking", "gp_baselines"): "gp_baselines_timed",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNER_DIR = os.path.join(BASE_DIR, "experiments")


def resolve_script(dataset: str, algo: str) -> str:
    dataset = dataset.lower()
    algo = ALIASES.get(algo.lower(), algo.lower())

    if dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset: {dataset}. Choose from {DATASETS}.")
    if algo not in ALGOS:
        raise SystemExit(
            f"Unknown algo: {algo}. Choose from {sorted(ALGOS)} "
            f"(aliases: {sorted(ALIASES)})."
        )

    suffix = SUFFIX_OVERRIDES.get((dataset, algo), ALGOS[algo])
    script = os.path.join(RUNNER_DIR, f"run_{dataset}_{suffix}.py")
    if not os.path.isfile(script):
        raise SystemExit(f"No runner for dataset={dataset} algo={algo}: {script} not found.")
    return script


def main():
    parser = argparse.ArgumentParser(
        description="Dispatch a bandit run to the right dataset/algorithm script.",
        epilog="All other flags are passed straight through to the target runner.",
    )
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS,
                        help="qm9 (reward = y[target_idx]), zinc (reward = penalized logP), "
                             "or docking (Parquet-derived; reward set by --objective)")
    parser.add_argument("--algo", type=str, required=True,
                        help=f"One of {sorted(ALGOS)}; aliases: {sorted(ALIASES)}")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the command that would run, then exit.")

    args, passthrough = parser.parse_known_args()
    script = resolve_script(args.dataset, args.algo)
    cmd = [sys.executable, script] + passthrough

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    print(f"[main] {args.dataset} / {args.algo} -> {os.path.basename(script)}")
    return subprocess.call(cmd, cwd=BASE_DIR)


if __name__ == "__main__":
    sys.exit(main())
