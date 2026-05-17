"""Shared helpers for reading experiment JSON files and summarizing runs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RunGroup:
    label: str
    directory: Path
    curves: tuple[np.ndarray, ...]
    seeds: tuple[int | None, ...]
    durations_min: tuple[float | None, ...]

    @property
    def n(self) -> int:
        return len(self.curves)


def _is_junk_path(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name.startswith("._") or path.name == ".DS_Store"


def _extract_regrets(payload: dict) -> np.ndarray:
    exp_results = payload.get("exp_results", {})
    if isinstance(exp_results, dict) and "regrets" in exp_results:
        return np.asarray(exp_results["regrets"], dtype=float)
    for key in ("cumulative_regret", "cum_regret", "regrets"):
        if key in payload:
            return np.asarray(payload[key], dtype=float)
    raise KeyError("Could not find a cumulative-regret curve in result JSON.")


def _extract_seed(payload: dict) -> int | None:
    value = payload.get("params", {}).get("seed", payload.get("seed"))
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _extract_duration_min(payload: dict) -> float | None:
    value = payload.get("duration_total")
    if isinstance(value, (int, float)):
        return float(value)
    for key in ("duration_sec", "duration_seconds", "wallclock_sec"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value) / 60.0
    return None


def load_group(label: str, directory: Path) -> RunGroup:
    paths = sorted(path for path in directory.rglob("*.json") if not _is_junk_path(path))
    if not paths:
        raise FileNotFoundError(f"No JSON result files found for {label!r} under {directory}")

    curves: list[np.ndarray] = []
    seeds: list[int | None] = []
    durations: list[float | None] = []
    for path in paths:
        with path.open("r") as handle:
            payload = json.load(handle)
        curve = _extract_regrets(payload).reshape(-1)
        if np.any(np.diff(curve) < -1e-9):
            curve = np.cumsum(curve)
        curves.append(curve)
        seeds.append(_extract_seed(payload))
        durations.append(_extract_duration_min(payload))

    return RunGroup(
        label=label,
        directory=directory,
        curves=tuple(curves),
        seeds=tuple(seeds),
        durations_min=tuple(durations),
    )


def mean_and_se(group: RunGroup) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = min(len(curve) for curve in group.curves)
    arr = np.stack([curve[:horizon] for curve in group.curves], axis=0)
    mean = arr.mean(axis=0)
    se = arr.std(axis=0, ddof=1) / math.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
    x = np.arange(1, horizon + 1)
    return x, mean, se


def runtime_row(group: RunGroup) -> dict[str, str | int | float]:
    finite = [value for value in group.durations_min if value is not None and np.isfinite(value)]
    total = float(np.sum(finite)) if finite else float("nan")
    avg = total / len(finite) if finite else float("nan")
    return {
        "method": group.label,
        "n_runs": group.n,
        "runtime_total_min": total,
        "runtime_avg_per_seed_min": avg,
    }


def write_runtime_csv(groups: Iterable[RunGroup], path: Path) -> None:
    rows = [runtime_row(group) for group in groups]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "n_runs", "runtime_total_min", "runtime_avg_per_seed_min"],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_runtime_table(groups: Iterable[RunGroup]) -> None:
    print("\nRuntime summary (minutes):")
    print(f"{'Method':48s} {'n':>3s} {'total':>12s} {'avg/seed':>12s}")
    print("-" * 80)
    for group in groups:
        row = runtime_row(group)
        total = row["runtime_total_min"]
        avg = row["runtime_avg_per_seed_min"]
        total_str = "n/a" if not np.isfinite(total) else f"{total:.3f}"
        avg_str = "n/a" if not np.isfinite(avg) else f"{avg:.3f}"
        print(f"{group.label:48s} {group.n:3d} {total_str:>12s} {avg_str:>12s}")


def require_subdir(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_dir():
        raise FileNotFoundError(f"Missing required result directory: {path}")
    return path
