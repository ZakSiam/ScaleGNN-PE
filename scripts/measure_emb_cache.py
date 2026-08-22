#!/usr/bin/env python
"""Measure a docking target's GP embedding-cache build in isolation (wall-clock, RSS, CUDA).

Builds emb_<target>_d128_L2_s0.npy into a SCRATCH directory -- the existing
results/emb_cache_public/ copy is never touched -- then verifies the fresh build is
byte-identical to it, so the timing describes the cache actually in use.

Mirrors experiments/run_docking_gp_baselines_timed.py's build_fixed_embeddings call exactly
(embed_dim 128, embed_depth 2, embedder_seed 0). The runner builds this cache BEFORE
its timing clock starts, so per-run JSONs never capture the cost.

    python scripts/measure_emb_cache.py --target 6T2W --out /path/to/scratch
"""
import argparse, json, os, resource, sys, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "experiments"))   # the runners moved under experiments/

import numpy as np
import torch

from docking_bandit_env import load_docking_domain
from run_docking_gp_baselines_timed import build_fixed_embeddings


class A:
    """Minimal args stand-in with exactly the fields build_fixed_embeddings reads."""
    def __init__(self, target, cache_dir):
        self.target = target
        self.emb_cache_dir = cache_dir
        self.embed_dim = 128
        self.embed_depth = 2
        self.embedder_seed = 0
        self.nn_aggr_feat = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="6T2W")
    ap.add_argument("--graph_dir", default=None)
    ap.add_argument("--out", required=True, help="scratch dir for the fresh build")
    ap.add_argument("--reference", default="results/emb_cache_public")
    args = ap.parse_args()

    graph_dir = args.graph_dir or f"data/graphs/{args.target}"
    os.makedirs(args.out, exist_ok=True)

    print(f"[emb] loading {args.target} domain...", flush=True)
    t_load = time.time()
    graphs, _, feat_dim = load_docking_domain(
        graph_dir=graph_dir, target=args.target, objective="dock_norm",
        num_actions=None, max_shards=None, seed=0, normalize_rewards=True)
    load_s = time.time() - t_load
    print(f"[emb] {len(graphs)} graphs, feat_dim={feat_dim}, load {load_s/60:.2f} min", flush=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    t0 = time.time()
    Z = build_fixed_embeddings(graphs, feat_dim, A(args.target, args.out))
    secs = time.time() - t0

    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cuda_alloc = cuda_resv = 0.0
    if torch.cuda.is_available():
        cuda_alloc = torch.cuda.max_memory_allocated() / 1024**2
        cuda_resv = torch.cuda.max_memory_reserved() / 1024**2

    fresh = os.path.join(args.out, f"emb_{args.target}_d128_L2_s0.npy")
    ref = os.path.join(BASE, args.reference, f"emb_{args.target}_d128_L2_s0.npy")
    identical = None
    if os.path.exists(ref):
        identical = bool(np.array_equal(np.load(fresh), np.load(ref)))

    rec = {
        "target": args.target, "cache": "gp_embedding", "n_arms": len(graphs),
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu",
        "domain_load_seconds": load_s,
        "build_seconds": secs, "build_minutes": secs / 60.0,
        "peak_rss_mb": rss1 / 1024.0, "rss_delta_mb": (rss1 - rss0) / 1024.0,
        "peak_cuda_alloc_mb": cuda_alloc, "peak_cuda_reserved_mb": cuda_resv,
        "shape": list(Z.shape), "dtype": str(Z.dtype),
        "identical_to_public_cache": identical,
        "fresh_path": fresh, "reference_path": ref,
    }
    out_json = os.path.join(BASE, "results", f".{args.target.lower()}_timed_logs",
                            "emb_cache_build.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(rec, open(out_json, "w"), indent=2)
    print(json.dumps(rec, indent=2), flush=True)
    print(f"\n[emb] -> {out_json}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
