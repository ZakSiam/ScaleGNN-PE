#!/usr/bin/env python
"""Run the full *timed* docking sweep for one target (3CLPro, rtcb, or 6T2W).

Same 8 method groups, same config, same 10 seeds (0-9) that Figure 1 and Table 4 use:
  GNN (experiments/run_docking_peft_timed.py, pivchol both m=1000, neuron 1024):
     fullntk (peft none), peft_lastlayer, peft_lora
  GP baselines (experiments/run_docking_gp_baselines_timed.py, embed_dim 128): random, gpucb, gpei, gpts
  GNN-SS (experiments/run_docking_gnnss.py, uncertainty=fjl): gnnss_fjl_B

Outputs land in results/docking_<TARGET>_dock_norm_timed/<group>/ (+ _gnnss_fjl_B), the
layout scripts/plot_hits_vs_time_thresholds.py and scripts/plot_docking_table.py expect.

Resumable: a (group, seed) whose result json already exists is skipped.
GNN jobs are pinned one-per-GPU (8x A100); GP baselines run CPU-only in parallel, so their
recorded peak_cuda_alloc_mb is 0 by construction -- scripts/run_docking_gp_gpu.py re-runs
those four on a visible GPU to get the real CUDA numbers for Table 4.
The pivchol scan cache is built ONCE up front (blocking) to avoid a concurrent-build race
across the parallel GNN jobs.

    python scripts/run_docking_timed.py --target 3CLPro
"""
import argparse, os, sys, glob, json, subprocess, threading, queue, time, resource

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)   # so the driver can import repo-root modules (docking_bandit_env, ...)

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--target", type=str, default="6T2W",
                 help="docking target: 3CLPro, rtcb, or 6T2W")
_ap.add_argument("--python", type=str,
                 default="/home/chong/miniconda3/envs/gnnpe/bin/python",
                 help="interpreter used for the child runs")
_ap.add_argument("--seeds", type=str, default="0-9",
                 help="comma list and/or a-b ranges, e.g. '0-9' or '0,1,2'")
_args, _ = _ap.parse_known_args()

def _parse_seeds(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out

PY = _args.python
TARGET = _args.target
GRAPH_DIR = f"data/graphs/{TARGET}"
SEEDS = _parse_seeds(_args.seeds)
LOG_DIR = os.path.join(BASE, "results", f".{TARGET.lower()}_timed_logs")
os.makedirs(LOG_DIR, exist_ok=True)
TIMED = f"results/docking_{TARGET}_dock_norm_timed"

COMMON = ["--graph_dir", GRAPH_DIR, "--target", TARGET, "--objective", "dock_norm",
          "--num_actions", "-1", "--T", "300"]

def gnn_cmd(sub, extra):
    return [PY, "experiments/run_docking_peft_timed.py", *COMMON,
            "--neuron_per_layer", "1024", "--num_mlp_layers_alg", "2",
            "--train_modes", "finetune",
            "--domain_scan_method", "pivchol", "--domain_scan_apply_to", "both",
            "--scan_pivchol_m", "1000", "--scan_matfree", "true",
            "--scan_cache_dir", "results/scan_cache_public",
            "--exp_result_folder", f"{TIMED}/{sub}", *extra]

def gp_cmd(baseline):
    return [PY, "experiments/run_docking_gp_baselines_timed.py", *COMMON,
            "--baseline", baseline,
            "--embed_dim", "128", "--embed_depth", "2", "--embedder_seed", "0",
            "--emb_cache_dir", "results/emb_cache_public",
            "--ucb_beta", "2.0", "--ei_xi", "0.0", "--rbf_length_scale", "1.0",
            "--init_random_steps", "5",
            "--exp_result_folder", f"{TIMED}/{baseline}"]

def gnnss_cmd():
    return [PY, "experiments/run_docking_gnnss.py", *COMMON,
            "--net", "GNN", "--neuron_per_layer", "1024", "--num_mlp_layers_alg", "2",
            "--uncertainty", "fjl", "--proj_dim", "2048",
            "--subsample_K", "1000", "--ss_batch", "1", "--explore_rounds", "40",
            "--ss_train_batch", "32", "--acquisition", "ucb", "--cov_scaling", "alg1",
            "--lap_pe_k", "0", "--paper_init", "false", "--train_modes", "finetune",
            "--domain_scan_method", "full",
            "--scan_cache_dir", "results/scan_cache_public",
            "--exp_result_folder", f"results/docking_{TARGET}_dock_norm_gnnss_fjl_B"]

# (group_name, result_dir, kind, base_cmd_fn)   kind in {"gpu","cpu"}
GROUPS = [
    ("fullntk",        f"{TIMED}/fullntk",        "gpu", lambda: gnn_cmd("fullntk", [])),
    ("peft_lastlayer", f"{TIMED}/peft_lastlayer", "gpu", lambda: gnn_cmd("peft_lastlayer", ["--peft", "last_layer"])),
    ("peft_lora",      f"{TIMED}/peft_lora",      "gpu", lambda: gnn_cmd("peft_lora", ["--peft", "lora", "--lora_rank", "8", "--lora_alpha", "16"])),
    ("gnnss_fjl_B",    f"results/docking_{TARGET}_dock_norm_gnnss_fjl_B", "gpu", gnnss_cmd),
    ("random",         f"{TIMED}/random",         "cpu", lambda: gp_cmd("random")),
    ("gpucb",          f"{TIMED}/gpucb",          "cpu", lambda: gp_cmd("gpucb")),
    ("gpei",           f"{TIMED}/gpei",           "cpu", lambda: gp_cmd("gpei")),
    ("gpts",           f"{TIMED}/gpts",           "cpu", lambda: gp_cmd("gpts")),
]

SCAN_CACHE = os.path.join(BASE, "results/scan_cache_public",
                          f"scan_{TARGET}_pivchol_wl2_m1000_fft300_km300.npz")
CACHE_TIMING = os.path.join(LOG_DIR, "scan_cache_build.json")

def build_and_time_scan_cache():
    """Build the target's pivchol scan cache in isolation, recording wall-clock + peak RSS.

    Mirrors the runner's build_scan_indexer call exactly (same params/cache path) so the
    resulting .npz is reused verbatim by every GNN job. Written separately because the
    runner builds the cache before its own timing clock starts, so per-run JSONs never
    capture this cost. Result saved to <LOG_DIR>/scan_cache_build.json.
    """
    if os.path.exists(SCAN_CACHE):
        print(f"Scan cache already present: {SCAN_CACHE}", flush=True)
        return
    from docking_bandit_env import load_docking_domain
    from graph_scan_accel import build_scan_indexer
    import numpy as np
    print(f"Loading {TARGET} domain for isolated scan-cache build...", flush=True)
    graph_data, _, _ = load_docking_domain(graph_dir=GRAPH_DIR, target=TARGET,
                                           objective="dock_norm", num_actions=None,
                                           max_shards=None, seed=0, normalize_rewards=True)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.time()
    build_scan_indexer(graphs=graph_data, method="pivchol", wl_h=2, fft_m=300,
                       kmeans_k=300, kmeans_iter=8, pivchol_m=1000, matfree=True,
                       random_state=np.random.RandomState(0), cache_path=SCAN_CACHE)
    secs = time.time() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rec = {"target": TARGET, "cache": "pivchol_scan", "n_arms": len(graph_data),
           "build_seconds": secs, "build_minutes": secs / 60.0,
           "peak_rss_mb": rss1 / 1024.0, "rss_delta_mb": (rss1 - rss0) / 1024.0,
           "path": SCAN_CACHE}
    json.dump(rec, open(CACHE_TIMING, "w"), indent=2)
    print(f"Scan cache built in {secs/60:.2f} min, peak RSS {rss1/1024.0:.0f} MB "
          f"-> {CACHE_TIMING}", flush=True)

def already_done(result_dir, seed):
    for f in glob.glob(os.path.join(BASE, result_dir, "**", "*.json"), recursive=True):
        try:
            if json.load(open(f)).get("params", {}).get("seed") == seed:
                return True
        except Exception:
            pass
    return False

def run_job(name, cmd, kind, gpu=None):
    env = dict(os.environ)
    if kind == "gpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["OMP_NUM_THREADS"] = "4"
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""      # force CPU for GP baselines
        env["OMP_NUM_THREADS"] = "16"         # 8 GP jobs x 16 threads = 128 < 255 cores
    log = os.path.join(LOG_DIR, f"{name}.log")
    t0 = time.time()
    with open(log, "w") as lf:
        r = subprocess.run(cmd, cwd=BASE, env=env, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    tag = "OK" if r.returncode == 0 else f"FAIL({r.returncode})"
    print(f"[{tag}] {name}  {dt/60:.1f} min  (log: {log})", flush=True)
    return r.returncode

def main():
    # Build full job list, skipping completed (group, seed).
    jobs = []  # (name, cmd, kind, result_dir, seed)
    for gname, rdir, kind, fn in GROUPS:
        for s in SEEDS:
            if already_done(rdir, s):
                continue
            jobs.append((f"{gname}_s{s}", fn() + ["--seed", str(s)], kind, rdir, s))
    print(f"Planned {len(jobs)} jobs "
          f"(gpu={sum(1 for j in jobs if j[2]=='gpu')}, cpu={sum(1 for j in jobs if j[2]=='cpu')})", flush=True)
    if not jobs:
        print("Nothing to do."); return

    # --- Prelude: build + time the pivchol scan cache once, in isolation, so the
    # build cost is recorded (the runner builds it before its own clock and never logs it).
    # Doing it here first also avoids a concurrent-build race across the parallel GNN jobs. ---
    gpu_jobs = [j for j in jobs if j[2] == "gpu"]
    cpu_jobs = [j for j in jobs if j[2] == "cpu"]
    if any(j[0].split("_s")[0] in ("fullntk", "peft_lastlayer", "peft_lora") for j in gpu_jobs):
        build_and_time_scan_cache()

    # --- GPU lane: 8 workers, one GPU each. CPU lane: 8 concurrent GP jobs (bounded threads),
    # kept low so wall-clock/memory stay faithful for the hits-vs-time figure. ---
    gq, cq = queue.Queue(), queue.Queue()
    for j in gpu_jobs: gq.put(j)
    for j in cpu_jobs: cq.put(j)

    def gpu_worker(gpu):
        while True:
            try: j = gq.get_nowait()
            except queue.Empty: return
            run_job(j[0], j[1], "gpu", gpu=gpu)
            gq.task_done()

    def cpu_worker():
        while True:
            try: j = cq.get_nowait()
            except queue.Empty: return
            run_job(j[0], j[1], "cpu")
            cq.task_done()

    threads = [threading.Thread(target=gpu_worker, args=(g,)) for g in range(8)]
    threads += [threading.Thread(target=cpu_worker) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    print("ALL DONE.", flush=True)

if __name__ == "__main__":
    main()
