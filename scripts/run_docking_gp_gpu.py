#!/usr/bin/env python
"""Re-run the docking GP baselines ON GPU, 10 seeds, recording runtime + CUDA allocation.

The main timed sweep (scripts/run_docking_timed.py) forces these four baselines to CPU
via CUDA_VISIBLE_DEVICES="", so their recorded peak_cuda_alloc_mb is 0 by construction
and their runtimes are CPU runtimes. But the runner does have a GPU path:
IncrementalRBF_GP is constructed with torch_device=cuda when a GPU is visible, and
set_candidates() moves the whole (N x 128) float64 candidate matrix onto the device
(~1 GB at N=1e6); _predict_gpu() then forms three (n, N) float64 arrays per round
(d2, K_star, v), so peak demand grows with the observation count. This script exposes a
GPU so those numbers become real measurements -- the source of the GP memory column in
Table 4.

Results go to results/docking_<TARGET>_dock_norm_timed_gpu/<baseline>/, leaving the
original CPU results untouched for side-by-side comparison.

    python scripts/run_docking_gp_gpu.py --target 6T2W --dry_run
    python scripts/run_docking_gp_gpu.py --target 6T2W --workers 8
"""
import argparse, os, queue, subprocess, sys, threading, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/chong/miniconda3/envs/gnnpe/bin/python"
SEEDS = list(range(10))
BASELINES = ["random", "gpucb", "gpei", "gpts"]

# set from --target in main()
TARGET = "6T2W"
OUT = STATE_DIR = LOG_DIR = None


def set_target(target, python_exe=None):
    global TARGET, PY, OUT, STATE_DIR, LOG_DIR
    TARGET = target
    if python_exe:
        PY = python_exe
    OUT = f"results/docking_{TARGET}_dock_norm_timed_gpu"
    STATE_DIR = os.path.join(BASE, "results", f".{TARGET.lower()}_gp_gpu_state")
    LOG_DIR = os.path.join(BASE, "results", f".{TARGET.lower()}_gp_gpu_logs")


def cmd_for(baseline, seed):
    return [PY, "experiments/run_docking_gp_baselines_timed.py",
            "--graph_dir", f"data/graphs/{TARGET}", "--target", TARGET,
            "--objective", "dock_norm", "--num_actions", "-1", "--T", "300",
            "--baseline", baseline,
            "--embed_dim", "128", "--embed_depth", "2", "--embedder_seed", "0",
            "--emb_cache_dir", "results/emb_cache_public",
            "--ucb_beta", "2.0", "--ei_xi", "0.0", "--rbf_length_scale", "1.0",
            "--init_random_steps", "5",
            "--seed", str(seed),
            "--exp_result_folder", f"{OUT}/{baseline}"]


def run_job(job, gpu):
    marker = os.path.join(STATE_DIR, job["name"] + ".done")
    if os.path.exists(marker):
        return "skip", job["name"], 0.0
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)     # the point of this script
    env["OMP_NUM_THREADS"] = "8"
    log = os.path.join(LOG_DIR, job["name"] + ".log")
    t0 = time.time()
    with open(log, "w") as lf:
        lf.write(" ".join(job["cmd"]) + f"\n[gpu {gpu}]\n\n")
        lf.flush()
        rc = subprocess.call(job["cmd"], cwd=BASE, env=env, stdout=lf,
                             stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60.0
    if rc == 0:
        with open(marker, "w") as f:
            f.write(f"{dt:.2f} min on gpu {gpu}\n")
        return "ok", job["name"], dt
    return "FAIL", job["name"], dt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=str, default="6T2W",
                    help="docking target: 3CLPro, rtcb, or 6T2W")
    ap.add_argument("--python", type=str, default=PY,
                    help="interpreter used for the child runs")
    ap.add_argument("--baselines", type=str, default=",".join(BASELINES))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    set_target(args.target, args.python)

    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    bl = [b.strip() for b in args.baselines.split(",") if b.strip()]
    jobs = [dict(name=f"{b}_s{s}", cmd=cmd_for(b, s)) for b in bl for s in SEEDS]
    todo = [j for j in jobs if not os.path.exists(
        os.path.join(STATE_DIR, j["name"] + ".done"))]

    print(f"[gp-gpu] {len(jobs)} jobs, {len(jobs)-len(todo)} done, {len(todo)} to run")
    if args.dry_run:
        print("   ", " ".join(todo[0]["cmd"]))
        return 0

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    q = queue.Queue()
    for j in todo:
        q.put(j)
    res = {"ok": 0, "FAIL": 0, "skip": 0}
    failed = []
    lock = threading.Lock()
    t_start = time.time()

    def worker(wid):
        gpu = gpus[wid % len(gpus)]
        while True:
            try:
                job = q.get_nowait()
            except queue.Empty:
                return
            status, name, dt = run_job(job, gpu)
            with lock:
                res[status] += 1
                if status == "FAIL":
                    failed.append(name)
                done = res["ok"] + res["FAIL"]
                el = (time.time() - t_start) / 60.0
                eta = (el / max(done, 1)) * (len(todo) - done)
                print(f"[{done}/{len(todo)}] {status:4s} {name} "
                      f"({dt:.1f}m gpu {gpu}) | elapsed {el:.0f}m eta {eta:.0f}m", flush=True)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(args.workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    print(f"\n[gp-gpu] done in {(time.time()-t_start)/60:.1f} min: "
          f"{res['ok']} ok, {res['FAIL']} failed")
    for f in failed:
        print("   FAILED:", f, f"-> {os.path.relpath(LOG_DIR, BASE)}/{f}.log")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
