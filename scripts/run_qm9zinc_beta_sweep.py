#!/usr/bin/env python
"""Exploration-coefficient (beta) sweep for GNN-PE + pivchol on QM9 and ZINC.

Sweeps --exploration_coef as multiples of the runner default
(0.001341321712103193), holding every other flag at the configuration used by the
reported results: pivchol scan, apply_to=both, neuron1024, revisit masking on, m=300.

The algorithm consumes the coefficient as sqrt(exploration_coef) (algorithms.py:502),
so a k-fold change in the coefficient is a sqrt(k)-fold change in the confidence width:

    coef x0.5 -> beta 0.0259    coef x1 -> beta 0.0366 (default)
    coef x2   -> beta 0.0518    coef x4 -> beta 0.0732

The x1 cell is NOT run here -- it already exists as
results/<ds>_phasedgp_T300_N1000_neuron1024_pivchol_both_masked (10 seeds, same 45
flags), and the summary script below picks it up as the anchor.

Folder naming follows the existing docking convention (`..._beta2x_...`), where the
multiplier refers to the COEFFICIENT, not to beta itself.

Results go to
  results/<ds>_phasedgp_T300_N1000_neuron1024_pivchol_both_masked_beta<K>x/finetune

    python scripts/run_qm9zinc_beta_sweep.py --dry_run
    python scripts/run_qm9zinc_beta_sweep.py --workers 8
"""

import argparse
import os
import queue
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(BASE_DIR, "results", ".qm9zinc_beta_state")
LOG_DIR = os.path.join(BASE_DIR, "results", ".qm9zinc_beta_logs")

# Prefer the project env; sys.executable is often base conda, which has no torch.
ENV_PY = "/home/chong/miniconda3/envs/gnnpe/bin/python"
DEFAULT_PY = ENV_PY if os.path.exists(ENV_PY) else sys.executable

DEFAULT_COEF = 0.001341321712103193
SEEDS = list(range(10))
MULTIPLIERS = [0.5, 2.0, 4.0]          # x1 already exists; not re-run
M_VALUE = 300                          # the reported compression setting
TAG = "T300_N1000_neuron1024_pivchol_both_masked"


def kstr(k):
    """Folder-safe multiplier label: 0.5 -> '0.5', 2.0 -> '2', 4.0 -> '4'."""
    return f"{k:g}"


def build_jobs(mults, datasets):
    jobs = []
    for ds in datasets:
        for k in mults:
            coef = DEFAULT_COEF * k
            folder = f"results/{ds}_phasedgp_{TAG}_beta{kstr(k)}x/finetune"
            for seed in SEEDS:
                jobs.append(dict(
                    name=f"{ds}_beta{kstr(k)}x_s{seed}",
                    args=["--dataset", ds, "--algo", "phasedgp",
                          "--train_modes", "finetune",
                          "--domain_scan_method", "pivchol",
                          "--domain_scan_apply_to", "both",
                          "--scan_pivchol_m", str(M_VALUE),
                          "--neuron_per_layer", "1024",
                          "--mask_revisit", "true",
                          "--exploration_coef", repr(coef),
                          "--seed", str(seed),
                          "--exp_result_folder", folder],
                ))
    return jobs


def run_job(job, gpu, python_exe):
    marker = os.path.join(STATE_DIR, job["name"] + ".done")
    if os.path.exists(marker):
        return "skip", job["name"], 0.0
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [python_exe, os.path.join(BASE_DIR, "main.py")] + job["args"]
    log_path = os.path.join(LOG_DIR, job["name"] + ".log")
    t0 = time.time()
    with open(log_path, "w") as log:
        log.write(" ".join(cmd) + f"\n[gpu {gpu}]\n\n")
        log.flush()
        rc = subprocess.call(cmd, cwd=BASE_DIR, env=env, stdout=log,
                             stderr=subprocess.STDOUT)
    dt = (time.time() - t0) / 60.0
    if rc == 0:
        with open(marker, "w") as f:
            f.write(f"{dt:.2f} min on gpu {gpu}\n")
        return "ok", job["name"], dt
    return "FAIL", job["name"], dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mults", type=str, default=",".join(kstr(k) for k in MULTIPLIERS),
                    help="Multipliers on the default exploration_coef.")
    ap.add_argument("--datasets", type=str, default="qm9,zinc")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--python", type=str, default=DEFAULT_PY)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    os.makedirs(STATE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    mults = [float(x) for x in args.mults.split(",") if x.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    jobs = build_jobs(mults, datasets)
    todo = [j for j in jobs if not os.path.exists(
        os.path.join(STATE_DIR, j["name"] + ".done"))]

    print(f"[beta] python={args.python}")
    for k in mults:
        print(f"[beta] x{kstr(k):<4} coef={DEFAULT_COEF*k:.12g}  beta={(DEFAULT_COEF*k)**0.5:.6f}")
    print(f"[beta] {len(jobs)} jobs, {len(jobs)-len(todo)} done, {len(todo)} to run")
    if args.dry_run:
        for j in todo[:4]:
            print("   ", j["name"], "|", " ".join(j["args"]))
        print(f"    ... ({len(todo)} total)")
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
            status, name, dt = run_job(job, gpu, args.python)
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

    print(f"\n[beta] done in {(time.time()-t_start)/60:.1f} min: "
          f"{res['ok']} ok, {res['FAIL']} failed")
    for f in failed:
        print("   FAILED:", f, f"-> results/.qm9zinc_beta_logs/{f}.log")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
