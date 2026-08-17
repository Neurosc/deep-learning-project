"""
03_run_alignment_all.py
=======================
Run the 03_run_alignment.py experiment for EVERY subject, spread across the
machine's GPUs so all ten finish in roughly the time one GPU needs for ~3
subjects (about a day) instead of ~4 days back-to-back on one GPU.

HOW IT WORKS (two roles, one file):
    * ORCHESTRATOR (default, when you just run the script): splits SUBJECTS
      across GPUS and launches one worker subprocess per GPU, each pinned to its
      GPU via CUDA_VISIBLE_DEVICES. It then waits for all of them.
    * WORKER (auto: when WORKER_SUBJECTS is set in the environment): imports the
      science from 03_run_alignment.py (model, training, loss -- unchanged) and
      runs the full alignment for each of its assigned subjects, writing the same
      five per-subject result CSVs 03 would, just named sub-XX instead of sub-01.

03_run_alignment.py itself is untouched and still runs a single subject (sub-01)
when launched on its own. This script only reuses its functions.

OUTPUT (per subject, in ~/things_eeg/results/):
    alignment_sub-XX_lr1e-4.csv          seed-averaged
    alignment_sub-XX_seeds_lr1e-4.csv    per-seed
    epoch_traces_sub-XX_lr1e-4.csv       per-epoch traces
    object_perception_sub-XX_lr1e-4.csv  deepest-layer accuracy
    sanity_check_sub-XX_lr1e-4.csv       full 1-second sanity check

Usage:
    # launch all subjects across all GPUs (run headless -- it takes ~a day):
    nohup python 03_run_alignment_all.py > align_all.log 2>&1 &
    # each GPU also writes its own detailed log: align_all_gpu<N>.log
"""

import os
import sys
import csv
import time
import subprocess

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUBJECTS = list(range(1, 11))        # subjects to run (1..10)
GPUS     = [0, 1, 2, 3]              # GPU ids to spread the subjects over
TAG      = "lr1e-4"                  # output filename suffix (matches 03_run_alignment.py)

# Path to the single-subject script whose functions we reuse.
HERE       = os.path.dirname(os.path.abspath(__file__))
ALIGN_FILE = os.path.join(HERE, "03_run_alignment.py")


def split_round_robin(subjects, n_groups):
    """Deal subjects out like cards so the groups differ in size by at most one
    (e.g. 10 subjects over 4 GPUs -> 3, 3, 2, 2)."""
    return [subjects[i::n_groups] for i in range(n_groups)]


# ===========================================================================
# WORKER: run the full experiment for one subject, reusing 03's functions
# ===========================================================================

def run_one_subject(A, sub):
    """Run 03's alignment for one subject.

    `A` is the imported 03_run_alignment module. This mirrors A.main() but is
    parametrised by `sub`: it loads sub-XX's EEG and writes sub-XX's CSVs. All
    the science (A.train_one, A.EEGProjectLayer, A.info_nce, A.sanity_check, the
    windows and seeds) comes straight from A, so the numbers match what 03 would
    produce for that subject.
    """
    import pandas as pd   # A already depends on pandas; imported here for the driver

    # Per-subject output paths (same names as 03, with the subject id swapped in).
    out_csv       = os.path.join(A.RES_DIR, f"alignment_sub-{sub:02d}_{TAG}.csv")
    out_csv_seeds = os.path.join(A.RES_DIR, f"alignment_sub-{sub:02d}_seeds_{TAG}.csv")
    epoch_csv     = os.path.join(A.RES_DIR, f"epoch_traces_sub-{sub:02d}_{TAG}.csv")
    obj_csv       = os.path.join(A.RES_DIR, f"object_perception_sub-{sub:02d}_{TAG}.csv")
    sanity_csv    = os.path.join(A.RES_DIR, f"sanity_check_sub-{sub:02d}_{TAG}.csv")

    print(f"\n########## SUB-{sub:02d} (GPU-visible: "
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}) ##########", flush=True)

    # ---- load this subject's prepared EEG ----
    train_eeg = np.load(os.path.join(A.EEG_DIR, f"sub-{sub:02d}_train_avg.npy"))
    test_eeg  = np.load(os.path.join(A.EEG_DIR, f"sub-{sub:02d}_test_avg.npy"))
    print("train EEG:", train_eeg.shape, "| test EEG:", test_eeg.shape, flush=True)

    assert train_eeg.shape == (16540, A.EXPECTED_NCHAN, A.EXPECTED_NTIMES)
    assert test_eeg.shape  == (200,   A.EXPECTED_NCHAN, A.EXPECTED_NTIMES)

    # ---- discover the feature sets (subject-independent, shared by all) ----
    targets = sorted(
        os.path.basename(f)[:-len("__train.npy")]
        for f in __import__("glob").glob(os.path.join(A.FEAT_DIR, "*__train.npy"))
    )
    print(f"{len(targets)} targets x {len(A.WINDOWS)} windows x {len(A.SEEDS)} seeds "
          f"= {len(targets) * len(A.WINDOWS) * len(A.SEEDS)} decoders", flush=True)

    raw = {}
    epoch_rows = []
    t0 = time.time()

    for target_name in targets:
        train_target = np.load(os.path.join(A.FEAT_DIR, f"{target_name}__train.npy"))
        test_target  = np.load(os.path.join(A.FEAT_DIR, f"{target_name}__test.npy"))

        assert train_target.shape[0] == train_eeg.shape[0]
        assert test_target.shape[0]  == test_eeg.shape[0]

        for window_name, window in A.WINDOWS.items():
            for seed in A.SEEDS:
                (val_loss, test_loss, best_epoch, top1, top5), trace = A.train_one(
                    train_eeg, train_target, test_eeg, test_target, window, seed)

                raw[(target_name, window_name, seed)] = (
                    best_epoch, val_loss, test_loss, top1, top5)

                for epoch_result in trace:                       # trace rows are dicts
                    epoch_rows.append(
                        {"target": target_name, "window": window_name,
                         "seed": seed, **epoch_result})

                print(f"sub{sub:02d} {target_name:28s} {window_name:9s} seed{seed} | "
                      f"ep{best_epoch:2d} val{val_loss:.3f} test{test_loss:.3f} "
                      f"top1{top1 * 100:4.1f}% top5{top5 * 100:4.1f}% "
                      f"[{time.time() - t0:.0f}s]", flush=True)

    # ---- per-epoch traces (pandas, same as 03) ----
    pd.DataFrame(epoch_rows).to_csv(epoch_csv, index=False)
    print("SAVED", epoch_csv, flush=True)

    # ---- per-seed raw results ----
    with open(out_csv_seeds, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "window", "seed", "best_epoch",
                    "val_loss", "test_loss", "top1", "top5"])
        for (tgt, wname, seed), (ep, vl, tl, t1, t5) in raw.items():
            w.writerow([tgt, wname, seed, ep,
                        round(vl, 4), round(tl, 4), round(t1, 4), round(t5, 4)])
    print("SAVED", out_csv_seeds, flush=True)

    # ---- averaged over seeds ----
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "window", "test_loss", "test_loss_std",
                    "top1", "top5", "n_seeds"])
        for target_name in targets:
            for window_name in A.WINDOWS:
                tls = [raw[(target_name, window_name, s)][2] for s in A.SEEDS]
                t1s = [raw[(target_name, window_name, s)][3] for s in A.SEEDS]
                t5s = [raw[(target_name, window_name, s)][4] for s in A.SEEDS]
                w.writerow([target_name, window_name,
                            round(float(np.mean(tls)), 4), round(float(np.std(tls)), 4),
                            round(float(np.mean(t1s)), 4), round(float(np.mean(t5s)), 4),
                            len(A.SEEDS)])
    print("SAVED", out_csv, flush=True)

    # ---- full 1-second sanity check (reuse A.sanity_check; point its global at
    #      this subject's file so it saves under sub-XX) ----
    if os.path.exists(os.path.join(A.FEAT_DIR, f"{A.DEEP_LAYER}__train.npy")):
        A.SANITY_CSV = sanity_csv        # override the module global for this subject
        print(f"\n=== Full 1-second sanity check (sub-{sub:02d}) ===", flush=True)
        A.sanity_check(train_eeg, test_eeg, A.DEEP_LAYER, seed=0)

    # ---- deepest-layer top-1/top-5 per window ----
    if any(k[0] == A.DEEP_LAYER for k in raw):
        with open(obj_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["target", "window", "top1", "top5"])
            best = (None, -1.0, -1.0)
            for window_name in A.WINDOWS:
                t1 = float(np.mean([raw[(A.DEEP_LAYER, window_name, s)][3] for s in A.SEEDS]))
                t5 = float(np.mean([raw[(A.DEEP_LAYER, window_name, s)][4] for s in A.SEEDS]))
                w.writerow([A.DEEP_LAYER, window_name, round(t1, 4), round(t5, 4)])
                if t1 > best[1]:
                    best = (window_name, t1, t5)
        print(f"=== Object-perception (sub-{sub:02d}) {A.DEEP_LAYER}: best window "
              f"{best[0]} top1={best[1] * 100:.2f}% top5={best[2] * 100:.2f}% ===", flush=True)
        print("SAVED", obj_csv, flush=True)


def worker():
    """Worker entry point: run every subject listed in WORKER_SUBJECTS.

    CUDA_VISIBLE_DEVICES is already set by the orchestrator BEFORE this process
    started, so importing 03_run_alignment.py (which imports torch) binds to the
    right GPU. Each subject is wrapped in try/except so one failure does not stop
    this GPU's remaining subjects.
    """
    subs = [int(x) for x in os.environ["WORKER_SUBJECTS"].split()]

    # Import 03's functions AFTER the GPU is pinned.
    import importlib.util
    spec = importlib.util.spec_from_file_location("align_single", ALIGN_FILE)
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)

    print(f"[worker] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"device={A.DEVICE} subjects={subs}", flush=True)

    for sub in subs:
        try:
            run_one_subject(A, sub)
        except Exception as exc:                 # keep going with the next subject
            print(f"[worker] sub-{sub:02d} FAILED: {type(exc).__name__}: {exc}", flush=True)

    print(f"[worker] done: {subs}", flush=True)


# ===========================================================================
# ORCHESTRATOR: launch one worker per GPU and wait
# ===========================================================================

def orchestrate():
    """Split SUBJECTS across GPUS and launch a worker subprocess per GPU."""
    groups = split_round_robin(SUBJECTS, len(GPUS))

    print(f"orchestrator: {len(SUBJECTS)} subjects over {len(GPUS)} GPUs", flush=True)
    procs = []
    for gpu, subs in zip(GPUS, groups):
        if not subs:                             # more GPUs than subjects -> skip idle GPU
            continue

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)   # pin this worker to one GPU
        env["WORKER_SUBJECTS"] = " ".join(str(s) for s in subs)   # tells the child it's a worker

        log_path = os.path.join(HERE, f"align_all_gpu{gpu}.log")
        log = open(log_path, "w")

        # Re-invoke THIS script; the WORKER_SUBJECTS env makes it take the worker path.
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__)],
                             env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((gpu, subs, p, log))
        print(f"  launched GPU{gpu} pid {p.pid} subjects {subs} -> {log_path}", flush=True)

    # Wait for every GPU's worker to finish.
    failed = False
    for gpu, subs, p, log in procs:
        rc = p.wait()
        log.close()
        status = "ok" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  GPU{gpu} subjects {subs}: {status}", flush=True)
        failed = failed or rc != 0

    print("\nALL WORKERS DONE" + (" (with failures -- check the gpu logs)" if failed else ""),
          flush=True)


if __name__ == "__main__":
    # WORKER_SUBJECTS present -> we are a worker; otherwise we are the orchestrator.
    if os.environ.get("WORKER_SUBJECTS"):
        worker()
    else:
        orchestrate()
