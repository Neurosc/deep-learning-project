"""
03_run_alignment.py
===================
The main experiment. Train a small model to predict each image's network feature
from the brain signal, for every (feature set x time window) combination, and
record how well it works. Run AFTER 01_prepare_eeg.py and 02_extract_features.py.

IDEA: if the brain processes a glimpsed image in stages (simple features first,
meaning later), then early time windows of the EEG should best match the early
network layers, and late time windows should best match the deep layers.

MULTIPLE SEEDS: training involves randomness (weights, validation split, batch
order). We run the whole experiment over several seeds and average, to check the
early->late pattern is real and not a fluke of one lucky random start.

This file also covers:
  #2  asserts the decoder input for a 100 ms window is 63 ch x 10 samples = 630.
  #3  records train AND validation loss at every epoch (overfitting check).
  #4  a single-condition sanity routine reporting val/test loss + top-1/top-5.
  #5  saves top-1/top-5 for the deepest, most semantic layer (literature number).
  It auto-discovers every feature set in features/, so the sharp and foveated
  variants from 02 are both trained and end up in the same results file (#6).

OUTPUT (in ~/things_eeg/results/):
    alignment_sub-01_seeds.csv   one row per (target, window, seed)
    alignment_sub-01.csv         averaged over seeds (step 04 reads this)
    epoch_traces_sub-01.csv      train & val loss per epoch (for the overfitting plot)
    object_perception_sub-01.csv top-1/top-5 for the deepest layer, per window

Usage:
    python 03_run_alignment.py
    # Runtime scales with targets x windows x seeds; launch with nohup overnight.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")   # choose GPU before importing torch
import glob, csv, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT     = os.path.expanduser("~/things_eeg")
EEG_DIR  = os.path.join(ROOT, "eeg_prepared")
FEAT_DIR = os.path.join(ROOT, "features")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
OUT_CSV       = os.path.join(RES_DIR, "alignment_sub-01.csv")          # averaged
OUT_CSV_SEEDS = os.path.join(RES_DIR, "alignment_sub-01_seeds.csv")    # per-seed
EPOCH_CSV     = os.path.join(RES_DIR, "epoch_traces_sub-01.csv")       # per-epoch traces (#3)
OBJ_CSV       = os.path.join(RES_DIR, "object_perception_sub-01.csv")  # deepest-layer acc (#5)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS  = [0, 1, 2, 3, 4]    # the random seeds to run and average over
EPOCHS, BATCH = 50, 256

EXPECTED_NCHAN  = 63     # EEG channels after dropping 'stim' (verified in step 01)
WINDOW_SAMPLES  = 10     # samples per 100 ms window (100 Hz -> 10 ms per sample)
DEEP_LAYER = "ViT-L-14__block24"   # deepest / most semantic layer (for points #4, #5)

# Each window is a (start, end) range over the 100 EEG time points (10 ms each).
# Sample 20 is stimulus onset (time 0), so e.g. (30,40) = 100-200 ms after onset.
WINDOWS = {
    "baseline": (10, 20),   # -100..0 ms, before the image -> control, should be at chance
    "0_100":    (20, 30),
    "100_200":  (30, 40),
    "200_300":  (40, 50),
    "300_400":  (50, 60),
    "400_500":  (60, 70),
    "500_600":  (70, 80),
    "600_700":  (80, 90),
    "700_800":  (90, 100),
}


class EEGDataset(Dataset):
    """Pair each image's EEG window with that image's target feature vector."""
    def __init__(self, eeg, targets, window):
        s, e = window
        self.eeg     = torch.from_numpy(eeg[:, :, s:e]).float()   # [N, 63, window_len]
        self.targets = torch.from_numpy(targets).float()          # [N, feature_dim]
    def __len__(self):
        return len(self.eeg)
    def __getitem__(self, i):
        return self.eeg[i], self.targets[i]


class EEGDecoder(nn.Module):
    """A 2-layer MLP with a residual (skip) connection.

    Flattens the EEG window to one long vector, passes it through two linear
    layers (with GELU activation and dropout), and adds a direct linear shortcut.
    Output length = the feature vector's length.
    """
    def __init__(self, input_size, output_size, hidden_size=1024):
        super().__init__()
        self.layer1   = nn.Linear(input_size, hidden_size)
        self.layer2   = nn.Linear(hidden_size, output_size)
        self.gelu     = nn.GELU()
        self.dropout  = nn.Dropout(0.1)
        self.residual = nn.Linear(input_size, output_size)
    def forward(self, x):
        x   = x.view(x.shape[0], -1)                      # flatten [N, 63, win] -> [N, 63*win]
        out = self.dropout(self.gelu(self.layer1(x)))
        out = self.layer2(out)
        return out + self.residual(x)


def info_nce(a, b, t=0.07):
    """Symmetric InfoNCE contrastive loss.

    Normalises both sets of vectors, builds a similarity matrix, and rewards each
    EEG (row) for being most similar to its own image's feature (the diagonal),
    in both directions (EEG->feature and feature->EEG). t is the temperature.
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / t
    lab = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))


def retrieval_accuracy(pred, target, ks=(1, 5)):
    """Point #4: shared retrieval metric, reused everywhere accuracy is needed.

    For each predicted EEG embedding, rank all target features by cosine
    similarity and check whether the correct one is within the top k. Returns a
    dict {1: top1, 5: top5}. (Chance for 200 test images is 1/200 and 5/200.)
    """
    p = F.normalize(pred, dim=-1)
    t = F.normalize(target, dim=-1)
    sims = p @ t.t()
    n = sims.shape[0]
    idx = torch.arange(n, device=sims.device)
    out = {}
    for k in ks:
        kk = min(k, n)
        hit = (sims.topk(kk, dim=1).indices == idx[:, None]).any(1).float().mean().item()
        out[k] = hit
    return out


def to_batch(eeg, tgt, window):
    """Slice the time window and move the whole set to the GPU as two tensors."""
    s, e = window
    return (torch.from_numpy(eeg[:, :, s:e]).float().to(DEVICE),
            torch.from_numpy(tgt).float().to(DEVICE))


def train_one(train_eeg, train_tgt, test_eeg, test_tgt, window, seed):
    """Train one decoder for one (feature set x window x seed).

    Returns:
        best  : (val_loss, test_loss, best_epoch, top1, top5) at the epoch with
                lowest validation loss.
        trace : list of (epoch, train_loss, val_loss) for the overfitting plot (#3).
    """
    # The seed fixes ALL randomness for this run: weight init, validation split,
    # and batch order. Different seeds = different random starts.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Carve a 10% validation set out of the training images (for epoch selection).
    n = len(train_eeg)
    perm = np.random.permutation(n)
    nv = n // 10
    vi, ti = perm[:nv], perm[nv:]

    dl = DataLoader(EEGDataset(train_eeg[ti], train_tgt[ti], window),
                    batch_size=BATCH, shuffle=True, drop_last=True)
    ve, vt = to_batch(train_eeg[vi], train_tgt[vi], window)        # validation set
    te, tt = to_batch(test_eeg, test_tgt, window)                  # test set
    # A fixed training probe (same size as the validation set) so the per-epoch
    # train and val losses are on the same scale and directly comparable (#3).
    tpe, tpt = to_batch(train_eeg[ti[:nv]], train_tgt[ti[:nv]], window)

    in_dim  = train_eeg.shape[1] * (window[1] - window[0])    # 63 * window length
    out_dim = train_tgt.shape[1]                              # feature length (e.g. 512)
    model = EEGDecoder(in_dim, out_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best  = (float("inf"), None, 0, 0.0, 0.0)   # (val, test, epoch, top1, top5)
    trace = []
    for ep in range(1, EPOCHS + 1):
        # --- train for one epoch ---
        model.train()
        for eeg, tgt in dl:
            eeg, tgt = eeg.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()
            info_nce(model(eeg), tgt).backward()
            opt.step()
        sch.step()

        # --- record train + val loss every epoch (overfitting check, #3) ---
        model.eval()
        with torch.no_grad():
            tr_loss = info_nce(model(tpe), tpt).item()
            vl      = info_nce(model(ve),  vt).item()
        trace.append((ep, tr_loss, vl))

        # If validation improved, snapshot the test loss + retrieval accuracy.
        if vl < best[0]:
            with torch.no_grad():
                tl  = info_nce(model(te), tt).item()
                acc = retrieval_accuracy(model(te), tt)
            best = (vl, tl, ep, acc[1], acc[5])
    return best, trace


def sanity_check(train_eeg, test_eeg, target, window_name, seed=0):
    """Point #4: train ONE decoder for a chosen (network/layer, window) and report
    validation loss, test loss, and top-1/top-5 together, so we can confirm that
    loss and retrieval accuracy move together."""
    train_tgt = np.load(os.path.join(FEAT_DIR, f"{target}__train.npy"))
    test_tgt  = np.load(os.path.join(FEAT_DIR, f"{target}__test.npy"))
    (vl, tl, ep, top1, top5), _ = train_one(
        train_eeg, train_tgt, test_eeg, test_tgt, WINDOWS[window_name], seed)
    print(f"[sanity] {target} @ {window_name} (seed {seed}): "
          f"val_loss={vl:.4f}  test_loss={tl:.4f}  "
          f"top1={top1*100:.2f}%  top5={top5*100:.2f}%", flush=True)
    return {"target": target, "window": window_name,
            "val_loss": vl, "test_loss": tl, "top1": top1, "top5": top5}


def main():
    print("device:", DEVICE, "| seeds:", SEEDS, flush=True)

    # Load the prepared EEG once (shared across all targets/windows/seeds).
    train_eeg = np.load(os.path.join(EEG_DIR, "sub-01_train_avg.npy"))
    test_eeg  = np.load(os.path.join(EEG_DIR, "sub-01_test_avg.npy"))

    # ---- Point #2: verify the decoder input shape for a 100 ms window ----
    n_ch = train_eeg.shape[1]
    s, e = WINDOWS["100_200"]
    realised = train_eeg[:1, :, s:e].shape          # (1, 63, 10)
    in_dim_100ms = n_ch * (e - s)
    print(f"decoder input per 100 ms window: {n_ch} ch x {e - s} samples = "
          f"{in_dim_100ms}  (realised slice {realised})", flush=True)
    assert n_ch == EXPECTED_NCHAN, f"expected {EXPECTED_NCHAN} channels, got {n_ch}"
    assert (e - s) == WINDOW_SAMPLES, f"expected {WINDOW_SAMPLES} samples per window"
    assert in_dim_100ms == EXPECTED_NCHAN * WINDOW_SAMPLES == 630, "decoder input is not 630"

    # Discover every feature set produced by step 02 (sharp AND foveated, #6).
    targets = sorted(os.path.basename(f)[:-len("__train.npy")]
                     for f in glob.glob(os.path.join(FEAT_DIR, "*__train.npy")))
    n_dec = len(targets) * len(WINDOWS) * len(SEEDS)
    print(f"{len(targets)} targets x {len(WINDOWS)} windows x {len(SEEDS)} seeds "
          f"= {n_dec} decoders", flush=True)

    # raw[(target, window, seed)] = (best_epoch, val_loss, test_loss, top1, top5)
    raw, t0 = {}, time.time()
    efile = open(EPOCH_CSV, "w", newline="")           # stream per-epoch traces (#3)
    ew = csv.writer(efile)
    ew.writerow(["target", "window", "seed", "epoch", "train_loss", "val_loss"])

    for tgt_name in targets:
        train_tgt = np.load(os.path.join(FEAT_DIR, f"{tgt_name}__train.npy"))
        test_tgt  = np.load(os.path.join(FEAT_DIR, f"{tgt_name}__test.npy"))
        for wname, window in WINDOWS.items():
            for seed in SEEDS:
                (vl, tl, ep, top1, top5), trace = train_one(
                    train_eeg, train_tgt, test_eeg, test_tgt, window, seed)
                raw[(tgt_name, wname, seed)] = (ep, vl, tl, top1, top5)
                for (e_, trl, vl_) in trace:
                    ew.writerow([tgt_name, wname, seed, e_, round(trl, 4), round(vl_, 4)])
                print(f"{tgt_name:28s} {wname:9s} seed{seed} | ep{ep:2d} "
                      f"val{vl:.3f} test{tl:.3f} top1{top1*100:4.1f}% top5{top5*100:4.1f}% "
                      f"[{time.time()-t0:.0f}s]", flush=True)
    efile.close()
    print("\nSAVED", EPOCH_CSV, flush=True)

    # ---- per-seed raw results ----
    with open(OUT_CSV_SEEDS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "window", "seed", "best_epoch", "val_loss", "test_loss", "top1", "top5"])
        for (tgt, wname, seed), (ep, vl, tl, top1, top5) in raw.items():
            w.writerow([tgt, wname, seed, ep, round(vl, 4), round(tl, 4),
                        round(top1, 4), round(top5, 4)])
    print("SAVED", OUT_CSV_SEEDS, flush=True)

    # ---- averaged over seeds (step 04 reads this) ----
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "window", "test_loss", "test_loss_std", "top1", "top5", "n_seeds"])
        for tgt_name in targets:
            for wname in WINDOWS:
                tls  = [raw[(tgt_name, wname, s)][2] for s in SEEDS]
                t1s  = [raw[(tgt_name, wname, s)][3] for s in SEEDS]
                t5s  = [raw[(tgt_name, wname, s)][4] for s in SEEDS]
                w.writerow([tgt_name, wname,
                            round(float(np.mean(tls)), 4), round(float(np.std(tls)), 4),
                            round(float(np.mean(t1s)), 4), round(float(np.mean(t5s)), 4),
                            len(SEEDS)])
    print("SAVED", OUT_CSV, flush=True)

    # ---- best-window stability across seeds ----
    post = [w for w in WINDOWS if w != "baseline"]
    print("\n=== Best-window stability across seeds ===", flush=True)
    for tgt_name in targets:
        bps = [post[int(np.argmin([raw[(tgt_name, w, s)][2] for w in post]))] for s in SEEDS]
        vals, counts = np.unique(bps, return_counts=True)
        mode = vals[int(np.argmax(counts))]; agree = counts.max() / len(SEEDS)
        print(f"{tgt_name:28s} {str(bps):45s} {mode} ({agree*100:.0f}%)", flush=True)

    # ---- Point #4: single-condition sanity check (loss vs accuracy move together) ----
    if os.path.exists(os.path.join(FEAT_DIR, f"{DEEP_LAYER}__train.npy")):
        print("\n=== Sanity check (#4) ===", flush=True)
        sanity_check(train_eeg, test_eeg, DEEP_LAYER, "200_300", seed=0)

    # ---- Point #5: top-1/top-5 of the deepest, most semantic layer, per window ----
    if any(k[0] == DEEP_LAYER for k in raw):
        with open(OBJ_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["target", "window", "top1", "top5"])
            best = (None, -1.0, -1.0)
            for wname in WINDOWS:
                t1 = float(np.mean([raw[(DEEP_LAYER, wname, s)][3] for s in SEEDS]))
                t5 = float(np.mean([raw[(DEEP_LAYER, wname, s)][4] for s in SEEDS]))
                w.writerow([DEEP_LAYER, wname, round(t1, 4), round(t5, 4)])
                if t1 > best[1]:
                    best = (wname, t1, t5)
        print(f"\n=== Object-perception accuracy (#5): {DEEP_LAYER} ===", flush=True)
        print(f"best window {best[0]}: top1={best[1]*100:.2f}%  top5={best[2]*100:.2f}%", flush=True)
        print("SAVED", OBJ_CSV, flush=True)


if __name__ == "__main__":
    main()
