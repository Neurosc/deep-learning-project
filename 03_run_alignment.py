"""
03_run_alignment.py
===================
The main experiment. Train a small model to predict each image's network feature
from the brain signal, for every (feature set x time window) combination, and
record how well it works. Run AFTER 01_prepare_eeg.py and 02_extract_features.py.

IDEA: if the brain processes a glimpsed image in stages (simple features first,
meaning later), then early time windows of the EEG should best match the early
network layers, and late time windows should best match the deep layers.

MULTIPLE SEEDS: training involves randomness (the decoder's starting weights, the
validation split, the batch order). To check the early->late pattern is real and
not a fluke of one lucky random start, we run the WHOLE experiment several times
with different random seeds and average the results. SEEDS controls how many.

INPUT:
    ~/things_eeg/eeg_prepared/sub-01_{train,test}_avg.npy   (from step 01)
    ~/things_eeg/features/<network>__<layer>__{train,test}.npy   (from step 02)

OUTPUT (in ~/things_eeg/results/):
    alignment_sub-01_seeds.csv   one row per (target, window, seed) -> raw data
    alignment_sub-01.csv         averaged over seeds (mean + std test loss);
                                 this is the drop-in file step 04 reads
    Also prints a stability summary: for each layer, the best (lowest-loss)
    window per seed and how often they agree.

Usage:
    python 03_run_alignment.py
    # Runtime scales with the number of seeds: 18 targets x 9 windows x N seeds
    # decoders. Start with fewer seeds (e.g. SEEDS=[0]) for a quick first pass.
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS  = [0, 1, 2, 3, 4]    # the random seeds to run and average over
EPOCHS, BATCH = 50, 256

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


def to_batch(eeg, tgt, window):
    """Slice the time window and move the whole set to the GPU as two tensors."""
    s, e = window
    return (torch.from_numpy(eeg[:, :, s:e]).float().to(DEVICE),
            torch.from_numpy(tgt).float().to(DEVICE))


def train_one(train_eeg, train_tgt, test_eeg, test_tgt, window, seed):
    """Train one decoder for one (feature set x window x seed); return its best scores."""
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
    ve, vt = to_batch(train_eeg[vi], train_tgt[vi], window)   # validation set
    te, tt = to_batch(test_eeg, test_tgt, window)             # test set

    in_dim  = train_eeg.shape[1] * (window[1] - window[0])    # 63 * window length
    out_dim = train_tgt.shape[1]                              # feature length (e.g. 512)
    model = EEGDecoder(in_dim, out_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best = (float("inf"), None, 0, 0.0)   # (val_loss, test_loss, epoch, top1)
    for ep in range(1, EPOCHS + 1):
        # --- train for one epoch ---
        model.train()
        for eeg, tgt in dl:
            eeg, tgt = eeg.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()
            info_nce(model(eeg), tgt).backward()
            opt.step()
        sch.step()

        # --- check validation loss; if it improved, record test loss + top-1 ---
        model.eval()
        with torch.no_grad():
            vl = info_nce(model(ve), vt).item()
            if vl < best[0]:
                tl = info_nce(model(te), tt).item()
                emb = model(te)
                sims = F.normalize(emb, dim=-1) @ F.normalize(tt, dim=-1).t()
                top1 = (sims.argmax(1) == torch.arange(len(emb), device=DEVICE)).float().mean().item()
                best = (vl, tl, ep, top1)
    return best


def main():
    print("device:", DEVICE, "| seeds:", SEEDS, flush=True)

    # Load the prepared EEG once (shared across all targets/windows/seeds).
    train_eeg = np.load(os.path.join(EEG_DIR, "sub-01_train_avg.npy"))
    test_eeg  = np.load(os.path.join(EEG_DIR, "sub-01_test_avg.npy"))

    # Discover every feature set produced by step 02 (one name per network+layer).
    targets = sorted(os.path.basename(f)[:-len("__train.npy")]
                     for f in glob.glob(os.path.join(FEAT_DIR, "*__train.npy")))
    n_dec = len(targets) * len(WINDOWS) * len(SEEDS)
    print(f"{len(targets)} targets x {len(WINDOWS)} windows x {len(SEEDS)} seeds "
          f"= {n_dec} decoders", flush=True)

    # raw[(target, window, seed)] = (best_epoch, val_loss, test_loss, top1)
    raw, t0 = {}, time.time()
    for tgt_name in targets:
        train_tgt = np.load(os.path.join(FEAT_DIR, f"{tgt_name}__train.npy"))
        test_tgt  = np.load(os.path.join(FEAT_DIR, f"{tgt_name}__test.npy"))
        for wname, window in WINDOWS.items():
            for seed in SEEDS:
                vl, tl, ep, top1 = train_one(train_eeg, train_tgt, test_eeg, test_tgt,
                                             window, seed)
                raw[(tgt_name, wname, seed)] = (ep, vl, tl, top1)
                print(f"{tgt_name:24s} {wname:9s} seed{seed} | ep{ep:2d} "
                      f"val{vl:.3f} test{tl:.3f} top1{top1 * 100:4.1f}% "
                      f"[{time.time() - t0:.0f}s]", flush=True)

    # --- write the per-seed raw results ---
    with open(OUT_CSV_SEEDS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "window", "seed", "best_epoch", "val_loss", "test_loss", "top1"])
        for (tgt, wname, seed), (ep, vl, tl, top1) in raw.items():
            w.writerow([tgt, wname, seed, ep, round(vl, 4), round(tl, 4), round(top1, 4)])
    print("\nSAVED", OUT_CSV_SEEDS, flush=True)

    # --- average over seeds and write the aggregated results (step 04 reads this) ---
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "window", "test_loss", "test_loss_std", "top1", "n_seeds"])
        for tgt_name in targets:
            for wname in WINDOWS:
                tls  = [raw[(tgt_name, wname, s)][2] for s in SEEDS]
                tops = [raw[(tgt_name, wname, s)][3] for s in SEEDS]
                w.writerow([tgt_name, wname,
                            round(float(np.mean(tls)), 4),
                            round(float(np.std(tls)), 4),
                            round(float(np.mean(tops)), 4),
                            len(SEEDS)])
    print("SAVED", OUT_CSV, flush=True)

    # --- stability check: per layer, which window is best for each seed? ---
    # If the same window wins across all seeds, the early->late pattern is robust.
    post = [w for w in WINDOWS if w != "baseline"]   # ignore the pre-stimulus control
    print("\n=== Best-window stability across seeds ===", flush=True)
    print(f"{'target':24s} {'best window per seed':>32s}   agreement", flush=True)
    for tgt_name in targets:
        best_per_seed = []
        for s in SEEDS:
            tls = [raw[(tgt_name, w, s)][2] for w in post]
            best_per_seed.append(post[int(np.argmin(tls))])
        vals, counts = np.unique(best_per_seed, return_counts=True)
        mode = vals[int(np.argmax(counts))]
        agree = counts.max() / len(SEEDS)
        print(f"{tgt_name:24s} {str(best_per_seed):>32s}   {mode} ({agree*100:.0f}%)",
              flush=True)


if __name__ == "__main__":
    main()
