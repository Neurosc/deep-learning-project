
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")  # set before importing torch
import glob, csv, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT     = os.path.expanduser("~/things_eeg")
EEG_DIR  = os.path.join(ROOT, "eeg_prepared")
FEAT_DIR = os.path.join(ROOT, "features")
RES_DIR  = os.path.join(ROOT, "results"); os.makedirs(RES_DIR, exist_ok=True)
OUT_CSV  = os.path.join(RES_DIR, "alignment_sub-01.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED, EPOCHS, BATCH = 0, 50, 256

WINDOWS = {
    "baseline": (10,20), "0_100": (20,30), "100_200": (30,40),
    "200_300": (40,50), "300_400": (50,60), "400_500": (60,70),
    "500_600": (70,80), "600_700": (80,90), "700_800": (90,100),
}

class EEGDataset(Dataset):
    def __init__(self, eeg, targets, window):
        s, e = window
        self.eeg     = torch.from_numpy(eeg[:, :, s:e]).float()
        self.targets = torch.from_numpy(targets).float()
    def __len__(self): return len(self.eeg)
    def __getitem__(self, i): return self.eeg[i], self.targets[i]

class EEGDecoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024):
        super().__init__()
        self.layer1   = nn.Linear(input_size, hidden_size)
        self.layer2   = nn.Linear(hidden_size, output_size)
        self.gelu     = nn.GELU(); self.dropout = nn.Dropout(0.1)
        self.residual = nn.Linear(input_size, output_size)
    def forward(self, x):
        x   = x.view(x.shape[0], -1)
        out = self.dropout(self.gelu(self.layer1(x)))
        out = self.layer2(out)
        return out + self.residual(x)

def info_nce(a, b, t=0.07):
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    logits = a @ b.t() / t
    lab = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))

def to_batch(eeg, tgt, window):
    s, e = window
    return (torch.from_numpy(eeg[:, :, s:e]).float().to(DEVICE),
            torch.from_numpy(tgt).float().to(DEVICE))

def train_one(train_eeg, train_tgt, test_eeg, test_tgt, window):
    # same seed everywhere -> identical train/val split and init across all cells (fair comparison)
    torch.manual_seed(SEED); np.random.seed(SEED)
    n = len(train_eeg); perm = np.random.permutation(n); nv = n // 10
    vi, ti = perm[:nv], perm[nv:]
    dl = DataLoader(EEGDataset(train_eeg[ti], train_tgt[ti], window),
                    batch_size=BATCH, shuffle=True, drop_last=True)
    ve, vt = to_batch(train_eeg[vi], train_tgt[vi], window)
    te, tt = to_batch(test_eeg, test_tgt, window)
    in_dim = train_eeg.shape[1] * (window[1] - window[0]); out_dim = train_tgt.shape[1]
    model = EEGDecoder(in_dim, out_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best = (float("inf"), None, 0, 0.0)   # (val_loss, test_loss, epoch, top1)
    for ep in range(1, EPOCHS + 1):
        model.train()
        for eeg, tgt in dl:
            eeg, tgt = eeg.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad(); info_nce(model(eeg), tgt).backward(); opt.step()
        sch.step()
        model.eval()
        with torch.no_grad():
            vl = info_nce(model(ve), vt).item()
            if vl < best[0]:
                tl = info_nce(model(te), tt).item()
                emb = model(te); sims = F.normalize(emb, dim=-1) @ F.normalize(tt, dim=-1).t()
                top1 = (sims.argmax(1) == torch.arange(len(emb), device=DEVICE)).float().mean().item()
                best = (vl, tl, ep, top1)
    return best

def main():
    print("device:", DEVICE, flush=True)
    train_eeg = np.load(os.path.join(EEG_DIR, "sub-01_train_avg.npy"))
    test_eeg  = np.load(os.path.join(EEG_DIR, "sub-01_test_avg.npy"))
    targets = sorted(os.path.basename(f)[:-len("__train.npy")]
                     for f in glob.glob(os.path.join(FEAT_DIR, "*__train.npy")))
    print(f"{len(targets)} targets x {len(WINDOWS)} windows = {len(targets)*len(WINDOWS)} decoders", flush=True)
    rows, t0 = [], time.time()
    for tgt_name in targets:
        train_tgt = np.load(os.path.join(FEAT_DIR, f"{tgt_name}__train.npy"))
        test_tgt  = np.load(os.path.join(FEAT_DIR, f"{tgt_name}__test.npy"))
        for wname, window in WINDOWS.items():
            vl, tl, ep, top1 = train_one(train_eeg, train_tgt, test_eeg, test_tgt, window)
            rows.append((tgt_name, wname, ep, round(vl,4), round(tl,4), round(top1,4)))
            print(f"{tgt_name:24s} {wname:9s} | ep{ep:2d} val{vl:.3f} test{tl:.3f} top1{top1*100:4.1f}% "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["target","window","best_epoch","val_loss","test_loss","top1"])
        w.writerows(rows)
    print("\nSAVED", OUT_CSV, flush=True)

if __name__ == "__main__":
    main()
