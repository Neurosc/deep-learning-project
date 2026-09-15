import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.expanduser("~/things_eeg")

EEG_DIRS = {
    "250": os.path.join(
        ROOT,
        "preprocessed_data",
        "Preprocessed_data_250Hz_whiten",
        "sub-01",
    ),
    "1000": os.path.join(
        ROOT,
        "preprocessed_data",
        "Preprocessed_data_1000Hz_whiten",
        "sub-01",
    ),
}

FEAT_DIR = os.path.join(ROOT, "features")
RES_DIR = os.path.join(ROOT, "results")

os.makedirs(RES_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(RES_DIR, "sampling_rate_comparison_sub-01.csv")
EPOCH_CSV = os.path.join(RES_DIR, "sampling_rate_comparison_epochs_sub-01.csv")
PAIRED_CSV = os.path.join(RES_DIR, "sampling_rate_comparison_paired_sub-01.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
FIXED_TEMPERATURE = 0.07
BATCH = 256
DROPOUT = 0.3
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 8
SEEDS = [0, 1]

EXPECTED_NCHAN = 63
EXPECTED_NTRAIN = 16540
EXPECTED_NTEST = 200

SAMPLING_RATES = {
    "250": {"sfreq": 250, "n_times": 250},
    "1000": {"sfreq": 1000, "n_times": 1000},
}

TARGETS = [
    "RN50__stem",
    "RN50__attnpool",
    "ViT-B-16__block2",
    "ViT-B-16__block12",
    "ViT-L-14__block4",
    "ViT-L-14__block24",
]

WINDOWS_MS = [(100, 200), (300, 400), (500, 600)]


def ms_to_samples(start_ms, end_ms, sfreq):
    return (
        int(round(start_ms * sfreq / 1000.0)),
        int(round(end_ms * sfreq / 1000.0)),
    )


class EEGDataset(Dataset):
    def __init__(self, eeg, targets, window):
        s, e = window
        self.eeg = torch.from_numpy(eeg[:, :, s:e]).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, index):
        return self.eeg[index], self.targets[index]


def to_batch(eeg, target, window):
    s, e = window
    eeg_tensor = torch.from_numpy(eeg[:, :, s:e]).float().to(DEVICE)
    target_tensor = torch.from_numpy(target).float().to(DEVICE)
    return eeg_tensor, target_tensor


class ResidualAdd(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return x + self.function(x)


class ResidualDecoder(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.3):
        super().__init__()
        self.input_size = input_size
        self.project = nn.Sequential(
            nn.Linear(input_size, output_size),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(output_size, output_size),
                    nn.Dropout(dropout),
                )
            ),
            nn.LayerNorm(output_size),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], self.input_size)
        return self.project(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def info_nce_fixed(prediction, target, temperature=0.07):
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = (prediction @ target.t()) / temperature
    labels = torch.arange(len(prediction), device=prediction.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )


def retrieval_accuracy(prediction, target, ks=(1, 5)):
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)
    similarities = prediction @ target.t()
    n = similarities.shape[0]
    correct = torch.arange(n, device=similarities.device)
    out = {}
    for k in ks:
        idx = similarities.topk(min(k, n), dim=1).indices
        out[k] = (idx == correct[:, None]).any(dim=1).float().mean().item()
    return out


@torch.no_grad()
def evaluate(model, eeg, target):
    model.eval()
    pred = model(eeg)
    loss = info_nce_fixed(pred, target, FIXED_TEMPERATURE).item()
    acc = retrieval_accuracy(pred, target)
    return loss, acc


def load_eeg(rate_name):
    train_path = os.path.join(
        EEG_DIRS[rate_name],
        "train.pt",
    )

    test_path = os.path.join(
        EEG_DIRS[rate_name],
        "test.pt",
    )

    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            f"Missing {rate_name} Hz train EEG: {train_path}"
        )

    if not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"Missing {rate_name} Hz test EEG: {test_path}"
        )

    train_loaded = torch.load(
        train_path,
        map_location="cpu",
        weights_only=False,
    )

    test_loaded = torch.load(
        test_path,
        map_location="cpu",
        weights_only=False,
    )

    train_eeg = train_loaded["eeg"].float().cpu().numpy()
    test_eeg = test_loaded["eeg"].float().cpu().numpy()

    expected_t = SAMPLING_RATES[
        rate_name
    ]["n_times"]

    assert train_eeg.shape == (
        EXPECTED_NTRAIN,
        EXPECTED_NCHAN,
        expected_t,
    ), train_eeg.shape

    assert test_eeg.shape == (
        EXPECTED_NTEST,
        EXPECTED_NCHAN,
        expected_t,
    ), test_eeg.shape

    print(
        f"{rate_name} Hz EEG loaded:",
        train_eeg.shape,
        test_eeg.shape,
    )

    return train_eeg, test_eeg

def train_one(train_eeg, train_tgt, test_eeg, test_tgt, window, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n = len(train_eeg)
    perm = np.random.permutation(n)
    n_val = n // 10
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]

    loader = DataLoader(
        EEGDataset(train_eeg[tr_idx], train_tgt[tr_idx], window),
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
    )

    val_eeg, val_tgt = to_batch(train_eeg[val_idx], train_tgt[val_idx], window)
    test_eeg_t, test_tgt_t = to_batch(test_eeg, test_tgt, window)

    probe_idx = tr_idx[:n_val]
    probe_eeg, probe_tgt = to_batch(train_eeg[probe_idx], train_tgt[probe_idx], window)

    input_dim = train_eeg.shape[1] * (window[1] - window[0])
    output_dim = train_tgt.shape[1]

    model = ResidualDecoder(input_dim, output_dim, DROPOUT).to(DEVICE)
    n_params = count_parameters(model)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best_val = float("inf")
    best = None
    bad_epochs = 0
    trace = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for eeg_b, tgt_b in loader:
            eeg_b = eeg_b.to(DEVICE)
            tgt_b = tgt_b.to(DEVICE)
            opt.zero_grad()
            pred = model(eeg_b)
            loss = info_nce_fixed(pred, tgt_b, FIXED_TEMPERATURE)
            loss.backward()
            opt.step()

        sched.step()

        train_loss, train_acc = evaluate(model, probe_eeg, probe_tgt)
        val_loss, val_acc = evaluate(model, val_eeg, val_tgt)
        gap = val_loss - train_loss
        improved = val_loss < best_val

        if improved:
            test_loss, test_acc = evaluate(model, test_eeg_t, test_tgt_t)
            best_val = val_loss
            best = {
                "best_epoch": epoch,
                "best_val_loss": val_loss,
                "test_loss_at_best": test_loss,
                "test_top1_at_best": test_acc[1],
                "test_top5_at_best": test_acc[5],
                "train_loss_at_best": train_loss,
                "gap_at_best": gap,
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        trace.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gap": gap,
            "train_top1": train_acc[1],
            "train_top5": train_acc[5],
            "val_top1": val_acc[1],
            "val_top5": val_acc[5],
            "current_lr": opt.param_groups[0]["lr"],
            "is_best_epoch": improved,
        })

        if bad_epochs >= EARLY_STOP_PATIENCE:
            break

    best.update({
        "epochs_run": len(trace),
        "final_train_loss": trace[-1]["train_loss"],
        "final_val_loss": trace[-1]["val_loss"],
        "final_gap": trace[-1]["gap"],
        "input_dimension": input_dim,
        "n_parameters": n_params,
    })
    return best, trace


def main():
    print("device:", DEVICE)
    print("lr:", LEARNING_RATE, "| wd:", WEIGHT_DECAY)
    print("fixed temperature:", FIXED_TEMPERATURE)

    eeg = {rate: load_eeg(rate) for rate in SAMPLING_RATES}

    summary_rows = []
    epoch_rows = []
    t0 = time.time()

    for target in TARGETS:
        train_tgt = np.load(os.path.join(FEAT_DIR, f"{target}__train.npy"))
        test_tgt = np.load(os.path.join(FEAT_DIR, f"{target}__test.npy"))

        for start_ms, end_ms in WINDOWS_MS:
            window_name = f"{start_ms}_{end_ms}"

            for seed in SEEDS:
                for rate_name, info in SAMPLING_RATES.items():
                    window = ms_to_samples(start_ms, end_ms, info["sfreq"])
                    train_eeg, test_eeg = eeg[rate_name]

                    result, trace = train_one(
                        train_eeg, train_tgt, test_eeg, test_tgt, window, seed
                    )

                    result.update({
                        "sampling_rate": int(rate_name),
                        "target": target,
                        "window": window_name,
                        "seed": seed,
                        "samples_in_window": window[1] - window[0],
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                    })
                    summary_rows.append(result)

                    for row in trace:
                        epoch_rows.append({
                            "sampling_rate": int(rate_name),
                            "target": target,
                            "window": window_name,
                            "seed": seed,
                            **row,
                        })

                    print(
                        f"{rate_name}Hz {target:24s} {window_name} seed{seed} | "
                        f"in={result['input_dimension']} params={result['n_parameters']} "
                        f"ep={result['best_epoch']} val={result['best_val_loss']:.3f} "
                        f"test={result['test_loss_at_best']:.3f} "
                        f"top1={100*result['test_top1_at_best']:.1f}% "
                        f"top5={100*result['test_top5_at_best']:.1f}% "
                        f"gap={result['gap_at_best']:.3f}"
                    )

    summary = pd.DataFrame(summary_rows)
    epochs = pd.DataFrame(epoch_rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    epochs.to_csv(EPOCH_CSV, index=False)

    keys = ["target", "window", "seed"]
    metrics = [
        "best_epoch",
        "best_val_loss",
        "test_loss_at_best",
        "test_top1_at_best",
        "test_top5_at_best",
        "gap_at_best",
        "final_gap",
        "input_dimension",
        "n_parameters",
    ]

    a = summary[summary.sampling_rate == 250][keys + metrics].copy()
    b = summary[summary.sampling_rate == 1000][keys + metrics].copy()
    a = a.rename(columns={m: f"{m}_250Hz" for m in metrics})
    b = b.rename(columns={m: f"{m}_1000Hz" for m in metrics})

    paired = a.merge(b, on=keys, validate="one_to_one")
    paired["delta_test_loss_1000_minus_250"] = (
        paired["test_loss_at_best_1000Hz"] - paired["test_loss_at_best_250Hz"]
    )
    paired["delta_test_top1_1000_minus_250"] = (
        paired["test_top1_at_best_1000Hz"] - paired["test_top1_at_best_250Hz"]
    )
    paired["delta_test_top5_1000_minus_250"] = (
        paired["test_top5_at_best_1000Hz"] - paired["test_top5_at_best_250Hz"]
    )
    paired["delta_gap_1000_minus_250"] = (
        paired["gap_at_best_1000Hz"] - paired["gap_at_best_250Hz"]
    )
    paired["parameter_ratio_1000_over_250"] = (
        paired["n_parameters_1000Hz"] / paired["n_parameters_250Hz"]
    )
    paired.to_csv(PAIRED_CSV, index=False)

    print("\n=== Mean summary ===")
    print(summary.groupby("sampling_rate").agg(
        best_val=("best_val_loss", "mean"),
        test_loss=("test_loss_at_best", "mean"),
        top1=("test_top1_at_best", "mean"),
        top5=("test_top5_at_best", "mean"),
        gap=("gap_at_best", "mean"),
        final_gap=("final_gap", "mean"),
        best_epoch=("best_epoch", "mean"),
        input_dim=("input_dimension", "mean"),
        params=("n_parameters", "mean"),
    ))

    print("\nSaved:")
    print(SUMMARY_CSV)
    print(EPOCH_CSV)
    print(PAIRED_CSV)
    print("Total seconds:", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
