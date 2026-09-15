"""
03_diagnose_overfitting.py
==========================
Fast diagnostic run for the existing 250 Hz EEG alignment pipeline.

Purpose
-------
Test two targeted hypotheses without rerunning the full experiment:

1. Current residual decoder + FIXED temperature (0.07)
   -> tests whether the learnable temperature contributes to overfitting.

2. Simple linear decoder + FIXED temperature (0.07)
   -> tests whether decoder capacity contributes to overfitting.

This script reuses the already prepared EEG arrays and already extracted
vision-network features. It does NOT rerun EEG preprocessing or feature extraction.

Default diagnostic condition:
    target = RN50__attnpool
    window = 100_200 ms
    seed   = 0

Change TARGET or WINDOW_NAME below if you want to diagnose another condition.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
ROOT = os.path.expanduser("~/things_eeg")
EEG_DIR = os.path.join(ROOT, "eeg_prepared")
FEAT_DIR = os.path.join(ROOT, "features")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(RES_DIR, exist_ok=True)

OUT_CSV = os.path.join(RES_DIR, "diagnose_overfitting_sub-01.csv")
SUMMARY_CSV = os.path.join(RES_DIR, "diagnose_overfitting_summary_sub-01.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 0
EPOCHS = 15
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 0.01
DROPOUT = 0.3
FIXED_TEMPERATURE = 0.07

EXPECTED_NCHAN = 63
EXPECTED_NTIMES = 250

TARGET = "RN50__attnpool"
WINDOW_NAME = "100_200"

WINDOWS = {
    "0_100": (0, 25),
    "100_200": (25, 50),
    "200_300": (50, 75),
    "300_400": (75, 100),
    "400_500": (100, 125),
    "500_600": (125, 150),
    "600_700": (150, 175),
    "700_800": (175, 200),
    "800_900": (200, 225),
    "900_1000": (225, 250),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class EEGDataset(Dataset):
    def __init__(self, eeg, targets, window):
        s, e = window
        self.eeg = torch.from_numpy(eeg[:, :, s:e]).float()
        self.targets = torch.from_numpy(targets).float()

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, i):
        return self.eeg[i], self.targets[i]


def to_batch(eeg, target, window):
    s, e = window
    eeg_tensor = torch.from_numpy(eeg[:, :, s:e]).float().to(DEVICE)
    target_tensor = torch.from_numpy(target).float().to(DEVICE)
    return eeg_tensor, target_tensor


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ResidualAdd(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return x + self.function(x)


class ResidualDecoderFixedTemp(nn.Module):
    """Same residual decoder as the main pipeline, but temperature is fixed."""

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


class LinearDecoderFixedTemp(nn.Module):
    """Lower-capacity control: one linear projection followed by LayerNorm."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.project = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm(output_size),
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], self.input_size)
        return self.project(x)


# ---------------------------------------------------------------------------
# Loss and evaluation
# ---------------------------------------------------------------------------
def info_nce_fixed(a, b, temperature=0.07):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    logits = (a @ b.t()) / temperature
    labels = torch.arange(len(a), device=a.device)

    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )


def retrieval_accuracy(pred, target, ks=(1, 5)):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    similarities = pred @ target.t()

    n = similarities.shape[0]
    correct_index = torch.arange(n, device=similarities.device)
    output = {}

    for k in ks:
        actual_k = min(k, n)
        retrieved = similarities.topk(actual_k, dim=1).indices
        hit = (retrieved == correct_index[:, None]).any(dim=1)
        output[k] = hit.float().mean().item()

    return output


@torch.no_grad()
def evaluate(model, eeg, target):
    model.eval()
    prediction = model(eeg)
    loss = info_nce_fixed(
        prediction,
        target,
        temperature=FIXED_TEMPERATURE,
    ).item()
    accuracy = retrieval_accuracy(prediction, target)
    return loss, accuracy


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# One diagnostic run
# ---------------------------------------------------------------------------
def train_one_condition(
    condition_name,
    model_class,
    train_eeg,
    train_tgt,
    test_eeg,
    test_tgt,
    window,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # IMPORTANT: same validation split logic as the original pipeline.
    n = len(train_eeg)
    permutation = np.random.permutation(n)
    n_validation = n // 10

    validation_indices = permutation[:n_validation]
    training_indices = permutation[n_validation:]

    training_loader = DataLoader(
        EEGDataset(
            train_eeg[training_indices],
            train_tgt[training_indices],
            window,
        ),
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
    )

    validation_eeg, validation_target = to_batch(
        train_eeg[validation_indices],
        train_tgt[validation_indices],
        window,
    )

    test_eeg_tensor, test_target_tensor = to_batch(
        test_eeg,
        test_tgt,
        window,
    )

    # Same-size training probe as in the original pipeline.
    training_probe_indices = training_indices[:n_validation]
    training_probe_eeg, training_probe_target = to_batch(
        train_eeg[training_probe_indices],
        train_tgt[training_probe_indices],
        window,
    )

    input_dimension = train_eeg.shape[1] * (window[1] - window[0])
    output_dimension = train_tgt.shape[1]

    if model_class is ResidualDecoderFixedTemp:
        model = model_class(
            input_size=input_dimension,
            output_size=output_dimension,
            dropout=DROPOUT,
        ).to(DEVICE)
    else:
        model = model_class(
            input_size=input_dimension,
            output_size=output_dimension,
        ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    print(
        f"\n=== {condition_name} ===\n"
        f"target: {TARGET}\n"
        f"window: {WINDOW_NAME}\n"
        f"input dimension: {input_dimension}\n"
        f"output dimension: {output_dimension}\n"
        f"trainable parameters: {count_parameters(model):,}\n",
        flush=True,
    )

    rows = []
    best_val = float("inf")
    best_epoch = None
    best_test_loss = None
    best_test_top1 = None
    best_test_top5 = None

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()

        for eeg_batch, target_batch in training_loader:
            eeg_batch = eeg_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            optimizer.zero_grad()
            prediction = model(eeg_batch)
            loss = info_nce_fixed(
                prediction,
                target_batch,
                temperature=FIXED_TEMPERATURE,
            )
            loss.backward()
            optimizer.step()

        scheduler.step()

        train_loss, train_acc = evaluate(
            model,
            training_probe_eeg,
            training_probe_target,
        )
        val_loss, val_acc = evaluate(
            model,
            validation_eeg,
            validation_target,
        )

        gap = val_loss - train_loss
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val:
            test_loss, test_acc = evaluate(
                model,
                test_eeg_tensor,
                test_target_tensor,
            )
            best_val = val_loss
            best_epoch = epoch
            best_test_loss = test_loss
            best_test_top1 = test_acc[1]
            best_test_top5 = test_acc[5]

        rows.append(
            {
                "condition": condition_name,
                "target": TARGET,
                "window": WINDOW_NAME,
                "seed": SEED,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_minus_train": gap,
                "train_top1": train_acc[1],
                "train_top5": train_acc[5],
                "val_top1": val_acc[1],
                "val_top5": val_acc[5],
                "learning_rate": current_lr,
                "temperature": FIXED_TEMPERATURE,
                "n_parameters": count_parameters(model),
            }
        )

        print(
            f"epoch {epoch:02d} | "
            f"train {train_loss:.4f} | "
            f"val {val_loss:.4f} | "
            f"gap {gap:+.4f} | "
            f"val top1 {val_acc[1]*100:.2f}%",
            flush=True,
        )

    elapsed = time.time() - start_time

    summary = {
        "condition": condition_name,
        "target": TARGET,
        "window": WINDOW_NAME,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss_at_best_val": best_test_loss,
        "test_top1_at_best_val": best_test_top1,
        "test_top5_at_best_val": best_test_top5,
        "final_train_loss": rows[-1]["train_loss"],
        "final_val_loss": rows[-1]["val_loss"],
        "final_gap": rows[-1]["val_minus_train"],
        "n_parameters": count_parameters(model),
        "elapsed_s": elapsed,
    }

    return rows, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(
        "device:", DEVICE,
        "| diagnostic epochs:", EPOCHS,
        "| fixed temperature:", FIXED_TEMPERATURE,
        flush=True,
    )

    if WINDOW_NAME not in WINDOWS:
        raise KeyError(f"Unknown WINDOW_NAME: {WINDOW_NAME}")

    window = WINDOWS[WINDOW_NAME]

    train_eeg_path = os.path.join(EEG_DIR, "sub-01_train_avg.npy")
    test_eeg_path = os.path.join(EEG_DIR, "sub-01_test_avg.npy")
    train_tgt_path = os.path.join(FEAT_DIR, f"{TARGET}__train.npy")
    test_tgt_path = os.path.join(FEAT_DIR, f"{TARGET}__test.npy")

    for path in [
        train_eeg_path,
        test_eeg_path,
        train_tgt_path,
        test_tgt_path,
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    train_eeg = np.load(train_eeg_path)
    test_eeg = np.load(test_eeg_path)
    train_tgt = np.load(train_tgt_path)
    test_tgt = np.load(test_tgt_path)

    assert train_eeg.shape == (16540, EXPECTED_NCHAN, EXPECTED_NTIMES)
    assert test_eeg.shape == (200, EXPECTED_NCHAN, EXPECTED_NTIMES)
    assert train_tgt.shape[0] == train_eeg.shape[0]
    assert test_tgt.shape[0] == test_eeg.shape[0]

    print("train EEG:", train_eeg.shape)
    print("test EEG :", test_eeg.shape)
    print("train target:", train_tgt.shape)
    print("test target :", test_tgt.shape)

    diagnostic_conditions = [
        (
            "residual_fixed_temperature",
            ResidualDecoderFixedTemp,
        ),
        (
            "linear_fixed_temperature",
            LinearDecoderFixedTemp,
        ),
    ]

    all_rows = []
    summaries = []

    for condition_name, model_class in diagnostic_conditions:
        rows, summary = train_one_condition(
            condition_name,
            model_class,
            train_eeg,
            train_tgt,
            test_eeg,
            test_tgt,
            window,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(summaries).to_csv(SUMMARY_CSV, index=False)

    print("\n=== SUMMARY ===")
    summary_df = pd.DataFrame(summaries)
    print(
        summary_df[
            [
                "condition",
                "best_epoch",
                "best_val_loss",
                "test_loss_at_best_val",
                "final_gap",
                "n_parameters",
            ]
        ].to_string(index=False)
    )

    print("\nsaved ->", OUT_CSV)
    print("saved ->", SUMMARY_CSV)
    print("\nDIAGNOSTIC RUN COMPLETED")


if __name__ == "__main__":
    main()
