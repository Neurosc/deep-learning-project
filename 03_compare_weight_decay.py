"""
03_compare_weight_decay.py
==========================
Small 250 Hz benchmark to compare weight decay values after selecting lr=1e-4.

Compare:
    weight_decay = 1e-2   (current pipeline)
    weight_decay = 1e-4   (Things-EEG paper code)

Everything else stays fixed:
    - 250 Hz prepared EEG
    - learning rate = 1e-4
    - residual decoder
    - fixed InfoNCE temperature = 0.07
    - batch size = 256
    - same target/window/seed combinations
    - same validation split for a given seed

Representative targets:
    RN50__stem
    RN50__attnpool
    ViT-B-16__block2
    ViT-B-16__block12
    ViT-L-14__block4
    ViT-L-14__block24

Representative EEG windows:
    100_200
    300_400
    500_600

Seeds:
    0, 1

Maximum epochs:
    30

Early stopping:
    Stop when validation loss has not improved for 8 consecutive epochs.

Outputs:
    ~/things_eeg/results/weight_decay_comparison_sub-01.csv
    ~/things_eeg/results/weight_decay_comparison_epochs_sub-01.csv
    ~/things_eeg/results/weight_decay_comparison_paired_sub-01.csv
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


ROOT = os.path.expanduser("~/things_eeg")
EEG_DIR = os.path.join(ROOT, "eeg_prepared")
FEAT_DIR = os.path.join(ROOT, "features")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(RES_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(
    RES_DIR,
    "weight_decay_comparison_sub-01.csv",
)
EPOCH_CSV = os.path.join(
    RES_DIR,
    "weight_decay_comparison_epochs_sub-01.csv",
)
PAIRED_CSV = os.path.join(
    RES_DIR,
    "weight_decay_comparison_paired_sub-01.csv",
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-4
WEIGHT_DECAYS = [1e-2, 1e-4]

SEEDS = [0, 1]

MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 8

BATCH = 256
DROPOUT = 0.3
FIXED_TEMPERATURE = 0.07

EXPECTED_NCHAN = 63
EXPECTED_NTIMES = 250

TARGETS = [
    "RN50__stem",
    "RN50__attnpool",
    "ViT-B-16__block2",
    "ViT-B-16__block12",
    "ViT-L-14__block4",
    "ViT-L-14__block24",
]

WINDOWS = {
    "100_200": (25, 50),
    "300_400": (75, 100),
    "500_600": (125, 150),
}


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

    eeg_tensor = torch.from_numpy(
        eeg[:, :, s:e]
    ).float().to(DEVICE)

    target_tensor = torch.from_numpy(
        target
    ).float().to(DEVICE)

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
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def info_nce_fixed(prediction, target, temperature=0.07):
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)

    logits = (prediction @ target.t()) / temperature

    labels = torch.arange(
        len(prediction),
        device=prediction.device,
    )

    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )


def retrieval_accuracy(prediction, target, ks=(1, 5)):
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target, dim=-1)

    similarities = prediction @ target.t()
    n = similarities.shape[0]

    correct_index = torch.arange(
        n,
        device=similarities.device,
    )

    output = {}

    for k in ks:
        actual_k = min(k, n)

        retrieved = similarities.topk(
            actual_k,
            dim=1,
        ).indices

        hit = (
            retrieved == correct_index[:, None]
        ).any(dim=1)

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

    accuracy = retrieval_accuracy(
        prediction,
        target,
    )

    return loss, accuracy


def train_one(
    train_eeg,
    train_target,
    test_eeg,
    test_target,
    window,
    seed,
    weight_decay,
):
    # Same seed ensures the same split and initialisation sequence
    # across weight-decay conditions.
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n = len(train_eeg)

    permutation = np.random.permutation(n)
    n_validation = n // 10

    validation_indices = permutation[:n_validation]
    training_indices = permutation[n_validation:]

    training_loader = DataLoader(
        EEGDataset(
            train_eeg[training_indices],
            train_target[training_indices],
            window,
        ),
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
    )

    validation_eeg, validation_target = to_batch(
        train_eeg[validation_indices],
        train_target[validation_indices],
        window,
    )

    test_eeg_tensor, test_target_tensor = to_batch(
        test_eeg,
        test_target,
        window,
    )

    training_probe_indices = training_indices[:n_validation]

    training_probe_eeg, training_probe_target = to_batch(
        train_eeg[training_probe_indices],
        train_target[training_probe_indices],
        window,
    )

    input_dimension = (
        train_eeg.shape[1]
        * (window[1] - window[0])
    )

    output_dimension = train_target.shape[1]

    model = ResidualDecoder(
        input_size=input_dimension,
        output_size=output_dimension,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_parameters = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    best_test_loss = np.nan
    best_test_top1 = np.nan
    best_test_top5 = np.nan
    best_train_loss = np.nan
    best_gap = np.nan

    epochs_without_improvement = 0
    trace = []

    for epoch in range(1, MAX_EPOCHS + 1):
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

        train_loss, train_accuracy = evaluate(
            model,
            training_probe_eeg,
            training_probe_target,
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_eeg,
            validation_target,
        )

        current_gap = validation_loss - train_loss
        current_lr = optimizer.param_groups[0]["lr"]

        improved = validation_loss < best_validation_loss

        if improved:
            test_loss, test_accuracy = evaluate(
                model,
                test_eeg_tensor,
                test_target_tensor,
            )

            best_validation_loss = validation_loss
            best_epoch = epoch
            best_test_loss = test_loss
            best_test_top1 = test_accuracy[1]
            best_test_top5 = test_accuracy[5]
            best_train_loss = train_loss
            best_gap = current_gap

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        trace.append(
            {
                "epoch": epoch,
                "weight_decay_condition": weight_decay,
                "current_lr": current_lr,
                "train_loss": train_loss,
                "val_loss": validation_loss,
                "gap": current_gap,
                "train_top1": train_accuracy[1],
                "train_top5": train_accuracy[5],
                "val_top1": validation_accuracy[1],
                "val_top5": validation_accuracy[5],
                "is_best_epoch": improved,
            }
        )

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            break

    final = trace[-1]

    result = {
        "learning_rate": LEARNING_RATE,
        "weight_decay": weight_decay,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_run": len(trace),
        "best_val_loss": best_validation_loss,
        "test_loss_at_best": best_test_loss,
        "test_top1_at_best": best_test_top1,
        "test_top5_at_best": best_test_top5,
        "train_loss_at_best": best_train_loss,
        "gap_at_best": best_gap,
        "final_train_loss": final["train_loss"],
        "final_val_loss": final["val_loss"],
        "final_gap": final["gap"],
        "n_parameters": n_parameters,
    }

    return result, trace


def main():
    print("device:", DEVICE, flush=True)
    print("learning rate:", LEARNING_RATE, flush=True)
    print("weight decays:", WEIGHT_DECAYS, flush=True)
    print("targets:", TARGETS, flush=True)
    print("windows:", list(WINDOWS.keys()), flush=True)
    print("seeds:", SEEDS, flush=True)

    print(
        "maximum runs:",
        len(WEIGHT_DECAYS)
        * len(TARGETS)
        * len(WINDOWS)
        * len(SEEDS),
        flush=True,
    )

    train_eeg = np.load(
        os.path.join(
            EEG_DIR,
            "sub-01_train_avg.npy",
        )
    )

    test_eeg = np.load(
        os.path.join(
            EEG_DIR,
            "sub-01_test_avg.npy",
        )
    )

    assert train_eeg.shape == (
        16540,
        EXPECTED_NCHAN,
        EXPECTED_NTIMES,
    )

    assert test_eeg.shape == (
        200,
        EXPECTED_NCHAN,
        EXPECTED_NTIMES,
    )

    summary_rows = []
    epoch_rows = []

    experiment_start = time.time()

    for target_name in TARGETS:
        train_feature_path = os.path.join(
            FEAT_DIR,
            f"{target_name}__train.npy",
        )

        test_feature_path = os.path.join(
            FEAT_DIR,
            f"{target_name}__test.npy",
        )

        if not os.path.exists(train_feature_path):
            raise FileNotFoundError(
                f"Missing feature file: {train_feature_path}"
            )

        if not os.path.exists(test_feature_path):
            raise FileNotFoundError(
                f"Missing feature file: {test_feature_path}"
            )

        train_target = np.load(train_feature_path)
        test_target = np.load(test_feature_path)

        assert train_target.shape[0] == train_eeg.shape[0]
        assert test_target.shape[0] == test_eeg.shape[0]

        for window_name, window in WINDOWS.items():
            for seed in SEEDS:
                for weight_decay in WEIGHT_DECAYS:
                    start_time = time.time()

                    result, trace = train_one(
                        train_eeg=train_eeg,
                        train_target=train_target,
                        test_eeg=test_eeg,
                        test_target=test_target,
                        window=window,
                        seed=seed,
                        weight_decay=weight_decay,
                    )

                    result.update(
                        {
                            "target": target_name,
                            "window": window_name,
                        }
                    )

                    summary_rows.append(result)

                    for epoch_result in trace:
                        epoch_rows.append(
                            {
                                "target": target_name,
                                "window": window_name,
                                "seed": seed,
                                **epoch_result,
                            }
                        )

                    print(
                        f"{target_name:24s} "
                        f"{window_name:8s} "
                        f"seed{seed} "
                        f"wd={weight_decay:.0e} | "
                        f"best_ep={result['best_epoch']:2d} "
                        f"val={result['best_val_loss']:.3f} "
                        f"test={result['test_loss_at_best']:.3f} "
                        f"gap_best={result['gap_at_best']:.3f} "
                        f"top1={result['test_top1_at_best']*100:5.1f}% "
                        f"epochs={result['epochs_run']:2d} "
                        f"[{time.time()-start_time:.1f}s]",
                        flush=True,
                    )

    summary_df = pd.DataFrame(summary_rows)
    epoch_df = pd.DataFrame(epoch_rows)

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    epoch_df.to_csv(
        EPOCH_CSV,
        index=False,
    )

    index_columns = [
        "target",
        "window",
        "seed",
    ]

    value_columns = [
        "best_epoch",
        "best_val_loss",
        "test_loss_at_best",
        "test_top1_at_best",
        "test_top5_at_best",
        "train_loss_at_best",
        "gap_at_best",
        "final_gap",
    ]

    wd_1e2 = (
        summary_df[
            np.isclose(
                summary_df["weight_decay"],
                1e-2,
            )
        ][index_columns + value_columns]
        .copy()
    )

    wd_1e4 = (
        summary_df[
            np.isclose(
                summary_df["weight_decay"],
                1e-4,
            )
        ][index_columns + value_columns]
        .copy()
    )

    wd_1e2 = wd_1e2.rename(
        columns={
            column: f"{column}_wd1e-2"
            for column in value_columns
        }
    )

    wd_1e4 = wd_1e4.rename(
        columns={
            column: f"{column}_wd1e-4"
            for column in value_columns
        }
    )

    paired = wd_1e2.merge(
        wd_1e4,
        on=index_columns,
        how="inner",
        validate="one_to_one",
    )

    paired[
        "delta_best_val_wd1e-4_minus_wd1e-2"
    ] = (
        paired["best_val_loss_wd1e-4"]
        - paired["best_val_loss_wd1e-2"]
    )

    paired[
        "delta_test_loss_wd1e-4_minus_wd1e-2"
    ] = (
        paired["test_loss_at_best_wd1e-4"]
        - paired["test_loss_at_best_wd1e-2"]
    )

    paired[
        "delta_test_top1_wd1e-4_minus_wd1e-2"
    ] = (
        paired["test_top1_at_best_wd1e-4"]
        - paired["test_top1_at_best_wd1e-2"]
    )

    paired[
        "delta_gap_at_best_wd1e-4_minus_wd1e-2"
    ] = (
        paired["gap_at_best_wd1e-4"]
        - paired["gap_at_best_wd1e-2"]
    )

    paired.to_csv(
        PAIRED_CSV,
        index=False,
    )

    print(
        "\n=== Mean performance by weight decay ===",
        flush=True,
    )

    mean_summary = (
        summary_df
        .groupby("weight_decay")
        .agg(
            mean_best_epoch=("best_epoch", "mean"),
            mean_best_val_loss=("best_val_loss", "mean"),
            mean_test_loss=("test_loss_at_best", "mean"),
            mean_test_top1=("test_top1_at_best", "mean"),
            mean_test_top5=("test_top5_at_best", "mean"),
            mean_gap_at_best=("gap_at_best", "mean"),
            mean_final_gap=("final_gap", "mean"),
        )
        .reset_index()
    )

    print(
        mean_summary.to_string(index=False),
        flush=True,
    )

    print(
        "\nInterpret paired deltas as:",
        flush=True,
    )
    print(
        "negative delta test loss = wd=1e-4 better",
        flush=True,
    )
    print(
        "positive delta test top1 = wd=1e-4 better",
        flush=True,
    )
    print(
        "negative delta gap = wd=1e-4 smaller gap",
        flush=True,
    )

    print("\nSAVED:", SUMMARY_CSV, flush=True)
    print("SAVED:", EPOCH_CSV, flush=True)
    print("SAVED:", PAIRED_CSV, flush=True)

    print(
        f"\nTOTAL TIME: {time.time()-experiment_start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
