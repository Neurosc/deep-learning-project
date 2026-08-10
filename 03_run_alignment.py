"""
03_run_alignment.py
===================
The main experiment. Train a small model to predict each image's network feature
from the brain signal, for every (feature set x time window) combination, and
record how well it works. Run AFTER 01_prepare_eeg.py and 02_extract_features.py.

Changes in this version:
    1. EEG is now 250 Hz, so each 100 ms window contains 25 samples.
    2. Uses EEGProjectLayer with a standard residual connection and LayerNorm.
    3. Uses a learnable contrastive temperature.
    4. Saves train/validation loss, top-1, top-5, learning rate and temperature
       for every epoch using pandas.
    5. Runs one full 1-second sanity check.

Usage:
    python 03_run_alignment.py
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import glob
import csv
import time
import numpy as np
import pandas as pd                          # CHANGED: pandas epoch log
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


ROOT     = os.path.expanduser("~/things_eeg")
EEG_DIR  = os.path.join(ROOT, "eeg_prepared")
FEAT_DIR = os.path.join(ROOT, "features")
RES_DIR  = os.path.join(ROOT, "results")
os.makedirs(RES_DIR, exist_ok=True)

OUT_CSV       = os.path.join(RES_DIR, "alignment_sub-01_lr1e-4.csv")
OUT_CSV_SEEDS = os.path.join(RES_DIR, "alignment_sub-01_seeds_lr1e-4.csv")
EPOCH_CSV     = os.path.join(RES_DIR, "epoch_traces_sub-01_lr1e-4.csv")
OBJ_CSV       = os.path.join(RES_DIR, "object_perception_sub-01_lr1e-4.csv")
SANITY_CSV    = os.path.join(RES_DIR, "sanity_check_sub-01_lr1e-4.csv")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [0, 1, 2, 3, 4]

EPOCHS = 50
BATCH  = 256

EXPECTED_NCHAN  = 63
EXPECTED_NTIMES = 250                       # CHANGED
WINDOW_SAMPLES  = 25                        # CHANGED: 100 ms at 250 Hz

DEEP_LAYER = "ViT-L-14__block24"


# ---------------------------------------------------------------------------
# EEG windows
# ---------------------------------------------------------------------------

# CHANGED:
# The prepared EEG starts at 0 seconds and contains 250 samples.
# At 250 Hz, 25 samples correspond to 100 ms.
#
# There is no baseline window because the prepared arrays contain only
# 0 to approximately 996 ms after stimulus onset.
WINDOWS = {
    "0_100":     (0, 25),
    "100_200":   (25, 50),
    "200_300":   (50, 75),
    "300_400":   (75, 100),
    "400_500":   (100, 125),
    "500_600":   (125, 150),
    "600_700":   (150, 175),
    "700_800":   (175, 200),
    "800_900":   (200, 225),
    "900_1000":  (225, 250),
}

# CHANGED:
# Used only for the full temporal-resolution sanity check.
FULL_WINDOW = (0, 250)


class EEGDataset(Dataset):
    """Pair each image's EEG window with that image's target feature vector."""

    def __init__(self, eeg, targets, window):
        s, e = window

        self.eeg = torch.from_numpy(
            eeg[:, :, s:e]
        ).float()

        self.targets = torch.from_numpy(
            targets
        ).float()

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, i):
        return self.eeg[i], self.targets[i]


# ---------------------------------------------------------------------------
# EEGProjectLayer
# ---------------------------------------------------------------------------

# CHANGED:
# Standard residual connection:
#
# output = x + function(x)
#
# Unlike the old architecture, the residual branch does not use a separate
# linear transformation of the original EEG input.
class ResidualAdd(nn.Module):

    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return x + self.function(x)


# CHANGED:
# This replaces EEGDecoder, LinearDecoder and make_decoder.
class EEGProjectLayer(nn.Module):
    """
    Project flattened EEG into the image-feature space.

    Structure:
        EEG
        -> flatten
        -> Linear(input_size, output_size)
        -> residual block
        -> LayerNorm

    The model also stores a learnable logit scale. This is equivalent to
    learning the temperature used by InfoNCE.
    """

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

        # Start from the old temperature value, 0.07.
        #
        # scale = 1 / temperature
        # logit_scale = log(1 / temperature)
        self.logit_scale = nn.Parameter(
            torch.tensor(
                np.log(1.0 / 0.07),
                dtype=torch.float32,
            )
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], self.input_size)
        return self.project(x)

    def temperature(self):
        """Return the current learned temperature."""

        scale = self.logit_scale.exp().clamp(max=100.0)
        return 1.0 / scale


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

# CHANGED:
# The fixed t=0.07 argument has been removed.
# logit_scale is learned by EEGProjectLayer.
def info_nce(a, b, logit_scale):
    """Symmetric InfoNCE contrastive loss with learnable temperature."""

    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    scale = logit_scale.exp().clamp(max=100.0)

    logits = scale * (a @ b.t())

    labels = torch.arange(
        len(a),
        device=a.device,
    )

    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )


def retrieval_accuracy(pred, target, ks=(1, 5)):
    """
    For each predicted EEG embedding, rank all target features by cosine
    similarity and check whether the correct target is within the top k.
    """

    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    similarities = pred @ target.t()

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


def to_batch(eeg, target, window):
    """Slice one EEG window and move the complete set to the GPU."""

    s, e = window

    eeg_tensor = torch.from_numpy(
        eeg[:, :, s:e]
    ).float().to(DEVICE)

    target_tensor = torch.from_numpy(
        target
    ).float().to(DEVICE)

    return eeg_tensor, target_tensor


# CHANGED:
# Shared evaluation function for loss and retrieval accuracy.
@torch.no_grad()
def evaluate(model, eeg, target):

    prediction = model(eeg)

    loss = info_nce(
        prediction,
        target,
        model.logit_scale,
    ).item()

    accuracy = retrieval_accuracy(
        prediction,
        target,
    )

    return loss, accuracy


def train_one(
    train_eeg,
    train_tgt,
    test_eeg,
    test_tgt,
    window,
    seed,
):
    """
    Train one EEGProjectLayer for one feature set, window and seed.

    Returns:
        best:
            val_loss, test_loss, best_epoch, test_top1, test_top5

        trace:
            per-epoch train/validation metrics
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Carve a 10% validation set out of the training images.
    n = len(train_eeg)

    permutation = np.random.permutation(n)

    n_validation = n // 10

    validation_indices = permutation[:n_validation]
    training_indices   = permutation[n_validation:]

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

    # Fixed training probe with the same size as validation.
    training_probe_indices = training_indices[:n_validation]

    training_probe_eeg, training_probe_target = to_batch(
        train_eeg[training_probe_indices],
        train_tgt[training_probe_indices],
        window,
    )

    input_dimension = (
        train_eeg.shape[1]
        * (window[1] - window[0])
    )

    output_dimension = train_tgt.shape[1]

    # CHANGED: EEGProjectLayer replaces the old decoder selection.
    model = EEGProjectLayer(
        input_size=input_dimension,
        output_size=output_dimension,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4, # changed this to 1e-4 from 1e-3
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best = (
        float("inf"),
        None,
        0,
        0.0,
        0.0,
    )

    trace = []

    for epoch in range(1, EPOCHS + 1):

        # ---------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------

        model.train()

        for eeg_batch, target_batch in training_loader:

            eeg_batch = eeg_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(eeg_batch)

            loss = info_nce(
                prediction,
                target_batch,
                model.logit_scale,
            )

            loss.backward()
            optimizer.step()

        scheduler.step()

        # ---------------------------------------------------------------
        # Per-epoch train and validation logs
        # ---------------------------------------------------------------

        model.eval()

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

        temperature = float(
            model.temperature().detach().cpu()
        )

        learning_rate = optimizer.param_groups[0]["lr"]

        # CHANGED:
        # More complete per-epoch information.
        trace.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation_loss,
                "train_top1": train_accuracy[1],
                "train_top5": train_accuracy[5],
                "val_top1": validation_accuracy[1],
                "val_top5": validation_accuracy[5],
                "temperature": temperature,
                "learning_rate": learning_rate,
            }
        )

        # ---------------------------------------------------------------
        # Test only when validation improves
        # ---------------------------------------------------------------

        if validation_loss < best[0]:

            test_loss, test_accuracy = evaluate(
                model,
                test_eeg_tensor,
                test_target_tensor,
            )

            best = (
                validation_loss,
                test_loss,
                epoch,
                test_accuracy[1],
                test_accuracy[5],
            )

    return best, trace


# ---------------------------------------------------------------------------
# Full temporal-resolution sanity check
# ---------------------------------------------------------------------------

# CHANGED:
# This now trains one decoder using all 250 samples rather than one 100 ms
# window.
def sanity_check(
    train_eeg,
    test_eeg,
    target,
    seed=0,
):
    """
    Train one decoder using the complete one-second EEG signal.
    """

    train_tgt = np.load(
        os.path.join(
            FEAT_DIR,
            f"{target}__train.npy",
        )
    )

    test_tgt = np.load(
        os.path.join(
            FEAT_DIR,
            f"{target}__test.npy",
        )
    )

    start_time = time.time()

    (
        validation_loss,
        test_loss,
        best_epoch,
        top1,
        top5,
    ), _ = train_one(
        train_eeg,
        train_tgt,
        test_eeg,
        test_tgt,
        FULL_WINDOW,
        seed,
    )

    elapsed = time.time() - start_time

    print(
        f"[sanity] {target} @ full_1s "
        f"(seed {seed}): "
        f"val_loss={validation_loss:.4f} "
        f"test_loss={test_loss:.4f} "
        f"top1={top1 * 100:.2f}% "
        f"top5={top5 * 100:.2f}% "
        f"[{elapsed:.1f}s]",
        flush=True,
    )

    sanity_result = pd.DataFrame(
        [
            {
                "target": target,
                "window": "full_1s",
                "seed": seed,
                "best_epoch": best_epoch,
                "val_loss": validation_loss,
                "test_loss": test_loss,
                "top1": top1,
                "top5": top5,
                "elapsed_s": elapsed,
            }
        ]
    )

    sanity_result.to_csv(
        SANITY_CSV,
        index=False,
    )

    print(
        "SAVED",
        SANITY_CSV,
        flush=True,
    )


def main():

    print(
        "device:",
        DEVICE,
        "| seeds:",
        SEEDS,
        "| decoder: EEGProjectLayer",
        flush=True,
    )

    # Load prepared EEG.
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

    print(
        "train EEG:",
        train_eeg.shape,
        "| test EEG:",
        test_eeg.shape,
        flush=True,
    )

    # CHANGED: verify the new 250-point EEG shape.
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

    # -----------------------------------------------------------------------
    # Verify the decoder input for one 100 ms window
    # -----------------------------------------------------------------------

    n_channels = train_eeg.shape[1]

    start, end = WINDOWS["100_200"]

    realised = train_eeg[
        :1,
        :,
        start:end,
    ].shape

    input_dimension_100ms = (
        n_channels
        * (end - start)
    )

    print(
        f"decoder input per 100 ms window: "
        f"{n_channels} channels x "
        f"{end - start} samples = "
        f"{input_dimension_100ms} "
        f"(realised slice {realised})",
        flush=True,
    )

    assert n_channels == EXPECTED_NCHAN

    assert (
        end - start
    ) == WINDOW_SAMPLES

    # CHANGED: 63 × 25 = 1575.
    assert input_dimension_100ms == (
        EXPECTED_NCHAN
        * WINDOW_SAMPLES
    ) == 1575

    print(
        f"full 1-second decoder input: "
        f"{EXPECTED_NCHAN} channels x "
        f"{EXPECTED_NTIMES} samples = "
        f"{EXPECTED_NCHAN * EXPECTED_NTIMES}",
        flush=True,
    )

    # Discover sharp and foveated feature sets.
    targets = sorted(
        os.path.basename(feature_file)[
            :-len("__train.npy")
        ]
        for feature_file in glob.glob(
            os.path.join(
                FEAT_DIR,
                "*__train.npy",
            )
        )
    )

    n_decoders = (
        len(targets)
        * len(WINDOWS)
        * len(SEEDS)
    )

    print(
        f"{len(targets)} targets x "
        f"{len(WINDOWS)} windows x "
        f"{len(SEEDS)} seeds = "
        f"{n_decoders} decoders",
        flush=True,
    )

    raw = {}

    # CHANGED:
    # Store epoch information and save it with pandas at the end.
    epoch_rows = []

    experiment_start = time.time()

    for target_name in targets:

        train_target = np.load(
            os.path.join(
                FEAT_DIR,
                f"{target_name}__train.npy",
            )
        )

        test_target = np.load(
            os.path.join(
                FEAT_DIR,
                f"{target_name}__test.npy",
            )
        )

        assert train_target.shape[0] == train_eeg.shape[0]
        assert test_target.shape[0] == test_eeg.shape[0]

        for window_name, window in WINDOWS.items():

            for seed in SEEDS:

                (
                    validation_loss,
                    test_loss,
                    best_epoch,
                    top1,
                    top5,
                ), trace = train_one(
                    train_eeg,
                    train_target,
                    test_eeg,
                    test_target,
                    window,
                    seed,
                )

                raw[
                    (
                        target_name,
                        window_name,
                        seed,
                    )
                ] = (
                    best_epoch,
                    validation_loss,
                    test_loss,
                    top1,
                    top5,
                )

                # CHANGED: trace now contains dictionaries.
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
                    f"{target_name:28s} "
                    f"{window_name:9s} "
                    f"seed{seed} | "
                    f"ep{best_epoch:2d} "
                    f"val{validation_loss:.3f} "
                    f"test{test_loss:.3f} "
                    f"top1{top1 * 100:4.1f}% "
                    f"top5{top5 * 100:4.1f}% "
                    f"[{time.time() - experiment_start:.0f}s]",
                    flush=True,
                )

    # CHANGED: Save complete epoch log using pandas.
    epoch_dataframe = pd.DataFrame(
        epoch_rows
    )

    epoch_dataframe.to_csv(
        EPOCH_CSV,
        index=False,
    )

    print(
        "\nSAVED",
        EPOCH_CSV,
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Per-seed raw results
    # -----------------------------------------------------------------------

    with open(
        OUT_CSV_SEEDS,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "target",
                "window",
                "seed",
                "best_epoch",
                "val_loss",
                "test_loss",
                "top1",
                "top5",
            ]
        )

        for (
            target_name,
            window_name,
            seed,
        ), (
            best_epoch,
            validation_loss,
            test_loss,
            top1,
            top5,
        ) in raw.items():

            writer.writerow(
                [
                    target_name,
                    window_name,
                    seed,
                    best_epoch,
                    round(validation_loss, 4),
                    round(test_loss, 4),
                    round(top1, 4),
                    round(top5, 4),
                ]
            )

    print(
        "SAVED",
        OUT_CSV_SEEDS,
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Averaged over seeds
    # -----------------------------------------------------------------------

    with open(
        OUT_CSV,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "target",
                "window",
                "test_loss",
                "test_loss_std",
                "top1",
                "top5",
                "n_seeds",
            ]
        )

        for target_name in targets:

            for window_name in WINDOWS:

                test_losses = [
                    raw[
                        (
                            target_name,
                            window_name,
                            seed,
                        )
                    ][2]
                    for seed in SEEDS
                ]

                top1_values = [
                    raw[
                        (
                            target_name,
                            window_name,
                            seed,
                        )
                    ][3]
                    for seed in SEEDS
                ]

                top5_values = [
                    raw[
                        (
                            target_name,
                            window_name,
                            seed,
                        )
                    ][4]
                    for seed in SEEDS
                ]

                writer.writerow(
                    [
                        target_name,
                        window_name,
                        round(float(np.mean(test_losses)), 4),
                        round(float(np.std(test_losses)), 4),
                        round(float(np.mean(top1_values)), 4),
                        round(float(np.mean(top5_values)), 4),
                        len(SEEDS),
                    ]
                )

    print(
        "SAVED",
        OUT_CSV,
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Best-window stability across seeds
    # -----------------------------------------------------------------------

    print(
        "\n=== Best-window stability across seeds ===",
        flush=True,
    )

    for target_name in targets:

        best_windows = [
            list(WINDOWS.keys())[
                int(
                    np.argmin(
                        [
                            raw[
                                (
                                    target_name,
                                    window_name,
                                    seed,
                                )
                            ][2]
                            for window_name in WINDOWS
                        ]
                    )
                )
            ]
            for seed in SEEDS
        ]

        values, counts = np.unique(
            best_windows,
            return_counts=True,
        )

        mode = values[
            int(np.argmax(counts))
        ]

        agreement = counts.max() / len(SEEDS)

        print(
            f"{target_name:28s} "
            f"{str(best_windows):55s} "
            f"{mode} "
            f"({agreement * 100:.0f}%)",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Full one-second sanity check
    # -----------------------------------------------------------------------

    if os.path.exists(
        os.path.join(
            FEAT_DIR,
            f"{DEEP_LAYER}__train.npy",
        )
    ):

        print(
            "\n=== Full 1-second sanity check ===",
            flush=True,
        )

        sanity_check(
            train_eeg,
            test_eeg,
            DEEP_LAYER,
            seed=0,
        )

    # -----------------------------------------------------------------------
    # Deepest-layer top-1/top-5
    # -----------------------------------------------------------------------

    if any(
        key[0] == DEEP_LAYER
        for key in raw
    ):

        with open(
            OBJ_CSV,
            "w",
            newline="",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "target",
                    "window",
                    "top1",
                    "top5",
                ]
            )

            best = (
                None,
                -1.0,
                -1.0,
            )

            for window_name in WINDOWS:

                top1 = float(
                    np.mean(
                        [
                            raw[
                                (
                                    DEEP_LAYER,
                                    window_name,
                                    seed,
                                )
                            ][3]
                            for seed in SEEDS
                        ]
                    )
                )

                top5 = float(
                    np.mean(
                        [
                            raw[
                                (
                                    DEEP_LAYER,
                                    window_name,
                                    seed,
                                )
                            ][4]
                            for seed in SEEDS
                        ]
                    )
                )

                writer.writerow(
                    [
                        DEEP_LAYER,
                        window_name,
                        round(top1, 4),
                        round(top5, 4),
                    ]
                )

                if top1 > best[1]:

                    best = (
                        window_name,
                        top1,
                        top5,
                    )

        print(
            f"\n=== Object-perception accuracy: "
            f"{DEEP_LAYER} ===",
            flush=True,
        )

        print(
            f"best window {best[0]}: "
            f"top1={best[1] * 100:.2f}% "
            f"top5={best[2] * 100:.2f}%",
            flush=True,
        )

        print(
            "SAVED",
            OBJ_CSV,
            flush=True,
        )


if __name__ == "__main__":
    main()