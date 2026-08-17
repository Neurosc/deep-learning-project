"""
01_prepare_eeg.py
=================
Prepare every subject's already-preprocessed EEG for decoding.

Run AFTER:
    00_precheck.py

INPUT:
    ~/things_eeg/preprocessed_data/
        Preprocessed_data_250Hz_whiten/
        sub-XX/                       # one folder per subject (sub-01 .. sub-10)
            train.pt
            test.pt

The preprocessing pipeline has already:
    1. Removed the non-brain stim channel.
    2. Epoched the EEG from 0 to approximately 1 second.
    3. Applied baseline correction.
    4. Resampled the EEG from 1000 Hz to 250 Hz.
    5. Applied MVNN whitening.
    6. Averaged the repetitions belonging to each image.

Expected input shapes:
    train EEG: [16540, 63, 250]
    test EEG:  [200,   63, 250]

WHAT THIS SCRIPT DOES:
    1. Loads the new .pt files.
    2. Verifies EEG shapes, channels, sampling frequency and time points.
    3. Verifies that the data contain no NaN or infinite values.
    4. Saves NumPy arrays for the later extraction/alignment pipelines.

OUTPUT:
    ~/things_eeg/eeg_prepared/
        sub-XX_train_avg.npy          # one pair per subject
        sub-XX_test_avg.npy
        eeg_channels.npy              # subject-independent, saved once
        eeg_times.npy                 # subject-independent, saved once

Usage:
    python 01_prepare_eeg.py
"""

import os

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = os.path.expanduser("~/things_eeg")

# Subjects to prepare: all ten (sub-01 .. sub-10).                    # NEW: which subjects to loop over
SUBJECTS = range(1, 11)                                               # NEW: range end is exclusive -> 1..10

# Root of the preprocessed data; the per-subject folder is appended   # NEW: shared parent path,
# inside the loop below.                                              # NEW: subject folder added per-iteration
PREPROCESSED_ROOT = os.path.join(                                     # NEW
    ROOT,                                                             # NEW
    "preprocessed_data",                                             # NEW
    "Preprocessed_data_250Hz_whiten",                               # NEW
)                                                                    # NEW

OUT_DIR = os.path.join(
    ROOT,
    "eeg_prepared",
)

os.makedirs(
    OUT_DIR,
    exist_ok=True,
)


# ---------------------------------------------------------------------------
# Expected data properties
# ---------------------------------------------------------------------------

EXPECTED_SFREQ = 250
EXPECTED_NTIMES = 250
EXPECTED_NCHAN = 63

EXPECTED_NTRAIN = 16540
EXPECTED_NTEST = 200

# At 250 Hz:
# one sample = 4 ms
# 100 ms = 25 samples
WINDOW_SAMPLES = 25


# ---------------------------------------------------------------------------
# Load one split
# ---------------------------------------------------------------------------

def load_split(path, split):
    """
    Load one already-preprocessed EEG split.

    The EEG is already repetition-averaged and has shape:
        [image, channel, time]
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{split} EEG file does not exist: {path}"
        )

    loaded = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "eeg",
        "label",
        "img",
        "text",
        "ch_names",
        "times",
        "sfreq",
    }

    missing_keys = required_keys.difference(
        loaded.keys()
    )

    if missing_keys:
        raise KeyError(
            f"{split} file is missing keys: "
            f"{sorted(missing_keys)}"
        )

    eeg = loaded["eeg"]
    labels = loaded["label"]
    images = loaded["img"]
    texts = loaded["text"]
    ch_names = list(loaded["ch_names"])
    times = loaded["times"]
    sfreq = float(loaded["sfreq"])

    if not isinstance(eeg, torch.Tensor):
        raise TypeError(
            f"{split} EEG must be a torch.Tensor."
        )

    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            f"{split} labels must be a torch.Tensor."
        )

    if not isinstance(times, torch.Tensor):
        times = torch.as_tensor(
            times,
            dtype=torch.float32,
        )

    eeg = eeg.to(torch.float32)
    times = times.to(torch.float32)

    return {
        "eeg": eeg,
        "label": labels,
        "img": images,
        "text": texts,
        "ch_names": ch_names,
        "times": times,
        "sfreq": sfreq,
    }


# ---------------------------------------------------------------------------
# Prepare every subject
# ---------------------------------------------------------------------------

saved_shared = False                                                 # NEW: track whether the subject-independent
                                                                     # NEW: files (channels/times) were written yet

for sub in SUBJECTS:                                                 # NEW: one iteration per subject

    # This subject's folder of preprocessed .pt files.               # NEW
    eeg_dir = os.path.join(                                          # NEW
        PREPROCESSED_ROOT,                                          # NEW
        f"sub-{sub:02d}",                                          # NEW: zero-padded folder, e.g. sub-03
    )                                                              # NEW

    train_path = os.path.join(eeg_dir, "train.pt")                  # NEW: per-subject train file
    test_path = os.path.join(eeg_dir, "test.pt")                    # NEW: per-subject test file

    print(f"\n=== PREPARING SUB-{sub:02d} ===")                     # NEW: label each subject in the log

    # -----------------------------------------------------------------------
    # Load training and test data
    # -----------------------------------------------------------------------

    train = load_split(
        train_path,
        "train",
    )

    test = load_split(
        test_path,
        "test",
    )

    train_eeg = train["eeg"]
    test_eeg = test["eeg"]

    ch_names = train["ch_names"]
    times = train["times"]
    sfreq = train["sfreq"]

    # -----------------------------------------------------------------------
    # Shape checks
    # -----------------------------------------------------------------------

    print("train EEG:", tuple(train_eeg.shape))
    print("test EEG :", tuple(test_eeg.shape))
    print("channels :", len(ch_names))
    print("time pts :", len(times))
    print("sfreq    :", sfreq)

    if train_eeg.ndim != 3:
        raise ValueError(
            "Training EEG must have shape "
            "[image, channel, time], but found "
            f"{tuple(train_eeg.shape)}."
        )

    if test_eeg.ndim != 3:
        raise ValueError(
            "Test EEG must have shape "
            "[image, channel, time], but found "
            f"{tuple(test_eeg.shape)}."
        )

    assert train_eeg.shape == (
        EXPECTED_NTRAIN,
        EXPECTED_NCHAN,
        EXPECTED_NTIMES,
    ), (
        "Unexpected training EEG shape: "
        f"{tuple(train_eeg.shape)}"
    )

    assert test_eeg.shape == (
        EXPECTED_NTEST,
        EXPECTED_NCHAN,
        EXPECTED_NTIMES,
    ), (
        "Unexpected test EEG shape: "
        f"{tuple(test_eeg.shape)}"
    )

    # -----------------------------------------------------------------------
    # Metadata checks
    # -----------------------------------------------------------------------

    assert test["ch_names"] == ch_names, (
        "Training and test channel orders differ."
    )

    assert np.isclose(
        test["sfreq"],
        sfreq,
    ), (
        "Training and test sampling frequencies differ."
    )

    assert torch.allclose(
        test["times"],
        times,
    ), (
        "Training and test time vectors differ."
    )

    assert len(train["label"]) == EXPECTED_NTRAIN
    assert len(train["img"]) == EXPECTED_NTRAIN
    assert len(train["text"]) == EXPECTED_NTRAIN

    assert len(test["label"]) == EXPECTED_NTEST
    assert len(test["img"]) == EXPECTED_NTEST
    assert len(test["text"]) == EXPECTED_NTEST

    # -----------------------------------------------------------------------
    # Channel checks
    # -----------------------------------------------------------------------

    assert len(ch_names) == EXPECTED_NCHAN, (
        f"Expected {EXPECTED_NCHAN} channels, "
        f"but found {len(ch_names)}."
    )

    assert "stim" not in ch_names, (
        "The stim trigger channel is still present."
    )

    # -----------------------------------------------------------------------
    # Sampling checks
    # -----------------------------------------------------------------------

    times_np = times.cpu().numpy()

    dt = float(
        np.median(
            np.diff(times_np)
        )
    )

    calculated_sfreq = 1.0 / dt

    print(
        "times     :",
        round(float(times_np.min()), 3),
        "..",
        round(float(times_np.max()), 3),
        "seconds",
    )

    print(
        f"sampling  : dt={dt * 1000:.2f} ms "
        f"-> {calculated_sfreq:.1f} Hz"
    )

    assert abs(
        calculated_sfreq - EXPECTED_SFREQ
    ) < 1.0, (
        f"Expected approximately {EXPECTED_SFREQ} Hz, "
        f"but calculated {calculated_sfreq:.2f} Hz."
    )

    assert abs(
        sfreq - EXPECTED_SFREQ
    ) < 1.0, (
        f"Saved sfreq is {sfreq}, but "
        f"{EXPECTED_SFREQ} Hz was expected."
    )

    assert train_eeg.shape[-1] == EXPECTED_NTIMES
    assert test_eeg.shape[-1] == EXPECTED_NTIMES
    assert len(times_np) == EXPECTED_NTIMES

    # -----------------------------------------------------------------------
    # Data validity checks
    # -----------------------------------------------------------------------

    assert torch.isfinite(
        train_eeg
    ).all(), (
        "Training EEG contains NaN or infinite values."
    )

    assert torch.isfinite(
        test_eeg
    ).all(), (
        "Test EEG contains NaN or infinite values."
    )

    print(
        "train mean/std:",
        float(train_eeg.mean()),
        float(train_eeg.std()),
    )

    print(
        "test mean/std :",
        float(test_eeg.mean()),
        float(test_eeg.std()),
    )

    # -----------------------------------------------------------------------
    # Decoder input dimensionality
    # -----------------------------------------------------------------------

    window_inputs = (
        EXPECTED_NCHAN
        * WINDOW_SAMPLES
    )

    full_epoch_inputs = (
        EXPECTED_NCHAN
        * EXPECTED_NTIMES
    )

    print(
        f"100 ms window: "
        f"{EXPECTED_NCHAN} channels x "
        f"{WINDOW_SAMPLES} samples = "
        f"{window_inputs} decoder inputs"
    )

    print(
        f"full 1-second epoch: "
        f"{EXPECTED_NCHAN} channels x "
        f"{EXPECTED_NTIMES} samples = "
        f"{full_epoch_inputs} decoder inputs"
    )

    # -----------------------------------------------------------------------
    # Save arrays for later pipelines
    # -----------------------------------------------------------------------

    train_np = train_eeg.cpu().numpy().astype(
        np.float32,
    )

    test_np = test_eeg.cpu().numpy().astype(
        np.float32,
    )

    np.save(
        os.path.join(
            OUT_DIR,
            f"sub-{sub:02d}_train_avg.npy",                          # NEW: per-subject filename (was sub-01)
        ),
        train_np,
    )

    np.save(
        os.path.join(
            OUT_DIR,
            f"sub-{sub:02d}_test_avg.npy",                           # NEW: per-subject filename (was sub-01)
        ),
        test_np,
    )

    # Channel names and time vector are identical for every subject,  # NEW
    # so write them just once instead of ten times.                   # NEW
    if not saved_shared:                                             # NEW: only on the first subject processed
        np.save(                                                    # NEW
            os.path.join(                                           # NEW
                OUT_DIR,                                            # NEW
                "eeg_channels.npy",                                # NEW
            ),                                                     # NEW
            np.asarray(ch_names),                                  # NEW
        )                                                          # NEW

        np.save(                                                   # NEW
            os.path.join(                                          # NEW
                OUT_DIR,                                           # NEW
                "eeg_times.npy",                                  # NEW
            ),                                                    # NEW
            times_np.astype(np.float32),                          # NEW
        )                                                         # NEW

        saved_shared = True                                       # NEW: don't rewrite these for later subjects

    print("\nsaved ->", OUT_DIR)
    print(f"sub-{sub:02d}_train_avg.npy:", train_np.shape)          # NEW: filename now carries the subject id
    print(f"sub-{sub:02d}_test_avg.npy :", test_np.shape)           # NEW: filename now carries the subject id
    print("eeg_channels.npy    :", len(ch_names))
    print("eeg_times.npy       :", times_np.shape)


print("\nEEG PREPARATION COMPLETED SUCCESSFULLY")                    # runs once, after every subject is done
