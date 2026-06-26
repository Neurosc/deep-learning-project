"""
01_prepare_eeg.py
=================
Prepare one subject's EEG for decoding. Run AFTER 00_precheck.py.

INPUT (from the THINGS-EEG2 dataset):
    ~/things_eeg/eeg_data/sub-01_63/sub-01__63_channels/preprocessed_eeg_training.npy
    ~/things_eeg/eeg_data/sub-01_63/sub-01__63_channels/preprocessed_eeg_test.npy
    Each file stores a dict whose 'preprocessed_eeg_data' array has shape
    [image, repetition, channel, time].

WHAT IT DOES:
    1. Drops the non-brain 'stim' trigger channel (64 -> 63 channels). 
    that one is to note when things happened during the experiment — for example, 
    the exact moment an image appeared on the screen. 
    2. Averages over the repetitions of each image (each image was shown several
       times; averaging cancels random noise and keeps the brain's response to
       the image).

OUTPUT (written to ~/things_eeg/eeg_prepared/):
    sub-01_train_avg.npy   shape (16540, 63, 100)   one clean trace per training image
    sub-01_test_avg.npy    shape (200,   63, 100)   one clean trace per test image
    eeg_channels.npy       the 63 channel names
    eeg_times.npy          the 100 time points, in seconds (used to define windows)

Note: the EEG has 100 time points spaced 10 ms apart, with the image appearing
at the sample closest to time 0 (the "onset"). Step 03 slices these 100 points
into time windows (0-100 ms, 100-200 ms, ...).

Usage:
    python 01_prepare_eeg.py
"""

import os
import numpy as np

ROOT    = os.path.expanduser("~/things_eeg")
EEG_DIR = os.path.join(ROOT, "eeg_data/sub-01_63/sub-01__63_channels")
OUT_DIR = os.path.join(ROOT, "eeg_prepared")
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_average(split):
    """Load one split ('train' or 'test'), drop the stim channel, average repetitions."""
    fname = "preprocessed_eeg_training.npy" if split == "train" else "preprocessed_eeg_test.npy"
    d = np.load(os.path.join(EEG_DIR, fname), allow_pickle=True).item()

    data     = d["preprocessed_eeg_data"]   # [image, repetition, 64, time]
    ch_names = list(d["ch_names"])
    times    = np.asarray(d["times"])        # the time (in seconds) of each sample

    # 1) Drop the 'stim' channel: it is a trigger marker, not brain activity.
    keep   = [i for i, c in enumerate(ch_names) if c != "stim"]
    data   = data[:, :, keep, :]             # -> [image, repetition, 63, time]
    eeg_ch = [ch_names[i] for i in keep]

    # 2) Average over the repetition axis (axis=1) -> one trace per image.
    data_avg = data.mean(axis=1).astype(np.float32)   # -> [image, 63, time]
    return data_avg, eeg_ch, times


# Prepare both splits.
train_avg, ch, times = load_and_average("train")
test_avg,  _,  _     = load_and_average("test")

print("train_avg:", train_avg.shape)   # (16540, 63, 100)
print("test_avg :", test_avg.shape)    # (200, 63, 100)
print("channels :", len(ch), "| stim removed:", "stim" not in ch)
print("times    :", round(float(times.min()), 3), "..", round(float(times.max()), 3),
      "s | onset idx:", int(np.argmin(np.abs(times))))   # onset = sample nearest time 0

# Save the prepared arrays for steps 02/03 to use.
np.save(os.path.join(OUT_DIR, "sub-01_train_avg.npy"), train_avg)
np.save(os.path.join(OUT_DIR, "sub-01_test_avg.npy"),  test_avg)
np.save(os.path.join(OUT_DIR, "eeg_channels.npy"), np.array(ch))
np.save(os.path.join(OUT_DIR, "eeg_times.npy"),    times)
print("saved ->", OUT_DIR)
