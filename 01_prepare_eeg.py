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

# ---------------------------------------------------------------------------
# Downsampling verification (point #2)
# ---------------------------------------------------------------------------
# The THINGS-EEG2 preprocessed files are already at 100 Hz. This script does NOT
# decimate or resample on top of that -- it only drops the stim channel and
# averages repetitions. We verify the native rate here so any accidental
# resampling (introduced upstream or by a future edit) is caught immediately.
EXPECTED_SFREQ  = 100   # Hz, the dataset's native preprocessed sampling rate
EXPECTED_NTIMES = 100   # time points per trial
EXPECTED_NCHAN  = 63    # channels remaining after dropping 'stim'
WINDOW_SAMPLES  = 10    # samples per 100 ms window (100 Hz -> 10 ms per sample)

dt      = float(np.median(np.diff(times)))   # seconds between consecutive samples
sfreq   = 1.0 / dt
n_times = train_avg.shape[2]
print(f"sampling : dt={dt*1000:.2f} ms -> {sfreq:.1f} Hz | n_times={n_times}")

assert abs(sfreq - EXPECTED_SFREQ) < 1.0, \
    f"expected ~{EXPECTED_SFREQ} Hz but found {sfreq:.2f} Hz -- has the data been resampled?"
assert n_times == EXPECTED_NTIMES, \
    f"expected {EXPECTED_NTIMES} time points but found {n_times} -- has the data been decimated?"
assert train_avg.shape[1] == EXPECTED_NCHAN, \
    f"expected {EXPECTED_NCHAN} channels but found {train_avg.shape[1]}"

# A 100 ms decoder window therefore carries 63 channels x 10 samples = 630 inputs.
print(f"verified : {EXPECTED_NCHAN} channels x {WINDOW_SAMPLES} samples = "
      f"{EXPECTED_NCHAN * WINDOW_SAMPLES} decoder inputs per 100 ms window (no resampling)")

# Save the prepared arrays for steps 02/03 to use.
np.save(os.path.join(OUT_DIR, "sub-01_train_avg.npy"), train_avg)
np.save(os.path.join(OUT_DIR, "sub-01_test_avg.npy"),  test_avg)
np.save(os.path.join(OUT_DIR, "eeg_channels.npy"), np.array(ch))
np.save(os.path.join(OUT_DIR, "eeg_times.npy"),    times)
print("saved ->", OUT_DIR)
