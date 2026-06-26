"""
00_precheck.py
==============
One-off environment check. Run this FIRST, before the rest of the pipeline.

It verifies three things:
    1. The EEG data loads and has the expected shape.
    2. The GPU is available and can run a computation.
    3. The three vision networks download and open correctly.

If every section prints without an "[!]" error line, the environment is ready.

Usage:
    python 00_precheck.py
"""

import os
# Choose which physical GPU to use. This MUST be set before torch is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. EEG data check
# ---------------------------------------------------------------------------
print("=== numpy / data ===")
print("numpy:", np.__version__)

# One subject's preprocessed EEG file (63-channel).
eeg_path = os.path.expanduser(
    "~/things_eeg/eeg_data/sub-01_63/sub-01__63_channels/preprocessed_eeg_training.npy"
)
data = np.load(eeg_path, allow_pickle=True).item()   # the file stores a Python dict
print("keys:", list(data.keys()))

# 'preprocessed_eeg_data' has shape [image, repetition, channel, time].
print("train shape:", data["preprocessed_eeg_data"].shape,
      "| n_channels:", len(data["ch_names"]))


# ---------------------------------------------------------------------------
# 2. GPU check
# ---------------------------------------------------------------------------
print("\n=== torch / gpu ===")
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          "| capability:", torch.cuda.get_device_capability(0))
    try:
        # Multiply two large matrices on the GPU and confirm the result is a valid number.
        x = torch.randn(2000, 2000, device="cuda")
        y = (x @ x).sum().item()
        torch.cuda.synchronize()
        print("GPU matmul OK, finite:", bool(np.isfinite(y)))
    except Exception as e:
        print("[!] GPU op FAILED:", repr(e))


# ---------------------------------------------------------------------------
# 3. Vision network check
# ---------------------------------------------------------------------------
# Load each network with the OpenAI pretrained weights and print its structure.
# All three use the SAME 'openai' weights, so architecture is the only thing
# that differs between them (important for a fair later comparison).
print("\n=== open_clip ===")
import open_clip
print("open_clip:", open_clip.__version__)

for arch in ["RN50", "ViT-B-16", "ViT-L-14"]:
    try:
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained="openai")
        model.eval()
        visual = model.visual
        if hasattr(visual, "transformer"):
            # Vision Transformer: report how many transformer blocks it has.
            print(f"{arch}: ViT, resblocks =", len(visual.transformer.resblocks))
        else:
            # ResNet: list its top-level stages (stem, layer1..4, attnpool).
            print(f"{arch}: RN50, children =", [k for k, _ in visual.named_children()])
    except Exception as e:
        print(f"[!] {arch} failed:", repr(e))

print("\nDONE")
