"""
00_precheck.py
==============
One-off environment check. Run this FIRST, before the rest of the pipeline.

It verifies four things:
    1. The preprocessed EEG train and test files load correctly.
    2. The EEG tensors have valid shapes and contain no NaN or infinite values.
    3. The GPU is available and can run a computation.
    4. The three vision networks download and open correctly.

Usage:
    python 00_precheck.py
"""

import os

# Choose which physical GPU to use.
# This must be set before torch is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. EEG data check
# ---------------------------------------------------------------------------
print("=== EEG data ===")
print("numpy:", np.__version__)
print("torch:", torch.__version__)

train_path = (
    "/home/feyzanur_mbb/things_eeg/preprocessed_data/"
    "Preprocessed_data_250Hz_whiten/sub-01/train.pt"
)

test_path = (
    "/home/feyzanur_mbb/things_eeg/preprocessed_data/"
    "Preprocessed_data_250Hz_whiten/sub-01/test.pt"
)


def inspect_tensor(name, tensor):
    """
    Print the shape, dtype and basic statistics of one tensor.

    This also checks whether the tensor contains NaN or infinite values.
    """

    print(f"\n{name}")
    print("shape:", tuple(tensor.shape))
    print("dtype:", tensor.dtype)

    is_finite = torch.isfinite(tensor).all().item()
    print("all values finite:", bool(is_finite))

    if not is_finite:
        print("[!] Tensor contains NaN or infinite values")
        return

    # Statistics require a floating-point tensor.
    tensor_float = tensor.float()

    print("min :", tensor_float.min().item())
    print("max :", tensor_float.max().item())
    print("mean:", tensor_float.mean().item())
    print("std :", tensor_float.std().item())

    # At 250 Hz, one second contains 250 time points.
    if tensor.ndim >= 1:
        n_timepoints = tensor.shape[-1]
        print("last dimension:", n_timepoints, "time points")

        if n_timepoints == 250:
            print("time dimension is consistent with 1 second at 250 Hz")
        else:
            print(
                "note: the last dimension is not 250. "
                "This may still be correct if the saved epoch is not exactly 1 second."
            )


def inspect_file(label, path):
    """
    Load one .pt file and print its internal structure.

    The file may contain:
        - a single tensor
        - a dictionary containing several tensors
        - a list or tuple
    """

    print(f"\n=== {label} ===")
    print("path:", path)

    if not os.path.exists(path):
        print("[!] File does not exist")
        return

    loaded = torch.load(path, map_location="cpu")

    print("loaded object type:", type(loaded))

    if isinstance(loaded, torch.Tensor):
        inspect_tensor("tensor", loaded)

    elif isinstance(loaded, dict):
        print("keys:", list(loaded.keys()))

        for key, value in loaded.items():
            if isinstance(value, torch.Tensor):
                inspect_tensor(key, value)

            elif isinstance(value, np.ndarray):
                print(f"\n{key}")
                print("type: numpy.ndarray")
                print("shape:", value.shape)
                print("dtype:", value.dtype)

                value_tensor = torch.from_numpy(value)
                inspect_tensor(f"{key} converted to tensor", value_tensor)

            else:
                print(f"\n{key}")
                print("type:", type(value))
                print("value:", value)

    elif isinstance(loaded, (list, tuple)):
        print("number of elements:", len(loaded))

        for index, value in enumerate(loaded):
            if isinstance(value, torch.Tensor):
                inspect_tensor(f"element {index}", value)
            else:
                print(f"element {index}: type =", type(value))

    else:
        print("[!] Unrecognised file structure")


inspect_file("training data", train_path)
inspect_file("test data", test_path)


# ---------------------------------------------------------------------------
# 2. GPU check
# ---------------------------------------------------------------------------
print("\n=== torch / gpu ===")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        "device:",
        torch.cuda.get_device_name(0),
        "| capability:",
        torch.cuda.get_device_capability(0),
    )

    try:
        # Multiply two matrices on the GPU and verify that the result is valid.
        x = torch.randn(2000, 2000, device="cuda")
        y = (x @ x).sum().item()

        torch.cuda.synchronize()

        print("GPU matmul OK, finite:", bool(np.isfinite(y)))

    except Exception as e:
        print("[!] GPU operation failed:", repr(e))

else:
    print("[!] CUDA is not available")


# ---------------------------------------------------------------------------
# 3. Vision network check
# ---------------------------------------------------------------------------
print("\n=== open_clip ===")

import open_clip

print("open_clip:", open_clip.__version__)

for arch in ["RN50", "ViT-B-16", "ViT-L-14"]:
    try:
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            pretrained="openai",
        )

        model.eval()
        visual = model.visual

        if hasattr(visual, "transformer"):
            print(
                f"{arch}: ViT, resblocks =",
                len(visual.transformer.resblocks),
            )

        else:
            print(
                f"{arch}: RN50, children =",
                [name for name, _ in visual.named_children()],
            )

        # Delete the model before loading the next architecture.
        del model
        del visual

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"[!] {arch} failed:", repr(e))


print("\nDONE")
