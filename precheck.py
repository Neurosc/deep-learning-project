import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
import numpy as np, torch

print("=== numpy / data ===")
print("numpy:", np.__version__)
p = os.path.expanduser("~/things_eeg/eeg_data/sub-01_63/sub-01__63_channels/preprocessed_eeg_training.npy")
d = np.load(p, allow_pickle=True).item()
print("keys:", list(d.keys()))
print("train shape:", d["preprocessed_eeg_data"].shape, "| n_ch:", len(d["ch_names"]))

print("\n=== torch / gpu ===")
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
    try:
        x = torch.randn(2000, 2000, device="cuda")
        y = (x @ x).sum().item()
        torch.cuda.synchronize()
        print("GPU matmul OK, finite:", bool(np.isfinite(y)))
    except Exception as e:
        print("[!] GPU op FAILED:", repr(e))

print("\n=== open_clip ===")
import open_clip
print("open_clip:", open_clip.__version__)
for arch in ["RN50", "ViT-B-16", "ViT-L-14"]:
    try:
        m, _, _ = open_clip.create_model_and_transforms(arch, pretrained="openai")
        m.eval(); v = m.visual
        if hasattr(v, "transformer"):
            print(f"{arch}: ViT, resblocks =", len(v.transformer.resblocks))
        else:
            print(f"{arch}: RN50, children =", [k for k,_ in v.named_children()])
    except Exception as e:
        print(f"[!] {arch} failed:", repr(e))
print("\nDONE")
