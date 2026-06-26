"""
02_extract_features.py
======================
Turn every stimulus image into numerical feature vectors, taken from several
depths inside three vision networks. Run AFTER 01_prepare_eeg.py.

WHY: a vision network processes an image in stages, from low-level (edges,
textures) in its early layers to high-level (object identity, meaning) in its
late layers. By recording the image's representation at 6 depths in each of 3
networks, we get features that span this low-to-high range. Step 03 then tests
which depth the brain signal matches at each moment in time.

INPUT:
    ~/things_eeg/image_metadata.npy        list of all train/test images
    ~/things_eeg/images/{training,test}_images/<concept>/<file>.jpg

WHAT IT DOES, per network:
    1. Loads the network with OpenAI pretrained weights.
    2. Attaches "hooks" to 6 chosen layers (a hook copies that layer's output
       into a dict whenever an image passes through; it does not change the net).
    3. Runs every image through the network's vision tower in batches.
    4. Reduces each layer's output to one vector per image:
         - conv layers  -> Global Average Pooling (average over the spatial grid)
         - transformer  -> the CLS token (the token that summarises the whole image)
    5. Shrinks each vector to PCA_DIM numbers with PCA (fit on train, applied to test).

VARIANTS (point #6): each image is processed twice -- once sharp, once foveated
(centre-sharp, periphery-blurred) -- so every feature set has a sharp and a
foveated version that step 03 can compare.

OUTPUT (written to ~/things_eeg/features/):
    <network>__<layer>__train.npy         sharp     features  (16540, PCA_DIM)
    <network>__<layer>__test.npy
    <network>__<layer>__fov__train.npy     foveated  features  (16540, PCA_DIM)
    <network>__<layer>__fov__test.npy
    layer_dimensions.csv                   native vs post-PCA dimension per layer

Usage:
    python 02_extract_features.py
    # Tip: set LIMIT = 200 below for a quick smoke test, then set it back to None.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")   # choose GPU before importing torch

import csv
import numpy as np
import torch
import open_clip
from PIL import Image, ImageFilter
from sklearn.decomposition import PCA

ROOT    = os.path.expanduser("~/things_eeg")
IMG_DIR = os.path.join(ROOT, "images")
META    = os.path.join(ROOT, "image_metadata.npy")
OUT_DIR = os.path.join(ROOT, "features")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
BATCH   = 64      # number of images processed at once
LIMIT   = None    # set to a small number (e.g. 200) for a quick test; None = all images

# PCA target dimension -- the single place to change the post-PCA feature length.
# PENDING supervisor confirmation: shallow layers (e.g. RN50 stem) have a native
# dimension below this, which the dimensionality check below will flag (see #1).
PCA_DIM = 512

# --- Foveation parameters (point #6) --- PENDING supervisor confirmation -------
# A centre-sharp, periphery-degraded blur applied to each image before the network.
FOVEATE        = True          # also produce a foveated variant of every feature set
FOV_FIXATION   = (0.5, 0.5)    # gaze point (x, y) in relative image coords; (0.5,0.5)=centre
FOV_FOVEA_FRAC = 0.15          # radius (fraction of the image) kept fully sharp
FOV_MAX_BLUR   = 6.0           # Gaussian blur radius (px) reached at the far periphery
FOV_FALLOFF    = 2.0           # how fast blur grows with eccentricity (higher = sharper centre)
# ------------------------------------------------------------------------------

# The 6 layers ("taps") to record from each network, ordered shallow -> deep.
# RN50 taps are named stages; ViT taps are transformer block numbers (1-indexed).
NETWORKS = {
    "RN50-quickgelu":     ["stem", "layer1", "layer2", "layer3", "layer4", "attnpool"],
    "ViT-B-16-quickgelu": [2, 4, 6, 8, 10, 12],     # ViT-B-16 has 12 blocks
    "ViT-L-14-quickgelu": [4, 8, 12, 16, 20, 24],   # ViT-L-14 has 24 blocks
}


def load_metadata():
    """Return two lists of (concept, filename) pairs: one for train, one for test."""
    m = np.load(META, allow_pickle=True).item()
    train = list(zip(m["train_img_concepts"], m["train_img_files"]))
    test  = list(zip(m["test_img_concepts"],  m["test_img_files"]))
    return train, test


def image_path(split, concept, fname):
    """Build the full path to one image file."""
    sub = "training_images" if split == "train" else "test_images"
    return os.path.join(IMG_DIR, sub, concept, fname)


def foveate(img):
    """Return a centre-sharp, periphery-blurred version of a PIL RGB image.

    Blends the sharp image with a strongly blurred copy using a radial weight:
    weight 0 inside the central fovea, rising to 1 at the periphery. The fovea
    size, blur strength and fall-off are set by the FOV_* parameters above.
    """
    W, H   = img.size
    sharp  = np.asarray(img, dtype=np.float32)
    strong = np.asarray(img.filter(ImageFilter.GaussianBlur(FOV_MAX_BLUR)), dtype=np.float32)

    fx, fy = FOV_FIXATION
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = (xx - fx * W) / W
    dy = (yy - fy * H) / H
    ecc = np.sqrt(dx * dx + dy * dy)                 # eccentricity: 0 at fixation
    ecc = ecc / ecc.max()                            # normalise to [0, 1]

    # alpha = 0 within the fovea, ramps to 1 at the far periphery
    a = np.clip((ecc - FOV_FOVEA_FRAC) / (1.0 - FOV_FOVEA_FRAC), 0.0, 1.0) ** FOV_FALLOFF
    a = a[..., None]
    out = (1.0 - a) * sharp + a * strong
    return Image.fromarray(out.astype(np.uint8))


# Image variants to extract: name suffix -> transform applied before the network.
VARIANTS = {"": None}
if FOVEATE:
    VARIANTS["__fov"] = foveate


def reduce_activation(act, bs):
    """Collapse one layer's raw output into a single vector per image.

    Three cases, depending on the output's shape:
      - 4-D [batch, channels, H, W]  (a conv feature map)
            -> Global Average Pooling: average over H and W -> [batch, channels]
      - 3-D (transformer tokens)
            -> take the CLS token. The `if` handles both possible orderings
               ([seq, batch, dim] vs [batch, seq, dim]) so we always pick the
               CLS token and get [batch, dim].
      - already 2-D [batch, dim]  (e.g. RN50 attnpool output)
            -> pass through unchanged.
    """
    if act.dim() == 4:
        return act.mean(dim=(2, 3))
    if act.dim() == 3:
        return act[0] if act.shape[1] == bs else act[:, 0]
    return act


def build_hooks(model, arch, layers):
    """Attach forward hooks to the chosen layers.

    Returns:
        feats   : dict that the hooks fill with each layer's latest output
        handles : the hook handles (so we can remove them afterwards)
        names   : the string name used for each tapped layer
    """
    feats, handles = {}, []

    def mk(name):
        # A hook that stores this layer's output into feats[name].
        def hook(mod, inp, out):
            feats[name] = out.detach()
        return hook

    if arch.startswith("RN50"):
        v = model.visual
        # NB: RN50 'stem' and 'layer1..4' are conv stages (GAP-pooled); 'attnpool'
        # is the attention-pooling head and is therefore labelled separately.
        mods = {"stem": v.avgpool, "layer1": v.layer1, "layer2": v.layer2,
                "layer3": v.layer3, "layer4": v.layer4, "attnpool": v.attnpool}
        names = layers
        for n in names:
            handles.append(mods[n].register_forward_hook(mk(n)))
    else:
        # Vision Transformer: tap specific transformer blocks (1-indexed -> i-1).
        blocks = model.visual.transformer.resblocks
        names = [f"block{i}" for i in layers]
        for i, n in zip(layers, names):
            handles.append(blocks[i - 1].register_forward_hook(mk(n)))

    return feats, handles, names


def probe_native_dims(train_items):
    """Point #1: record every layer's native dimension (just before PCA).

    Runs a tiny 2-image batch through each network and reads the pooled size of
    each tapped layer. For ViT taps this is the CLS-token dimension; for RN50 it
    is the channel count of the GAP-pooled conv stage (with attnpool separate).
    Returns a list of (network, layer, native_dim, post_pca_dim) rows.
    """
    sample = train_items[:2]
    rows = []
    for arch, layers in NETWORKS.items():
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained="openai")
        model.eval().to(DEVICE)
        feats, handles, names = build_hooks(model, arch, layers)

        imgs = [preprocess(Image.open(image_path("train", c, f)).convert("RGB"))
                for c, f in sample]
        x = torch.stack(imgs).to(DEVICE)
        feats.clear()
        with torch.no_grad():
            model.visual(x)
        bs = x.shape[0]

        tag = arch.replace("-quickgelu", "")
        for nm in names:
            native = int(reduce_activation(feats[nm], bs).shape[1])
            rows.append((tag, str(nm), native, min(PCA_DIM, native)))

        for h in handles:
            h.remove()
        del model
        torch.cuda.empty_cache()
    return rows


@torch.no_grad()   # we are only reading the network, never training it
def extract_split(model, preprocess, arch, items, split, names, feats, transform=None):
    """Run all images of one split through the network and collect the tapped outputs.

    If `transform` is given (e.g. foveate), it is applied to each image before
    preprocessing. The key trick: model.visual(x) is run only to make the hooks
    fire; its return value is ignored, the data lands in `feats` as a side effect.
    """
    if LIMIT:
        items = items[:LIMIT]
    n = len(items)
    buf = {nm: [] for nm in names}   # one collecting list per tapped layer

    for s in range(0, n, BATCH):
        batch = items[s:s + BATCH]
        imgs = []
        for c, f in batch:
            img = Image.open(image_path(split, c, f)).convert("RGB")
            if transform is not None:
                img = transform(img)
            imgs.append(preprocess(img))
        x = torch.stack(imgs).to(DEVICE)

        feats.clear()        # clear last batch's outputs
        model.visual(x)      # run the vision tower -> hooks fill `feats`
        bs = x.shape[0]      # actual batch size (last batch may be smaller)

        for nm in names:
            buf[nm].append(reduce_activation(feats[nm], bs).float().cpu().numpy())

        if (s // BATCH) % 20 == 0:
            print(f"  [{arch}/{split}] {s + bs}/{n}", flush=True)

    return {nm: np.concatenate(buf[nm], 0) for nm in names}


def main():
    print("device:", DEVICE, "| LIMIT:", LIMIT, "| variants:", list(VARIANTS), flush=True)
    train_items, test_items = load_metadata()
    print(f"train {len(train_items)} | test {len(test_items)}", flush=True)

    # ---- Point #1: dimensionality table + PCA sanity check (before any heavy work) ----
    dim_rows = probe_native_dims(train_items)
    dim_csv = os.path.join(OUT_DIR, "layer_dimensions.csv")
    with open(dim_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network", "layer", "native_dim", "post_pca_dim"])
        w.writerows(dim_rows)

    print("\n=== Layer dimensionality (native -> post-PCA) ===", flush=True)
    print(f"{'network':14s} {'layer':10s} {'native':>7s} {'post_pca':>9s}", flush=True)
    for net, layer, native, post in dim_rows:
        print(f"{net:14s} {layer:10s} {native:7d} {post:9d}", flush=True)
    print("saved ->", dim_csv, flush=True)

    # Stop (do not silently proceed) if any layer is narrower than the PCA target,
    # because PCA cannot expand dimensions and the PCA target then needs revising.
    smallest = min(native for _, _, native, _ in dim_rows)
    offenders = [(n, l, d) for n, l, d, _ in dim_rows if d < PCA_DIM]
    if offenders:
        msg = ", ".join(f"{n}/{l}={d}" for n, l, d in offenders)
        raise SystemExit(
            f"\n[STOP] PCA_DIM={PCA_DIM} exceeds the native dimension of: {msg}.\n"
            f"        Smallest native dimension is {smallest}. PCA cannot expand "
            f"dimensions, so either lower PCA_DIM to <= {smallest}, or drop these "
            f"shallow layers. Set the PCA target with the supervisor, then re-run.")

    # ---- Feature extraction (sharp + foveated variants) ----
    for arch, layers in NETWORKS.items():
        print(f"\n=== {arch} ===", flush=True)
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained="openai")
        model.eval().to(DEVICE)
        feats, handles, names = build_hooks(model, arch, layers)
        tag = arch.replace("-quickgelu", "")

        for suffix, transform in VARIANTS.items():
            label = "sharp" if suffix == "" else "foveated"
            print(f"  -- variant: {label} --", flush=True)
            tr = extract_split(model, preprocess, arch, train_items, "train", names, feats, transform)
            te = extract_split(model, preprocess, arch, test_items,  "test",  names, feats, transform)

            # PCA fit on train, applied to test (no leakage). Saved per variant.
            for nm in names:
                pca = PCA(n_components=min(PCA_DIM, tr[nm].shape[1], tr[nm].shape[0]))
                Xtr = pca.fit_transform(tr[nm]).astype(np.float32)
                Xte = pca.transform(te[nm]).astype(np.float32)
                np.save(os.path.join(OUT_DIR, f"{tag}__{nm}{suffix}__train.npy"), Xtr)
                np.save(os.path.join(OUT_DIR, f"{tag}__{nm}{suffix}__test.npy"),  Xte)
                print(f"    {tag}__{nm}{suffix}: train{Xtr.shape} test{Xte.shape} "
                      f"(raw {tr[nm].shape[1]}, var {pca.explained_variance_ratio_.sum():.3f})",
                      flush=True)
            del tr, te

        for h in handles:
            h.remove()
        del model
        torch.cuda.empty_cache()

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
