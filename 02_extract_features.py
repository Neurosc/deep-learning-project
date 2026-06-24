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
    2. Attaches "hooks" to 6 chosen layers. A hook copies that layer's output
       into a dict whenever an image passes through (it does not change the network).
    3. Runs every image through the network's vision tower in batches.
    4. Reduces each layer's output to one vector per image:
         - conv layers  -> Global Average Pooling (average over the spatial grid)
         - transformer  -> the CLS token (the token that summarises the whole image)
    5. Shrinks each vector to 512 numbers with PCA (fit on train, applied to test).

OUTPUT (written to ~/things_eeg/features/):
    <network>__<layer>__train.npy   shape (16540, 512)
    <network>__<layer>__test.npy    shape (200,   512)
    3 networks x 6 layers x 2 splits = 36 files.

Usage:
    python 02_extract_features.py
    # Tip: set LIMIT = 200 below for a quick smoke test, then set it back to None.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")   # choose GPU before importing torch

import numpy as np
import torch
import open_clip
from PIL import Image
from sklearn.decomposition import PCA

ROOT    = os.path.expanduser("~/things_eeg")
IMG_DIR = os.path.join(ROOT, "images")
META    = os.path.join(ROOT, "image_metadata.npy")
OUT_DIR = os.path.join(ROOT, "features")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
BATCH   = 64      # number of images processed at once
PCA_DIM = 512     # final length of every feature vector
LIMIT   = None    # set to a small number (e.g. 200) for a quick test; None = all images

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


@torch.no_grad()   # we are only reading the network, never training it
def extract_split(model, preprocess, arch, items, split, names, feats):
    """Run all images of one split through the network and collect the tapped outputs.

    Note the key trick: model.visual(x) is run only to make the hooks fire.
    Its return value is ignored; the data we want lands in `feats` as a side effect.
    """
    if LIMIT:
        items = items[:LIMIT]
    n = len(items)
    buf = {nm: [] for nm in names}   # one collecting list per tapped layer

    for s in range(0, n, BATCH):
        batch = items[s:s + BATCH]
        # Open and preprocess this batch of images.
        imgs = [preprocess(Image.open(image_path(split, c, f)).convert("RGB"))
                for c, f in batch]
        x = torch.stack(imgs).to(DEVICE)

        feats.clear()        # clear last batch's outputs
        model.visual(x)      # run the vision tower -> hooks fill `feats`
        bs = x.shape[0]      # actual batch size (last batch may be smaller)

        # Pool each tapped layer's output and move it to CPU as a NumPy array.
        for nm in names:
            buf[nm].append(reduce_activation(feats[nm], bs).float().cpu().numpy())

        if (s // BATCH) % 20 == 0:
            print(f"  [{arch}/{split}] {s + bs}/{n}", flush=True)

    # Glue the per-batch chunks into one array per layer: shape (n_images, dim).
    return {nm: np.concatenate(buf[nm], 0) for nm in names}


def main():
    print("device:", DEVICE, "| LIMIT:", LIMIT, flush=True)
    train_items, test_items = load_metadata()
    print(f"train {len(train_items)} | test {len(test_items)}", flush=True)

    for arch, layers in NETWORKS.items():
        print(f"\n=== {arch} ===", flush=True)

        # Load the network (vision + text), keep it in eval mode on the GPU.
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained="openai")
        model.eval().to(DEVICE)

        # Attach hooks, extract both splits, then remove the hooks.
        feats, handles, names = build_hooks(model, arch, layers)
        tr = extract_split(model, preprocess, arch, train_items, "train", names, feats)
        te = extract_split(model, preprocess, arch, test_items,  "test",  names, feats)
        for h in handles:
            h.remove()

        # PCA-reduce each layer to 512 dims and save. PCA is FIT on train only,
        # then APPLIED to test, so no test information leaks into the reduction.
        tag = arch.replace("-quickgelu", "")
        for nm in names:
            pca = PCA(n_components=min(PCA_DIM, tr[nm].shape[1], tr[nm].shape[0]))
            Xtr = pca.fit_transform(tr[nm]).astype(np.float32)
            Xte = pca.transform(te[nm]).astype(np.float32)
            np.save(os.path.join(OUT_DIR, f"{tag}__{nm}__train.npy"), Xtr)
            np.save(os.path.join(OUT_DIR, f"{tag}__{nm}__test.npy"),  Xte)
            print(f"  {tag}__{nm}: train{Xtr.shape} test{Xte.shape} "
                  f"(raw {tr[nm].shape[1]}, var {pca.explained_variance_ratio_.sum():.3f})",
                  flush=True)

        # Free the network before loading the next one.
        del model, tr, te
        torch.cuda.empty_cache()

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
