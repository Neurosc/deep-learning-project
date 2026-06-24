
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

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
BATCH   = 64
PCA_DIM = 512
LIMIT   = None      # <- önce smoke test: 200 görselle dene. Her şey doğruysa None yap, tam çalıştır.

# her ağ için 6 tap katmanı
NETWORKS = {
    "RN50-quickgelu":     ["stem", "layer1", "layer2", "layer3", "layer4", "attnpool"],
    "ViT-B-16-quickgelu": [2, 4, 6, 8, 10, 12],      # resblock çıkışı (1-indexed)
    "ViT-L-14-quickgelu": [4, 8, 12, 16, 20, 24],
}

def load_metadata():
    m = np.load(META, allow_pickle=True).item()
    train = list(zip(m["train_img_concepts"], m["train_img_files"]))
    test  = list(zip(m["test_img_concepts"],  m["test_img_files"]))
    return train, test

def image_path(split, concept, fname):
    sub = "training_images" if split == "train" else "test_images"
    return os.path.join(IMG_DIR, sub, concept, fname)

def reduce_activation(act, bs):
    if act.dim() == 4:                       # [B,C,H,W] -> GAP
        return act.mean(dim=(2, 3))
    if act.dim() == 3:                       # transformer token'ları
        return act[0] if act.shape[1] == bs else act[:, 0]   # CLS token (LND/NLD)
    return act                               # [B,D] zaten pooled (attnpool)

def build_hooks(model, arch, layers):
    feats, handles = {}, []
    def mk(name):
        def hook(mod, inp, out): feats[name] = out.detach()
        return hook
    if arch.startswith("RN50"):
        v = model.visual
        mods = {"stem": v.avgpool, "layer1": v.layer1, "layer2": v.layer2,
                "layer3": v.layer3, "layer4": v.layer4, "attnpool": v.attnpool}
        names = layers
        for n in names: handles.append(mods[n].register_forward_hook(mk(n)))
    else:
        blocks = model.visual.transformer.resblocks
        names = [f"block{i}" for i in layers]
        for i, n in zip(layers, names): handles.append(blocks[i-1].register_forward_hook(mk(n)))
    return feats, handles, names

@torch.no_grad()
def extract_split(model, preprocess, arch, items, split, names, feats):
    if LIMIT: items = items[:LIMIT]
    n = len(items)
    buf = {nm: [] for nm in names}
    for s in range(0, n, BATCH):
        batch = items[s:s+BATCH]
        imgs = [preprocess(Image.open(image_path(split, c, f)).convert("RGB")) for c, f in batch]
        x = torch.stack(imgs).to(DEVICE)
        feats.clear()
        model.visual(x)
        bs = x.shape[0]
        for nm in names:
            buf[nm].append(reduce_activation(feats[nm], bs).float().cpu().numpy())
        if (s // BATCH) % 20 == 0:
            print(f"  [{arch}/{split}] {s+bs}/{n}", flush=True)
    return {nm: np.concatenate(buf[nm], 0) for nm in names}

def main():
    print("device:", DEVICE, "| LIMIT:", LIMIT, flush=True)
    train_items, test_items = load_metadata()
    print(f"train {len(train_items)} | test {len(test_items)}", flush=True)

    for arch, layers in NETWORKS.items():
        print(f"\n=== {arch} ===", flush=True)
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained="openai")
        model.eval().to(DEVICE)
        feats, handles, names = build_hooks(model, arch, layers)

        tr = extract_split(model, preprocess, arch, train_items, "train", names, feats)
        te = extract_split(model, preprocess, arch, test_items,  "test",  names, feats)
        for h in handles: h.remove()

        tag = arch.replace("-quickgelu", "")
        for nm in names:
            pca = PCA(n_components=min(PCA_DIM, tr[nm].shape[1], tr[nm].shape[0]))
            Xtr = pca.fit_transform(tr[nm]).astype(np.float32)
            Xte = pca.transform(te[nm]).astype(np.float32)
            np.save(os.path.join(OUT_DIR, f"{tag}__{nm}__train.npy"), Xtr)
            np.save(os.path.join(OUT_DIR, f"{tag}__{nm}__test.npy"),  Xte)
            print(f"  {tag}__{nm}: train{Xtr.shape} test{Xte.shape} "
                  f"(raw {tr[nm].shape[1]}, var {pca.explained_variance_ratio_.sum():.3f})", flush=True)

        del model, tr, te
        torch.cuda.empty_cache()
    print("\nALL DONE", flush=True)

if __name__ == "__main__":
    main()
