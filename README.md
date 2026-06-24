# Temporal Alignment of EEG and Vision-Network Layers

Deep Learning project — Justus-Liebig-Universität Gießen.

## What this project does

We test whether the brain processes a glimpsed image in stages — simple visual
features first, meaning later — by aligning EEG over time with the layer hierarchy
of vision networks.

For each image we record its representation at **6 depths** inside **3 networks**
(`RN50`, `ViT-B-16`, `ViT-L-14`, all with OpenAI weights, so architecture is the
only variable). We then train small decoders to map a subject's EEG, in **9 time
windows** after the image appears, onto each of those representations, and measure
how well each window matches each depth. Early windows matching early layers and
late windows matching deep layers would indicate a temporal hierarchy.

Dataset: **THINGS-EEG2** (Gifford et al., 2022), 63-channel version, subject 01.

## Pipeline (run in order)

| Step | File | Reads | Writes |
|------|------|-------|--------|
| 0 | `00_precheck.py` | — | nothing (sanity check only) |
| 1 | `01_prepare_eeg.py` | `eeg_data/` | `eeg_prepared/` |
| 2 | `02_extract_features.py` | `images/`, `image_metadata.npy` | `features/` |
| 3 | `03_run_alignment.py` | `eeg_prepared/`, `features/` | `results/alignment_sub-01.csv` |
| 4 | `04_visualisation.ipynb` | `results/` | figures |

```bash
python 00_precheck.py          # confirm data, GPU, and networks are available
python 01_prepare_eeg.py       # clean + average EEG  -> eeg_prepared/
python 02_extract_features.py  # images -> layer features (slow) -> features/
python 03_run_alignment.py     # train decoders -> results/alignment_sub-01.csv
# then open 04_visualisation.ipynb and run all cells
```

Steps 2 and 3 are long-running; on the server, launch them in the background, e.g.:
```bash
nohup python 02_extract_features.py > extract.log 2>&1 &
```

## Expected data layout

The code expects the dataset under `~/things_eeg/` (this folder also holds the
code). Only the code is tracked in git; the data is not (see `.gitignore`).

```
~/things_eeg/
├── image_metadata.npy
├── images/{training_images,test_images}/<concept>/<file>.jpg
├── eeg_data/sub-01_63/sub-01__63_channels/preprocessed_eeg_{training,test}.npy
├── eeg_prepared/      # created by step 01
├── features/          # created by step 02
└── results/           # created by step 03
```

## Environment

```bash
conda create -n deepl python=3.10
conda activate deepl
pip install torch torchvision open_clip_torch
pip install numpy scikit-learn pillow
pip install jupyter matplotlib pandas    # for the visualisation notebook
```

## Notes for a replicator

- **GPU selection:** each script sets `CUDA_VISIBLE_DEVICES = "3"` at the top.
  Change this to the GPU index you want to use (e.g. `"0"`).
- **Smoke test:** in `02_extract_features.py`, set `LIMIT = 200` to process only
  the first 200 images and confirm the whole pipeline runs, then set it back to
  `None` for the full run.
- **Same seed everywhere:** `03_run_alignment.py` fixes the random seed for every
  decoder, so the train/validation split and initialisation are identical across
  all conditions — differences in score come from the data, not from randomness.
- **No data leakage:** PCA (step 02) and the validation split (step 03) are both
  fit on training data only and then applied to the test set.

## Currently implemented vs. planned

Implemented: multi-layer feature extraction (RN50 / ViT-B-16 / ViT-L-14, OpenAI
weights, GAP / CLS pooling), repetition-averaged EEG, 9 time windows, the
alignment decoding for subject 01.

Planned (see project roadmap): binarized-edge and highlighted-edge conditions,
foveated-blur condition, ImageNet-trained encoders for object decoding, multiple
random seeds, task-specific metrics (edge F1 / object accuracy / semantic
sentence-level), and re-running RSA on the updated pipeline.
```
