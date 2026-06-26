# Methods — Temporal alignment of EEG and vision-network layers

This document records the design choices realised in the pipeline so that a
second reader can reproduce it. Concrete values are taken from the code as it
currently stands (`01_prepare_eeg.py`, `02_extract_features.py`,
`03_run_alignment.py`). Where a choice is still provisional it is marked as such.

## Dataset and provenance

The study uses the THINGS-EEG2 dataset (Gifford, Dwivedi, Roig, & Cichy, 2022),
in which participants viewed natural object images under rapid serial visual
presentation while 64-channel EEG was recorded. The dataset provides ten
subjects. Each image was presented several times — four repetitions per image in
the training set and eighty in the test set — and the public release supplies the
data already epoched and preprocessed. The training set comprises 16,540 distinct
images and the test set 200 distinct images, and the two sets share no images.
This disjoint structure is what makes the evaluation a 200-way zero-shot
retrieval problem: a decoder trained on the 16,540 training images is asked, for
each held-out test trial, to identify the correct image out of the 200 it has
never been trained on. The current pipeline is configured for subject 01; the
remaining subjects can be run by repointing the input paths, and extending the
analysis across subjects is the natural next step for establishing robustness.

## EEG preprocessing as realised

The preprocessing performed here is deliberately minimal because the public files
are already cleaned and epoched. For each subject the preprocessed array has the
shape image × repetition × channel × time. Two operations are applied. First, the
non-brain stimulus-trigger channel, labelled "stim", is removed, leaving the 63
genuine EEG channels. Second, the repetitions of each image are averaged, which
suppresses the trial-to-trial noise that dominates single EEG trials while
preserving the stimulus-locked response; this averaging is the reason the test
set, with eighty repetitions, yields cleaner signals than the training set with
four. The result is one averaged trace per image, of shape 16,540 × 63 × 100 for
training and 200 × 63 × 100 for test.

The temporal axis carries 100 samples. The data are sampled at the dataset's
native 100 Hz, that is one sample every 10 ms, and no further decimation or
resampling is applied on top of this; the preparation script verifies the rate
explicitly and halts if it is not 100 Hz, and confirms that 100 time points
remain. Stimulus onset falls at the sample nearest time zero, which is sample
index 20. The decoding analysis slices these 100 samples into nine non-
overlapping 100 ms windows: a pre-stimulus baseline spanning samples 10 to 20
(−100 to 0 ms), which serves as a control expected to perform at chance, followed
by eight post-stimulus windows covering 0–100, 100–200, 200–300, 300–400,
400–500, 500–600, 600–700 and 700–800 ms. Because each window is ten samples wide
and 63 channels are retained, the decoder input for one window is 63 × 10 = 630
dimensions; the training script asserts this 630-dimensional shape at startup.

## Vision networks

Three networks supply the target representations: a ResNet-50 (RN50) and two
Vision Transformers, ViT-B-16 and ViT-L-14. All three are the OpenCLIP
implementations loaded with the OpenAI pretrained weights, using the quickgelu
variants required by those weights. Using a single common training source (the
OpenAI image–text data) for all three architectures is a deliberate control: it
isolates architecture as the variable of interest and avoids confounds that would
arise from comparing networks trained on different datasets, at different
resolutions, or under different objectives.

## Six-layer feature extraction

For each network, representations are extracted at six depths spread from shallow
to deep, so that the targets span the low-level-to-high-level processing
hierarchy. For RN50 the six taps are the stem and the four residual stages
(layer1 through layer4), each pooled by global average pooling over its spatial
map, together with the final attention-pooling head; the attention-pooling head
is treated and labelled separately from the convolutional stages because it is
not a convolutional stage. For the Vision Transformers the six taps are evenly
spaced transformer blocks — blocks 2, 4, 6, 8, 10 and 12 for ViT-B-16, and blocks
4, 8, 12, 16, 20 and 24 for ViT-L-14 — and at each tapped block the class (CLS)
token is taken as the image's representation, since it is the token that
aggregates information across all patches.

Each extracted representation is reduced to a fixed length by principal component
analysis fitted on the training images and then applied unchanged to the test
images, so that no test information enters the reduction. The PCA target
dimension is currently set to 512 and is treated as a single named variable; this
value is provisional and pending confirmation with the supervisor. Before any
features are written, the extraction script records the native dimension of every
tapped layer — the CLS-token dimension for the Vision Transformers and the
channel count of each global-average-pooled stage for RN50 — and writes them,
alongside the post-PCA dimension, to `features/layer_dimensions.csv`, which should
be consulted for the authoritative per-layer figures. The script asserts that the
smallest native dimension is at least the PCA target and stops rather than
proceeding silently if any layer is narrower than the target, because PCA cannot
expand dimensionality and the target would then need to be revised. The shallow
RN50 stem in particular has a small native dimension, so this check is expected to
force an explicit decision about the PCA target before a full run.

## Decoder, objective and optimisation

For each combination of one feature set and one time window, a small decoder is
trained to map the EEG window onto that feature set. The decoder is a two-layer
multilayer perceptron: the 630-dimensional flattened EEG window passes through a
linear layer to a hidden width of 1024 with a GELU nonlinearity and dropout of
0.1, then a second linear layer to the feature dimension, with an additional
linear residual shortcut added to the output. Training uses a symmetric InfoNCE
contrastive objective at temperature 0.07, which, within each mini-batch, pulls
each EEG embedding towards its own image's feature and pushes it away from the
other images' features, in both the EEG-to-feature and feature-to-EEG directions.
Optimisation uses AdamW with a learning rate of 1e-3 and weight decay of 0.01, a
cosine-annealing learning-rate schedule over the full training length, a batch
size of 256, and 50 epochs.

Ten percent of the training images are held out as a validation set used only for
checkpoint selection: the epoch with the lowest validation loss is taken as the
final model, and the test loss and retrieval accuracy reported for a condition are
those measured at that epoch. To guard against results depending on a single
lucky initialisation, the entire procedure is repeated over five random seeds
(0 through 4); the seed fixes the weight initialisation, the validation split and
the batch order, and results are averaged across seeds with their spread retained.
In addition, the training and validation losses are recorded at every epoch and
persisted, so that the gap between them can be inspected directly as an
overfitting diagnostic.

## Evaluation protocol and metrics

The primary metric is the InfoNCE loss on the test set, reported for every
combination of layer and time window. Loss is used as the principal quantity
because it is continuous and comparable across the whole layer-by-time matrix.
Its use is justified by a sanity check that, for a single representative
condition, reports the validation loss, the test loss and the retrieval accuracy
together and confirms that loss and accuracy move in step. Retrieval accuracy is
computed by a single shared function as top-1 and top-5 within the 200-way
zero-shot setting: for each test trial the 200 candidate features are ranked by
cosine similarity to the decoder's prediction, and a hit is scored when the
correct image falls within the top one or top five. This same function provides
the object-perception number reported for the deepest, most semantic layer
(the final ViT-L-14 block), which is the figure intended for comparison with the
literature.

## Foveation condition

To probe the role of peripheral image degradation, a foveation transform is
applied to each stimulus before it enters the networks. The transform keeps a
central region sharp and progressively blurs the periphery by blending the sharp
image with a Gaussian-blurred copy under a radial weight that is zero inside a
central fovea and rises towards the edge. Its parameters — the fixation point, the
fraction of the image kept fully sharp, the maximum peripheral blur radius and the
rate at which blur grows with eccentricity — are exposed as named variables with
default values of a central fixation, a foveal fraction of 0.15, a maximum blur
radius of 6 pixels and a fall-off exponent of 2; these values are provisional and
pending confirmation with the supervisor. Foveation produces a parallel,
foveated set of features for every network and layer in addition to the sharp
set, so that the feature-extraction stage yields two variants of each condition.
The alignment training discovers and trains on both variants, and because their
results share the same results file and a consistent naming scheme — the foveated
targets carry a "fov" suffix on the otherwise identical name — the test loss of a
sharp condition and its foveated counterpart can be compared directly. Adding the
foveated variant roughly doubles the number of conditions and therefore the
training time, so a full run combining both variants and all seeds is intended to
be launched overnight.

## Provisional choices

Two choices in the above are explicitly provisional and should be settled with the
supervisor before the final run. The first is the PCA target dimension, currently
512, which interacts with the native dimensionality of the shallow layers and is
guarded by the assertion described above. The second is the set of foveation
parameters, whose defaults are reasonable but not yet calibrated to a specific
visual-angle specification. The concrete native dimensions of every layer are
produced by the extraction script in `features/layer_dimensions.csv`, and that
file rather than this narrative should be treated as the authoritative record of
the dimensionality table.
