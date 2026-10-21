# HipMRI 2D Improved U-Net - Panupong

COMP3710 individual project: compare a standard 2D U-Net baseline with the course-specified 2D Improved U-Net on HipMRI. The selected difficulty is Normal.

## Current development status

- Project planning and the initial feasibility review have been prepared.
- VS Code execution through Google Colab was verified using PyTorch CUDA tensor arithmetic on a Tesla T4.
- Independent online research identified **stored value 5 as Prostate** in the official CSIRO dataset documentation. All 60 processed masks for Case 006 / Week 0 preserve the original numeric labels in a direct array comparison. This finding was not obtained from or confirmed by a tutor; other series have not been individually matched to original 3D masks.
- `dataset.py`, `modules.py`, `metrics.py` and `train.py` are implemented. The data pipeline was checked on CPU against the full dataset audit. The models were checked with random tensors only (output shapes, gradients, loss range).
- Both models completed short real-data timing runs on a Colab Tesla T4 (100 training batches each, followed by validation). Full training and final test evaluation are still pending.

## Progress update: Colab T4 timing runs (21 October 2026)

The user supplied the Colab console outputs for the pipeline checks and timing runs below. The underlying run artifacts have not been independently inspected for this update.

- All 11,460 training, 660 validation and 540 test image-mask pairs passed the pipeline's shape, range and one-hot checks on Colab. Case counts were 31 / 4 / 3, and series counts were 191 / 11 / 9. The expected 60 training slices from Case 019 / Week 1 were cropped.
- Runtime: Linux, Python 3.13.15, PyTorch 2.11.0+cu130, CUDA available, Tesla T4 with 15,360 MiB. Both timing runs used batch size 16, AMP, seed 42 and 100 training batches, followed by one complete validation pass. A full training epoch contains 716 batches.

| Timing measurement | Baseline U-Net | Improved U-Net |
| --- | ---: | ---: |
| Parameters | 31,036,806 | 3,189,282 |
| Training time for 100 batches | 27 s | 16 s |
| Validation time (rounded) | 3 s | 2 s |
| Steady-state seconds per training batch | 0.187 | 0.082 |
| Estimated full epoch | 137 s | 61 s |
| Estimated 40-epoch training time | 1.5 h | 0.7 h |
| Peak allocated GPU memory | 4.85 GiB | 0.92 GiB |
| Validation prostate Dice after short run | 0.0000 | 0.5839 |

These are timing experiments, not completed full-training results or a final model comparison. The Dice values are the mean per-series 3D prostate Dice on validation after only 100 training batches. No model evaluation on the test split was performed (`test=false`); the test pipeline checks above only validate data integrity. Timing estimates exclude checkpoint/plot saving and other overhead and may vary across runs.

Next: train both models from scratch for 40 full epochs each on Colab T4, using separate output directories from the timing runs. Full-run completion has not yet been reported. Inspect the completed training histories before the final test evaluation.

## Progress update: prostate label provenance (21 October 2026)

The dataset publisher's [BriefDataDescription.txt](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034), attached to the [CSIRO collection](https://doi.org/10.25919/45t8-p065), explicitly defines `0=Background`, `1=Body`, `2=Bones`, `3=Bladder`, `4=Rectum`, and **`5=Prostate`**.

This milestone came from **independent online source research**, followed by a local comparison of existing masks. It is separate from the tutor's earlier acceptance of the 2D project direction and is not a claim of tutor confirmation. See the [source, verification method and remaining limitations](LABEL_MAPPING.md).

## Files

- `dataset.py`: image-mask pairing, the case-ID split check, preprocessing, paired augmentation and DataLoaders.
- `modules.py`: the standard 2D U-Net baseline, the 2D Improved U-Net and the multiclass soft Dice loss.
- `metrics.py`: per-series 3D Dice and the other fixed evaluation rules.
- `train.py`: training, validation, checkpoint selection, saving, loss/metric curves and the one-time test evaluation.
- `predict.py` (planned): saved-model inference and visualisation.

This README is still a progress document. It will become the full report once verified experiments exist.

## Dependencies

These are the versions the code was checked with locally. Each training run also records its own versions in `config.json`.

| Package | Version |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.13.0 (CUDA 13.0 build) |
| torchvision | 0.28.0 |
| NumPy | 2.5.2 |
| nibabel | 5.4.2 |
| matplotlib | 3.11.1 |

## Usage

`--data-root` points to the supplied `keras_slices_data` folder.

```bash
pip install nibabel==5.4.2              # torch/torchvision/matplotlib are preinstalled on Colab
python dataset.py --data-root keras_slices_data                      # pipeline self-check
python train.py --model unet --epochs 1 --max-train-batches 100      # timing run, no test
python train.py --model improved --epochs 40 --out-dir runs/improved --test
```

The test split is evaluated only with `--test`, after the full epoch budget. It is evaluated once for the best-validation checkpoint and once for the last checkpoint. Shortened runs refuse `--test`, and existing test results are never overwritten.

## Fixed preprocessing and evaluation rules

These rules were set before any training:

- **Splits:** the supplied train/validation/test splits are kept. Case IDs are disjoint across splits.
- **Shape:** 60 training slices (Case 019 / Week 1) are 256x144. They are centre-cropped to 256x128, which keeps every bladder, rectum and prostate pixel. Nothing is resized.
- **Intensity:** per-slice z-score, as in the course loader, clipped to [-5, 5]. No statistics are fitted on any split.
- **Labels:** labels 0-5 are kept as stored and one-hot encoded. Label 5 is the prostate.
- **Augmentation:** training slices only. Rotation +/-15 degrees, scale 0.85-1.15, a flip along the 256-pixel axis, and gamma 0.7-1.5, each with p = 0.5.
- **Loss:** both models use the multiclass soft Dice loss of Isensee et al. (2018, Eq. 1), summed over the whole batch. The batch size is at least 8.
- **Primary metric:** 3D prostate Dice per case/week series, averaged over series. It is also used for validation checkpoint selection. The global, case-weighted, per-class Dice/IoU and slice-level metrics are reported alongside it.

## Deviations from the reference papers

**Baseline U-Net (Ronneberger et al., 2015):**

- `padding=1` replaces unpadded convolutions.
- Batch normalisation is added.
- Training uses the soft Dice loss with Adam instead of weighted cross-entropy with SGD.
- The 0.5 dropout at the end of the contracting path is kept.

**Improved U-Net (Isensee et al., 2018):**

- 3D convolutions become 2D, and the model takes one input channel and produces six outputs.
- The paper does not specify several details, so these are assumptions:
  - the normalisation/activation order outside the context modules;
  - the 1x1 segmentation layers;
  - the width of the localisation 3x3 convolution.
- The training schedule is adapted to full passes over the data (learning-rate decay rescaled).
- Elastic deformation, augmentation attenuation, test-time mirroring, MC-dropout averaging and the 5-fold ensemble are not used.
- Per-slice normalisation replaces per-patient normalisation within a brain mask.

## Data and model storage

Keep the HipMRI dataset and saved model weights outside Git version control. Preserve the supplied train/validation/test splits while verifying patient grouping and spatial alignment.

## Artificial Intelligence Usage Disclosure

AI assistance (Anthropic Claude, through Claude Code) was used for the following:

- locating the publisher's online documentation;
- preparing and executing the mask-comparison and dataset-audit code;
- drafting the data/metric policy;
- drafting `dataset.py`, `modules.py`, `metrics.py` and `train.py`;
- drafting this documentation.

The AI-suggested code was verified in four ways:

- running the data pipeline over every slice and comparing its counts with an independent audit;
- tracing the architectures against the papers' text and Fig. 1;
- comparing the metrics with an independent NumPy reference on synthetic data;
- running synthetic end-to-end training/resume/test checks on CPU.

OpenAI Codex also assisted with the Colab execution guidance and this progress update, using the console outputs supplied by the user. Only short timing-run training is reported so far; full-training and final test results remain pending.
