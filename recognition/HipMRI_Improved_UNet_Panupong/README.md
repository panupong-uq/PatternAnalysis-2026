# HipMRI 2D Improved U-Net - Panupong

COMP3710 individual project: compare a standard 2D U-Net baseline with the course-specified 2D Improved U-Net on HipMRI. The selected difficulty is Normal.

## Current development status

- Project planning and the initial feasibility review have been prepared.
- VS Code execution through Google Colab was verified using PyTorch CUDA tensor arithmetic on a Tesla T4.
- Independent online research identified **stored value 5 as Prostate** in the official CSIRO dataset documentation. All 60 processed masks for Case 006 / Week 0 preserve the original numeric labels in a direct array comparison. This finding was not obtained from or confirmed by a tutor; other series have not been individually matched to original 3D masks.
- `dataset.py`, `modules.py`, `metrics.py` and `train.py` are implemented. The data pipeline was checked on CPU against the full dataset audit. The models were first checked with random tensors (output shapes, gradients, loss range) and then in short real-data timing runs (below).
- Both models completed short real-data timing runs on a Colab Tesla T4 (100 training batches each, followed by validation). These remain timing evidence only.
- Final training moved to the UQ Rangpur cluster. The data pipeline passed its full check there.
- Both models completed 40 epochs of training on the same Rangpur node (NVIDIA A100-PCIE-40GB). Validation results are below.
- The one-time test evaluation of both models has been submitted, and its results are still pending.

## Progress update: Rangpur training complete (22 October 2026)

Both models were trained from scratch on Rangpur node `a100-6` (NVIDIA A100-PCIE-40GB), with code at commit `92f51d4`. Both used the same settings: 40 epochs, batch size 16, AMP and seed 42. The values below come from each run's `history.json` and `config.json` and from the Slurm logs. The user copied these files back from Rangpur. **All Dice values in this section are on the validation split. No test result is reported here.**

| Measurement | Baseline U-Net | Improved U-Net |
| --- | ---: | ---: |
| Parameters | 31,036,806 | 3,189,282 |
| Median training time per epoch | 29.0 s | 24.0 s |
| Median seconds per training batch (steady state) | 0.040 | 0.033 |
| Peak allocated GPU memory | 2.32 GiB | 0.96 GiB |
| Training + validation time, 40 epochs | 20.4 min | 17.1 min |
| Job time, script start to finish (from the log) | 28 min 57 s | 19 min 13 s |
| Best validation prostate Dice, per-series mean (epoch) | **0.8686** (28) | **0.8564** (34) |
| At that epoch: case-weighted / global prostate Dice | 0.8331 / 0.8697 | 0.8157 / 0.8534 |
| At that epoch: foreground Dice (classes 1-5) | 0.9202 | 0.9200 |
| Last epoch (40): validation prostate Dice | 0.8626 | 0.8556 |
| Last 10 epochs: mean ± SD validation prostate Dice | 0.8628 ± 0.0018 | 0.8548 ± 0.0013 |

The job time also includes one-time dataset loading and the per-epoch checkpoint, history and plot saving. Most of the larger baseline overhead is probably from writing its larger checkpoints: about 0.5 GB per epoch, against about 0.05 GB for Improved. This overhead was not timed separately.

Observations, to be revisited with the test results:

- **Accuracy:** on validation, the baseline's prostate Dice is slightly higher. The gap is about 0.012 at the best epoch and 0.008 over the last 10 epochs. The foreground Dice over classes 1-5 (body, bones, bladder, rectum and prostate) is nearly identical: 0.9202 against 0.9200 at the best epoch.
- **Cost:** the Improved U-Net uses about 10 times fewer parameters, about 17% less time per training batch and about 60% less peak GPU memory.
- **Caveats:**
  - Validation has only 11 series from 4 cases, and each model was trained with a single seed, so these gaps have not been tested for significance.
  - The best epoch was chosen on the same validation split, so the best-epoch scores are optimistic.
- **Training curves:** validation loss levels off after about 10-15 epochs, while training loss keeps decreasing (about -0.95 against -0.78 for validation). There is no late drop in validation Dice.
- **Fallback rule:** the pre-specified rule for switching to Dice plus cross-entropy was not triggered. It applied if training loss fell by less than 10% between epochs 1 and 5, or if validation prostate Dice was below 0.3 at epoch 5. Training loss fell by 18.7% for the baseline and 14.6% for Improved, and epoch-5 validation Dice was 0.853 and 0.838.

**Next:** each model's best-validation and last checkpoints are evaluated once on the test split (`--resume --test`). Nothing will be tuned after the test results are seen.

## Progress update: Rangpur setup and data check (22 October 2026)

The assignment directs compute-intensive work to the Rangpur Slurm cluster, which also hosts the 2D data at `/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`. Rangpur is now the platform for the final results of both models. The user supplied the Rangpur console outputs for this update.

**Environment** (interactive Slurm session on node `a100-6`, code at commit `92f51d4`):

- NVIDIA A100-PCIE-40GB, driver 590.48.01 (CUDA 13.1).
- Python 3.12.13 and PyTorch 2.13.0+cu130; a test operation on the GPU succeeded.

**Data pipeline check** (`python dataset.py` on the course copy of the data, 95 s):

| Split | Pairs | Cases | Series | Prostate-positive slices |
| --- | ---: | ---: | ---: | ---: |
| Train | 11,460 | 31 | 191 | 4,152 |
| Validation | 660 | 4 | 11 | 254 |
| Test | 540 | 3 | 9 | 195 |

- Every item of every split passed the shape, range and one-hot checks.
- The 60 training slices from Case 019 / Week 1 were cropped.
- Per-class pixel counts match the earlier full-dataset audit of the local copy, with one expected difference. The training Background, Body and Bones counts are lower by 245,760 pixels in total, which is exactly the cropped area (60 slices x 256 x 16 columns). Bladder, rectum and prostate pixels are unchanged, so the Rangpur copy and the audited copy agree on these summaries.

**Status:** both models were submitted as Slurm jobs (one GPU each) with the default settings: 40 epochs, batch size 16, AMP, seed 42. The jobs do not evaluate the test split. No training results are reported yet.

(Updated on 22 October 2026: both training jobs completed, as described above.)

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

(Superseded on 22 October 2026: final training moved to Rangpur, as described above.)

## Progress update: prostate label provenance (21 October 2026)

The dataset publisher's [BriefDataDescription.txt](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034), attached to the [CSIRO collection](https://doi.org/10.25919/45t8-p065), explicitly defines `0=Background`, `1=Body`, `2=Bones`, `3=Bladder`, `4=Rectum`, and **`5=Prostate`**.

This milestone came from **independent online source research**, followed by a local comparison of existing masks. It is separate from the tutor's earlier acceptance of the 2D project direction and is not a claim of tutor confirmation. See the [source, verification method and remaining limitations](LABEL_MAPPING.md).

## Files

- `dataset.py`: image-mask pairing, the case-ID split check, preprocessing, paired augmentation and DataLoaders.
- `modules.py`: the standard 2D U-Net baseline, the 2D Improved U-Net and the multiclass soft Dice loss.
- `metrics.py`: per-series 3D Dice and the other fixed evaluation rules.
- `train.py`: training, validation, checkpoint selection, saving, loss/metric curves and the one-time test evaluation.
- `predict.py`: reloads the validation-selected checkpoints of both models after the test evaluation. It draws side-by-side test predictions and failure cases, rechecks the test prostate Dice and times inference. It does not choose a checkpoint or tune anything; the test predictions are used only to pick which slices to display.

This README is still a progress document. It will become the full report once verified experiments exist.

## Dependencies

These are the versions the code was checked with locally and the versions installed on Rangpur. Each training run also records its Python and PyTorch versions in `config.json`.

| Package | Local check | Rangpur |
| --- | --- | --- |
| Python | 3.12.13 | 3.12.13 |
| PyTorch | 2.13.0 (CUDA 13.0 build) | 2.13.0+cu130 |
| torchvision | 0.28.0 | 0.28.0+cu130 |
| NumPy | 2.5.2 | 2.5.1 |
| nibabel | 5.4.2 | 5.4.2 |
| matplotlib | 3.11.1 | 3.11.2 |

## Usage

`--data-root` points to the supplied `keras_slices_data` folder.

```bash
pip install nibabel==5.4.2              # torch/torchvision/matplotlib are preinstalled on Colab
python dataset.py --data-root keras_slices_data                      # pipeline self-check
python train.py --model unet --epochs 1 --max-train-batches 100      # timing run, no test
python train.py --model improved --epochs 40 --out-dir runs/improved --test
python predict.py --runs-dir runs   # after both models are tested; writes runs/predict_best/
```

On Rangpur, `train.py` runs inside a Slurm job (`--partition=comp3710 --account=comp3710 --gres=gpu:1`) and reads the course copy of the data directly:

```bash
python train.py --model improved --data-root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data --out-dir ~/hipmri_runs/improved
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
- preparing the Rangpur environment-setup and Slurm job scripts (kept outside this repository);
- drafting this documentation.

The AI-suggested code was verified in four ways:

- running the data pipeline over every slice and comparing its counts with an independent audit;
- tracing the architectures against the papers' text and Fig. 1;
- comparing the metrics with an independent NumPy reference on synthetic data;
- running synthetic end-to-end training/resume/test checks on CPU.

OpenAI Codex also assisted with the Colab execution guidance and this progress update, using the console outputs supplied by the user. Claude Code summarised the Rangpur run files supplied by the user for the training update. The short timing runs and the full 40-epoch training (validation only) are reported; final test results remain pending.
