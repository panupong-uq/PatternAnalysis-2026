# HipMRI 2D Improved U-Net - Panupong

COMP3710 individual project: compare a standard 2D U-Net baseline with the course-specified 2D Improved U-Net on HipMRI. The selected difficulty is Normal.

## Current development status

- Project planning and the initial feasibility review have been prepared.
- VS Code execution through Google Colab was verified using PyTorch CUDA tensor arithmetic on a Tesla T4.
- Independent online research identified **stored value 5 as Prostate** in the official CSIRO dataset documentation. All 60 processed masks for Case 006 / Week 0 preserve the original numeric labels in a direct array comparison. This finding was not obtained from or confirmed by a tutor; other series still require the planned dataset audit.
- The model implementations, training and segmentation evaluation have not started.

## Progress update: prostate label provenance (21 October 2026)

The dataset publisher's [BriefDataDescription.txt](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034), attached to the [CSIRO collection](https://doi.org/10.25919/45t8-p065), explicitly defines `0=Background`, `1=Body`, `2=Bones`, `3=Bladder`, `4=Rectum`, and **`5=Prostate`**.

This milestone came from **independent online source research**, followed by a local comparison of existing masks. It is separate from the tutor's earlier acceptance of the 2D project direction and is not a claim of tutor confirmation. See the [source, verification method and remaining limitations](LABEL_MAPPING.md).

## Planned submission files

- `modules.py`: baseline and Improved U-Net components.
- `dataset.py`: image-mask pairing, preprocessing and data loading.
- `train.py`: training, validation, testing and checkpoint saving.
- `predict.py`: saved-model inference and visualisation.
- `README.md`: project report, feasibility review, reproducibility instructions, results and AI usage disclosure.

This README is an initial project scaffold. The final report will be expanded as implementation and verified experiments progress.

## Data and model storage

Keep the HipMRI dataset and saved model weights outside Git version control. Preserve the supplied train/validation/test splits while verifying patient grouping and spatial alignment.

## Artificial Intelligence Usage Disclosure

AI assistance was used to locate the publisher's online documentation, prepare and execute the mask-comparison code, and draft this progress documentation. The linked primary source and recorded array comparisons support the stated finding. This update contains documentation and verification metadata; no model training or segmentation performance result is reported.
