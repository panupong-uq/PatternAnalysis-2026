# HipMRI 2D Improved U-Net - Panupong

COMP3710 individual project: compare a standard 2D U-Net baseline with the course-specified 2D Improved U-Net on HipMRI. The selected difficulty is Normal.

## Current development status

- Project planning and the initial feasibility review have been prepared.
- VS Code execution through Google Colab was verified using PyTorch CUDA tensor arithmetic on a Tesla T4.
- The anatomical label mapping for the processed 2D masks, especially the prostate label ID, is awaiting tutor confirmation.
- The model implementations, training and segmentation evaluation have not started.

## Planned submission files

- `modules.py`: baseline and Improved U-Net components.
- `dataset.py`: image-mask pairing, preprocessing and data loading.
- `train.py`: training, validation, testing and checkpoint saving.
- `predict.py`: saved-model inference and visualisation.
- `README.md`: project report, feasibility review, reproducibility instructions, results and AI usage disclosure.

This README is an initial project scaffold. The final report will be expanded as implementation and verified experiments progress.

## Data and model storage

Keep the HipMRI dataset and saved model weights outside Git version control. Preserve the supplied train/validation/test splits while verifying patient grouping and spatial alignment.
