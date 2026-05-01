# COVID-19 Chest X-ray Segmentation and Classification

This repository reconstructs the workflow from the paper **“COVID-19 Detection from Chest X-ray Images Using Deep Learning Approach”** and packages the research code into a clean GitHub-ready project.

## What is included

- A cleaned repo structure
- The original lung-segmentation notebook in `notebooks/`
- A U-Net implementation for lung mask prediction
- Transfer-learning classifiers for:
  - VGG19
  - DenseNet121
  - ResNet50
  - InceptionV2 / InceptionV3 fallback
  - Xception
- Training and inference scripts
- A paper summary with the key methodology and reported results

## Repository layout

```text
.
├── configs/
├── data/
├── docs/
├── notebooks/
├── src/
└── requirements.txt
```

## Methodology followed from the paper

### 1) Lung segmentation
The paper uses a U-Net model trained on Montgomery County chest X-rays with masks. The workflow in this repo matches that setup:

- input size: `256 × 256`
- grayscale preprocessing
- U-Net from scratch
- Adam optimizer
- learning rate: `0.0005`
- batch size: `16`
- early stopping patience: `10`

### 2) Classification
The paper evaluates both plain X-rays and lung-segmented X-rays using transfer learning. This repo provides a training pipeline for the same model family and hyperparameter style:

- batch size: `16`
- learning rate: `5e-4`
- dropout: `0.5`
- early stopping + reduce-on-plateau + best-checkpoint saving

## Expected data layout

```text
data/
├── raw/
│   ├── segmentation/
│   │   ├── CXR_png/
│   │   └── masks/
│   └── classification/
│       ├── binary/
│       │   ├── covid/
│       │   └── non_covid/
│       └── multiclass/
│           ├── covid/
│           ├── normal/
│           └── pneumonia/
└── processed/
    └── segmented_images/
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Train lung segmentation

```bash
python -m covidcxr.segmentation.train --config configs/default.yaml
```

## Apply lung segmentation

```bash
python -m covidcxr.segmentation.infer   --model data/models/unet_lungs.keras   --input_dir data/raw/classification/binary/covid   --output_dir data/processed/segmented_images/covid
```

## Train a classifier

```bash
python -m covidcxr.classification.train   --config configs/default.yaml   --architecture resnet50   --dataset segmented
```

## Notes

- The paper mentions **InceptionV2**; current Keras application models expose **InceptionV3** and **InceptionResNetV2**. This repo maps that slot to **InceptionV3** so the pipeline stays executable with current Keras APIs.
- The repo is structured so you can add your other Colab notebooks under `notebooks/` and convert them into scripts later if needed.

## Reported paper results

| Task | Best model | Accuracy |
|---|---:|---:|
| Binary classification on segmented data | ResNet50 | 86% |
| Three-class classification on segmented data | ResNet50 | 79% |
| Lung segmentation | U-Net | ~90% Dice |

## Source files

- Paper: `Final-paper.pdf`
- Notebook: `FinalLungSegmentation.ipynb`
