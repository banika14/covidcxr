# COVID-19 Chest X-ray Segmentation and Classification

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

## Methodology

### 1) Lung segmentation
Our paper uses a U-Net model trained on Montgomery County chest X-rays with masks.

- input size: `256 × 256`
- grayscale preprocessing
- U-Net from scratch
- Adam optimizer
- learning rate: `0.0005`
- batch size: `16`
- early stopping patience: `10`

### 2) Classification
Our paper evaluates both plain X-rays and lung-segmented X-rays using transfer learning. This repo provides a training pipeline for the same model family and hyperparameter style:

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

## Reported paper results

| Task | Best model | Accuracy |
|---|---:|---:|
| Binary classification on segmented data | ResNet50 | 86% |
| Three-class classification on segmented data | ResNet50 | 79% |
| Lung segmentation | U-Net | ~90% Dice |

## Source files

- Paper: `Final-paper.pdf`
