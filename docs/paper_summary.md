# Paper summary

## Goal
Build a computer-aided diagnosis pipeline for COVID-19 and pneumonia detection from chest X-rays using:
1. U-Net lung segmentation.
2. Transfer-learning classification on both plain and segmented images.

## Segmentation details
- Dataset: Montgomery County chest X-rays with masks
- Image size: 256 × 256
- Model: U-Net from scratch
- Optimizer: Adam
- Learning rate: 0.0005
- Batch size: 16
- Early stopping patience: 10
- Output: lung mask + segmented image via bitwise operation / masking

## Classification details
- Binary task: COVID-19 vs non-COVID
- Multiclass task: COVID-19 vs normal vs pneumonia
- Models: VGG19, DenseNet121, ResNet50, InceptionV2 (mapped to InceptionV3 in this repo), Xception
- Dropout: 0.5
- Optimizer: Adam
- Learning rate: 5e-4
- Batch size: 16

## Reported results from the paper
- Segmentation Dice coefficient: about 90%
- Binary classification on segmented dataset: 86% accuracy with ResNet50
- Multiclass classification on segmented dataset: 79% accuracy with ResNet50
