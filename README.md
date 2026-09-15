# Explainable Deep Learning for Multiclass Kidney Disease Classification from CT Images

Undergraduate thesis project — multiclass classification of kidney CT images (Normal, Cyst, Tumor, Stone) using transfer learning across multiple architectures (ResNet50, EfficientNetB0, ConvNeXtTiny, ViT-B16), with Grad-CAM explainability to interpret model predictions.

## Dataset

[CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) (Kaggle) — 12,446 abdominal CT images across 4 classes:

| Class  | Images |
|--------|--------|
| Normal | 5,077  |
| Cyst   | 3,709  |
| Tumor  | 2,283  |
| Stone  | 1,377  |

Base paper for this dataset: Islam et al. (2022), *"Vision transformer and explainable transfer learning models for auto detection of kidney cyst, stone and tumor from CT-radiography,"* Scientific Reports.

## Approach

- **Split:** Stratified 70% train / 15% validation / 15% test (image-level; no patient ID available in metadata)
- **Preprocessing:** Resize to 224×224, each model's own dedicated `preprocess_input`/normalization scheme
- **Augmentation:** Conservative — horizontal flip, small rotation, zoom, translation (train set only, to preserve anatomical validity)
- **Architectures compared:**
  - **ResNet50** — ImageNet-pretrained, frozen backbone + GlobalAveragePooling → Dropout(0.3) → Dense(4, softmax)
  - **EfficientNetB0** — same head, frozen backbone
  - **ConvNeXtTiny** — frozen backbone, then fine-tuned (last ~20 layers unfrozen, lr = 1e-5)
  - **ViT-B16** — PyTorch/`timm`, fully fine-tuned (lr = 1e-5)
- **Class imbalance:** For ResNet50 and EfficientNetB0, trained both a baseline model and a class-weighted model (`compute_class_weight('balanced')`) to compare their effect on minority-class (Stone, Tumor) recall
- **Explainability:** Grad-CAM on the final convolutional block of each CNN backbone, to visualize which image regions drive each prediction, including analysis of misclassified samples

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| ResNet50 (baseline) | 91.23% | Best CNN result; low recall on Stone (0.71) / Tumor (0.82) |
| ResNet50 (class-weighted) | 89.15% | Improved Stone/Tumor recall (0.81 / 0.86) at the cost of overall accuracy |
| EfficientNetB0 (baseline) | 81.77% | Frozen backbone |
| EfficientNetB0 (class-weighted) | 69.05% | Class weights over-corrected, precision collapse on Stone |
| ConvNeXtTiny (fine-tuned) | 82.20% | Frozen backbone alone only reached 72.80%; fine-tuning the last 20 layers improved it |
| ViT-B16 (fine-tuned) | ~100%\* | See data-quality note below |

\* **Data-quality note:** while investigating the unexpectedly high ViT accuracy, a duplicate-image check found **517 exact-duplicate images (4.15% of the dataset)** leaking across the train/val/test splits (identical CT images saved under different filenames). This affects all models to some degree, but was most exploited by ViT's full fine-tuning. A deduplicated version of the dataset (11,929 unique images) has been prepared; re-training all models on it is a planned next step to obtain fully corrected numbers.

**Grad-CAM finding (CNN models):** the dominant error pattern (Stone/Tumor misclassified as Cyst) shows the model's attention consistently centered on the round, central kidney region shared by all three classes — suggesting the models rely more on coarse shape/brightness cues than on fine texture (e.g. calcification density, boundary irregularity) that would normally distinguish these conditions.

## Repository Contents

- `Kidney_Classification_ResNet50_GradCAM.ipynb` — full pipeline: data loading, preprocessing, training (baseline + class-weighted), evaluation, Grad-CAM explainability, error analysis.
- `Kidney_Classification_EfficientNetB0_.ipynb` — EfficientNetB0 baseline and class-weighted training/evaluation.
- `Kidney_Classification_ConvNeXtTiny_.ipynb` — ConvNeXtTiny frozen baseline and fine-tuning experiments.
- `Kidney_Classification_ViT_B16.ipynb` — ViT-B16 fine-tuning (PyTorch/`timm`) and the duplicate-image investigation.

Cell outputs are saved in each notebook, so results are visible without re-running.

## Environment

Developed and run on Google Colab. TensorFlow/Keras (ResNet50, EfficientNetB0, ConvNeXtTiny), PyTorch + `timm` (ViT-B16), scikit-learn, OpenCV.
