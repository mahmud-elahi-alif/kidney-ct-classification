 Explainable Deep Learning for Multiclass Kidney Disease Classification from CT Images

Undergraduate thesis project — multiclass classification of kidney CT images (Normal, Cyst, Tumor, Stone) using transfer learning, with Grad-CAM explainability to interpret model predictions.

## Dataset

[CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) (Kaggle) — 12,446 abdominal CT images across 4 classes:

| Class  | Images |
|--------|--------|
| Normal | 5,077  |
| Cyst   | 3,709  |
| Tumor  | 2,283  |
| Stone  | 1,377  |

## Approach

- **Split:** Stratified 70% train / 15% validation / 15% test (image-level; no patient ID available in metadata)
- **Preprocessing:** Resize to 224×224, ResNet50-specific `preprocess_input` normalization
- **Augmentation:** Conservative — horizontal flip, small rotation, zoom, translation (train set only, to preserve anatomical validity)
- **Model:** ResNet50 (ImageNet-pretrained, frozen) + GlobalAveragePooling → Dropout(0.3) → Dense(4, softmax)
- **Class imbalance:** Trained both a baseline model and a class-weighted model (`compute_class_weight('balanced')`) to compare their effect on minority-class (Stone, Tumor) recall
- **Explainability:** Grad-CAM on the final convolutional block (`conv5_block3_out`) to visualize which image regions drive each prediction, including analysis of misclassified samples

## Results

| Metric | Baseline | Class-Weighted |
|---|---|---|
| Test Accuracy | 91.23% | 89.15% |
| Stone Recall | 0.71 | 0.81 |
| Tumor Recall | 0.82 | 0.86 |

Class weighting improved recall on the minority classes (Stone, Tumor) at a moderate cost to overall accuracy and Stone precision — an accuracy/fairness trade-off discussed further in the report.

**Grad-CAM finding:** the dominant error pattern (Stone/Tumor misclassified as Cyst) shows the model's attention consistently centered on the round, central kidney region shared by all three classes — suggesting it relies more on coarse shape/brightness cues than on fine texture (e.g. calcification density, boundary irregularity) that would normally distinguish these conditions.

## Repository Contents

- `Kidney_Classification_ResNet50_GradCAM.ipynb` — full pipeline: data loading, preprocessing, training (baseline + class-weighted), evaluation, Grad-CAM explainability, error analysis. Cell outputs are saved, so results are visible without re-running.

## Environment

Developed and run on Google Colab (TensorFlow/Keras, scikit-learn, OpenCV).
