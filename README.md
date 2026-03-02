# Diabetic Retinopathy 5-Class Grading (Research-Grade)

## Overview
This project implements a research-grade deep learning system for automated 5-class Diabetic Retinopathy (DR) grading using retinal fundus images.

The system follows strict evaluation protocols including:
- 80/20 Stratified Train-Test Split
- 5-Fold Cross-Validation (inside training set)
- Ensemble Learning
- Data Leakage Prevention
- Grad-CAM Explainability

---

## Model Architecture
Backbone: EfficientNet-B0 (pretrained on ImageNet)  
Loss: CrossEntropyLoss  
Optimizer: Adam  

---

## Training Strategy

1. Dataset split into 80% train and 20% hold-out test.
2. Stratified 5-Fold cross-validation performed on training set.
3. Best model per fold saved.
4. Final prediction via softmax averaging ensemble.

---

## Evaluation Metrics

- Multi-class AUC (One-vs-Rest)
- Weighted F1 Score
- Confusion Matrix
- Per-Class Sensitivity & Specificity

---

## Results (Leakage-Free Evaluation)

- AUC-ROC: ~0.83
- Weighted F1 Score: ~0.71
- Accuracy: ~0.76

---

## How To Run

### Train 5-Fold Models
