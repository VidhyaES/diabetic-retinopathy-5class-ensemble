import os
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    f1_score,
    classification_report
)

from models.model import get_model
from utils import DRDataset


# ==========================================================
# 1. Device
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================================================
# 2. Load TRUE Hold-Out Test Split
# ==========================================================
test_paths = np.load("splits/test_paths.npy", allow_pickle=True)
test_labels = np.load("splits/test_labels.npy", allow_pickle=True)

print("Test samples:", len(test_paths))


# ==========================================================
# 3. Dataset & Loader
# ==========================================================
test_dataset = DRDataset(test_paths, test_labels, transform=None)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# ==========================================================
# 4. Load All 5 Fold Models
# ==========================================================
models = []

for fold in range(1, 6):
    model_path = f"checkpoints_kfold/model_fold_{fold}.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    model = get_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    models.append(model)

print("All 5 fold models loaded successfully.")


# ==========================================================
# 5. Ensemble Prediction
# ==========================================================
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels_batch in test_loader:

        images = images.to(device)

        fold_outputs = []

        for model in models:
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            fold_outputs.append(probs)

        # Stack shape: [5, batch, classes]
        stacked = torch.stack(fold_outputs)

        # Mean across folds
        ensemble_probs = torch.mean(stacked, dim=0)

        all_probs.extend(ensemble_probs.cpu().numpy())
        all_labels.extend(labels_batch.numpy())


all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

pred_classes = np.argmax(all_probs, axis=1)


# ==========================================================
# 6. Metrics
# ==========================================================
auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
f1 = f1_score(all_labels, pred_classes, average="weighted")
cm = confusion_matrix(all_labels, pred_classes)


print("\n========== 5-Fold Ensemble Evaluation ==========")
print("AUC-ROC:", round(auc, 4))
print("Weighted F1 Score:", round(f1, 4))
print("Confusion Matrix:\n", cm)
print("------------------------------------------------")


# ==========================================================
# 7. Classification Report
# ==========================================================
print("\nClassification Report:\n")
print(
    classification_report(
        all_labels,
        pred_classes,
        zero_division=0
    )
)


# ==========================================================
# 8. Per-Class Sensitivity & Specificity
# ==========================================================
print("\nPer-Class Sensitivity & Specificity:")

num_classes = cm.shape[0]

for i in range(num_classes):

    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FN + FP)

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print(
        f"Class {i} -> "
        f"Sensitivity: {sensitivity:.4f}, "
        f"Specificity: {specificity:.4f}"
    )

print("=================================================")
