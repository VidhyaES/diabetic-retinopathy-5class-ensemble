import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    f1_score,
    classification_report
)

from models.model import get_model, get_resnet
from utils import DRDataset


# =========================================================
# 1️⃣ Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# 2️⃣ Check Checkpoints
# =========================================================
print("Available checkpoints:", os.listdir("checkpoints"))


# =========================================================
# 3️⃣ Load Dataset (Kaggle Format)
# =========================================================
data_dir = "data/train"
csv_path = "data/trainLabels.csv"

df = pd.read_csv(csv_path)

image_paths = []
labels = []

for _, row in df.iterrows():
    img_name = row["image"] + ".jpeg"
    label = row["level"]
    full_path = os.path.join(data_dir, img_name)

    if os.path.exists(full_path):
        image_paths.append(full_path)
        labels.append(label)

# Same split as training
_, val_paths, _, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

val_dataset = DRDataset(val_paths, val_labels)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# =========================================================
# 4️⃣ Load Models
# =========================================================

# EfficientNet
eff_model = get_model().to(device)
eff_model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
eff_model.eval()

# ResNet
res_model = get_resnet().to(device)
res_model.load_state_dict(torch.load("checkpoints/resnet_best.pth", map_location=device))
res_model.eval()

print("Both models loaded successfully.")


# =========================================================
# 5️⃣ Ensemble Evaluation
# =========================================================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels_batch in val_loader:
        images = images.to(device)

        eff_out = torch.softmax(eff_model(images), dim=1)
        res_out = torch.softmax(res_model(images), dim=1)

        # Weighted ensemble (slightly favor EfficientNet)
        ensemble_out = (0.6 * eff_out + 0.4 * res_out)

        all_preds.extend(ensemble_out.cpu().numpy())
        all_labels.extend(labels_batch.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# =========================================================
# 6️⃣ Metrics
# =========================================================

val_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
pred_classes = np.argmax(all_preds, axis=1)
f1 = f1_score(all_labels, pred_classes, average='weighted')
cm = confusion_matrix(all_labels, pred_classes)

print("\n========== Ensemble Evaluation Results ==========")
print("AUC-ROC:", val_auc)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n")
print(classification_report(all_labels, pred_classes))


# =========================================================
# 7️⃣ Sensitivity & Specificity Per Class
# =========================================================
print("\nPer-Class Sensitivity & Specificity:")

for i in range(cm.shape[0]):
    TP = cm[i, i]
    FN = cm[i, :].sum() - TP
    FP = cm[:, i].sum() - TP
    TN = cm.sum() - (TP + FN + FP)

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print(f"Class {i} -> Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

print("=================================================\n")
