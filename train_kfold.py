import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

from models.model import get_model
from utils import DRDataset


# ======================================================
# 1. Setup
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("checkpoints_kfold", exist_ok=True)
os.makedirs("splits", exist_ok=True)


# ======================================================
# 2. Load Full Dataset
# ======================================================
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

image_paths = np.array(image_paths)
labels = np.array(labels)

print("Total samples:", len(image_paths))


# ======================================================
# 3. Create TRUE Hold-Out Test Set (20%)
# ======================================================
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

print("Train size:", len(train_paths))
print("Test size:", len(test_paths))

# Save test split for evaluation script
np.save("splits/test_paths.npy", test_paths)
np.save("splits/test_labels.npy", test_labels)


# ======================================================
# 4. Transforms
# ======================================================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ======================================================
# 5. Stratified 5-Fold ONLY on Training Set
# ======================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_paths, train_labels)):

    print(f"\n========== Fold {fold+1} ==========")

    fold_train_paths = train_paths[train_idx]
    fold_val_paths = train_paths[val_idx]

    fold_train_labels = train_labels[train_idx]
    fold_val_labels = train_labels[val_idx]

    train_dataset = DRDataset(fold_train_paths, fold_train_labels, transform=train_transform)
    val_dataset = DRDataset(fold_val_paths, fold_val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_auc = 0

    for epoch in range(15):

        # -------------------
        # Training
        # -------------------
        model.train()
        running_loss = 0

        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # -------------------
        # Validation
        # -------------------
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels_batch.numpy())

        val_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')

        print(f"Fold {fold+1} | Epoch {epoch+1} | AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                model.state_dict(),
                f"checkpoints_kfold/model_fold_{fold+1}.pth"
            )

    print(f"Best AUC for Fold {fold+1}: {best_auc:.4f}")

print("\n5-Fold Training Completed WITHOUT data leakage.")
