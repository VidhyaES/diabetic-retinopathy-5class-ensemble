import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from models.model import get_model
from utils import DRDataset


# =========================================================
# 1️⃣ Reproducibility
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =========================================================
# 2️⃣ Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# 3️⃣ Data Loading (Kaggle Format)
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

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=SEED
)


# =========================================================
# 4️⃣ Data Augmentation
# =========================================================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = DRDataset(train_paths, train_labels, transform=train_transform)
val_dataset = DRDataset(val_paths, val_labels, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# =========================================================
# 5️⃣ Model
# =========================================================
model = get_model().to(device)


# =========================================================
# 6️⃣ Class Weights
# =========================================================
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(weights, dtype=torch.float).to(device)


# =========================================================
# 7️⃣ Focal Loss
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


criterion = FocalLoss(alpha=None, gamma=1.5)


# =========================================================
# 8️⃣ Two-Stage Fine-Tuning
# =========================================================

# -------- Stage 1: Freeze backbone --------
for param in model.parameters():
    param.requires_grad = False

for param in model._fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)


# =========================================================
# 9️⃣ Scheduler
# =========================================================
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    patience=3,
    factor=0.3
)


# =========================================================
# 🔟 Training Loop
# =========================================================
os.makedirs("checkpoints", exist_ok=True)
best_auc = 0
num_epochs = 30

for epoch in range(num_epochs):

    # Unfreeze after 5 epochs
    if epoch == 5:
        print("Unfreezing full model...")
        for param in model.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # -------------------
    # Training
    # -------------------
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # -------------------
    # Validation
    # -------------------
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    val_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')

    scheduler.step(val_auc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} "
          f"Val AUC: {val_auc:.4f}")

    # Save Best Model
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print("Best model saved.")


print("Training Completed.")
print("Best Validation AUC:", best_auc)
