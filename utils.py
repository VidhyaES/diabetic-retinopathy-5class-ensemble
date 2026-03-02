import torch
from torch.utils.data import Dataset
import cv2

class DRDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.resize(img, (224,224))
        img = img / 255.0

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute(2,0,1).float()

        label = torch.tensor(self.labels[idx]).long()
        return img, label
