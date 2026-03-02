import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

def get_model():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 5)
    return model
def get_resnet():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 5)
    return model