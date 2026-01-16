# src/model.py

import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES

def build_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model
