import torch
import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 101)  # Assuming 101 classes for Food101 dataset
    return model