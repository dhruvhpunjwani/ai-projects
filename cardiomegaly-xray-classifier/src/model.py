import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def build_model(num_classes: int = 2):
    """
    ResNet50 transfer learning model for binary classification.
    """
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model
