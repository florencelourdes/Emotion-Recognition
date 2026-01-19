from torchvision.models import (
    efficientnet_b0,
    efficientnet_b3,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    ResNet50_Weights,
    resnet50,
)
from torch import nn


class B0Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = efficientnet_b0(weights=self.weights)
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(1280, 7))
        self.name = "efficientnet_b0"

    def forward(self, x):
        return self.model(x)


class B3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        self.model = efficientnet_b3(weights=self.weights)
        self.model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(1536, 7))
        self.name = "efficientnet_b3"

    def forward(self, x):
        return self.model(x)


class Resnet50Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = resnet50(weights=self.weights)
        self.model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(self.model.fc.in_features, 7))
        self.name = "resnet50"

    def forward(self, x):
        return self.model(x)
