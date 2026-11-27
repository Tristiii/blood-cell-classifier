import torch
from torch import nn

from torchvision.models import (
    DenseNet121_Weights,
    ResNet18_Weights,
    densenet121,
    resnet18,
)

class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
        )

        # Adjust to fit BloodMNIST
        self.model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DenseNet121(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
        )

        # Preserve pretrained filters while retaining more detail from 28x28 images.
        self.model.features.conv0.stride = (1, 1)
        self.model.features.pool0 = nn.Identity()
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.model.classifier.in_features, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)