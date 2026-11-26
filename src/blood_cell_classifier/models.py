import torch
from torch import nn

from torchvision.models import resnet18, ResNet18_Weights

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