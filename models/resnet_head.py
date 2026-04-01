import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetHead(nn.Module):
    """ResNet18 with a custom regression head for self-driving prediction.

    Uses a pretrained ResNet18 backbone and replaces the final classification
    layer with a lightweight regression head that outputs steering and throttle.

    Input:  (batch, 3, 224, 224) — ImageNet-normalized RGB images
    Output: (batch, 2)           — [steering, throttle]
    """

    def __init__(self):
        super().__init__()

        # Load pretrained ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove the original fully-connected layer (replace with identity)
        self.backbone.fc = nn.Identity()

        # Regression head: 512 -> 2
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        """Freeze all ResNet backbone layers for initial head-only training."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the last two residual blocks (layer3, layer4) for fine-tuning."""
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
