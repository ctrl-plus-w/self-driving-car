import torch.nn as nn


class PilotNet(nn.Module):
    """NVIDIA PilotNet architecture for end-to-end self-driving.

    Input:  (batch, 3, 66, 200) — YUV preprocessed images
    Output: (batch, 2)          — [steering, throttle]
    """

    def __init__(self):
        super().__init__()

        # 5 convolutional layers with ELU activation
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
