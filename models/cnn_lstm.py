import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """CNN-LSTM temporal model for end-to-end self-driving.

    A CNN feature extractor processes each frame independently, the per-frame
    features are concatenated with the corresponding speed scalar, and the
    resulting sequence is fed through an LSTM.  The last hidden state is
    decoded into steering and throttle predictions.

    Input:
        images: (batch, seq_len, 3, H, W) — preprocessed YUV frames
        speeds: (batch, seq_len)           — normalised speed per frame

    Output:
        (batch, 2) — [steering, throttle]
    """

    def __init__(self, hidden_size: int = 128, num_layers: int = 1):
        super().__init__()

        # --- CNN feature extractor (shared across timesteps) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # -> (batch, 256)
        )

        # --- LSTM over the temporal sequence ---
        # Each timestep vector: 256 (CNN features) + 1 (speed) = 257
        self.lstm = nn.LSTM(
            input_size=257,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # --- Prediction head ---
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(
        self, images: torch.Tensor, speeds: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = images.shape[:2]

        # Process each frame through the CNN independently.
        # Merge batch and seq dims -> (batch*seq_len, C, H, W)
        imgs_flat = images.view(batch_size * seq_len, *images.shape[2:])
        features = self.cnn(imgs_flat)  # (batch*seq_len, 256)
        features = features.view(batch_size, seq_len, -1)  # (batch, seq_len, 256)

        # Concatenate speed as an extra feature per timestep.
        speeds_expanded = speeds.unsqueeze(-1)  # (batch, seq_len, 1)
        lstm_input = torch.cat([features, speeds_expanded], dim=-1)  # (batch, seq_len, 257)

        # Run through LSTM and take the last hidden state.
        lstm_out, (h_n, _) = self.lstm(lstm_input)
        last_hidden = h_n[-1]  # (batch, hidden_size)

        return self.head(last_hidden)  # (batch, 2)
