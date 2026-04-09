from __future__ import annotations

import torch
import torch.nn as nn


class DiffusionMusicStub(nn.Module):
    """
    Minimal placeholder for future diffusion-based music generation.

    This is intentionally lightweight so the project structure remains complete.
    It is not used in the main experiments yet.
    """

    def __init__(
        self,
        input_dim: int = 88,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    x = torch.rand(4, 128, 88)
    model = DiffusionMusicStub(input_dim=88, hidden_dim=256)
    y = model(x)
    print("Input shape: ", x.shape)
    print("Output shape:", y.shape)