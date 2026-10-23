import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ConvSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImpalaEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        channels: tuple[int, ...] = (16, 32, 32),
    ) -> None:
        super().__init__()
        in_channels = (3, *channels[:-1])
        side = 64 // 2 ** len(channels)
        self.net = nn.Sequential(
            *[ConvSequence(i, o) for i, o in zip(in_channels, channels)],
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels[-1] * side * side, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
