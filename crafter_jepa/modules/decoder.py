import torch
from torch import nn

from crafter_jepa.modules.encoder import ResidualBlock


class DeconvSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ResidualBlock(in_channels),
            ResidualBlock(in_channels),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImpalaDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        channels: tuple[int, ...] = (16, 32, 32),
    ) -> None:
        super().__init__()
        side = 64 // 2 ** len(channels)
        self.side = side
        self.start_channels = channels[-1]

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, channels[-1] * side * side),
            nn.ReLU(),
        )

        dec_in = list(reversed(channels))
        dec_out = list(reversed(channels[:-1])) + [3]

        layers: list[nn.Module] = []
        for i, (in_c, out_c) in enumerate(zip(dec_in, dec_out)):
            layers.append(DeconvSequence(in_c, out_c))
            if i < len(dec_in) - 1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())

        self.conv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, self.start_channels, self.side, self.side)
        return self.conv(h)
