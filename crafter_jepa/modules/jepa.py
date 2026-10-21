import copy

import torch
import torch.nn.functional as F
from torch import nn


class JEPAPredictor(nn.Module):
    def __init__(self, latent_dim: int = 256, action_dim: int = 17) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(action_dim + latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([latent, action], dim=-1))


class JEPA(nn.Module):
    def __init__(
        self, encoder: nn.Module, latent_dim: int = 256, action_dim: int = 17
    ) -> None:
        super().__init__()

        self.predictor = JEPAPredictor(latent_dim, action_dim)
        self.online_encoder = encoder
        self.action_dim = action_dim
        self.target = copy.deepcopy(encoder).requires_grad_(False)

    def update_target(self, tau: float) -> None:
        with torch.no_grad():
            for p_t, p_o in zip(
                self.target.parameters(), self.online_encoder.parameters()
            ):
                p_t.lerp_(p_o, 1 - tau)

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> torch.Tensor:
        x = self.online_encoder(obs)
        pred = self.predictor(x, F.one_hot(action, self.action_dim).float())

        with torch.no_grad():
            y = self.target(next_obs)

        return ((pred - y) ** 2).mean()
