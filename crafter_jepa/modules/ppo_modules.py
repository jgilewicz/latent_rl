import math

import torch
from torch import nn
from torch.distributions import Categorical


def _init_linear(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.out_layer = nn.Linear(hidden_dim, action_dim)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _init_linear(layer, math.sqrt(2))
        _init_linear(self.out_layer, 0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.net(state))


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )
        self.out_layer = nn.Linear(hidden_dim, 1)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                _init_linear(layer, math.sqrt(2))
        _init_linear(self.out_layer, 1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.net(state)).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.critic = Critic(state_dim, hidden_dim, dropout)
        self.actor = Actor(state_dim, action_dim, hidden_dim)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self(state)

        if action is None:
            action = dist.sample()

        return (action, dist.log_prob(action), dist.entropy(), value)

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        dist = Categorical(logits=self.actor(state))
        v_s = self.get_value(state)

        return (dist, v_s)
