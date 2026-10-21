from typing import Any

import torch
import torch.nn.functional as F

from crafter_jepa.modules import JEPA
from crafter_jepa.trainers.ppo_trainer import PPOTrainer


class JEPATrainer(PPOTrainer):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        *,
        jepa_coef: float,
        ema_tau: float,
        var_coef: float,
        **ppo_kwargs: Any,
    ) -> None:
        super().__init__(feature_dim, action_dim, **ppo_kwargs)
        self.jepa = JEPA(self.encoder, feature_dim, action_dim).to(self.device)
        self.jepa_coef = jepa_coef
        self.ema_tau = ema_tau
        self.var_coef = var_coef

    def _parameters(self) -> list[torch.nn.Parameter]:
        return [*super()._parameters(), *self.jepa.predictor.parameters()]

    def _extra_loss(
        self, batch: dict[str, torch.Tensor], features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        jepa_loss = self.jepa.loss(batch["obs"], batch["actions"], batch["next_obs"])
        std = features.std(dim=0)
        var_loss = F.relu(1.0 - std).mean()
        stats = {
            "jepa_loss": jepa_loss.item(),
            "var_loss": var_loss.item(),
            "latent_std": std.detach().mean().item(),
        }
        return self.jepa_coef * jepa_loss + self.var_coef * var_loss, stats

    def _checkpoint_state(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_state(),
            "predictor": self.jepa.predictor.state_dict(),
            "target_encoder": self.jepa.target.state_dict(),
        }

    def _after_step(self) -> None:
        self.jepa.update_target(self.ema_tau)
