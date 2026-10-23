from typing import Any

import torch

from crafter_jepa.modules import SAE
from crafter_jepa.trainers.ppo_trainer import PPOTrainer


class SAETrainer(PPOTrainer):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        *,
        recon_coef: float = 1.0,
        sparsity_coef: float = 0.001,
        sparsity_target: float = 0.05,
        **ppo_kwargs: Any,
    ) -> None:
        super().__init__(feature_dim, action_dim, **ppo_kwargs)
        self.sae = SAE(
            self.encoder,
            latent_dim=feature_dim,
            sparsity_target=sparsity_target,
        ).to(self.device)
        self.recon_coef = recon_coef
        self.sparsity_coef = sparsity_coef

    def _parameters(self) -> list[torch.nn.Parameter]:
        return [*super()._parameters(), *self.sae.extra_parameters()]

    def _extra_loss(
        self, batch: dict[str, torch.Tensor], features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        recon_loss, sparsity_loss, _ = self.sae.loss(batch["obs"], features=features)
        sae_loss = self.recon_coef * recon_loss + self.sparsity_coef * sparsity_loss
        stats = {
            "recon_loss": recon_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "sae_loss": sae_loss.item(),
            "latent_std": features.std(dim=0).detach().mean().item(),
        }
        return sae_loss, stats

    def _checkpoint_state(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_state(),
            "sae_decoder": self.sae.decoder.state_dict(),
            "sae_fc_enc": self.sae.fc_enc.state_dict(),
        }
