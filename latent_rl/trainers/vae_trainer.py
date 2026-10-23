from typing import Any

import torch

from latent_rl.modules import VAE
from latent_rl.trainers.ppo_trainer import PPOTrainer


class VAETrainer(PPOTrainer):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        *,
        recon_coef: float = 1.0,
        kl_coef: float = 0.001,
        **ppo_kwargs: Any,
    ) -> None:
        super().__init__(feature_dim, action_dim, **ppo_kwargs)
        self.vae = VAE(self.encoder, latent_dim=feature_dim).to(self.device)
        self.recon_coef = recon_coef
        self.kl_coef = kl_coef

    def _parameters(self) -> list[torch.nn.Parameter]:
        return [*super()._parameters(), *self.vae.extra_parameters()]

    def _extra_loss(
        self, batch: dict[str, torch.Tensor], features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        recon_loss, kl_loss, _ = self.vae.loss(batch["obs"], features=features)
        vae_loss = self.recon_coef * recon_loss + self.kl_coef * kl_loss
        stats = {
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item(),
            "vae_loss": vae_loss.item(),
            "latent_std": features.std(dim=0).detach().mean().item(),
        }
        return vae_loss, stats

    def _checkpoint_state(self) -> dict[str, Any]:
        return {
            **super()._checkpoint_state(),
            "vae_decoder": self.vae.decoder.state_dict(),
            "vae_fc_mu": self.vae.fc_mu.state_dict(),
            "vae_fc_logvar": self.vae.fc_logvar.state_dict(),
        }
