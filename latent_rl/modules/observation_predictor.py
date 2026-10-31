import torch
import torch.nn.functional as F
from torch import nn

from latent_rl.modules.decoder import ImpalaDecoder
from latent_rl.modules.jepa import JEPAPredictor


class ObservationPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 17,
        decoder: nn.Module | None = None,
        predictor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.predictor = (
            predictor
            if predictor is not None
            else JEPAPredictor(latent_dim, action_dim)
        )
        self.decoder = (
            decoder if decoder is not None else ImpalaDecoder(latent_dim=latent_dim)
        )

    def forward(
        self, latent: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action.ndim == 1:
            action_one_hot = F.one_hot(action, self.action_dim).float()
        else:
            action_one_hot = action.float()

        next_latent = self.predictor(latent, action_one_hot)
        next_obs = self.decoder(next_latent)
        return next_latent, next_obs

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        encoder: nn.Module,
        *,
        latent_loss_coef: float = 1.0,
        recon_loss_coef: float = 1.0,
        pred_obs_loss_coef: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            z_t = encoder(obs)
            z_next_true = encoder(next_obs)

        z_next_pred, o_next_pred = self(z_t, action)
        o_t_recon = self.decode(z_t)

        recon_curr_loss = F.mse_loss(o_t_recon, obs)
        recon_next_loss = F.mse_loss(o_next_pred, next_obs)
        latent_loss = F.mse_loss(z_next_pred, z_next_true)

        total_loss = (
            recon_loss_coef * recon_curr_loss
            + pred_obs_loss_coef * recon_next_loss
            + latent_loss_coef * latent_loss
        )

        stats = {
            "loss/total": total_loss.item(),
            "loss/recon_curr": recon_curr_loss.item(),
            "loss/recon_next": recon_next_loss.item(),
            "loss/latent": latent_loss.item(),
        }
        return total_loss, stats
