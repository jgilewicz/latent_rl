import torch
import torch.nn.functional as F
from torch import nn

from latent_rl.modules.decoder import ImpalaDecoder


class VAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module | None = None,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.latent_dim = latent_dim
        self.decoder = decoder if decoder is not None else ImpalaDecoder(latent_dim)

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)

        nn.init.orthogonal_(self.fc_mu.weight, 1.0)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def extra_parameters(self) -> list[nn.Parameter]:
        return [
            *self.fc_mu.parameters(),
            *self.fc_logvar.parameters(),
            *self.decoder.parameters(),
        ]

    def encode_features(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fc_mu(features), self.fc_logvar(features)

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        return self.encode_features(features)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        deterministic: bool | None = None,
    ) -> torch.Tensor:
        is_deterministic = not self.training if deterministic is None else deterministic
        if is_deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(obs)
        z = self.reparameterize(mu, logvar, deterministic=deterministic)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss(
        self,
        obs: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features is not None:
            mu, logvar = self.encode_features(features)
        else:
            mu, logvar = self.encode(obs)

        z = self.reparameterize(mu, logvar, deterministic=False)
        recon = self.decode(z)

        recon_loss = F.mse_loss(recon, obs)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss, kl_loss, recon
