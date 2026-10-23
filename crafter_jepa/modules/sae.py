import torch
import torch.nn.functional as F
from torch import nn

from crafter_jepa.modules.decoder import ImpalaDecoder


class SAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module | None = None,
        latent_dim: int = 256,
        sparsity_target: float = 0.05,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.latent_dim = latent_dim
        self.decoder = decoder if decoder is not None else ImpalaDecoder(latent_dim)
        self.sparsity_target = sparsity_target

        self.fc_enc = nn.Linear(latent_dim, latent_dim)
        nn.init.xavier_uniform_(self.fc_enc.weight)
        nn.init.zeros_(self.fc_enc.bias)
        self.activation = nn.Sigmoid()

    def extra_parameters(self) -> list[nn.Parameter]:
        return [
            *self.fc_enc.parameters(),
            *self.decoder.parameters(),
        ]

    def encode_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.fc_enc(features))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)
        return self.encode_features(features)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(obs)
        recon = self.decode(encoded)
        return recon, encoded

    def sparsity_penalty(self, encoded: torch.Tensor) -> torch.Tensor:
        rho_hat = torch.mean(encoded, dim=0)
        rho = self.sparsity_target
        epsilon = 1e-8
        rho_hat = torch.clamp(rho_hat, min=epsilon, max=1.0 - epsilon)
        kl_divergence = rho * torch.log(rho / rho_hat) + (1.0 - rho) * torch.log(
            (1.0 - rho) / (1.0 - rho_hat)
        )
        return torch.sum(kl_divergence)

    def loss(
        self,
        obs: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features is not None:
            encoded = self.encode_features(features)
        else:
            encoded = self.encode(obs)

        recon = self.decode(encoded)
        recon_loss = F.mse_loss(recon, obs)
        sparsity_loss = self.sparsity_penalty(encoded)
        return recon_loss, sparsity_loss, recon
