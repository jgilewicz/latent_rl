from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from crafter_jepa.app.constants import CRAFTER_ACTIONS
from crafter_jepa.modules import (
    ActorCritic,
    ImpalaEncoder,
    ObservationPredictor,
    SAE,
    VAE,
)


@dataclass
class StepPrediction:
    recon_curr: torch.Tensor | None
    pred_next: torch.Tensor | None
    mse_val: float | None
    v_est: float
    act_prob: float
    act_name: str
    act_idx: int
    algo: str


class ModelSession:
    def __init__(self, algo: str = "jepa", checkpoint_path: str | None = None) -> None:
        self.algo = algo.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = 256
        self.action_dim = len(CRAFTER_ACTIONS)
        self.hidden_dim = 256

        self.encoder = ImpalaEncoder(self.feature_dim).to(self.device).eval()
        self.actor_critic = (
            ActorCritic(self.feature_dim, self.action_dim, self.hidden_dim, 0.0)
            .to(self.device)
            .eval()
        )
        self.decoder: torch.nn.Module | None = None
        self.predictor_model: ObservationPredictor | None = None
        self.vae_model: VAE | None = None
        self.sae_model: SAE | None = None
        self.loaded_ckpt = "Inicjalizacja losowa (brak checkpointa)"

        self._setup_algo()
        if checkpoint_path and Path(checkpoint_path).is_file():
            self.load_checkpoint(checkpoint_path)

    def _setup_algo(self) -> None:
        if self.algo == "vae":
            self.vae_model = (
                VAE(self.encoder, latent_dim=self.feature_dim).to(self.device).eval()
            )
            self.decoder = self.vae_model.decoder
        elif self.algo == "sae":
            self.sae_model = (
                SAE(self.encoder, latent_dim=self.feature_dim).to(self.device).eval()
            )
            self.decoder = self.sae_model.decoder
        elif self.algo == "jepa":
            self.predictor_model = (
                ObservationPredictor(
                    latent_dim=self.feature_dim, action_dim=self.action_dim
                )
                .to(self.device)
                .eval()
            )
            self.decoder = self.predictor_model.decoder

    def load_checkpoint(self, path: str) -> str:
        p = Path(path)
        if not p.is_file():
            return f"Nie znaleziono pliku: {path}"
        try:
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            if "encoder" in ckpt:
                self.encoder.load_state_dict(ckpt["encoder"])
            if "actor_critic" in ckpt:
                self.actor_critic.load_state_dict(ckpt["actor_critic"])

            if self.algo == "vae":
                if "vae_decoder" in ckpt and self.vae_model is not None:
                    self.vae_model.decoder.load_state_dict(ckpt["vae_decoder"])
                if "vae_fc_mu" in ckpt and self.vae_model is not None:
                    self.vae_model.fc_mu.load_state_dict(ckpt["vae_fc_mu"])
            elif self.algo == "sae":
                if "sae_decoder" in ckpt and self.sae_model is not None:
                    self.sae_model.decoder.load_state_dict(ckpt["sae_decoder"])
                if "sae_fc_enc" in ckpt and self.sae_model is not None:
                    self.sae_model.fc_enc.load_state_dict(ckpt["sae_fc_enc"])
            elif self.algo == "jepa":
                if "predictor" in ckpt and self.predictor_model is not None:
                    self.predictor_model.predictor.load_state_dict(ckpt["predictor"])

            pred_ckpt_path = p.parent / "predictor" / "predictor_decoder.pt"
            if pred_ckpt_path.is_file():
                pred_ckpt = torch.load(
                    pred_ckpt_path, map_location=self.device, weights_only=False
                )
                if self.predictor_model is None:
                    self.predictor_model = (
                        ObservationPredictor(
                            latent_dim=self.feature_dim, action_dim=self.action_dim
                        )
                        .to(self.device)
                        .eval()
                    )
                self.predictor_model.predictor.load_state_dict(pred_ckpt["predictor"])
                self.predictor_model.decoder.load_state_dict(pred_ckpt["decoder"])
                self.decoder = self.predictor_model.decoder

            self.loaded_ckpt = str(p)
            return f"Wczytano: {p.name}"
        except Exception as e:
            return f"Błąd wczytywania: {e}"

    @torch.no_grad()
    def select_agent_action(self, obs_np: np.ndarray) -> tuple[int, float, float]:
        t_obs = (
            torch.as_tensor(obs_np, device=self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255
        )
        feat = self.encoder(t_obs)
        act_t, _, _, value = self.actor_critic.get_action_and_value(feat)
        act_idx = int(act_t.item())

        dist, _ = self.actor_critic(feat)
        prob = float(torch.exp(dist.log_prob(act_t)).item())
        v_est = float(value.item())
        return act_idx, prob, v_est

    @torch.no_grad()
    def process_step(
        self,
        obs_t_np: np.ndarray,
        action: int,
        next_obs_np: np.ndarray,
    ) -> StepPrediction:
        t_obs = (
            torch.as_tensor(obs_t_np, device=self.device).permute(2, 0, 1).unsqueeze(0)
        )
        t_obs_norm = t_obs.float() / 255
        t_next = (
            torch.as_tensor(next_obs_np, device=self.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        t_next_norm = t_next.float() / 255

        features = self.encoder(t_obs_norm)
        _, _, _, value = self.actor_critic.get_action_and_value(features)
        v_est = float(value.item())

        dist, _ = self.actor_critic(features)
        act_t = torch.tensor([action], device=self.device)
        act_prob = float(torch.exp(dist.log_prob(act_t)).item())
        act_name = CRAFTER_ACTIONS[action]

        if self.algo == "ppo":
            return StepPrediction(
                recon_curr=None,
                pred_next=None,
                mse_val=None,
                v_est=v_est,
                act_prob=act_prob,
                act_name=act_name,
                act_idx=action,
                algo="ppo",
            )

        recon_curr: torch.Tensor | None = None
        pred_next: torch.Tensor | None = None

        if self.algo == "vae" and self.vae_model is not None:
            mu, _ = self.vae_model.encode_features(features)
            recon_curr = self.vae_model.decode(mu)

            if self.predictor_model is not None:
                _, pred_next = self.predictor_model(mu, act_t)
            else:
                next_feat = self.encoder(t_next_norm)
                next_mu, _ = self.vae_model.encode_features(next_feat)
                pred_next = self.vae_model.decode(next_mu)

        elif self.algo == "sae" and self.sae_model is not None:
            encoded = self.sae_model.encode_features(features)
            recon_curr = self.sae_model.decode(encoded)

            if self.predictor_model is not None:
                _, pred_next = self.predictor_model(encoded, act_t)
            else:
                next_feat = self.encoder(t_next_norm)
                next_enc = self.sae_model.encode_features(next_feat)
                pred_next = self.sae_model.decode(next_enc)

        elif self.algo == "jepa":
            if self.predictor_model is not None:
                recon_curr = self.predictor_model.decode(features)
                _, pred_next = self.predictor_model(features, act_t)
            elif self.decoder is not None:
                recon_curr = self.decoder(features)
                pred_next = self.decoder(self.encoder(t_next_norm))

        mse_val: float | None = None
        if pred_next is not None:
            mse_val = float(F.mse_loss(pred_next, t_next_norm).item())

        return StepPrediction(
            recon_curr=recon_curr,
            pred_next=pred_next,
            mse_val=mse_val,
            v_est=v_est,
            act_prob=act_prob,
            act_name=act_name,
            act_idx=action,
            algo=self.algo,
        )
