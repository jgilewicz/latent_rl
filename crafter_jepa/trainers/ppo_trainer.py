from typing import Any

import numpy as np
import torch

from crafter_jepa.modules import ActorCritic, CNNEncoder
from crafter_jepa.utils.logger import log_metrics
from crafter_jepa.utils.rollout_buffer import RolloutBuffer

OBS_SHAPE = (3, 64, 64)


class PPOTrainer:
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        dropout: float,
        *,
        n_steps: int = 2048,
        n_iterations: int = 500,
        batch_size: int = 64,
        epochs: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        epsilon: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_norm: float = 0.5,
    ) -> None:
        self.encoder = CNNEncoder(feature_dim)
        self.actor_critic = ActorCritic(feature_dim, action_dim, hidden_dim, dropout)
        self.device = next(self.actor_critic.parameters()).device
        self.n_steps = n_steps
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_norm = max_norm
        self._episode_return = 0.0

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        pixels = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return pixels.permute(2, 0, 1) / 255.0

    def _parameters(self) -> list[torch.nn.Parameter]:
        return [*self.encoder.parameters(), *self.actor_critic.parameters()]

    @staticmethod
    @torch.no_grad()
    def _advantage_estimation(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: torch.Tensor,
        gamma: float,
        lam: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        last_gae = torch.zeros_like(last_value, dtype=torch.float32)
        dones = dones.float()
        n_steps = len(advantages)

        for t in reversed(range(n_steps)):
            next_value = last_value if t == n_steps - 1 else values[t + 1]
            non_terminal = 1 - dones[t]

            delta = rewards[t] + gamma * next_value * non_terminal - values[t]
            last_gae = delta + gamma * lam * non_terminal * last_gae

            advantages[t] = last_gae

        return advantages, advantages + values

    @torch.no_grad()
    def _collect_rollout(
        self, env: Any, obs: np.ndarray, buffer: RolloutBuffer
    ) -> tuple[np.ndarray, torch.Tensor, list[float]]:
        episode_returns: list[float] = []
        self.encoder.eval()
        self.actor_critic.eval()

        for _ in range(self.n_steps):
            obs_t = self._to_tensor(obs)
            features = self.encoder(obs_t.unsqueeze(0))
            action, log_prob, _, value = self.actor_critic.get_action_and_value(
                features
            )
            action, log_prob, value = action[0], log_prob[0], value[0]
            next_obs, reward, done, _ = env.step(action.item())

            buffer.add(obs_t, action, log_prob, value, reward, done)

            self._episode_return += reward
            if done:
                episode_returns.append(self._episode_return)
                self._episode_return = 0.0

            obs = env.reset() if done else next_obs

        features = self.encoder(self._to_tensor(obs).unsqueeze(0))
        last_value = self.actor_critic.get_value(features)[0]
        return obs, last_value, episode_returns

    def _update(
        self, buffer: RolloutBuffer, optimizer: torch.optim.Optimizer
    ) -> dict[str, float]:
        stats: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }
        self.encoder.train()
        self.actor_critic.train()

        for _ in range(self.epochs):
            for batch in buffer.minibatches(self.batch_size):
                advantages = batch["advantages"]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                features = self.encoder(batch["obs"])
                _, log_prob, entropy, value = self.actor_critic.get_action_and_value(
                    features, batch["actions"]
                )
                ratio = torch.exp(log_prob - batch["log_probs"])

                policy_loss = -torch.min(
                    ratio * advantages,
                    torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages,
                ).mean()
                value_loss = ((value - batch["returns"]) ** 2).mean()
                entropy_bonus = entropy.mean()

                total_loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy_bonus
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._parameters(), self.max_norm)
                optimizer.step()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy_bonus.item())

        return {name: sum(vals) / len(vals) for name, vals in stats.items()}

    def train(self, env: Any) -> None:
        optimizer = torch.optim.AdamW(self._parameters(), lr=self.lr)
        buffer = RolloutBuffer(self.n_steps, OBS_SHAPE, self.device)
        obs = env.reset()

        for iteration in range(1, self.n_iterations + 1):
            obs, last_value, episode_returns = self._collect_rollout(env, obs, buffer)

            buffer.advantages, buffer.returns = self._advantage_estimation(
                buffer.rewards,
                buffer.values,
                buffer.dones,
                last_value,
                self.gamma,
                self.lam,
            )

            metrics = self._update(buffer, optimizer)
            if episode_returns:
                metrics["episode_return"] = sum(episode_returns) / len(episode_returns)
            log_metrics(iteration * self.n_steps, metrics)
            buffer.reset()
