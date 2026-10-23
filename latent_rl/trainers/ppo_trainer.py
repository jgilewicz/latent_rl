import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium.vector import VectorEnv

from latent_rl.envs.crafter_env import LABEL_NAMES
from latent_rl.modules import ActorCritic, ImpalaEncoder
from latent_rl.utils.diagnostics import (
    OnlineProbe,
    effective_rank,
    explained_variance,
    linear_cka,
)
from latent_rl.utils.logger import log_metrics, log_model_artifact
from latent_rl.utils.rollout_buffer import RolloutBuffer
from latent_rl.utils.schedule import LinearSchedule

OBS_SHAPE = (3, 64, 64)
MILESTONE_STEPS = 1_000_000


class PPOTrainer:
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        dropout: float,
        *,
        device: str = "cpu",
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
        ent_coef_final: float = 0.005,
        ent_decay_steps: int = 5_000_000,
        max_norm: float = 0.5,
        checkpoint_dir: str = "checkpoints",
        checkpoint_every: int = 50,
        cka_lag: int = 10,
        diag_frames: int = 1024,
        artifact_name: str,
        checkpoint_meta: dict[str, Any],
    ) -> None:
        self.device = self._resolve_device(device)
        self.encoder = ImpalaEncoder(feature_dim).to(self.device)
        self.actor_critic = ActorCritic(
            feature_dim, action_dim, hidden_dim, dropout
        ).to(self.device)
        self.n_steps = n_steps
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.vf_coef = vf_coef
        self.ent_schedule = LinearSchedule(ent_coef, ent_coef_final, ent_decay_steps)
        self.max_norm = max_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_every = checkpoint_every
        self.artifact_name = artifact_name
        self.checkpoint_meta = checkpoint_meta
        self.cka_lag = cka_lag
        self.diag_frames = diag_frames
        self.probe = OnlineProbe(feature_dim, len(LABEL_NAMES), self.device)
        self._episode_return = np.zeros(0)
        self._labels = torch.zeros(0)
        self._cka_frames: torch.Tensor | None = None
        self._cka_prev: torch.Tensor | None = None

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {name!r}")
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device 'cuda' requested but CUDA is unavailable; "
                "set trainer.device: cpu in the config"
            )
        return torch.device(name)

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        pixels = torch.as_tensor(obs, device=self.device)
        return pixels.permute(0, 3, 1, 2)

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

    def _real_next_obs(
        self, obs: np.ndarray, done: np.ndarray, info: dict[str, Any]
    ) -> torch.Tensor:
        next_obs = obs.copy()
        for i in np.flatnonzero(done):
            next_obs[i] = info["final_obs"][i]
        return self._to_tensor(next_obs)

    @torch.no_grad()
    def _truncation_bonus(
        self, truncated: np.ndarray, info: dict[str, Any]
    ) -> torch.Tensor:
        bonus = torch.zeros(len(truncated), device=self.device)
        idx = np.flatnonzero(truncated)
        if len(idx) == 0:
            return bonus
        final_obs = np.stack([info["final_obs"][i] for i in idx])
        features = self.encoder(self._to_tensor(final_obs).float() / 255)
        bonus[idx] = self.gamma * self.actor_critic.get_value(features)
        return bonus

    @torch.no_grad()
    def _collect_rollout(
        self, env: VectorEnv, obs: np.ndarray, buffer: RolloutBuffer
    ) -> tuple[np.ndarray, torch.Tensor, list[float], list[dict[str, int]]]:
        episode_returns: list[float] = []
        episode_achievements: list[dict[str, int]] = []
        self.encoder.eval()
        self.actor_critic.eval()

        for _ in range(self.n_steps):
            obs_t = self._to_tensor(obs)
            features = self.encoder(obs_t.float() / 255)
            action, log_prob, _, value = self.actor_critic.get_action_and_value(
                features
            )
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = terminated | truncated
            reward_t = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            reward_t += self._truncation_bonus(truncated & ~terminated, info)

            buffer.add(
                obs_t,
                self._real_next_obs(obs, done, info),
                action,
                log_prob,
                value,
                reward_t,
                torch.as_tensor(done, dtype=torch.float32, device=self.device),
                self._labels,
            )
            self._labels = torch.as_tensor(info["labels"], device=self.device)

            self._episode_return += reward
            for i in np.flatnonzero(done):
                episode_returns.append(float(self._episode_return[i]))
                self._episode_return[i] = 0.0
                episode_achievements.append(
                    {
                        k: int(v[i])
                        for k, v in info["final_info"]["achievements"].items()
                        if not k.startswith("_")
                    }
                )

        features = self.encoder(self._to_tensor(obs).float() / 255)
        last_value = self.actor_critic.get_value(features)
        return obs, last_value, episode_returns, episode_achievements

    @staticmethod
    def _achievement_metrics(episodes: list[dict[str, int]]) -> dict[str, float]:
        if not episodes:
            return {}
        rates = {
            name: sum(ep[name] > 0 for ep in episodes) / len(episodes)
            for name in episodes[0]
        }
        log_mean = sum(math.log(1 + 100 * r) for r in rates.values()) / len(rates)
        return {
            **{f"achievement/{name}": rate for name, rate in rates.items()},
            "crafter_score": math.exp(log_mean) - 1,
        }

    def _diagnostics(self, buffer: RolloutBuffer, iteration: int) -> dict[str, float]:
        self.encoder.eval()
        total = buffer.n_steps * buffer.n_envs
        idx = torch.randperm(total, device=self.device)[: self.diag_frames]
        obs = buffer.obs.flatten(0, 1)[idx].float() / 255
        labels = buffer.labels.flatten(0, 1)[idx]
        with torch.no_grad():
            features = self.encoder(obs)

        metrics = {"diag/effective_rank": effective_rank(features)}
        metrics.update(self.probe.fit_and_score(features, labels))
        metrics.update(self._cka_metrics(obs, iteration))
        return metrics

    @torch.no_grad()
    def _cka_metrics(self, obs: torch.Tensor, iteration: int) -> dict[str, float]:
        if self._cka_frames is None:
            self._cka_frames = obs
        current = self.encoder(self._cka_frames)
        metrics = {}
        if self._cka_prev is not None:
            metrics["diag/cka_prev"] = linear_cka(current, self._cka_prev)
        if self._cka_prev is None or iteration % self.cka_lag == 0:
            self._cka_prev = current
        return metrics

    def _extra_loss(
        self, batch: dict[str, torch.Tensor], features: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return features.new_zeros(()), {}

    def _after_step(self) -> None:
        pass

    def _checkpoint_state(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "actor_critic": self.actor_critic.state_dict(),
        }

    def _save_checkpoint(self, iteration: int, env_steps: int, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            **self.checkpoint_meta,
            "iteration": iteration,
            "env_steps": env_steps,
            **self._checkpoint_state(),
        }
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        tmp.replace(path)

    def _checkpoint(self, iteration: int, env_steps: int, prev_steps: int) -> None:
        if prev_steps < MILESTONE_STEPS <= env_steps:
            milestone = self.checkpoint_dir / "1M" / "last.pt"
            self._save_checkpoint(iteration, env_steps, milestone)
            log_model_artifact(milestone, self.artifact_name, ["1M"])
        is_last = iteration == self.n_iterations
        if iteration % self.checkpoint_every == 0 or is_last:
            final = self.checkpoint_dir / "last.pt"
            self._save_checkpoint(iteration, env_steps, final)
        if is_last:
            budget = f"{round(env_steps / 1e6, 1):g}M"
            log_model_artifact(final, self.artifact_name, [budget, "latest"])

    def _update(
        self,
        buffer: RolloutBuffer,
        optimizer: torch.optim.Optimizer,
        ent_coef: float,
    ) -> dict[str, float]:
        stats: defaultdict[str, list[float]] = defaultdict(list)
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
                extra_loss, extra_stats = self._extra_loss(batch, features)

                total_loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - ent_coef * entropy_bonus
                    + extra_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._parameters(), self.max_norm)
                optimizer.step()
                self._after_step()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy_bonus.item())
                for name, value_ in extra_stats.items():
                    stats[name].append(value_)

        return {name: sum(vals) / len(vals) for name, vals in stats.items()}

    def train(self, env: VectorEnv, n_envs: int) -> None:
        self._episode_return = np.zeros(n_envs)
        optimizer = torch.optim.AdamW(self._parameters(), lr=self.lr)
        buffer = RolloutBuffer(
            self.n_steps, n_envs, OBS_SHAPE, self.device, len(LABEL_NAMES)
        )
        obs, info = env.reset()
        self._labels = torch.as_tensor(info["labels"], device=self.device)

        for iteration in range(1, self.n_iterations + 1):
            obs, last_value, episode_returns, achievements = self._collect_rollout(
                env, obs, buffer
            )

            buffer.advantages, buffer.returns = self._advantage_estimation(
                buffer.rewards,
                buffer.values,
                buffer.dones,
                last_value,
                self.gamma,
                self.lam,
            )

            metrics: dict[str, float] = {
                "explained_variance": explained_variance(buffer.values, buffer.returns)
            }
            metrics.update(self._diagnostics(buffer, iteration))

            prev_steps = (iteration - 1) * self.n_steps * n_envs
            env_steps = iteration * self.n_steps * n_envs
            ent_coef = self.ent_schedule(prev_steps)
            metrics.update(self._update(buffer, optimizer, ent_coef))
            metrics["ent_coef"] = ent_coef
            if episode_returns:
                metrics["episode_return"] = sum(episode_returns) / len(episode_returns)
            metrics.update(self._achievement_metrics(achievements))
            metrics["env_steps"] = env_steps
            log_metrics(env_steps, metrics)
            buffer.reset()
            self._checkpoint(iteration, env_steps, prev_steps)
