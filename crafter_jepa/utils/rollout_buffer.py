from collections.abc import Iterator

import torch


class RolloutBuffer:
    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple[int, ...],
        device: torch.device,
        n_labels: int = 0,
    ) -> None:
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.device = device
        self.obs = torch.zeros(
            (n_steps, n_envs, *obs_shape), dtype=torch.uint8, device=device
        )
        self.next_obs = torch.zeros_like(self.obs)
        self.actions = torch.zeros(n_steps, n_envs, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        self.values = torch.zeros(n_steps, n_envs, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, device=device)
        self.dones = torch.zeros(n_steps, n_envs, device=device)
        self.advantages = torch.zeros(n_steps, n_envs, device=device)
        self.returns = torch.zeros(n_steps, n_envs, device=device)
        self.labels = torch.zeros(n_steps, n_envs, n_labels, device=device)
        self.pos = 0

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> None:
        if self.pos >= self.n_steps:
            raise RuntimeError(
                f"buffer full ({self.n_steps} steps); call reset() after the PPO update"
            )
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        if labels is not None:
            self.labels[self.pos] = labels
        self.pos += 1

    def reset(self) -> None:
        self.pos = 0

    def minibatches(self, batch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        if self.pos < self.n_steps:
            raise RuntimeError(
                f"buffer holds {self.pos}/{self.n_steps} steps; "
                "fill it before iterating"
            )

        total = self.n_steps * self.n_envs
        indices = torch.randperm(total, device=self.device)

        obs = self.obs.flatten(0, 1)
        next_obs = self.next_obs.flatten(0, 1)
        flat = {
            "actions": self.actions.flatten(),
            "log_probs": self.log_probs.flatten(),
            "advantages": self.advantages.flatten(),
            "returns": self.returns.flatten(),
        }

        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": obs[idx].float() / 255,
                "next_obs": next_obs[idx].float() / 255,
                **{name: tensor[idx] for name, tensor in flat.items()},
            }
