from collections.abc import Iterator

import torch


class RolloutBuffer:
    def __init__(
        self, n_steps: int, obs_shape: tuple[int, ...], device: torch.device
    ) -> None:
        self.n_steps = n_steps
        self.device = device
        self.obs = torch.zeros((n_steps, *obs_shape), device=device)
        self.actions = torch.zeros(n_steps, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(n_steps, device=device)
        self.values = torch.zeros(n_steps, device=device)
        self.rewards = torch.zeros(n_steps, device=device)
        self.dones = torch.zeros(n_steps, device=device)
        self.advantages = torch.zeros(n_steps, device=device)
        self.returns = torch.zeros(n_steps, device=device)
        self.pos = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
    ) -> None:
        if self.pos >= self.n_steps:
            raise RuntimeError(
                f"buffer full ({self.n_steps} steps); call reset() after the PPO update"
            )
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.pos += 1

    def reset(self) -> None:
        self.pos = 0

    def minibatches(self, batch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        if self.pos < self.n_steps:
            raise RuntimeError(
                f"buffer holds {self.pos}/{self.n_steps} steps; "
                "fill it before iterating"
            )
        indices = torch.randperm(self.n_steps, device=self.device)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs": self.obs[idx],
                "actions": self.actions[idx],
                "log_probs": self.log_probs[idx],
                "advantages": self.advantages[idx],
                "returns": self.returns[idx],
            }
