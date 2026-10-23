from dataclasses import dataclass, field
from typing import Any

import numpy as np

from crafter_jepa.envs.crafter_env import CrafterGymEnv


@dataclass
class EnvSnapshot:
    prev_obs: np.ndarray
    next_obs: np.ndarray
    reward: float
    done: bool
    step_num: int
    total_reward: float
    health: int = 9
    food: int = 9
    water: int = 9
    energy: int = 9
    info: dict[str, Any] = field(default_factory=dict)


class CrafterSession:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.env = CrafterGymEnv(seed=seed)
        self.obs, _ = self.env.reset()
        self.prev_obs = self.obs.copy()
        self.step_num = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.last_info: dict[str, Any] = {}

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = seed
        self.env.close()
        self.env = CrafterGymEnv(seed=self.seed)
        self.obs, _ = self.env.reset()
        self.prev_obs = self.obs.copy()
        self.step_num = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.done = False
        self.last_info = {}
        return self.obs

    def step(self, action: int) -> EnvSnapshot:
        self.prev_obs = self.obs.copy()
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.done = bool(terminated or truncated)
        self.total_reward += float(reward)
        self.last_reward = float(reward)
        self.step_num += 1
        self.obs = next_obs
        self.last_info = info

        inv = info.get("inventory", {})
        return EnvSnapshot(
            prev_obs=self.prev_obs,
            next_obs=self.obs,
            reward=self.last_reward,
            done=self.done,
            step_num=self.step_num,
            total_reward=self.total_reward,
            health=int(inv.get("health", 9)),
            food=int(inv.get("food", 9)),
            water=int(inv.get("water", 9)),
            energy=int(inv.get("energy", 9)),
            info=info,
        )
