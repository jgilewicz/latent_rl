from typing import Any

import crafter
import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, VectorEnv

N_ACTIONS = 17
OBS_SHAPE_HWC = (64, 64, 3)


class CrafterGymEnv(gym.Env[np.ndarray, int]):
    observation_space = gym.spaces.Box(0, 255, OBS_SHAPE_HWC, dtype=np.uint8)
    action_space = gym.spaces.Discrete(N_ACTIONS)

    def __init__(self, seed: int, **crafter_kwargs: Any) -> None:
        self._env = crafter.Env(seed=seed, **crafter_kwargs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return self._env.reset(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        # crafter sets discount=0 only on death; time-limit ends keep discount=1
        terminated = done and info["discount"] == 0
        truncated = done and not terminated
        return obs, float(reward), terminated, truncated, info


def make_vec_env(n_envs: int, seed: int, **crafter_kwargs: Any) -> VectorEnv:
    return AsyncVectorEnv(
        [
            lambda i=i: CrafterGymEnv(seed * n_envs + i, **crafter_kwargs)
            for i in range(n_envs)
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
