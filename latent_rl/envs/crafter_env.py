from typing import Any

import crafter
import crafter.constants
import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, VectorEnv

N_ACTIONS = 17
LABEL_NAMES = (
    "health",
    "food",
    "drink",
    "energy",
    "sapling",
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
    "daylight",
)
MAX_INVENTORY = 9.0
OBS_SHAPE_HWC = (64, 64, 3)
# crafter.constants sets its attributes from data.yaml at import time
ACHIEVEMENT_NAMES = tuple(
    crafter.constants.achievements  # ty: ignore[unresolved-attribute]
)
# crafter.engine.SemanticView ids: materials, then object types
SEMANTIC_NAMES = (
    "void",
    "water",
    "grass",
    "stone",
    "path",
    "sand",
    "tree",
    "lava",
    "coal",
    "iron",
    "diamond",
    "table",
    "furnace",
    "player",
    "cow",
    "zombie",
    "skeleton",
    "arrow",
    "plant",
)
# world cells drawn in the frame (x, y); the bottom 2 of 9 rows show inventory
VIEW_GRID = (9, 7)
CELL_PX = OBS_SHAPE_HWC[0] // 9


class CrafterGymEnv(gym.Env[np.ndarray, int]):
    observation_space = gym.spaces.Box(0, 255, OBS_SHAPE_HWC, dtype=np.uint8)
    action_space = gym.spaces.Discrete(N_ACTIONS)

    def __init__(self, seed: int, **crafter_kwargs: Any) -> None:
        self._env = crafter.Env(seed=seed, **crafter_kwargs)

    def _labels(self, inventory: dict[str, int]) -> np.ndarray:
        # probe targets for the current frame, scaled to [0, 1]
        items = [inventory[name] / MAX_INVENTORY for name in LABEL_NAMES[:-1]]
        return np.array([*items, self._env._world.daylight], dtype=np.float32)

    def _visible_semantic(self) -> np.ndarray:
        offset = np.array(VIEW_GRID) // 2
        pad = int(offset.max())
        world = np.pad(self._env._sem_view(), pad)
        x, y = np.array(self._env._player.pos) + pad - offset
        crop = world[x : x + VIEW_GRID[0], y : y + VIEW_GRID[1]]
        return crop.T.astype(np.uint8)

    def state(self) -> dict[str, Any]:
        player = self._env._player
        return {
            "inventory": np.array(
                [player.inventory[name] for name in LABEL_NAMES[:-1]], np.uint8
            ),
            "achievements": np.array(
                [player.achievements[name] > 0 for name in ACHIEVEMENT_NAMES]
            ),
            "daylight": np.float32(self._env._world.daylight),
            "sleeping": bool(player.sleeping),
            # [row, col] = [y, x], aligned with the frame's pixel layout
            "semantic": self._visible_semantic(),
            "player_pos": np.array(player.pos, np.int16),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        obs = self._env.reset()
        return obs, {"labels": self._labels(self._env._player.inventory)}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        # crafter sets discount=0 only on death; time-limit ends keep discount=1
        terminated = done and info["discount"] == 0
        truncated = done and not terminated
        info["labels"] = self._labels(info["inventory"])
        return obs, float(reward), terminated, truncated, info


def make_vec_env(n_envs: int, seed: int, **crafter_kwargs: Any) -> VectorEnv:
    return AsyncVectorEnv(
        [
            lambda i=i: CrafterGymEnv(seed * n_envs + i, **crafter_kwargs)
            for i in range(n_envs)
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
