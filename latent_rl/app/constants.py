import glob

import numpy as np
import torch
from PIL import Image

CRAFTER_ACTIONS: list[str] = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]

BLANK_IMAGE = Image.fromarray(np.full((128, 128, 3), 10, dtype=np.uint8))


def tensor_to_img(t: torch.Tensor | None, size: int = 256) -> Image.Image:
    if t is None:
        return BLANK_IMAGE.resize((size, size), Image.Resampling.NEAREST)
    if t.ndim == 4:
        t = t[0]
    arr = t.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    img = Image.fromarray((arr * 255).astype(np.uint8))
    return img.resize((size, size), Image.Resampling.NEAREST)


def obs_to_img(obs: np.ndarray, size: int = 256) -> Image.Image:
    img = Image.fromarray(obs)
    return img.resize((size, size), Image.Resampling.NEAREST)


def find_checkpoints() -> list[str]:
    files = sorted(glob.glob("checkpoints/**/last.pt", recursive=True))
    return files if files else ["Brak zapisanych checkpointów"]
