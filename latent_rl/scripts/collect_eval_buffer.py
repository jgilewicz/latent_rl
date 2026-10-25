import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from latent_rl.envs.crafter_env import (
    ACHIEVEMENT_NAMES,
    CELL_PX,
    LABEL_NAMES,
    SEMANTIC_NAMES,
    VIEW_GRID,
    CrafterGymEnv,
)
from latent_rl.modules import ActorCritic, ImpalaEncoder
from latent_rl.utils.device import resolve_device
from latent_rl.utils.logger import LOGGER_NAME, setup_logging

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "eval_buffer_config.yaml"
)
CHECKPOINT_KEYS = {"algo", "seed", "env_steps", "config", "encoder", "actor_critic"}
logger = logging.getLogger(LOGGER_NAME)

Policy = tuple[ImpalaEncoder, ActorCritic]
Arrays = dict[str, np.ndarray]


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    cfg = yaml.safe_load(path.read_text())
    unknown = {m for ms in cfg["biomes"].values() for m in ms} - set(SEMANTIC_NAMES)
    if unknown:
        raise ValueError(
            f"unknown biome materials {sorted(unknown)} in {path}; "
            f"choose from {SEMANTIC_NAMES}"
        )
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_checkpoint(
    path: Path, algo: str, min_env_steps: int, device: torch.device
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{algo} checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    missing = CHECKPOINT_KEYS - ckpt.keys()
    if missing:
        raise KeyError(f"{path} lacks {sorted(missing)}; not a train_policy.py ckpt")
    if ckpt["algo"] != algo:
        raise ValueError(f"{path} holds {ckpt['algo']!r}, config lists it as {algo}")
    if ckpt["env_steps"] < min_env_steps:
        raise ValueError(
            f"{path} is at {ckpt['env_steps']} env steps (< {min_env_steps}); "
            "point the config at the final checkpoint, not seed<s>/1M"
        )
    return ckpt


def build_policy(ckpt: dict[str, Any], device: torch.device) -> Policy:
    trainer = ckpt["config"]["trainer"]
    encoder = ImpalaEncoder(trainer["feature_dim"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    actor_critic = ActorCritic(
        trainer["feature_dim"], trainer["action_dim"], trainer["hidden_dim"], 0.0
    ).to(device)
    actor_critic.load_state_dict(ckpt["actor_critic"])
    return encoder.eval(), actor_critic.eval()


def to_input(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    pixels = torch.as_tensor(frames, device=device).permute(0, 3, 1, 2)
    return pixels.float() / 255


@torch.no_grad()
def rollout(policy: Policy, cfg: dict[str, Any], device: torch.device) -> Arrays:
    encoder, actor_critic = policy
    envs = [CrafterGymEnv(cfg["env_seed"] + i) for i in range(cfg["n_envs"])]
    obs = np.stack([env.reset()[0] for env in envs])
    states = [env.state() for env in envs]
    episode = list(range(len(envs)))
    next_episode = len(envs)
    t = [0] * len(envs)
    records: defaultdict[str, list[Any]] = defaultdict(list)
    for _ in range(cfg["n_steps"]):
        features = encoder(to_input(obs, device))
        actions = actor_critic.get_action_and_value(features)[0].cpu().numpy()
        for i, env in enumerate(envs):
            next_obs, reward, terminated, truncated, _ = env.step(int(actions[i]))
            for key, value in states[i].items():
                records[key].append(value)
            records["obs"].append(obs[i].copy())
            records["next_obs"].append(next_obs)
            records["action"].append(actions[i])
            records["reward"].append(reward)
            records["terminated"].append(terminated)
            records["truncated"].append(truncated)
            records["env"].append(i)
            records["episode"].append(episode[i])
            records["t"].append(t[i])
            t[i] += 1
            if terminated or truncated:
                next_obs, _ = env.reset()
                episode[i], next_episode = next_episode, next_episode + 1
                t[i] = 0
            obs[i] = next_obs
            states[i] = env.state()
    return {key: np.stack(values) for key, values in records.items()}


def biome_labels(semantic: np.ndarray, biomes: dict[str, list[str]]) -> np.ndarray:
    cells = semantic.reshape(len(semantic), -1)
    counts = [
        np.isin(cells, [SEMANTIC_NAMES.index(m) for m in materials]).sum(axis=1)
        for materials in biomes.values()
    ]
    return np.argmax(np.stack(counts, axis=1), axis=1).astype(np.uint8)


@torch.no_grad()
def encode(
    encoder: ImpalaEncoder, frames: np.ndarray, batch: int, device: torch.device
) -> np.ndarray:
    chunks = [
        encoder(to_input(frames[i : i + batch], device)).cpu().numpy()
        for i in range(0, len(frames), batch)
    ]
    return np.concatenate(chunks).astype(np.float32)


def merge_rollouts(rollouts: list[Arrays]) -> Arrays:
    # episode ids restart per policy; offset them so they are global
    offset = 0
    for source, arrays in enumerate(rollouts):
        arrays["episode"] = arrays["episode"] + offset
        arrays["source"] = np.full(len(arrays["obs"]), source)
        offset = int(arrays["episode"].max()) + 1
    merged = {key: np.concatenate([r[key] for r in rollouts]) for key in rollouts[0]}
    order = np.lexsort((merged["t"], merged["episode"]))
    return {key: value[order] for key, value in merged.items()}


DTYPES = {
    "action": np.uint8,
    "reward": np.float32,
    "env": np.uint8,
    "episode": np.int32,
    "t": np.int32,
    "source": np.uint8,
}


def write(out: Path, arrays: Arrays, meta: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: v.astype(DTYPES.get(k, v.dtype)) for k, v in arrays.items()}
    # ty matches **kwargs against the `allow_pickle: bool` parameter
    np.savez_compressed(
        out / "buffer.npz",
        **arrays,  # ty: ignore[invalid-argument-type]
    )
    meta["fields"] = {
        key: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for key, value in arrays.items()
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    cfg = load_config(parse_args().config)
    setup_logging()
    device = resolve_device(cfg["device"], "device")
    policies: dict[str, Policy] = {}
    sources: dict[str, dict[str, Any]] = {}
    rollouts = []
    for algo, path in cfg["checkpoints"].items():
        ckpt = load_checkpoint(Path(path), algo, cfg["min_env_steps"], device)
        policies[algo] = build_policy(ckpt, device)
        sources[algo] = {
            "checkpoint": path,
            **{k: ckpt[k] for k in ("seed", "env_steps")},
        }
        torch.manual_seed(cfg["seed"])
        rollouts.append(rollout(policies[algo], cfg, device))
        logger.info("rolled out %s: %d frames", algo, len(rollouts[-1]["obs"]))

    arrays = merge_rollouts(rollouts)
    arrays["biome"] = biome_labels(arrays["semantic"], cfg["biomes"])
    for algo, (encoder, _) in policies.items():
        for frames, latent in (("obs", "z"), ("next_obs", "z_next")):
            z = encode(encoder, arrays[frames], cfg["encode_batch"], device)
            arrays[f"{latent}_{algo}"] = z
        logger.info("encoded all frames with the %s encoder", algo)

    meta = {
        "sources": sources,
        "algos": list(cfg["checkpoints"]),
        "inventory_names": list(LABEL_NAMES[:-1]),
        "achievement_names": list(ACHIEVEMENT_NAMES),
        "semantic_names": list(SEMANTIC_NAMES),
        "biome_names": list(cfg["biomes"]),
        "view_grid_xy": list(VIEW_GRID),
        "cell_px": CELL_PX,
        "config": cfg,
    }
    write(Path(cfg["out_dir"]), arrays, meta)
    logger.info("wrote %d frames to %s", len(arrays["obs"]), cfg["out_dir"])


if __name__ == "__main__":
    main()
