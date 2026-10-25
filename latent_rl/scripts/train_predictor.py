import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from gymnasium.vector import VectorEnv

from latent_rl.envs.crafter_env import make_vec_env
from latent_rl.modules import ActorCritic, ImpalaEncoder, ObservationPredictor
from latent_rl.utils.device import resolve_device
from latent_rl.utils.logger import (
    LOGGER_NAME,
    finish_wandb,
    init_wandb,
    log_metrics,
    log_model_artifact,
    setup_logging,
)
from latent_rl.utils.rollout_buffer import RolloutBuffer

OBS_SHAPE = (3, 64, 64)
DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "predictor_config.yaml"
)
logger = logging.getLogger(LOGGER_NAME)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="policy last.pt written by train_policy.py",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tag",
        default="dev",
        help="W&B experiment tag, e.g. predictor-5m",
    )
    return parser.parse_args()


def load_policy_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"policy checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    missing = {"algo", "seed", "encoder", "actor_critic"} - ckpt.keys()
    if missing:
        raise KeyError(
            f"{path} lacks {sorted(missing)}; "
            "retrain with the current train_policy.py, which stores them"
        )
    return ckpt


def build_models(
    ckpt: dict[str, Any], model_cfg: dict[str, Any], device: torch.device
) -> tuple[ImpalaEncoder, ActorCritic, ObservationPredictor]:
    feature_dim = model_cfg["feature_dim"]
    action_dim = model_cfg["action_dim"]

    encoder = ImpalaEncoder(feature_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval().requires_grad_(False)

    actor_critic = ActorCritic(
        feature_dim, action_dim, model_cfg["hidden_dim"], 0.0
    ).to(device)
    actor_critic.load_state_dict(ckpt["actor_critic"])
    actor_critic.eval().requires_grad_(False)

    model = ObservationPredictor(latent_dim=feature_dim, action_dim=action_dim)
    model.to(device)
    if "predictor" in ckpt:
        logger.info("warm-starting latent predictor from checkpoint")
        model.predictor.load_state_dict(ckpt["predictor"])
    for key in ("vae_decoder", "sae_decoder"):
        if key in ckpt:
            logger.info("warm-starting decoder from %s", key)
            model.decoder.load_state_dict(ckpt[key])
    return encoder, actor_critic, model


def to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(obs, device=device).permute(0, 3, 1, 2)


def real_next_obs(
    obs: np.ndarray, done: np.ndarray, info: dict[str, Any], device: torch.device
) -> torch.Tensor:
    next_obs = obs.copy()
    for i in np.flatnonzero(done):
        next_obs[i] = info["final_obs"][i]
    return to_tensor(next_obs, device)


@torch.no_grad()
def collect(
    envs: VectorEnv,
    obs: np.ndarray,
    buffer: RolloutBuffer,
    encoder: ImpalaEncoder,
    actor_critic: ActorCritic,
) -> np.ndarray:
    device = buffer.device
    for _ in range(buffer.n_steps):
        obs_t = to_tensor(obs, device)
        features = encoder(obs_t.float() / 255)
        action, log_prob, _, value = actor_critic.get_action_and_value(features)
        obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
        done = terminated | truncated
        buffer.add(
            obs_t,
            real_next_obs(obs, done, info, device),
            action,
            log_prob,
            value,
            torch.as_tensor(reward, dtype=torch.float32, device=device),
            torch.as_tensor(done, dtype=torch.float32, device=device),
        )
    return obs


def update(
    model: ObservationPredictor,
    encoder: ImpalaEncoder,
    buffer: RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
) -> dict[str, float]:
    model.train()
    stats: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(train_cfg["epochs"]):
        for batch in buffer.minibatches(train_cfg["batch_size"]):
            loss, batch_stats = model.loss(
                batch["obs"],
                batch["actions"],
                batch["next_obs"],
                encoder,
                latent_loss_coef=train_cfg["latent_loss_coef"],
                recon_loss_coef=train_cfg["recon_loss_coef"],
                pred_obs_loss_coef=train_cfg["pred_obs_loss_coef"],
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, value in batch_stats.items():
                stats[name].append(value)
    return {name: sum(vals) / len(vals) for name, vals in stats.items()}


def save(model: ObservationPredictor, path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(
        {
            **state,
            "predictor": model.predictor.state_dict(),
            "decoder": model.decoder.state_dict(),
        },
        tmp,
    )
    tmp.replace(path)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    device = resolve_device(train_cfg["device"], "training.device")
    torch.manual_seed(args.seed)

    ckpt = load_policy_checkpoint(args.checkpoint, device)
    algo, policy_seed = ckpt["algo"], ckpt["seed"]
    encoder, actor_critic, model = build_models(ckpt, cfg["model"], device)
    out_path = (
        args.checkpoint.parent / "predictor" / f"seed{args.seed}"
    ) / "predictor_decoder.pt"

    env_cfg = dict(cfg["env"])
    n_envs = env_cfg.pop("n_envs")
    envs = make_vec_env(n_envs, args.seed, **env_cfg)
    buffer = RolloutBuffer(train_cfg["n_steps"], n_envs, OBS_SHAPE, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])
    run_config = {**cfg, "seed": args.seed, "policy_checkpoint": str(args.checkpoint)}
    init_wandb(
        **cfg["wandb"],
        name=f"pred_{algo}_crafter_s{policy_seed}_p{args.seed}",
        group=f"pred-{algo}",
        config=run_config,
        tags=[args.tag, algo, "crafter"],
    )

    n_iterations = train_cfg["n_iterations"]
    try:
        obs, _ = envs.reset()
        for iteration in range(1, n_iterations + 1):
            buffer.reset()
            obs = collect(envs, obs, buffer, encoder, actor_critic)
            metrics = update(model, encoder, buffer, optimizer, train_cfg)
            log_metrics(iteration * buffer.n_steps * n_envs, metrics)
            if iteration % train_cfg["save_every"] == 0 or iteration == n_iterations:
                save(model, out_path, {"iteration": iteration, **run_config})
        log_model_artifact(out_path, f"pred-{algo}-s{policy_seed}", [f"p{args.seed}"])
    finally:
        envs.close()
        finish_wandb()


if __name__ == "__main__":
    main()
