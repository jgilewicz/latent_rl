from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from crafter_jepa.envs.crafter_env import make_vec_env
from crafter_jepa.modules import ActorCritic, ImpalaEncoder, ObservationPredictor
from crafter_jepa.utils.logger import setup_logging
from crafter_jepa.utils.rollout_buffer import RolloutBuffer
from crafter_jepa.utils.visualization import save_rollout_gif, save_rollout_grid

OBS_SHAPE = (3, 64, 64)


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text())


def get_config_path() -> Path:
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return Path(sys.argv[1])
    return Path(__file__).resolve().parent.parent / "config" / "predictor_config.yaml"


def to_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    pixels = torch.as_tensor(obs, device=device)
    return pixels.permute(0, 3, 1, 2)


def real_next_obs(
    obs: np.ndarray, done: np.ndarray, info: dict[str, Any], device: torch.device
) -> torch.Tensor:
    next_obs = obs.copy()
    for i in np.flatnonzero(done):
        next_obs[i] = info["final_obs"][i]
    return to_tensor(next_obs, device)


@torch.no_grad()
def evaluate_rollout(
    env_fn: Any,
    encoder: ImpalaEncoder,
    actor_critic: ActorCritic,
    model: ObservationPredictor,
    horizon: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Runs one episode in a single env and compares ground truth with imagined rollout."""
    encoder.eval()
    actor_critic.eval()
    model.eval()

    eval_env = env_fn()
    obs_np, _ = eval_env.reset()

    gt_frames: list[torch.Tensor] = []
    actions: list[int] = []

    obs_t = to_tensor(obs_np[None], device).float() / 255
    gt_frames.append(obs_t[0])

    curr_obs_np = obs_np
    for _ in range(horizon):
        curr_t = to_tensor(curr_obs_np[None], device).float() / 255
        feat = encoder(curr_t)
        action, _, _, _ = actor_critic.get_action_and_value(feat)
        act_int = int(action.item())
        actions.append(act_int)

        curr_obs_np, _, done, truncated, info = eval_env.step(act_int)
        if done or truncated:
            actual_next = info["final_obs"]
        else:
            actual_next = curr_obs_np

        gt_frames.append(to_tensor(actual_next[None], device).float()[0] / 255)
        if done or truncated:
            break

    eval_env.close()

    actual_horizon = len(actions)
    if actual_horizon == 0:
        return torch.empty(0), torch.empty(0), []

    initial_obs = gt_frames[0].unsqueeze(0)
    actions_t = torch.tensor([actions], dtype=torch.long, device=device)

    _, pred_obs = model.rollout(initial_obs, actions_t, encoder)

    gt_tensor = torch.stack(gt_frames[1:], dim=0)
    pred_tensor = pred_obs[0]

    step_mses = [
        float(F.mse_loss(pred_tensor[t], gt_tensor[t]).item())
        for t in range(actual_horizon)
    ]
    return gt_tensor, pred_tensor, step_mses


def main() -> None:
    setup_logging()
    config_path = get_config_path()
    cfg = load_config(config_path)

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)

    ckpt_path = Path(cfg["checkpoint"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {ckpt_path}")

    output_dir_val = cfg.get("output_dir")
    if output_dir_val:
        out_dir = Path(output_dir_val)
    else:
        out_dir = ckpt_path.parent / "predictor"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = dict(cfg.get("env", {}))
    n_envs = env_cfg.pop("n_envs", 8)

    model_cfg = dict(cfg.get("model", {}))
    feature_dim = model_cfg.get("feature_dim", 256)
    action_dim = model_cfg.get("action_dim", 17)
    hidden_dim = model_cfg.get("hidden_dim", 256)

    train_cfg = dict(cfg.get("training", {}))
    device_name = train_cfg.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    n_steps = train_cfg.get("n_steps", 256)
    n_iterations = train_cfg.get("n_iterations", 100)
    batch_size = train_cfg.get("batch_size", 256)
    epochs = train_cfg.get("epochs", 4)
    lr = float(train_cfg.get("lr", 5e-4))
    latent_loss_coef = float(train_cfg.get("latent_loss_coef", 1.0))
    recon_loss_coef = float(train_cfg.get("recon_loss_coef", 1.0))
    pred_obs_loss_coef = float(train_cfg.get("pred_obs_loss_coef", 1.0))
    save_every = train_cfg.get("save_every", 25)

    eval_cfg = dict(cfg.get("evaluation", {}))
    visualize_every = eval_cfg.get("visualize_every", 25)
    visualize_horizon = eval_cfg.get("visualize_horizon", 8)

    print(f"Loaded config from {config_path}")
    print(f"Loading policy checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    encoder = ImpalaEncoder(feature_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval().requires_grad_(False)

    actor_critic = ActorCritic(feature_dim, action_dim, hidden_dim, 0.0).to(device)
    actor_critic.load_state_dict(ckpt["actor_critic"])
    actor_critic.eval().requires_grad_(False)

    model = ObservationPredictor(latent_dim=feature_dim, action_dim=action_dim).to(
        device
    )

    if "predictor" in ckpt:
        print("Warm-starting latent predictor weights from checkpoint.")
        model.predictor.load_state_dict(ckpt["predictor"])
    if "vae_decoder" in ckpt:
        print("Warm-starting decoder weights from checkpoint.")
        model.decoder.load_state_dict(ckpt["vae_decoder"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    buffer = RolloutBuffer(n_steps, n_envs, OBS_SHAPE, device)
    envs = make_vec_env(n_envs, seed, **env_cfg)

    obs, _ = envs.reset()

    print(
        f"Training observation predictor on {device} for {n_iterations} iterations..."
    )

    try:
        for iteration in range(1, n_iterations + 1):
            model.train()
            buffer.reset()

            with torch.no_grad():
                for _ in range(n_steps):
                    obs_t = to_tensor(obs, device)
                    features = encoder(obs_t.float() / 255)
                    action, log_prob, _, value = actor_critic.get_action_and_value(
                        features
                    )

                    next_obs_np, reward, term, trunc, info = envs.step(
                        action.cpu().numpy()
                    )
                    done = term | trunc
                    reward_t = torch.as_tensor(
                        reward, dtype=torch.float32, device=device
                    )
                    done_t = torch.as_tensor(done, dtype=torch.float32, device=device)

                    buffer.add(
                        obs_t,
                        real_next_obs(next_obs_np, done, info, device),
                        action,
                        log_prob,
                        value,
                        reward_t,
                        done_t,
                    )
                    obs = next_obs_np

            epoch_stats: defaultdict[str, list[float]] = defaultdict(list)
            for _ in range(epochs):
                for batch in buffer.minibatches(batch_size):
                    loss, stats = model.loss(
                        batch["obs"],
                        batch["actions"],
                        batch["next_obs"],
                        encoder,
                        latent_loss_coef=latent_loss_coef,
                        recon_loss_coef=recon_loss_coef,
                        pred_obs_loss_coef=pred_obs_loss_coef,
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    for k, v in stats.items():
                        epoch_stats[k].append(v)

            mean_stats = {k: sum(v) / len(v) for k, v in epoch_stats.items()}
            print(
                f"[Iter {iteration:03d}/{n_iterations}] "
                f"total_loss: {mean_stats['loss/total']:.4f} | "
                f"recon_curr: {mean_stats['loss/recon_curr']:.4f} | "
                f"recon_next: {mean_stats['loss/recon_next']:.4f} | "
                f"latent_loss: {mean_stats['loss/latent']:.4f}"
            )

            if iteration % visualize_every == 0 or iteration == n_iterations:
                from crafter_jepa.envs.crafter_env import CrafterGymEnv

                gt_rollout, pred_rollout, step_mses = evaluate_rollout(
                    lambda: CrafterGymEnv(seed + 1000, **env_cfg),
                    encoder,
                    actor_critic,
                    model,
                    visualize_horizon,
                    device,
                )
                if len(step_mses) > 0:
                    mse_str = ", ".join(
                        f"t+{t + 1}: {m:.4f}" for t, m in enumerate(step_mses)
                    )
                    print(f"  Multi-step rollout MSE: [{mse_str}]")

                    grid_path = out_dir / f"rollout_iter_{iteration:04d}.png"
                    save_rollout_grid(gt_rollout, pred_rollout, grid_path)
                    save_rollout_grid(
                        gt_rollout, pred_rollout, out_dir / "latest_rollout.png"
                    )
                    save_rollout_gif(
                        gt_rollout, pred_rollout, out_dir / "latest_rollout.gif"
                    )
                    print(f"  Visual rollout saved to {grid_path}")

            if iteration % save_every == 0 or iteration == n_iterations:
                save_path = out_dir / "predictor_decoder.pt"
                torch.save(
                    {
                        "iteration": iteration,
                        "predictor": model.predictor.state_dict(),
                        "decoder": model.decoder.state_dict(),
                        "policy_checkpoint": str(ckpt_path),
                        "config": cfg,
                    },
                    save_path,
                )
                print(f"  Checkpoint saved to {save_path}")

    finally:
        envs.close()


if __name__ == "__main__":
    main()
