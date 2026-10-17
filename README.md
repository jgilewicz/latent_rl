# crafter-jepa

PPO baseline for [Crafter](https://github.com/danijar/crafter); planned ablation of the visual encoder: end-to-end CNN vs. VAE vs. JEPA.

## Layout

- `main.py` — entry point; `--algo` selects trainer + `config/<algo>_config.yaml`
- `config/ppo_config.yaml` — env and trainer hyperparameters
- `crafter_jepa/modules/` — `CNNEncoder` (pixels → features), `ActorCritic` (MLP heads)
- `crafter_jepa/trainers/ppo_trainer.py` — rollout, GAE, clipped PPO update
- `crafter_jepa/utils/` — `RolloutBuffer`, `log_metrics` (stdlib logging; to be replaced by wandb)

## Pipeline

- Obs `64x64x3` → `CNNEncoder` → `feature_dim` vector → `ActorCritic` (Categorical policy + value)
- Encoder trained end-to-end by the PPO loss (single AdamW over encoder + actor-critic)
- Each iteration: `n_steps` rollout → GAE → `epochs` × minibatch updates
- Logged per iteration: `policy_loss`, `value_loss`, `entropy`, `episode_return`

## Usage

- `uv sync`
- `just train` — default `ppo`
- `just train PPO` / `just train algo=PPO` — case-insensitive
- `just check` — ruff format/check + ty
- `just test` — pytest

## Known limitations

- Single env, no parallel workers
- Time-limit truncation treated as terminal
- No tests yet
