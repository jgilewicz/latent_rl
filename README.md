# Latent RL Benchmark

PPO on [Crafter](https://github.com/danijar/crafter) with different latent-space objectives for the encoder: plain PPO (done), JEPA (done, joint mode), VAE and SAE (planned).

## Layout

- `main.py` — entry point; `--algo` selects trainer + `config/<algo>_config.yaml`
- `config/<algo>_config.yaml` (`ppo`, `jepa`) — env (`n_envs`, crafter kwargs) and trainer (`device`, hyperparameters)
- `crafter_jepa/modules/` — `ImpalaEncoder` (IMPALA ResNet + MLP + LayerNorm, pixels → features), `ActorCritic` (MLP heads), `JEPA` (predictor, EMA target encoder, latent-prediction loss)
- `crafter_jepa/envs/crafter_env.py` — `CrafterGymEnv` (Gymnasium API), `make_vec_env` (`AsyncVectorEnv`, `SAME_STEP` autoreset)
- `crafter_jepa/trainers/ppo_trainer.py` — vectorized rollout, GAE, clipped PPO update
- `crafter_jepa/trainers/jepa_trainer.py` — `JEPATrainer(PPOTrainer)`: adds JEPA + variance loss via `_extra_loss` / `_after_step` hooks
- `crafter_jepa/utils/` — `RolloutBuffer` (`[T, M]`, `uint8` `obs` + `next_obs`), `log_metrics` (stdlib logging; to be replaced by wandb)

## Pipeline

- `M` envs step in parallel; obs `[M,64,64,3]` `uint8` → `ImpalaEncoder` → `feature_dim` vector → `ActorCritic` (Categorical policy + value)
- Encoder trained end-to-end by the PPO loss (single AdamW over encoder + actor-critic)
- JEPA (`--algo jepa`): `loss = L_ppo + jepa_coef * L_jepa + var_coef * L_var`
  - `L_jepa` = MSE(predictor(enc(o_t), onehot(a_t)), sg(target_enc(o_{t+1})))
  - target encoder = Polyak/EMA copy of the shared encoder (`ema_tau`), updated after every optimizer step
  - `L_var` = hinge on per-dim latent std (anti-collapse)
  - `next_obs` = `info["final_obs"]` for finished envs (autoreset returns the next episode's first frame otherwise)
- Each iteration: `n_steps` × `M` rollout (shared buffer) → GAE → `epochs` × minibatch updates
- `trainer.device`: `cpu` or `cuda` (fails fast if CUDA unavailable); models + buffer live on it
- Checkpoints: weights only (`encoder`, `actor_critic`, + `predictor`, `target_encoder` for JEPA) → `<checkpoint_dir>/last.pt`, overwritten every `checkpoint_every` iterations and at the end; no resume
- Config budget: `n_steps=512` × 8 envs × 1221 iterations ≈ 5M env steps, `ent_coef` 0.02 → 0.005 linear over `ent_decay_steps` env steps (`LinearSchedule`, logged as `ent_coef`)
- W&B logs `achievement/<name>` success rates per iteration
- Metrics go to stdout and W&B (`wandb:` section of the config; run name `<algo>-seed<seed>`, W&B group = `<algo>`; needs `wandb login`)
- Time-limit truncation bootstrapped with `gamma * V(final_obs)`
- Logged per iteration: `policy_loss`, `value_loss`, `entropy`, `episode_return`; JEPA adds `jepa_loss`, `var_loss`, `latent_std` (→ 0 means collapse)

## Usage

- `uv sync`
- `just train` — default `ppo`; `just train jepa` for JEPA
- `just train PPO` / `just train algo=PPO` — case-insensitive
- `just check` — ruff format/check + ty
- `just test` — pytest

## Known limitations

- Env stepping is CPU-bound; GPU only helps the forward/update
- No tests yet
- Seeds: `just train <algo> <seed>` (`--seed`, default 0); `just sweep <algo>` runs seeds 0-4 sequentially; seed sets `torch.manual_seed` and env seeds `seed * n_envs + i` (no overlap between seeds)
- Checkpoints go to `<checkpoint_dir>/seed<seed>/last.pt`
- W&B also logs `crafter_score` (geometric-mean score, %) from episodes finished in the iteration
- Roadmap: `todo.md` (VAE, SAE, 5-seed experiments, XAI candidates)
