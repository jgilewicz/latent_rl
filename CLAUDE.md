# CLAUDE.md

Project: latent-rl — PPO on Crafter; encoder ablation (PPO vs JEPA vs VAE vs SAE). Global rules in `~/.claude/CLAUDE.md`.

## Structure

- `crafter_jepa/scripts/train_policy.py` — `--algo`, `--seed` → `crafter_jepa/config/<algo>_config.yaml`; builds vec env, trainer; checkpoint dir gets `/seed<seed>`
- `crafter_jepa/scripts/app.py` — launches Gradio visualizer
- `crafter_jepa/scripts/train_predictor.py` — trains `ObservationPredictor` on frozen policy checkpoint (configured via YAML)
- `crafter_jepa/config/` — `ppo_config.yaml`, `jepa_config.yaml`, `vae_config.yaml`, `sae_config.yaml`, `predictor_config.yaml`
- `crafter_jepa/envs/crafter_env.py` — `CrafterGymEnv`, `make_vec_env`
- `crafter_jepa/modules/` — `ImpalaEncoder`, `ImpalaDecoder`, `ActorCritic`, `JEPA`, `VAE`, `SAE`, `ObservationPredictor`
- `crafter_jepa/trainers/ppo_trainer.py` — `PPOTrainer` (hooks `_extra_loss`, `_after_step`)
- `crafter_jepa/trainers/jepa_trainer.py` — `JEPATrainer(PPOTrainer)`; config `crafter_jepa/config/jepa_config.yaml` (`jepa_coef`, `ema_tau`, `var_coef`)
- `crafter_jepa/trainers/vae_trainer.py` — `VAETrainer(PPOTrainer)`; config `crafter_jepa/config/vae_config.yaml` (`recon_coef`, `kl_coef`)
- `crafter_jepa/trainers/sae_trainer.py` — `SAETrainer(PPOTrainer)`; config `crafter_jepa/config/sae_config.yaml` (`recon_coef`, `sparsity_coef`, `sparsity_target`)
- `crafter_jepa/utils/` — `RolloutBuffer`, `log_metrics`, `LinearSchedule` (entropy-coef decay), `diagnostics.py` (`OnlineProbe`, `effective_rank`, `linear_cka`, `explained_variance`)

## Latent diagnostics (training-time)

- `PPOTrainer._diagnostics` every iteration on `diag_frames` rollout frames: `diag/effective_rank`, `probe/r2`, `probe/mse`, `diag/cka_prev`; plus `explained_variance`
- Probe labels: `CrafterGymEnv` puts `info["labels"]` (`LABEL_NAMES`: inventory/9 + daylight) on reset and step; buffer stores labels of `obs_t`
- `OnlineProbe` must stay detached with its own optimizer: it must not affect the encoder (fair comparison)
- CKA: fixed frames from iteration 1, reference snapshot refreshed every `cka_lag` iterations
- XAI is post-hoc only (frozen checkpoints); methods listed in `todo.md`

## Conventions

- PyTorch, class-based; no tests (user decision)
- Checks: `just check` (ruff + ty)
- Scripts using `AsyncVectorEnv` need an `if __name__ == "__main__":` guard (spawn) and `PYTHONPATH=.`

## Vectorization

- `M = env.n_envs`; one `RolloutBuffer` shaped `[T, M, ...]`, flattened to `T*M` only in `minibatches`
- Buffer obs `uint8`; `/255` per minibatch / before encoder
- `terminated`: crafter `discount == 0`; else `truncated`
- Autoreset `SAME_STEP`; truncation bootstrap via `info["final_obs"]` added to reward
- `n_steps` is per env; samples per iteration = `n_steps * M`

## JEPA

- Encoder lives in the trainer and is shared by reference with `JEPA.online_encoder`; `target` is a Polyak copy (no grad)
- Optimizer params: trainer params + `jepa.predictor` only (not `jepa.parameters()`: duplicates encoder)
- Buffer stores `next_obs` (true successor via `info["final_obs"]` on done), so no mask
- Watch `latent_std`: → 0 = collapse

## Checkpoints / logging

- `PPOTrainer._save_checkpoint` → `<checkpoint_dir>/seed<seed>/last.pt` (atomic replace); subclasses extend `_checkpoint_state`
- Per-iteration `achievement/<name>` = fraction of finished episodes with that achievement (from `info["final_info"]["achievements"]`, `_`-prefixed mask keys skipped)
- `crafter_score` logged per iteration (from `_achievement_metrics`); W&B group = algo
- Seeds: env seed `seed * n_envs + i`; `just sweep <algo>` = seeds 0-4
- Roadmap in `todo.md`
- `log_metrics` logs to stdout + W&B; `init_wandb` called in `crafter_jepa/scripts/train_policy.py` from the `wandb:` config section

## Device

- `trainer.device` ∈ {`cpu`, `cuda`}; validated in `PPOTrainer._resolve_device`
