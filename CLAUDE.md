# CLAUDE.md

Project: latent-rl — PPO on Crafter; encoder ablation (PPO vs JEPA vs VAE vs SAE). Global rules in `~/.claude/CLAUDE.md`.

## Structure

- `latent_rl/scripts/train_policy.py` — `--algo`, `--seed`, `--tag` (W&B experiment tag, default `dev`), `--n-iterations` (override, smoke tests) → `latent_rl/config/<algo>_config.yaml`; builds vec env, trainer; checkpoint dir gets `/seed<seed>`
- `latent_rl/scripts/app.py` — launches Gradio visualizer
- `latent_rl/scripts/train_predictor.py` — `--checkpoint <last.pt>`, `--seed`, `--tag`; trains `ObservationPredictor` on a frozen policy (visualization only, not a comparison metric); warm-starts from the checkpoint's predictor/decoder; writes `<ckpt dir>/predictor/seed<s>/predictor_decoder.pt` + W&B artifact `pred-<algo>-s<policy seed>:p<s>`
- `latent_rl/config/` — `ppo_config.yaml`, `jepa_config.yaml`, `vae_config.yaml`, `sae_config.yaml`, `predictor_config.yaml`
- `latent_rl/envs/crafter_env.py` — `CrafterGymEnv`, `make_vec_env`
- `latent_rl/modules/` — `ImpalaEncoder`, `ImpalaDecoder`, `ActorCritic`, `JEPA`, `VAE`, `SAE`, `ObservationPredictor`
- `latent_rl/trainers/ppo_trainer.py` — `PPOTrainer` (hooks `_extra_loss`, `_after_step`)
- `latent_rl/trainers/jepa_trainer.py` — `JEPATrainer(PPOTrainer)`; config `latent_rl/config/jepa_config.yaml` (`jepa_coef`, `ema_tau`, `var_coef`)
- `latent_rl/trainers/vae_trainer.py` — `VAETrainer(PPOTrainer)`; config `latent_rl/config/vae_config.yaml` (`recon_coef`, `kl_coef`)
- `latent_rl/trainers/sae_trainer.py` — `SAETrainer(PPOTrainer)`; config `latent_rl/config/sae_config.yaml` (`recon_coef`, `sparsity_coef`, `sparsity_target`)
- `latent_rl/utils/` — `RolloutBuffer`, `log_metrics`, `LinearSchedule` (entropy-coef decay), `diagnostics.py` (`OnlineProbe`, `effective_rank`, `linear_cka`, `explained_variance`)

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
- `numba` dependency: without it `opensimplex` runs Crafter worldgen in pure Python (reset ~700 ms vs ~50 ms on WCSS); sync vec env stalls on every reset

## JEPA

- Encoder lives in the trainer and is shared by reference with `JEPA.online_encoder`; `target` is a Polyak copy (no grad)
- Optimizer params: trainer params + `jepa.predictor` only (not `jepa.parameters()`: duplicates encoder)
- Buffer stores `next_obs` (true successor via `info["final_obs"]` on done), so no mask
- Watch `latent_std`: → 0 = collapse

## Checkpoints / logging

- `PPOTrainer._save_checkpoint` → `<checkpoint_dir>/seed<seed>/last.pt` (atomic replace, every `checkpoint_every` + final); subclasses extend `_checkpoint_state`
- `.pt` = all weights + `algo`, `seed`, `config`, `iteration`, `env_steps` (no optimizer state)
- First iteration ≥1M steps → `seed<s>/1M/last.pt`; W&B artifact `ckpt-<algo>-s<seed>` aliases `1M`, final `<budget>M` + `latest`; file inside is always `last.pt`
- Per-iteration `achievement/<name>` = fraction of finished episodes with that achievement (from `info["final_info"]["achievements"]`, `_`-prefixed mask keys skipped)
- `crafter_score` logged per iteration (from `_achievement_metrics`); `env_steps` logged as a metric (x-axis)
- W&B: project `latent_rl`, run `<algo>_crafter_s<seed>`, group = algo, tags `[<tag>, algo, crafter, <budget>M]`
- Seeds: env seed `seed * n_envs + i`; `just sweep <algo>` = seeds 0-4
- Roadmap in `todo.md`
- `log_metrics` logs to stdout + W&B; `init_wandb` called in `latent_rl/scripts/train_policy.py` from the `wandb:` config section

## Device

- `trainer.device` ∈ {`cpu`, `cuda`}; validated in `PPOTrainer._resolve_device`

## WCSS (SLURM)

- Account `hpc-jakgil6519-1780267238`, `lem-gpu-short`, 1×H100 + 4 cores per run (env-bound; 9 cores is only ~7% faster)
- `slurm/sync.sh` rsyncs the working tree to `wcss:~/latent_rl` (no git on the cluster) and writes `CODE_VERSION`
- `slurm/env.sh` (sourced): venv, uv cache, managed Python, checkpoints, logs all in `/lustre/pd03/<account>/latent_rl/` ($HOME is near quota); cwd = that dir
- `slurm/setup.sbatch` — `uv sync --frozen` into the PD venv; rerun after dependency changes
- `slurm/train.sbatch` — array 0-11: task i → `ALGOS[i/3]`, seed `i%3`; env `TAG`, `N_ITER`
- `slurm/predictor.sbatch <last.pt>` — array 0-2 = predictor seeds
- Experiment spec + record: `experiments/<name>/` (wcss-experiment skill)
