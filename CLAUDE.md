# CLAUDE.md

Project: latent-rl — PPO on Crafter; encoder ablation (PPO vs JEPA vs VAE vs SAE). Global rules in `~/.claude/CLAUDE.md`.

## Structure

- `latent_rl/scripts/train_policy.py` — `--algo`, `--seed`, `--tag` (W&B experiment tag, default `dev`), `--n-iterations` (override, smoke tests) → `latent_rl/config/<algo>_config.yaml`; builds vec env, trainer; checkpoint dir gets `/seed<seed>`
- `latent_rl/scripts/app.py` — launches Gradio visualizer
- `latent_rl/scripts/compare_methods.py` — `just compare`; config `latent_rl/config/compare_config.yaml` (all constants + Polish plot text, no diacritics); W&B runs by tag → rliable Mean/IQM + P(X>Y) CIs + Welch → `plots/` (PNG+PDF, CSV); plotting in `latent_rl/utils/plotting.py`
- `latent_rl/scripts/collect_eval_buffer.py` — `just eval-buffer`; config `latent_rl/config/eval_buffer_config.yaml` (5M checkpoint paths, `min_env_steps` guard, biome groups); one shared post-hoc dataset → `eval_buffer/buffer.npz` + `meta.json`; frames encoded by all 4 encoders (`z_<algo>`, `z_next_<algo>`); labels via `CrafterGymEnv.state()`
- `latent_rl/scripts/latent_comparison.py` — `just latent-comparison`; config `latent_rl/config/latent_comparison_config.yaml`; thin orchestration only (config, `check_algos`, CSV + plots via `write_intervals`); metrics live in `latent_rl/analysis/`
- `latent_rl/analysis/` — post-hoc metrics on `EvalBuffer`, plain functions grouped by metric family:
  - `bootstrap.py` — `Statistic` (row indices → one value per algo), `bootstrap` (episode resampling, same seed for every metric), `summarize` (full, mean, 95% CI), `metric_table` (long table algo × metric)
  - `similarity.py` — `cka` → 4×4 linear CKA (`plots/cka_matrix.*`, `cka.csv`)
  - `dimensions.py` — `two_nn_ratios` (dedupe, Theiler window), `dimensions` → TwoNN + effective rank (`plots/dimensions.*`)
  - `temporal.py` — `temporal` → `latent_step`, `step_correlation`, `r2_next`, `r2_delta` (lstsq fit once on non-held-out episodes, scored on held-out rows of each resample) (`plots/temporal.*`)
  - `embedding.py` — `umap_embeddings` (`umap:` params), `umap_colorings` (biome classes must match `meta.json` order) (`plots/umap.*`)
  - new metric = function returning `metric_table(...)` + a panel entry in the config section named like the CSV
- `latent_rl/scripts/train_predictor.py` — `--checkpoint <last.pt>`, `--seed`, `--tag`; trains `ObservationPredictor` on a frozen policy (visualization only, not a comparison metric); warm-starts from the checkpoint's predictor/decoder; writes `<ckpt dir>/predictor/seed<s>/predictor_decoder.pt` + W&B artifact `pred-<algo>-s<policy seed>:p<s>`
- `latent_rl/config/` — `ppo_config.yaml`, `jepa_config.yaml`, `vae_config.yaml`, `sae_config.yaml`, `predictor_config.yaml`, `compare_config.yaml`, `eval_buffer_config.yaml`, `latent_comparison_config.yaml`
- `latent_rl/envs/crafter_env.py` — `CrafterGymEnv` (`state()`: full labels of the current frame), `make_vec_env`; `SEMANTIC_NAMES`, `VIEW_GRID` (9×7 world cells), `CELL_PX`
- `latent_rl/modules/` — `ImpalaEncoder`, `ImpalaDecoder`, `ActorCritic`, `JEPA`, `VAE`, `SAE`, `ObservationPredictor`
- `latent_rl/trainers/ppo_trainer.py` — `PPOTrainer` (hooks `_extra_loss`, `_after_step`)
- `latent_rl/trainers/jepa_trainer.py` — `JEPATrainer(PPOTrainer)`; config `latent_rl/config/jepa_config.yaml` (`jepa_coef`, `ema_tau`, `var_coef`)
- `latent_rl/trainers/vae_trainer.py` — `VAETrainer(PPOTrainer)`; config `latent_rl/config/vae_config.yaml` (`recon_coef`, `kl_coef`)
- `latent_rl/trainers/sae_trainer.py` — `SAETrainer(PPOTrainer)`; config `latent_rl/config/sae_config.yaml` (`recon_coef`, `sparsity_coef`, `sparsity_target`)
- `latent_rl/utils/` — `resolve_device`, `EvalBuffer` (post-hoc loader: lazy cached fields, `latents`, `resample_episodes`), `RolloutBuffer`, `log_metrics`, `LinearSchedule` (entropy-coef decay), `diagnostics.py` (`OnlineProbe`, `effective_rank`, `linear_cka`, `linear_cka_matrix`, `nn_distance_ratios`, `two_nn_dimension`, `explained_variance`), `plotting.py` (`compare_methods` figures + `cka_matrix`, `metric_intervals`, `umap_grid`)

## Post-hoc latent comparison

- Only 1 seed per algo in the buffer → uncertainty = episode bootstrap (stratified by `source`), not across seeds
- Plotted value = bootstrap mean; `full` column = full-sample estimate (effective rank mean sits slightly below `full`: duplicated episodes)
- `effective_rank` uses `eigvalsh` of the Gram in float64 (≈10× faster than `svdvals`, same values) — keep float64
- TwoNN: float64 distances (float32 gives r1 = 0) + Theiler window 50 (without it ID ≈ 1.5 = trajectory); plateau from ~10 up to whole episode
- UMAP: `init: pca` (spectral init fails on small eigengap); `n_jobs: 1` explicit (else warning with `random_state`)
- `umap-learn==0.5.12` pulls scikit-learn, pynndescent

## Latent diagnostics (training-time)

- `PPOTrainer._diagnostics` every iteration on `diag_frames` rollout frames: `diag/effective_rank`, `probe/r2`, `probe/mse`, `diag/cka_prev`; plus `explained_variance`
- Probe labels: `CrafterGymEnv` puts `info["labels"]` (`LABEL_NAMES`: inventory/9 + daylight) on reset and step; buffer stores labels of `obs_t`
- `OnlineProbe` must stay detached with its own optimizer: it must not affect the encoder (fair comparison)
- CKA: fixed frames from iteration 1, reference snapshot refreshed every `cka_lag` iterations
- XAI is post-hoc only (frozen checkpoints); methods listed in `todo.md`

## Conventions

- PyTorch, class-based; no tests (user decision)
- Checks: `just check` (ruff + ty)
- `pandas==2.3.3` pinned: rliable caps `arch<8`, and arch 7 fails to import on pandas 3
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

- `trainer.device` ∈ {`cpu`, `cuda`}; validated by `latent_rl.utils.device.resolve_device`

## WCSS (SLURM)

- Account `hpc-jakgil6519-1780267238`, `lem-gpu-short`, 1×H100 + 4 cores per run (env-bound; 9 cores is only ~7% faster)
- `slurm/sync.sh` rsyncs the working tree to `wcss:~/latent_rl` (no git on the cluster) and writes `CODE_VERSION`
- `slurm/env.sh` (sourced): venv, uv cache, managed Python, checkpoints, logs all in `/lustre/pd03/<account>/latent_rl/` ($HOME is near quota); cwd = that dir
- `slurm/setup.sbatch` — `uv sync --frozen` into the PD venv; rerun after dependency changes
- `slurm/train.sbatch` — array 0-19: task i → `ALGOS[i/5]`, seed `i%5` (main-5m: 0-2 first, then 3,4,8,9,13,14,18,19); env `TAG`, `N_ITER`
- `slurm/predictor.sbatch <last.pt>` — array 0-2 = predictor seeds
- Experiment spec + record: `experiments/<name>/` (wcss-experiment skill)
