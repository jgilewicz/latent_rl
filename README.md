# Latent RL Benchmark

PPO on [Crafter](https://github.com/danijar/crafter): which encoder objective gives the best latent space?

| `--algo` | Extra encoder loss |
| --- | --- |
| `ppo` | none (PPO only) |
| `jepa` | predict the next latent (EMA target) + variance hinge |
| `vae` | reconstruction + KL to N(0, I) |
| `sae` | reconstruction + sparsity |

## Quickstart

```bash
uv sync
wandb login
just train jepa 0        # algo, seed
just sweep jepa "0 1 2"  # several seeds (default 0-4)
just train-predictor checkpoints/jepa/seed0/last.pt 0  # policy ckpt, predictor seed
just app                 # Gradio visualizer
just compare             # rliable stats + plots → plots/
just eval-buffer         # frozen 5M policies → eval_buffer/ (post-hoc data)
just check               # ruff + ty
```

Config: `latent_rl/config/<algo>_config.yaml` (~5M env steps per run; identical PPO hyperparameters for all algos).

## Logged metrics (W&B)

- **RL**: `crafter_score`, `episode_return`, `achievement/<name>`, `explained_variance`
- **Latent health**:
  - `diag/effective_rank`: how many latent dimensions are in use (low = collapse)
  - `probe/r2`: linear probe on frozen features predicting inventory + daylight
  - `diag/cka_prev`: similarity to the representation 10 iterations earlier (drift)

## Method comparison

- `just compare` pulls `main-5m` runs from W&B → `plots/` (config `latent_rl/config/compare_config.yaml`, plot text in Polish)
- Score per run: Crafter score from success rates averaged over the last 250k-step window before each budget (1M, 5M)
- Aggregates: Mean + IQM with 95% stratified bootstrap CI (rliable)
- Significance: probability of improvement P(X > Y) per pair, significant when the CI excludes 0.5; Welch t-test alongside
- Plots (PNG + PDF): `aggregate_{1M,5M}`, `probability_of_improvement_{1M,5M}`, `sample_efficiency`, `latent_diagnostics`, `achievements`
- Tables (CSV): `scores`, `aggregate`, `significance`, `achievements`

## Evaluation buffer (post-hoc)

- `just eval-buffer` → `eval_buffer/buffer.npz` + `meta.json` (config `latent_rl/config/eval_buffer_config.yaml`)
- Each 5M checkpoint drives 8 envs × 1250 steps on the same env seeds (`env_seed + i`); stochastic actions
- Every frame encoded by every encoder: `z_<algo>`, `z_next_<algo>` → cross-method comparisons on one frame set
- Per frame: `obs`, `next_obs` (true successor), `action`, `reward`, `terminated`, `truncated`, `source` (policy index), `env`, `episode`, `t`
- Labels of `obs`: `inventory` (raw counts, incl. vitals), `achievements` (unlocked so far), `daylight`, `sleeping`, `semantic` (9×7 visible cells, [row, col] = frame layout), `player_pos`, `biome`
- Sorted by (`episode`, `t`): episodes are contiguous; `meta.json` holds names, sources, field shapes

## Layout

- `latent_rl/scripts/`: `train_policy.py`, `train_predictor.py`, `app.py`
- `latent_rl/trainers/`: `PPOTrainer` + one subclass per algo
- `latent_rl/modules/`: encoder, decoder, actor-critic, JEPA, VAE, SAE
- `latent_rl/utils/diagnostics.py`: latent metrics
- Checkpoints: `checkpoints/<algo>/seed<seed>/last.pt` (+ `1M/last.pt`)
- `slurm/`: WCSS job scripts, see below

## WCSS cluster

```bash
slurm/sync.sh                                    # rsync working tree to wcss:~/latent_rl
ssh wcss 'cd latent_rl && sbatch slurm/setup.sbatch'   # venv in PD, once
ssh wcss 'cd latent_rl && sbatch slurm/train.sbatch'   # 4 algos × 5 seeds, tag main-5m
ssh wcss 'cd latent_rl && sbatch slurm/predictor.sbatch <path/to/last.pt>'  # 3 seeds
```

- Smoke test: `sbatch --array=0,5,10,15 --time=0:30:00 --export=ALL,TAG=smoke,N_ITER=25 slurm/train.sbatch`
