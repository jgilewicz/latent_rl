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
just check               # ruff + ty
```

Config: `latent_rl/config/<algo>_config.yaml` (~5M env steps per run; identical PPO hyperparameters for all algos).

## Logged metrics (W&B)

- **RL**: `crafter_score`, `episode_return`, `achievement/<name>`, `explained_variance`
- **Latent health**:
  - `diag/effective_rank`: how many latent dimensions are in use (low = collapse)
  - `probe/r2`: linear probe on frozen features predicting inventory + daylight
  - `diag/cka_prev`: similarity to the representation 10 iterations earlier (drift)

## Layout

- `latent_rl/scripts/`: `train_policy.py`, `train_predictor.py`, `app.py`
- `latent_rl/trainers/`: `PPOTrainer` + one subclass per algo
- `latent_rl/modules/`: encoder, decoder, actor-critic, JEPA, VAE, SAE
- `latent_rl/utils/diagnostics.py`: latent metrics
- Checkpoints: `checkpoints/<algo>/seed<seed>/last.pt` (+ `1M/last.pt`)
- `slurm/`: WCSS job scripts, see below

## Checkpoints from W&B

```bash
wandb artifact get latent_rl/ckpt-jepa-s0:5M --root checkpoints/jepa/seed0      # or :1M
wandb artifact get latent_rl/pred-jepa-s0:p0 --root checkpoints/jepa/seed0/predictor
```

## WCSS cluster

```bash
slurm/sync.sh                                    # rsync working tree to wcss:~/latent_rl
ssh wcss 'cd latent_rl && sbatch slurm/setup.sbatch'   # venv in PD, once
ssh wcss 'cd latent_rl && sbatch slurm/train.sbatch'   # 4 algos × 3 seeds, tag main-5m
ssh wcss 'cd latent_rl && sbatch slurm/predictor.sbatch <path/to/last.pt>'  # 3 seeds
```

- Smoke test: `sbatch --array=0,5,10,15 --time=0:30:00 --export=ALL,TAG=smoke,N_ITER=25 slurm/train.sbatch`

## Next

See `todo.md`: seed sweep, then post-hoc XAI on the best seed per method.
