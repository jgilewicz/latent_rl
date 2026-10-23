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
just app                 # Gradio visualizer
just check               # ruff + ty
```

Config: `crafter_jepa/config/<algo>_config.yaml` (~5M env steps per run).

## Logged metrics (W&B)

- **RL**: `crafter_score`, `episode_return`, `achievement/<name>`, `explained_variance`
- **Latent health**:
  - `diag/effective_rank`: how many latent dimensions are in use (low = collapse)
  - `probe/r2`: linear probe on frozen features predicting inventory + daylight
  - `diag/cka_prev`: similarity to the representation 10 iterations earlier (drift)

## Layout

- `crafter_jepa/scripts/`: `train_policy.py`, `train_predictor.py`, `app.py`
- `crafter_jepa/trainers/`: `PPOTrainer` + one subclass per algo
- `crafter_jepa/modules/`: encoder, decoder, actor-critic, JEPA, VAE, SAE
- `crafter_jepa/utils/diagnostics.py`: latent metrics
- Checkpoints: `checkpoints/<algo>/seed<seed>/last.pt`

## Next

See `todo.md`: seed sweep, then post-hoc XAI on the best seed per method.
