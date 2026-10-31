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
just latent-comparison   # CKA, TwoNN, eff. rank, UMAP, temporal from eval_buffer/ → plots/
just xai                 # probes on frozen latents from eval_buffer/ → plots/
just occlusion           # 9×9 grid occlusion maps (Δz, probe output) → plots/
just ablation            # mean-ablation of z subspaces, KL of the policy → plots/
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
- Load with `EvalBuffer(Path("eval_buffer"))` (`latent_rl/utils/eval_buffer.py`): lazy per-field access, `latents(algo, next_step)`, `resample_episodes(rng)` (episode bootstrap, stratified by policy)

## Latent-space comparison (post-hoc)

- `just latent-comparison` → `plots/` (config `latent_rl/config/latent_comparison_config.yaml`)
- All metrics on all buffer frames; plotted value = mean of an episode bootstrap (1000 reps), 95% CI; CSV also has `full` (full-sample estimate)
- Linear CKA between the 4 encoders → `cka_matrix.{png,pdf}`, `cka.csv`
- TwoNN intrinsic dimension (Facco et al. 2017) + effective rank → `dimensions.{png,pdf}`, `dimensions.csv`
- UMAP of every encoder (all frames, fixed seed, PCA init), coloured by daylight, biome, achievements unlocked so far → `umap.{png,pdf}`
- Temporal → `temporal.{png,pdf}`, `temporal.csv`:
  - `latent_step`: mean ‖z_{t+1} − z_t‖ / mean distance to a random frame (lower = smoother)
  - `step_correlation`: Spearman between latent step and pixel step
  - `r2_next`, `r2_delta`: linear regression [z_t, one-hot a_t] → z_{t+1}, fit once, scored on held-out episodes (every `holdout_every`-th); R² of z_{t+1} and of the change z_{t+1} − z_t
- TwoNN uses a Theiler window (`theiler_window`, 50 steps): same-episode neighbours closer in time are skipped, else it measures the 1-D trajectory (ID ~1.5); duplicate latents dropped
- Code: `latent_rl/analysis/` (one module per metric group); the script only loads the config, runs them and writes CSV + plots
- Runtime ~4-5 min on a local CPU; mostly the 1000-rep bootstraps, UMAP ~10 s per encoder

## XAI (post-hoc)

- `just xai` → `plots/` (config `latent_rl/config/xai_config.yaml`)
- Probes: player vitals (`probes.targets`: health, food, drink, energy) regressed from frozen `z_<algo>`
  - `linear`: `StandardScaler` + `Ridge`; `mlp`: `StandardScaler` + `MLPRegressor` (1×256, early stopping)
  - fit once on train episodes; scored on held-out episodes (every `holdout_every`-th)
  - R² per target, episode bootstrap (1000 reps), 95% CI → `probe_r2.{png,pdf}`, `probe_r2.csv` (metric `<probe>_<target>`)
  - MLP fit with one seed: CI covers test-data noise, not probe-training noise
- Control task (Hewitt & Liang 2019): does the probe read the property or memorize?
  - frame "types": k-means (`control.kmeans`, 250 clusters) on 16×16 downsampled frames, encoder-agnostic
  - every type gets one random row of the real labels → control targets; same probes, same split
  - selectivity = R²(real) − max(R²(control), 0), on the same bootstrap resample → `probe_selectivity.{png,pdf,csv}`
  - control R² (metric `<probe>_control_<target>`, unclipped) in `probe_r2.csv`
- Runtime ~2 min on a local CPU (mostly the MLP fits)
- Occlusion (`just occlusion`, config section `occlusion`): which frame cells move z and the probe output
  - grid 9×9 cells of 7 px: 7 world rows + 2 HUD rows (inventory incl. vitals)
  - one cell at a time replaced by the mean buffer frame; 500 test-episode frames re-encoded by the checkpoint encoders (checked against `z_<algo>`)
  - maps: `latent` = mean ‖Δz‖ / RMS spread of z (scale-free across encoders); per target mean |Δŷ| of the linear probe (inventory units)
  - → `occlusion.{png,pdf}` (shared color scale per column), `occlusion.csv` (`algo, map, row, col, value`)
  - no CI (one checkpoint, mean over frames); runtime ~1.5 min on CPU
  - finding: vitals probes read their own HUD icon (+ correlated HUD cells); world cells barely matter
- Ablation (`just ablation`, config section `ablation`): which subspaces of z the policy depends on
  - z has no privileged basis (`Linear → LayerNorm`, then the actor's `Linear`) → data-defined subspaces, not single dims
  - mean-ablation: the deviation of z from its mean is projected out of an orthonormal subspace
  - subspaces: `vitals` (linear probe directions W/s, QR), `random` (same count, control), `pca_<k>` (top-k PCs; `pca_256` = z → z̄, the upper bound)
  - metric: mean KL(π ‖ π_abl) over all buffer frames, episode bootstrap → `ablation_kl.{png,pdf,csv}` (metric `kl_<subspace>`)
  - caveat: probe weights (filter) sit in low-variance directions (≤0.1% of z variance vs ~2% for random) → `vitals` removes only the linear readout, not the vitals pattern cov(z, ŷ)
  - findings: KL saturates at top-16 PCs for all encoders; JEPA's policy is the most concentrated (top-4 PCs ≈ 65% of the bound vs 37–47%)

## Layout

- `latent_rl/scripts/`: entry points
  - training: `train_policy.py`, `train_predictor.py`
  - post-hoc: `collect_eval_buffer.py`, `compare_methods.py`, `latent_comparison.py`, `xai_analysis.py`, `occlusion_analysis.py`, `ablation_analysis.py`
  - UI: `app.py`
- `latent_rl/trainers/`: `PPOTrainer` + one subclass per algo
- `latent_rl/modules/`: encoder, decoder, actor-critic, JEPA, VAE, SAE
- `latent_rl/analysis/`: post-hoc latent metrics on the eval buffer
  - `bootstrap.py`: episode bootstrap, metric tables
  - `similarity.py` (CKA), `dimensions.py` (TwoNN, effective rank)
  - `temporal.py` (smoothness, linear predictability), `embedding.py` (UMAP)
- `latent_rl/xai/`: post-hoc explainability on the eval buffer
  - `probe.py`: linear + MLP probes of player vitals, control tasks, selectivity
  - `occlusion.py`: policy loading from the buffer's checkpoints, grid occlusion maps
  - `ablation.py`: mean-ablation of z subspaces, policy KL
- `latent_rl/utils/`:
  - `diagnostics.py`: metric primitives (training-time + post-hoc)
  - `eval_buffer.py`: `EvalBuffer` loader
  - `plotting.py`: all comparison figures
- `latent_rl/config/`: one YAML per algo + `compare`, `eval_buffer`, `latent_comparison`, `xai`, `predictor`
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
