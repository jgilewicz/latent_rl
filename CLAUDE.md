# CLAUDE.md

latent-rl: PPO on Crafter, encoder ablation (PPO / JEPA / VAE / SAE). Global rules in `~/.claude/CLAUDE.md`. Commands and layout: `README.md`. Methods and results: `docs/results.md`; update it when an analysis changes.

## Conventions

- PyTorch, class-based; no tests (user decision); checks: `just check` (ruff + ty)
- One script per `just` recipe in `latent_rl/scripts/`, config `latent_rl/config/<name>_config.yaml`; scripts are thin orchestration, logic lives in `analysis/`, `xai/`, `trainers/`
- Plot text lives in config (Polish, no diacritics); plots → `plots/` (gitignored), figures used in docs are copied to `docs/figures/`
- Scripts using `AsyncVectorEnv` need `if __name__ == "__main__":` (spawn) and `PYTHONPATH=.`
- `pandas==2.3.3` pinned: rliable caps `arch<8`, arch 7 fails on pandas 3
- `numba` required: without it Crafter worldgen runs in pure Python (reset ~700 ms vs ~50 ms) and the vec env stalls

## Training

- `PPOTrainer` hooks `_extra_loss`, `_after_step`, `_checkpoint_state`; one subclass per algo
- `RolloutBuffer` shaped `[T, M, ...]` (`M = n_envs`), flattened only in `minibatches`; obs `uint8`, `/255` before the encoder
- `terminated` = crafter `discount == 0`, else `truncated`; autoreset `SAME_STEP`; truncation bootstraps via `info["final_obs"]`
- JEPA: encoder shared by reference with `JEPA.online_encoder`; optimizer gets trainer params + `jepa.predictor` only (`jepa.parameters()` would duplicate the encoder); watch `latent_std` → 0 = collapse
- `OnlineProbe` stays detached with its own optimizer: it must not train the encoder (fair comparison)
- Env seed `seed * n_envs + i`
- Checkpoint `<checkpoint_dir>/seed<s>/last.pt` (atomic, weights + `algo`, `seed`, `config`, `iteration`, `env_steps`, no optimizer); first iteration ≥1M also → `seed<s>/1M/last.pt`
- W&B: run `<algo>_crafter_s<seed>`, group = algo, tags `[<tag>, algo, crafter, <budget>M]`; artifact `ckpt-<algo>-s<seed>` aliases `1M`, `<budget>M`, `latest`
- Predictor (app only, not a metric): `train_predictor.py` → `<ckpt dir>/predictor/seed<s>/predictor_decoder.pt`; the app globs `predictor/**/predictor_decoder.pt`

## Post-hoc analysis

- Everything reads `EvalBuffer` (`eval_buffer/`): one 5M checkpoint per algo → uncertainty = episode bootstrap stratified by `source`, never across seeds
- New metric = function returning `metric_table(...)` + a panel section in the config named like the CSV; `write_intervals` reads `cfg[name]` as panels (hence `probe_r2` / `probe_selectivity`, not `probes`)
- `effective_rank`: `eigvalsh` of the Gram in float64 (same values as `svdvals`, ~10× faster)
- TwoNN: float64 distances (float32 → r1 = 0) + Theiler window 50 (without it ID ≈ 1.5 = trajectory)
- UMAP: `init: pca` (spectral fails on small eigengap), explicit `n_jobs: 1`
- `load_policies` refuses a checkpoint that doesn't reproduce `z_<algo>` → rerun `just eval-buffer` after replacing checkpoints
- Occlusion `GRID = (9, 9)`, `CELL_PX = 7`: 7 world rows + 2 HUD rows; HUD cells fixed per item (row 7: health, food, drink, energy, sapling, wood, stone, coal, iron; row 8: diamond, pickaxes, swords)
- Ablation: z has no privileged basis (`Linear → LayerNorm` → actor `Linear`) → ablate subspaces, not dims; SAE sparse code is not the policy input
- Ridge probe weights are a filter in low-variance directions; ablating them ≠ removing vitals information

## WCSS (SLURM)

- Account `hpc-jakgil6519-1780267238`, `lem-gpu-short`, 1×H100 + 4 cores per run (env-bound)
- No git on the cluster: `slurm/sync.sh` rsyncs and writes `CODE_VERSION`
- $HOME near quota: `slurm/env.sh` puts venv, uv cache, Python, checkpoints, logs in `/lustre/pd03/<account>/latent_rl/`
- `slurm/setup.sbatch` after dependency changes; `train.sbatch` array 0-19 → `ALGOS[i/5]`, seed `i%5`
- Experiment records: `experiments/<name>/` (wcss-experiment skill)
