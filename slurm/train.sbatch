#!/usr/bin/env bash
# task i -> algo ALGOS[i / 3], seed i % 3
# smoke: sbatch --array=0,3,6,9 --time=0:30:00 \
#   --export=ALL,TAG=smoke,N_ITER=25 slurm/train.sbatch
#SBATCH --job-name=latent-rl-train
#SBATCH --account=hpc-jakgil6519-1780267238
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=3:00:00
#SBATCH --array=0-11
#SBATCH --output=/lustre/pd03/hpc-jakgil6519-1780267238/latent_rl/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-jakgil6519-1780267238/latent_rl/logs/%x_%A_%a.err
set -euo pipefail

# shellcheck source=slurm/env.sh
source "$HOME/latent_rl/slurm/env.sh"
ALGOS=(ppo jepa vae sae)
SEEDS_PER_ALGO=3
algo="${ALGOS[$((SLURM_ARRAY_TASK_ID / SEEDS_PER_ALGO))]}"
seed=$((SLURM_ARRAY_TASK_ID % SEEDS_PER_ALGO))
tag="${TAG:-main-5m}"
extra=()
if [[ -n "${N_ITER:-}" ]]; then extra=(--n-iterations "$N_ITER"); fi

echo "run       ${algo} seed ${seed} tag ${tag} ${extra[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" "$CODE/latent_rl/scripts/train_policy.py" \
  --algo "$algo" --seed "$seed" --tag "$tag" "${extra[@]}"
