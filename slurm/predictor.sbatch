#!/usr/bin/env bash
# usage: sbatch slurm/predictor.sbatch <path to policy last.pt>
# task i -> predictor seed i
#SBATCH --job-name=latent-rl-predictor
#SBATCH --account=hpc-jakgil6519-1780267238
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=1:00:00
#SBATCH --array=0-2
#SBATCH --output=/lustre/pd03/hpc-jakgil6519-1780267238/latent_rl/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-jakgil6519-1780267238/latent_rl/logs/%x_%A_%a.err
set -euo pipefail

checkpoint="${1:?usage: sbatch slurm/predictor.sbatch <policy last.pt>}"
# shellcheck source=slurm/env.sh
source "$HOME/latent_rl/slurm/env.sh"
tag="${TAG:-predictor-5m}"

echo "run       predictor seed ${SLURM_ARRAY_TASK_ID} on ${checkpoint}"
"$PYTHON" "$CODE/latent_rl/scripts/train_predictor.py" \
  --checkpoint "$checkpoint" --seed "$SLURM_ARRAY_TASK_ID" --tag "$tag"
