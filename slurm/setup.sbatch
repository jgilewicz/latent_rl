#!/usr/bin/env bash
#SBATCH --job-name=latent-rl-setup
#SBATCH --account=hpc-jakgil6519-1780267238
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=0:30:00
#SBATCH --output=/lustre/pd03/hpc-jakgil6519-1780267238/latent_rl/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-jakgil6519-1780267238/latent_rl/logs/%x_%j.err
set -euo pipefail

# shellcheck source=slurm/env.sh
source "$HOME/latent_rl/slurm/env.sh"
cd "$CODE"
"$HOME/.local/bin/uv" sync --frozen
"$PYTHON" -c "import torch, crafter, wandb; print(torch.__version__, torch.version.cuda, wandb.__version__)"
# fills numba's on-disk cache once, so parallel training jobs skip the JIT
"$PYTHON" -c "import crafter; crafter.Env(seed=0).reset(); print('crafter worldgen ok')"
