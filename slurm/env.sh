#!/usr/bin/env bash
# sourced by every job script: PD paths, venv, W&B
PD="/lustre/pd03/${SLURM_JOB_ACCOUNT:-hpc-jakgil6519-1780267238}"
export CODE="$HOME/latent_rl"
export WORK="$PD/latent_rl"
# $HOME is nearly full: uv cache, managed Python and the venv live in PD
export UV_CACHE_DIR="$WORK/uv-cache"
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python"
export UV_PROJECT_ENVIRONMENT="$WORK/.venv"
export PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export PYTHONPATH="$CODE"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=online
export WANDB_DIR="${TMPDIR:-$WORK}"

mkdir -p "$WORK/logs"
cd "$WORK" || exit 1
echo "job       ${SLURM_JOB_ID:-local} on ${SLURM_JOB_NODELIST:-?}"
echo "code      $(cat "$CODE/CODE_VERSION" 2>/dev/null || echo unknown)"
