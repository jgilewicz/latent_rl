#!/usr/bin/env bash
# push the working tree (committed or not) to the cluster and record its version
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
ssh_opts=(-o BatchMode=yes -o NumberOfPasswordPrompts=0)
version="$(git -C "$root" rev-parse --short HEAD)$(git -C "$root" diff --quiet HEAD || echo "+dirty")"
echo "$version $(date -Iseconds)" >"$root/CODE_VERSION"
rsync -a --delete -e "ssh ${ssh_opts[*]}" \
  --exclude .venv --exclude .git --exclude checkpoints --exclude wandb \
  --exclude __pycache__ --exclude .ruff_cache --exclude .pytest_cache \
  "$root/" wcss:latent_rl/
echo "synced $version"
