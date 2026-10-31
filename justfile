default:
    @just --list

train algo="ppo" seed="0":
    uv run python latent_rl/scripts/train_policy.py --algo {{trim_start_match(algo, "algo=")}} --seed {{trim_start_match(seed, "seed=")}}

sweep algo="ppo" seeds="0 1 2 3 4":
    for s in {{seeds}}; do uv run python latent_rl/scripts/train_policy.py --algo {{trim_start_match(algo, "algo=")}} --seed $s; done

train-predictor checkpoint seed="0":
    uv run python latent_rl/scripts/train_predictor.py --checkpoint {{trim_start_match(checkpoint, "checkpoint=")}} --seed {{trim_start_match(seed, "seed=")}}

eval-buffer:
    PYTHONPATH=. uv run python latent_rl/scripts/collect_eval_buffer.py

latent-comparison:
    PYTHONPATH=. uv run python latent_rl/scripts/latent_comparison.py

xai:
    PYTHONPATH=. uv run python latent_rl/scripts/xai_analysis.py

occlusion:
    PYTHONPATH=. uv run python latent_rl/scripts/occlusion_analysis.py

ablation:
    PYTHONPATH=. uv run python latent_rl/scripts/ablation_analysis.py

compare:
    PYTHONPATH=. uv run python latent_rl/scripts/compare_methods.py

app:
    uv run python latent_rl/scripts/app.py

check:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check latent_rl

test:
    uv run pytest -q
