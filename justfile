default:
    @just --list

train algo="ppo" seed="0":
    uv run python crafter_jepa/scripts/train_policy.py --algo {{trim_start_match(algo, "algo=")}} --seed {{trim_start_match(seed, "seed=")}}

sweep algo="ppo" seeds="0 1 2 3 4":
    for s in {{seeds}}; do uv run python crafter_jepa/scripts/train_policy.py --algo {{trim_start_match(algo, "algo=")}} --seed $s; done

train-predictor config="crafter_jepa/config/predictor_config.yaml":
    uv run python crafter_jepa/scripts/train_predictor.py {{trim_start_match(config, "config=")}}


app:
    uv run python crafter_jepa/scripts/app.py

check:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check crafter_jepa

test:
    uv run pytest -q
