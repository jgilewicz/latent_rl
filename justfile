default:
    @just --list

train algo="ppo" seed="0":
    uv run python main.py --algo {{trim_start_match(algo, "algo=")}} --seed {{trim_start_match(seed, "seed=")}}

sweep algo="ppo" seeds="0 1 2 3 4":
    for s in {{seeds}}; do uv run python main.py --algo {{trim_start_match(algo, "algo=")}} --seed $s; done

check:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check crafter_jepa main.py

test:
    uv run pytest -q
