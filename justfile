default:
    @just --list

train algo="ppo":
    uv run python main.py --algo {{trim_start_match(algo, "algo=")}}

check:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check crafter_jepa main.py

test:
    uv run pytest -q
