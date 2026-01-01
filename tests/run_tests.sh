uv run ../pysces/set_config.py --wrapper_type none
uv run pytest $1
uv run ../pysces/set_config.py --wrapper_type jax
uv run --group jax pytest $1
uv run ../pysces/set_config.py --wrapper_type torch
uv run --group torch pytest $1
uv run ../pysces/set_config.py
