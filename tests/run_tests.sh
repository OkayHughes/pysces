uv run ../pysces/set_config.py --wrapper_type none
uv run pytest
uv run ../pysces/set_config.py --wrapper_type jax
uv run --group jax pytest
uv run ../pysces/set_config.py --wrapper_type torch
uv run --group torch pytest
uv run ../pysces/set_config.py
