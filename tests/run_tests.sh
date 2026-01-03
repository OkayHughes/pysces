uv run ../pysces/set_config.py --wrapper_type none --use_mpi
for num_proc in $(seq 1 6);
do
  uv run mpirun -n $num_proc pytest distributed_memory_tests
done
uv run ../pysces/set_config.py --wrapper_type none
uv run pytest $1
uv run ../pysces/set_config.py --wrapper_type jax
uv run --group jax pytest $1
uv run ../pysces/set_config.py --wrapper_type torch
uv run --group torch pytest $1
uv run ../pysces/set_config.py
