rm pysces_config.json
res=""
uv run ../pysces/set_config.py --wrapper_type none --use_mpi
for num_proc in $(seq 1 6);
do
  uv run mpirun -n $num_proc pytest distributed_memory_tests && res="${res}Numpy succeeded with nproc $num_proc \n"
done
uv run ../pysces/set_config.py --wrapper_type none
uv run pytest $1 && res="${res}Numpy local memory succeeded \n"
uv run ../pysces/set_config.py --wrapper_type jax
uv run --group jax pytest $1 && res="${res}Jax local memory succeeded \n"
uv run ../pysces/set_config.py --wrapper_type torch
uv run --group torch pytest $1
uv run ../pysces/set_config.py && res="${res}Torch local memory succeeded \n"
printf $res
