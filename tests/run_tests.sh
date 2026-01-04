rm pysces_config.json
res=""
uv run ../pysces/set_config.py --wrapper_type none --use_mpi
for num_proc in $(seq 1 6);
do
  uv run mpirun -n $num_proc pytest distributed_memory_tests && res="${res}Numpy succeeded with $num_proc processors \n"
done
uv run ../pysces/set_config.py --wrapper_type none
uv run pytest && res="${res}Numpy local memory succeeded \n"
uv run ../pysces/set_config.py --wrapper_type jax
uv run --group jax pytest && res="${res}Jax local memory succeeded \n"
uv run --group jax mpirun -n 2 pytest distributed_memory_tests && res="${res}Jax succeeded with 2 processors \n"
uv run ../pysces/set_config.py --wrapper_type torch
uv run --group torch pytest  && res="${res}Torch local memory succeeded \n"
uv run --group torch mpirun -n 2 pytest distributed_memory_tests && res="${res}Torch succeeded with 2 processors \n"
uv run ../pysces/set_config.py
echo -e $res
