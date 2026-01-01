from json import dumps
from argparse import ArgumentParser


if __name__ == "__main__":
  from config import write_config
  parser = ArgumentParser(prog='pysces_config',
                          description='Set computational configuration before running models.')
  parser.add_argument('-w', '--wrapper_type', default="none")
  parser.add_argument('-m', '--use_mpi', action='store_true', default=False)
  parser.add_argument('-g', '--use_gpu', action='store_true', default=False)
  parser.add_argument('-s', '--single_precision', action='store_true', default=False)

  args = parser.parse_args()
  wrapper_type = args.wrapper_type
  valid_wrappers = ["none", "torch", "jax"]
  assert wrapper_type in valid_wrappers, f"Invalid wrapper type: {wrapper_type}, must be one of {valid_wrappers}"
  if wrapper_type == "jax":
    use_wrapper = True
  elif wrapper_type == "torch":
    use_wrapper = True
  else:
    use_wrapper = False
  write_config(debug=True,
               use_mpi=args.use_mpi,
               use_wrapper=use_wrapper,
               wrapper_type=wrapper_type,
               use_cpu=not args.use_gpu,
               use_double=not args.single_precision)
