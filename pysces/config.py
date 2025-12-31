import os
from json import dumps
import numpy as np
from json import loads

def get_config_filepath():
  #return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
  return os.path.join(os.getcwd(), "config.json")

def write_config(debug=True,
                 use_mpi=False,
                 use_wrapper=False,
                 wrapper_type="none",
                 use_cpu=True,
                 use_double=True):
  config_struct = {"debug": debug,
                   "use_mpi": use_mpi,
                   "use_wrapper": use_wrapper,
                   "wrapper_type": wrapper_type,
                   "use_cpu": use_cpu,
                   "use_double": use_double}
  with open(get_config_filepath(), "w") as config_file:
    config_file.write(dumps(config_struct, indent=2))

def parse_config_file():
  config_filename = get_config_filepath()
  try:
    assert os.path.isfile(config_filename)
  except AssertionError:
    print("Config file is not written, writing serial config file as fallback")
    write_config()
  with open(config_filename, "r") as f:
    config_vars = loads(f.read())
  return config_vars


config_vars = parse_config_file()

DEBUG = config_vars["debug"]

has_mpi = config_vars["use_mpi"]


use_wrapper = config_vars["use_wrapper"]
wrapper_type = config_vars["wrapper_type"]
use_cpu = config_vars["use_cpu"]
use_double = config_vars["use_double"]

if use_double:
  eps = 1e-11
else:
  eps = 1e-6

if wrapper_type == "jax" and use_wrapper:
  import jax.numpy as jnp
  import jax
  if use_cpu:
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
  if use_double:
    jax.config.update("jax_enable_x64", True)

  def device_wrapper(x, dtype=jnp.float64):
    return jnp.array(x, dtype=dtype)

  def device_unwrapper(x):
    return np.asarray(x)
  jit = jax.jit

  def versatile_assert(should_be_true):
    return

  from jax.tree_util import Partial as partial

  def vmap_1d_apply(func, vector, in_axis, out_axis):
      return jax.vmap(func, in_axes=(in_axis), out_axes=(out_axis))(vector)

  def flip(array, axis):
    return jnp.flip(array, axis=axis)

  def remainder(array, divisor):
    return jnp.mod(array, divisor)

  def take_along_axis(array, idxs, axis):
    return jnp.take_along_axis(array, idxs, axis=axis)

  def cast_type(arr, dtype):
    return arr.astype(dtype)

elif wrapper_type == "torch" and use_wrapper:
  import torch as jnp
  import torch
  device = torch.device("mps:0" if torch.backends.mps.is_available() and not use_cpu else "cpu")
  print(device)

  if use_double:
    default_dtype = jnp.float64
  else:
    default_dtype = jnp.float32

  def device_wrapper(x, dtype=default_dtype):
    return jnp.tensor(x, dtype=dtype).to(device)

  def device_unwrapper(x):
    return x.cpu().detach().numpy()

  def jit(func, *_, **__):
     return func

  def versatile_assert(should_be_true):
    return

  from functools import partial

  def vmap_1d_apply(func, vector, in_axis, out_axis):
    return torch.vmap(func, in_dims=(in_axis), out_dims=(out_axis))(vector)

  def flip(array, axis):
    return jnp.flip(array, dims=(axis,))

  def remainder(array, divisor):
    return torch.remainder(array, divisor)

  def take_along_axis(array, idxs, axis):
    return torch.take_along_dim(array, idxs, dim=axis)

  def cast_type(arr, dtype):
    return arr.type(dtype)

else:
  import numpy as jnp

  def device_wrapper(x, dtype=np.float64):
    return np.array(x, dtype=dtype)

  def device_unwrapper(x):
    return x

  def jit(func, *_, **__):
    return func

  def versatile_assert(should_be_true):
    assert should_be_true

  from functools import partial

  def vmap_1d_apply(func, scalar, in_axis, out_axis):
    levs = []
    for lev_idx in range(scalar.shape[in_axis]):
      scalar_2d = scalar.take(indices=lev_idx, axis=in_axis)
      levs.append(func(scalar_2d))
    return np.stack(levs, axis=out_axis)

  def flip(array, axis):
    return jnp.flip(array, axis=axis)

  def remainder(array, divisor):
    return jnp.mod(array, divisor)

  def take_along_axis(array, idxs, axis):
    return jnp.take_along_axis(array, idxs, axis=axis)

  def cast_type(arr, dtype):
    return arr.astype(dtype)

if has_mpi:
  from mpi4py import MPI
  mpi_comm = MPI.COMM_WORLD
  mpi_rank = mpi_comm.Get_rank()
  mpi_size = mpi_comm.Get_size()

else:
  mpi_comm = None
  mpi_rank = 0
  mpi_size = 1

do_mpi_communication = mpi_size > 1
