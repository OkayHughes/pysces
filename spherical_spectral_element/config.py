import numpy as np


DEBUG = True
npt = 4

has_mpi = False

use_wrapper = True
wrapper_type = "jax"
use_cpu = True
use_double = True

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
    return x

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
  mpi_rank = comm.Get_rank()
  mpi_size = comm.Get_size()
else:
  mpi_rank = 0
  mpi_size = 1