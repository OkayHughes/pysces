from .config import np, use_wrapper, jit, wrapper_type, jnp
from functools import partial
if use_wrapper and wrapper_type=="jax":
    import jax
if use_wrapper and wrapper_type=="torch":
    import torch


def dss_scalar_for(f, grid, *args):
  metdet = grid["met_det"]
  inv_mass_mat = grid["mass_matrix_inv"]
  vert_redundancy_gll = grid["vert_redundancy"]
  gll_weights = grid["gll_weights"]
  workspace = f.copy()
  workspace *= metdet * (gll_weights[np.newaxis, :, np.newaxis] * gll_weights[np.newaxis, np.newaxis, :])
  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        workspace[remote_face_id, remote_i, remote_j] += (metdet[local_face_idx, local_i, local_j] *
                                                          f[local_face_idx, local_i, local_j] *
                                                          (gll_weights[local_i] * gll_weights[local_j]))
  workspace *= inv_mass_mat
  return workspace


def dss_scalar_sparse(f, grid, *args, scaled=True):
  if scaled:
    return (grid["dss_matrix"] @ f.flatten()).reshape(f.shape)
  else:
    return (grid["dss_matrix_unscaled"] @ f.flatten()).reshape(f.shape)


def segment_sum(data, segment_ids, N):
  data = np.asarray(data)
  s = np.zeros(N, dtype=data.dtype)
  np.add.at(s, segment_ids, data)
  return s

def dss_scalar_torch(f, grid, dims, scaled=True):
  (data, data_un, rows, cols) = grid["dss_triple"]
  if scaled:
    relevant_data = f.flatten()[cols] * data
  else:
    relevant_data = f.flatten()[cols] * data_un
  if use_wrapper and wrapper_type=="torch":
    return jnp.zeros_like(f.flatten()).scatter_add_(0, rows, relevant_data).reshape(dims["shape"])
  else:
    return segment_sum(relevant_data, rows, dims["N"]).reshape(dims["shape"])


@partial(jit, static_argnames=["dims", "scaled"])
def dss_scalar_jax(f, grid, dims, scaled=True):
  (data, data_un, rows, cols) = grid["dss_triple"]
  
  if scaled:
    relevant_data = f.flatten().take(cols) * data
  else:
    relevant_data = f.flatten().take(cols) * data_un
  if use_wrapper and wrapper_type=="jax":
    return jax.ops.segment_sum(relevant_data, rows, dims["N"]).reshape(dims["shape"])
  elif use_wrapper and wrapper_type=="torch":
    return jnp.zeros_like(f.flatten()).scatter_add_(0, rows, relevant_data).reshape(dims["shape"])
  else:
    return segment_sum(relevant_data, rows, dims["N"]).reshape(dims["shape"])




if use_wrapper and wrapper_type=="jax":
  dss_scalar = dss_scalar_jax
elif use_wrapper and wrapper_type=="torch":
  dss_scalar = dss_scalar_torch
else:
  dss_scalar = dss_scalar_sparse
