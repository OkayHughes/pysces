from .config import np, use_wrapper, jit, wrapper_type, jnp
from functools import partial
if use_wrapper and wrapper_type == "jax":
    import jax

def summation_local_for(f, grid, *args):
  vert_redundancy_gll = grid["vert_redundancy"]
  workspace = f.copy()
  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        workspace[remote_face_id, remote_i, remote_j] += f[local_face_idx, local_i, local_j]
  # this line works even for multi-processor decompositions.

  return workspace

def dss_scalar_for(f, grid, *args, scaled=True):
  # assumes that values from remote processors have already been accumulated
  metdet = grid["met_det"]
  inv_mass_mat = grid["mass_matrix_inv"]
  vert_redundancy_gll = grid["vert_redundancy"]
  gll_weights = grid["gll_weights"]
  workspace = f.copy()
  if scaled:
    workspace *= metdet * (gll_weights[np.newaxis, :, np.newaxis] * gll_weights[np.newaxis, np.newaxis, :])
  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        workspace[remote_face_id, remote_i, remote_j] += (metdet[local_face_idx, local_i, local_j] *
                                                          f[local_face_idx, local_i, local_j] *
                                                          (gll_weights[local_i] * gll_weights[local_j]))
  # this line works even for multi-processor decompositions.
  workspace *= inv_mass_mat
  return workspace


def dss_scalar_sparse(f, grid, *args, scaled=True):
  if scaled:
    ret = (grid["dss_matrix"] @ (f * grid["mass_matrix"]).flatten()).reshape(f.shape)
  else:
    ret = (grid["dss_matrix"] @ (f).flatten()).reshape(f.shape)
  return ret * grid["mass_matrix_inv"]


def segment_sum(data, segment_ids, N):
  data = np.asarray(data)
  s = np.zeros(N, dtype=data.dtype)
  np.add.at(s, segment_ids, data)
  return s


def dss_scalar_torch(f, grid, dims, scaled=True):
  (data, rows, cols) = grid["dss_triple"]
  if scaled:
    relevant_data = (f * grid["mass_matrix"]).flatten()[cols] * data
  else:
    relevant_data = f.flatten()[cols]
  if use_wrapper and wrapper_type == "torch":
    ret = jnp.zeros_like(f.flatten()).scatter_add_(0, rows, relevant_data).reshape(dims["shape"])
  else:
    ret = segment_sum(relevant_data, rows, dims["N"]).reshape(dims["shape"])
  return ret * grid["mass_matrix_inv"]


@partial(jit, static_argnames=["dims", "scaled"])
def dss_scalar_jax(f, grid, dims, scaled=True):
  (data, rows, cols) = grid["dss_triple"]

  if scaled:
    relevant_data = (f * grid["mass_matrix"]).flatten().take(cols) * data
  else:
    relevant_data = f.flatten().take(cols)

  if use_wrapper and wrapper_type == "jax":
    ret = jax.ops.segment_sum(relevant_data, rows, dims["N"]).reshape(dims["shape"])
  elif use_wrapper and wrapper_type == "torch":
    ret = jnp.zeros_like(f.flatten()).scatter_add_(0, rows, relevant_data).reshape(dims["shape"])
  else:
    ret = segment_sum(relevant_data, rows, dims["N"]).reshape(dims["shape"])
  return ret * grid["mass_matrix_inv"]


if use_wrapper and wrapper_type == "jax":
  dss_scalar = dss_scalar_jax
elif use_wrapper and wrapper_type == "torch":
  dss_scalar = dss_scalar_torch
else:
  dss_scalar = dss_scalar_sparse
