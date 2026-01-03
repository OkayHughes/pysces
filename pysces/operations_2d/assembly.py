from ..config import np, use_wrapper, jit, wrapper_type, jnp
from functools import partial


if use_wrapper and wrapper_type == "jax":
    import jax


def summation_local_for(f, grid, *args):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  vert_redundancy_gll = grid["vert_redundancy"]
  workspace = f.copy()
  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        workspace[remote_face_id, remote_i, remote_j] += f[local_face_idx, local_i, local_j]
  # this line works even for multi-processor decompositions.

  return workspace


def dss_scalar_for(f, grid, *args, scaled=True):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
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
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  if scaled:
    vals_scaled = f * grid["mass_matrix"]
    ret = vals_scaled + (grid["dss_matrix"] @ (vals_scaled).flatten()).reshape(f.shape)
  else:
    ret = f + (grid["dss_matrix"] @ (f).flatten()).reshape(f.shape)
  return ret * grid["mass_matrix_inv"]


def segment_sum(data, segment_ids, N):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  data = np.asarray(data)
  s = np.zeros(N, dtype=data.dtype)
  np.add.at(s, segment_ids, data)
  return s


@partial(jit, static_argnames=["dims", "scaled"])
def dss_scalar_wrapper(f, grid, dims, scaled=True):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  (data, rows, cols) = grid["assembly_triple"]

  if scaled:
    scaled_f = f * grid["mass_matrix"]
    relevant_data = (scaled_f).flatten().take(cols) * data
  else:
    scaled_f = f
    relevant_data = scaled_f.flatten().take(cols)

  if use_wrapper and wrapper_type == "jax":
    scaled_f = scaled_f.flatten().at[rows].add(relevant_data)
    scaled_f = scaled_f.reshape(dims["shape"])
  elif use_wrapper and wrapper_type == "torch":
    scaled_f = scaled_f.flatten()
    scaled_f = scaled_f.scatter_add_(0, rows, relevant_data)
    scaled_f = scaled_f.reshape(dims["shape"])
  else:
    scaled_f += segment_sum(relevant_data, rows, dims["N"]).reshape(dims["shape"])
  return scaled_f * grid["mass_matrix_inv"]


if use_wrapper and wrapper_type == "jax":
  dss_scalar = dss_scalar_wrapper
elif use_wrapper and wrapper_type == "torch":
  dss_scalar = dss_scalar_wrapper
else:
  dss_scalar = dss_scalar_sparse
