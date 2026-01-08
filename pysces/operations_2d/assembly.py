from ..config import np, use_wrapper, jit, wrapper_type
from functools import partial


def summation_local_for(f, grid, *args):
  """
  Sum processor-local redundant Degrees of Freedom into a scalar using a for loop.

  *This is for testing purposes, and should not be used in performance code*

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Processor local scalar field.
  grid : `SpectralElementGrid`
    Spectral element grid struct that contains coordinate and metric data.

  Notes
  -----
  * Return value is not normalized by, e.g., the global mass matrix.

  Returns
  -------
  f_summed: `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Scalar field with DOFs added.
  """
  vert_redundancy_gll = grid["vert_redundancy"]
  workspace = f.copy()
  for ((local_face_idx, local_i, local_j),
       (remote_face_id, remote_i, remote_j)) in vert_redundancy_gll:
    workspace[remote_face_id, remote_i, remote_j] += f[local_face_idx, local_i, local_j]
  # this line works even for multi-processor decompositions.

  return workspace


def project_scalar_for(f, grid, *args):
  """
  Project a potentially discontinuous scalar onto the continuous subspace using a for loop, assuming all data is processor-local.

  *This is used for testing. Do not use in performance code*

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Scalar field to project
  grid : `SpectralElementGrid`
    Spectral element grid struct that contains coordinate and metric data.

  Notes
  -----
  This is the most human-readable way to perform projection, and can be used to test
  if your grid is topologically malformed.

  Returns
  -------
  f_cont
      The globally continous scalar closest in norm to f. 
  """
  # assumes that values from remote processors have already been accumulated
  metdet = grid["met_det"]
  inv_mass_mat = grid["mass_matrix_inv"]
  vert_redundancy_gll = grid["vert_redundancy"]
  gll_weights = grid["gll_weights"]
  workspace = f.copy()
  workspace *= metdet * (gll_weights[np.newaxis, :, np.newaxis] * gll_weights[np.newaxis, np.newaxis, :])
  for ((local_face_idx, local_i, local_j),
       (remote_face_id, remote_i, remote_j)) in vert_redundancy_gll:
    workspace[remote_face_id, remote_i, remote_j] += (metdet[local_face_idx, local_i, local_j] *
                                                      f[local_face_idx, local_i, local_j] *
                                                      (gll_weights[local_i] * gll_weights[local_j]))
  # this line works even for multi-processor decompositions.
  workspace *= inv_mass_mat
  return workspace


def project_scalar_sparse(f, grid, matrix, *args, scaled=True):

  """
  Project a potentially discontinuous scalar onto the continuous subspace using a sparse matrix, assuming all data is processor-local.

  *This is used for testing. Do not use in performance code*

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Scalar field to project
  grid : `SpectralElementGrid`
    Spectral element grid struct that contains coordinate and metric data.
  scaled: `bool`, default=True
    Should f be scaled by the mass matrix before being summed?

  Notes
  -----
  * When using weak operators (i.e. in hyperviscosity),
  the resulting values are already scaled by the mass matrix.
  * In an ideal world, even performance device code would use a
  version of code that treats assembly, or Direct Stiffness Summation,
  as the application of a linear projection operator. 
  However, support for sparse matrices in automatic differentiation libraries
  is bizarrely spotty.
    
  Returns
  -------
  f_cont
      The globally continous scalar closest in norm to f. 
  """
  if scaled:
    vals_scaled = f * grid["mass_matrix"]
    ret = vals_scaled + (matrix @ (vals_scaled).flatten()).reshape(f.shape)
  else:
    ret = f + (matrix @ (f).flatten()).reshape(f.shape)
  return ret * grid["mass_matrix_inv"]


def segment_sum(data, segment_ids, N):
  """
  A function that provides a numpy equivalent of the `segment_sum` function
  from Jax and TensorFlow.

  Parameters
  ----------
  data : Array[tuple[point_idx], Float]
      The floating point values to sum over
  segment_ids : Array[tuple[point_idx], Int]
      The indices in the result array into which to sum data.
      That is, `result[segment_idx[p]] += data[p]
  N: int
      The number of bins in which to sum

  Returns
  -------
  s: Array[tuple[N], Float]
      arrays into which segments have been summed.
  """
  data = np.asarray(data)
  s = np.zeros(N, dtype=data.dtype)
  np.add.at(s, segment_ids, data)
  return s


@partial(jit, static_argnames=["dims", "scaled"])
def project_scalar_wrapper(f, grid, dims, scaled=True):
  """
  Project a potentially discontinuous scalar onto the continuous subspace using assembly triples,
  assuming all data is processor-local.

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Scalar field to project
  grid : `SpectralElementGrid`
    Spectral element grid struct that contains coordinate and metric data.
  scaled: `bool`, default=True
    Should f be scaled by the mass matrix before being summed?

  Notes
  -----
  * When using weak operators (i.e. in hyperviscosity),
  the resulting values are already scaled by the mass matrix.
  * This routine is allowed to depend on wrapper_type.

  Returns
  -------
  f_cont
      The globally continous scalar closest in norm to f. 
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


project_scalar = project_scalar_wrapper
