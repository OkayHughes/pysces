from ..config import np, use_wrapper, jit, wrapper_type
from ..distributed_memory.processor_decomposition import global_to_local, elem_idx_global_to_proc_idx
from scipy.sparse import coo_array
from functools import partial


def project_scalar_for(f, grid, *args):
  """
  Project a potentially discontinuous scalar onto the continuous
  subspace using a for loop, assuming all data is processor-local.

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
  metdet = grid["metric_determinant"]
  inv_mass_mat = grid["mass_matrix_denominator"]
  vert_redundancy_gll = grid["vertex_redundancy"]
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


def project_scalar_sparse(f, grid, matrix, *args):

  """
  Project a potentially discontinuous scalar onto the continuous
  subspace using a sparse matrix, assuming all data is processor-local.

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
  is bizarrely *ahem* sparse.

  Returns
  -------
  f_cont
      The globally continous scalar closest in norm to f.
  """
  vals_scaled = f * grid["mass_matrix"]
  ret = vals_scaled + (matrix @ (vals_scaled).flatten()).reshape(f.shape)
  return ret * grid["mass_matrix_denominator"]


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


@partial(jit, static_argnames=["dims"])
def project_scalar_wrapper(f, grid, dims):
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

  scaled_f = f * grid["mass_matrix"]
  relevant_data = (scaled_f).flatten().take(cols) * data

  if use_wrapper and wrapper_type == "jax":
    scaled_f = scaled_f.flatten().at[rows].add(relevant_data)
    scaled_f = scaled_f.reshape(dims["shape"])
  elif use_wrapper and wrapper_type == "torch":
    scaled_f = scaled_f.flatten()
    scaled_f = scaled_f.scatter_add_(0, rows, relevant_data)
    scaled_f = scaled_f.reshape(dims["shape"])
  else:
    scaled_f += segment_sum(relevant_data, rows, dims["N"]).reshape(dims["shape"])
  return scaled_f * grid["mass_matrix_denominator"]


project_scalar = project_scalar_wrapper


def init_assembly_matrix(NELEM, npt, assembly_triple):
  data, rows, cols = assembly_triple
  assembly_matrix = coo_array((data, (rows, cols)), shape=(NELEM * npt * npt, NELEM * npt * npt))
  return assembly_matrix


def init_assembly_local(NELEM, npt, vert_redundancy_local):
  # From this moment forward, we assume that
  # vert_redundancy_gll contains only the information
  # for processor-local GLL things,
  # and that remote_face_idx corresponds to processor local
  # ids.
  index_hack = np.zeros((NELEM, npt, npt), dtype=np.int64)
  # hack: easier than figuring out indexing conventions
  index_hack = np.arange(index_hack.size).reshape(index_hack.shape)

  data = []
  rows = []
  cols = []

  for ((local_face_idx, local_i, local_j),
       (remote_face_id, remote_i, remote_j)) in vert_redundancy_local:
    data.append(1.0)
    rows.append(index_hack[remote_face_id, remote_i, remote_j])
    cols.append(index_hack[local_face_idx, local_i, local_j])

  # print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")
  return (np.array(data, dtype=np.float64),
          np.array(rows, dtype=np.int64),
          np.array(cols, dtype=np.int64))


def init_assembly_global(NELEM, npt, vert_redundancy_send, vert_redundancy_receive):
  # From this moment forward, we assume that
  # vert_redundancy_gll contains only the information
  # for processor-local GLL things,
  # and that remote_face_idx corresponds to processor local
  # ids.
  index_hack = np.zeros((NELEM, npt, npt), dtype=np.int64)
  # hack: easier than figuring out indexing conventions
  index_hack = np.arange(index_hack.size, dtype=np.int64).reshape(index_hack.shape)

  # convention: when scaled=True, remote values are
  # pre-multiplied by numerator
  # divided by total mass matrix on receiving end
  triples_receive = {}
  triples_send = {}
  for vert_redundancy, triples, transpose in zip([vert_redundancy_receive, vert_redundancy_send],
                                                 [triples_receive, triples_send],
                                                 [False, True]):
    for source_proc_idx in vert_redundancy.keys():
      data = []
      rows = []
      cols = []
      for col_idx, (target_local_idx, target_i, target_j) in enumerate(vert_redundancy[source_proc_idx]):
        data.append(1.0)
        if transpose:
          cols.append(index_hack[target_local_idx, target_i, target_j])
          rows.append(col_idx)
        else:
          rows.append(index_hack[target_local_idx, target_i, target_j])
          cols.append(col_idx)
      triples[source_proc_idx] = (np.array(data, dtype=np.float64),
                                  np.array(rows, dtype=np.int64),
                                  np.array(cols, dtype=np.int64))
  # print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")
  return triples_send, triples_receive


def triage_vert_redundancy_flat(vert_redundancy_gll_flat,
                                proc_idx, decomp):
  # current understanding: this works because the outer
  # three for loops will iterate in exactly the same order for
  # the sending and recieving processor
  vert_redundancy_local = []
  vert_redundancy_send = {}
  vert_redundancy_receive = {}

  for ((target_global_idx, target_i, target_j),
       (source_global_idx, source_i, source_j)) in vert_redundancy_gll_flat:
    target_proc_idx = elem_idx_global_to_proc_idx(target_global_idx, decomp)
    source_proc_idx = elem_idx_global_to_proc_idx(source_global_idx, decomp)
    if (target_proc_idx == proc_idx and source_proc_idx == proc_idx):
      target_local_idx = int(global_to_local(target_global_idx, proc_idx, decomp))
      source_local_idx = int(global_to_local(source_global_idx, proc_idx, decomp))
      vert_redundancy_local.append(((target_local_idx, target_i, target_j),
                                    (source_local_idx, source_i, source_j)))
    elif (target_proc_idx == proc_idx and not
          source_proc_idx == proc_idx):
      target_local_idx = int(global_to_local(target_global_idx, proc_idx, decomp))
      if source_proc_idx not in vert_redundancy_receive.keys():
        vert_redundancy_receive[source_proc_idx] = []
      vert_redundancy_receive[source_proc_idx].append(((target_local_idx, target_i, target_j)))
    elif (not target_proc_idx == proc_idx and
          source_proc_idx == proc_idx):
      source_local_idx = int(global_to_local(source_global_idx, proc_idx, decomp))
      if target_proc_idx not in vert_redundancy_send.keys():
        vert_redundancy_send[target_proc_idx] = []
      vert_redundancy_send[target_proc_idx].append(((source_local_idx, source_i, source_j)))
  return vert_redundancy_local, vert_redundancy_send, vert_redundancy_receive
