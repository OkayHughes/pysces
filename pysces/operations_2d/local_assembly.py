from ..config import np, use_wrapper, jit, wrapper_type
from ..distributed_memory.processor_decomposition import global_to_local, elem_idx_global_to_proc_idx
from scipy.sparse import coo_array
from functools import partial


def project_scalar_sparse(f,
                          grid,
                          matrix,
                          *args):

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


def segment_sum(field,
                data,
                segment_ids):
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
  np.add.at(field, (segment_ids[0], segment_ids[1], segment_ids[2]), data)


@partial(jit, static_argnames=["dims"])
def project_scalar_wrapper(f,
                           grid,
                           dims):
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
  if use_wrapper and wrapper_type == "jax":
    relevant_data = (scaled_f).at[cols[0], cols[1], cols[2]].get()
  elif not use_wrapper:
    relevant_data = scaled_f[cols[0], cols[1], cols[2]]
  if use_wrapper and wrapper_type == "jax":
    scaled_f = scaled_f.at[rows[0], rows[1], rows[2]].add(relevant_data)
  elif use_wrapper and wrapper_type == "torch":
    # this is broken
    scaled_f = scaled_f.flatten()
    scaled_f = scaled_f.scatter_add_(0, rows, relevant_data)
    scaled_f = scaled_f.reshape(dims["shape"])
  else:
    segment_sum(scaled_f, relevant_data, rows)
  return scaled_f * grid["mass_matrix_denominator"]


project_scalar = project_scalar_wrapper


def init_assembly_matrix(NELEM,
                         npt,
                         assembly_triple):
  data, rows, cols = assembly_triple
  assembly_matrix = coo_array((data, (rows, cols)), shape=(NELEM * npt * npt, NELEM * npt * npt))
  return assembly_matrix


def init_assembly_local(vert_redundancy_local):
  # From this moment forward, we assume that
  # vert_redundancy_gll contains only the information
  # for processor-local GLL things,
  # and that remote_face_idx corresponds to processor local
  # ids.
  # hack: easier than figuring out indexing conventions

  data = []
  rows = [[], [], []]
  cols = [[], [], []]

  for ((local_face_idx, local_i, local_j),
       (remote_face_id, remote_i, remote_j)) in vert_redundancy_local:
    data.append(1.0)
    rows[0].append(remote_face_id)
    rows[1].append(remote_i)
    rows[2].append(remote_j)
    cols[0].append(local_face_idx)
    cols[1].append(local_i)
    cols[2].append(local_j)
  # print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")
  return (np.array(data, dtype=np.float64),
          [np.array(arr, dtype=np.int64) for arr in rows],
          [np.array(arr, dtype=np.int64) for arr in cols])


def init_assembly_global(vert_redundancy_send,
                         vert_redundancy_receive):
  # From this moment forward, we assume that
  # vert_redundancy_gll contains only the information
  # for processor-local GLL things,
  # and that remote_face_idx corresponds to processor local
  # ids.
  # hack: easier than figuring out indexing conventions

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
      rows = [[], [], []]
      cols = [[], [], []]
      for col_idx, (target_local_idx, target_i, target_j) in enumerate(vert_redundancy[source_proc_idx]):
        data.append(1.0)
        if transpose:
          cols[0].append(target_local_idx)
          cols[1].append(target_i)
          cols[2].append(target_j)
          rows[0].append(col_idx)
          rows[1].append(col_idx)
          rows[2].append(col_idx)
        else:
          rows[0].append(target_local_idx)
          rows[1].append(target_i)
          rows[2].append(target_j)
          cols[0].append(col_idx)
          cols[1].append(col_idx)
          cols[2].append(col_idx)
      triples[source_proc_idx] = (np.array(data, dtype=np.float64),
                                  [np.array(arr, dtype=np.int64) for arr in rows],
                                  [np.array(arr, dtype=np.int64) for arr in cols])
  # print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")
  return triples_send, triples_receive


def triage_vert_redundancy_flat(assembly_triple,
                                proc_idx,
                                decomp):
  # current understanding: this works because the outer
  # three for loops will iterate in exactly the same order for
  # the sending and recieving processor
  vert_redundancy_gll_flat = [((assembly_triple[1][0][k_idx],
                                assembly_triple[1][1][k_idx],
                                assembly_triple[1][2][k_idx]),
                               (assembly_triple[2][0][k_idx],
                                assembly_triple[2][1][k_idx],
                                assembly_triple[2][2][k_idx])) for k_idx in range(assembly_triple[0].shape[0])]
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
