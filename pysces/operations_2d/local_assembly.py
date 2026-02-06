from ..config import np, jnp, use_wrapper, jit, wrapper_type, DEBUG, device_wrapper
from ..mpi.processor_decomposition import global_to_local, elem_idx_global_to_proc_idx
from scipy.sparse import coo_array
from functools import partial
if use_wrapper and wrapper_type == "jax":
  from ..config import num_jax_devices, projection_sharding, extraction_sharding, usual_scalar_sharding, do_sharding


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


def segment_max(data,
                segment_ids,
                N):
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
  s = np.copy(data)
  np.max.at(s, segment_ids, data)
  return s


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
  shape = f.shape

  scaled_f = f * grid["mass_matrix"]
  if use_wrapper and wrapper_type == "jax":
    if do_sharding:
      scaled_f = scaled_f.reshape((num_jax_devices, -1, dims["npt"], dims["npt"]), out_sharding=projection_sharding)
      extraction_struct = grid["shard_extraction_map"]

      relevant_data = (scaled_f).at[extraction_struct["extract_from"]["shard_idx"],
                                    extraction_struct["extract_from"]["elem_idx"],
                                    extraction_struct["extract_from"]["i_idx"],
                                    extraction_struct["extract_from"]["j_idx"]].get(out_sharding=extraction_sharding)
      relevant_data *= extraction_struct["mask"]
      scaled_f = scaled_f.at[extraction_struct["sum_into"]["shard_idx"],
                             extraction_struct["sum_into"]["elem_idx"],
                             extraction_struct["sum_into"]["i_idx"],
                             extraction_struct["sum_into"]["j_idx"]].add(relevant_data,
                                                                         out_sharding=projection_sharding)
      scaled_f = scaled_f.reshape(shape, out_sharding=usual_scalar_sharding)
    else:
      relevant_data = (scaled_f).at[cols[0], cols[1], cols[2]].get()
      scaled_f = scaled_f.at[rows[0], rows[1], rows[2]].add(relevant_data)
  elif use_wrapper and wrapper_type == "torch":
    # this is broken
    scaled_f = scaled_f.flatten()
    scaled_f = scaled_f.scatter_add_(0, rows, relevant_data)
    scaled_f = scaled_f.reshape(dims["shape"])
    relevant_data = scaled_f[cols[0], cols[1], cols[2]]
  else:
    relevant_data = scaled_f[cols[0], cols[1], cols[2]]
    segment_sum(scaled_f, relevant_data, rows)
  return scaled_f * grid["mass_matrix_denominator"]


@partial(jit, static_argnames=["dims"])
def max_scalar(f,
               grid,
               dims):
  """
  """
  (_, rows, cols) = grid["assembly_triple"]

  relevant_data = (f).flatten().take(cols)
  if use_wrapper and wrapper_type == "jax":
    scaled_f = scaled_f.flatten().at[rows].max(relevant_data)
    scaled_f = scaled_f.reshape(dims["shape"])
  elif use_wrapper and wrapper_type == "torch":
    pass
    # scaled_f = scaled_f.flatten()
    # scaled_f = scaled_f.scatter_add_(0, rows, relevant_data)
    # scaled_f = scaled_f.reshape(dims["shape"])
  else:
    scaled_f = segment_max(relevant_data, rows, dims["N"]).reshape(dims["shape"])
  return scaled_f 

@partial(jit, static_argnames=["dims"])
def min_scalar(f,
               grid,
               dims):
  """
  """
  (_, rows, cols) = grid["assembly_triple"]

  relevant_data = (-f).flatten().take(cols)
  if use_wrapper and wrapper_type == "jax":
    scaled_f = scaled_f.flatten().at[rows].max(relevant_data)
    scaled_f = scaled_f.reshape(dims["shape"])
  elif use_wrapper and wrapper_type == "torch":
    pass
    # scaled_f = scaled_f.flatten()
    # scaled_f = scaled_f.scatter_add_(0, rows, relevant_data)
    # scaled_f = scaled_f.reshape(dims["shape"])
  else:
    scaled_f = segment_max(relevant_data, rows, dims["N"]).reshape(dims["shape"])
  return -scaled_f 

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


def init_shard_extraction_map(assembly_triple, num_devices, nelem_padded, dims, wrapped=True):
  # this will maybe eventually be rewritten to work for bigger grids?
  assert np.abs(np.round(nelem_padded / num_devices) - nelem_padded / num_devices) < 1e-6, "Did you pad your array?"

  if wrapped:
    wrapper = device_wrapper
  else:
    def wrapper(x, *args, **kwargs):
      return x

  shard_idx, elem_idx = np.meshgrid(np.arange(num_devices, dtype=np.int64),
                                    np.arange(np.round(nelem_padded / num_devices), dtype=np.int64))
  shard_flat = shard_idx.flatten(order="F")
  elem_flat = elem_idx.flatten(order="F")
  sum_into_shard = [[] for _ in range(num_devices)]
  extract_from_shard = [[] for _ in range(num_devices)]

  for ((f_row, i_row, j_row),
       (f_col, i_col, j_col)) in zip(zip(assembly_triple[1][0],
                                         assembly_triple[1][1],
                                         assembly_triple[1][2]),
                                     zip(assembly_triple[2][0],
                                         assembly_triple[2][1],
                                         assembly_triple[2][2])):
    sum_into_shard_idx = shard_flat[f_row]
    shard_local_elem_idx = elem_flat[f_row]
    extract_from_shard_idx = shard_flat[f_col]
    shard_remote_elem_idx = elem_flat[f_col]
    sum_into_shard[sum_into_shard_idx].append([sum_into_shard_idx, shard_local_elem_idx, i_row, j_row])
    extract_from_shard[sum_into_shard_idx].append([extract_from_shard_idx, shard_remote_elem_idx, i_col, j_col])
  max_dof = max([len(x) for x in sum_into_shard])
  if DEBUG:
    max_dof_maybe = max([len(x) for x in extract_from_shard])
    assert max_dof == max_dof_maybe
  size_of_comm = (num_devices, max_dof)
  coeff_mat = np.zeros(size_of_comm, dtype=np.float64)
  sum_into_shard_idxs = np.zeros(size_of_comm, dtype=np.int64)
  sum_into_elem_idxs = np.zeros(size_of_comm, dtype=np.int64)
  sum_into_i_idxs = np.zeros(size_of_comm, dtype=np.int64)
  sum_into_j_idxs = np.zeros(size_of_comm, dtype=np.int64)
  extract_from_shard_idxs = np.zeros(size_of_comm, dtype=np.int64)
  extract_from_elem_idxs = np.zeros(size_of_comm, dtype=np.int64)
  extract_from_i_idxs = np.zeros(size_of_comm, dtype=np.int64)
  extract_from_j_idxs = np.zeros(size_of_comm, dtype=np.int64)
  for shard_idx in range(num_devices):
    sum_into_data = np.array(sum_into_shard[shard_idx])
    extract_from_data = np.array(extract_from_shard[shard_idx])
    if sum_into_data.ndim == 1:
      sum_into_data = sum_into_data[np.newaxis, :]
    if extract_from_data.ndim == 1:
      extract_from_data = extract_from_data[np.newaxis, :]
    num_pts = sum_into_data.shape[0]
    if DEBUG:
      num_pts_maybe = extract_from_data.shape[0]
      assert num_pts == num_pts_maybe

    sum_into_shard_idxs[shard_idx, :num_pts] = sum_into_data[:, 0]
    sum_into_elem_idxs[shard_idx, :num_pts] = sum_into_data[:, 1]
    sum_into_i_idxs[shard_idx, :num_pts] = sum_into_data[:, 2]
    sum_into_j_idxs[shard_idx, :num_pts] = sum_into_data[:, 3]

    extract_from_shard_idxs[shard_idx, :num_pts] = extract_from_data[:, 0]
    extract_from_elem_idxs[shard_idx, :num_pts] = extract_from_data[:, 1]
    extract_from_i_idxs[shard_idx, :num_pts] = extract_from_data[:, 2]
    extract_from_j_idxs[shard_idx, :num_pts] = extract_from_data[:, 3]
    coeff_mat[shard_idx, :num_pts] = 1.0
  return {"sum_into": {"shard_idx": wrapper(sum_into_shard_idxs, dtype=jnp.int64, elem_sharding_axis=0),
                       "elem_idx": wrapper(sum_into_elem_idxs, dtype=jnp.int64, elem_sharding_axis=0),
                       "i_idx": wrapper(sum_into_i_idxs, dtype=jnp.int64, elem_sharding_axis=0),
                       "j_idx": wrapper(sum_into_j_idxs, dtype=jnp.int64, elem_sharding_axis=0)},
          "extract_from": {"shard_idx": wrapper(extract_from_shard_idxs, dtype=jnp.int64, elem_sharding_axis=0),
                           "elem_idx": wrapper(extract_from_elem_idxs, dtype=jnp.int64, elem_sharding_axis=0),
                           "i_idx": wrapper(extract_from_i_idxs, dtype=jnp.int64, elem_sharding_axis=0),
                           "j_idx": wrapper(extract_from_j_idxs, dtype=jnp.int64, elem_sharding_axis=0)},
          "mask": wrapper(coeff_mat, dtype=jnp.int64, elem_sharding_axis=0)}, max_dof
