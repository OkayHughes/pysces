from ..config import np, jnp, has_mpi, use_wrapper, wrapper_type, take_along_axis
from ..operations_2d.assembly import summation_local_for


if has_mpi:
  from mpi4py import MPI
  from ..config import mpi_comm


def dss_scalar_for_pack(fs_local, grid):
  """
  Extract 3d fields before distributed memory communication using a for loop.

  Parameters
  ----------
  fs_local : list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]
      List of scalar fields to extract for communication
  grid : dict[str, Any]
      Processor-local Spectral Element Grid struct.
      Contains "vert_redundancy_send" key, which
      has type dict[proc_idx, tuple[Array[tuple[local_point_idx], Float],
                                    Array[tuple[local_point_idx], Int],
                                    Array[tuple[local_point_idx], Int]]]

  Returns
  -------
  buffers: dict[proc_idx, list[Array[tuple[local_point_idx, level_idx], Float]]
      Mapping from remote processor idx to a three-tuple where
      the first item is the scaling applied to 

  See Also
  --------
  See pysces.operations_2d.se_grid.create_spectral_element_grid
  for example of what this struct looks like

  """
  buffers = extract_fields_for([f.reshape((*f.shape, 1)) for f in fs_local], grid["vert_redundancy_send"])
  return buffers


def dss_scalar_triple_pack(fs_local, grid):
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
  buffers = extract_fields_triple([f.reshape((*f.shape, 1)) for f in fs_local], grid["vert_redundancy_send"])
  return buffers


def dss_scalar_for_unpack(fs_local, buffers, grid, *args):
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
  return [f[:, :, :, 0] for f in accumulate_fields_for([f.reshape((*f.shape, 1)) for f in fs_local],
                                                       buffers, grid["vert_redundancy_receive"])]


def dss_scalar_for_stub(fs_global, grids):
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
  # This is primarily for testing!
  # do not use in model code!

  buffers = []
  for fs_local, grid in zip(fs_global, grids):
    buffers.append(dss_scalar_for_pack([f * grid["mass_matrix"] for f in fs_local], grid))

  fs_out = [[summation_local_for(f * grid["mass_matrix"], grid) for f in fs_local]
            for (fs_local, grid) in zip(fs_global, grids)]
  buffers = exchange_buffers_stub(buffers)

  for proc_idx in range(len(fs_out)):
    fs_out[proc_idx] = [f * grids[proc_idx]["mass_matrix_inv"]
                        for f in dss_scalar_for_unpack(fs_out[proc_idx], buffers[proc_idx], grids[proc_idx])]

  return fs_out


def dss_scalar_for_mpi(f, grid):
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
  # This is primarily for testing!
  # do not use in model code!
  buffer = dss_scalar_for_pack(f, grid)
  buffer = exchange_buffers_mpi(buffer)
  # f = dss_scalar_for_unscaled(dss_scalar_for_unpack(f, buffer, grid), grid)
  return f


def extract_fields_for(fijk_fields, vert_redundancy_send):
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
  buffers = {}
  for remote_proc_idx in vert_redundancy_send.keys():
    buffers[remote_proc_idx] = []
    for field_idx in range(len(fijk_fields)):
      data = []
      for (source_local_idx, source_i, source_j) in vert_redundancy_send[remote_proc_idx]:
        data.append(fijk_fields[field_idx][source_local_idx, source_i, source_j, :])
      buffers[remote_proc_idx].append(np.stack(data, axis=-1))
  return buffers


def accumulate_fields_for(fijk_fields, buffers, vert_redundancy_receive):
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
  # designed for device code to be tested against, but this is much more transparent
  for remote_proc_idx in buffers.keys():
    for field_idx in range(len(fijk_fields)):
      for col_idx, (target_local_idx, target_i, target_j) in enumerate(vert_redundancy_receive[remote_proc_idx]):
          fijk_fields[field_idx][target_local_idx,
                                 target_i,
                                 target_j, :] += buffers[remote_proc_idx][field_idx][:, col_idx]
  return fijk_fields


def exchange_buffers_stub(buffer_list):
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
  # assumes access to list of buffers for all grid chunks
  pairs = set()
  for source_proc_idx in range(len(buffer_list)):
    buffer = buffer_list[source_proc_idx]
    for target_proc_idx in buffer.keys():
      if (target_proc_idx, source_proc_idx) not in pairs:
        # Python names and lists are counter-intuitive
        # so I'm leaving this ugly for the moment.
        buffer[target_proc_idx], buffer_list[target_proc_idx][source_proc_idx] = (buffer_list[target_proc_idx][source_proc_idx],
                                                                                  buffer[target_proc_idx])
        pairs.add((source_proc_idx, target_proc_idx))
  return buffer_list


def exchange_buffers_mpi(buffer):
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
  reqs = []
  if not has_mpi:
    raise NotImplementedError("MPI communication called with has_mpi = False")
  for source_proc_idx in buffer.keys():
    for k_idx in range(len(buffer[source_proc_idx])):
      reqs.append(mpi_comm.Isendrecv_replace(buffer[source_proc_idx][k_idx],
                                             source_proc_idx,
                                             source=source_proc_idx,
                                             sendtag=k_idx,
                                             recvtag=k_idx))
  MPI.Request.Waitall(reqs)
  return buffer


def extract_fields_triple(fijk_fields, vert_redundancy_send):
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
  buffers = {}
  for remote_proc_idx in vert_redundancy_send.keys():
    buffers[remote_proc_idx] = []
    for field_idx in range(len(fijk_fields)):
      (data, rows, cols) = vert_redundancy_send[remote_proc_idx]
      relevant_data = take_along_axis(fijk_fields[field_idx].reshape((-1, fijk_fields[field_idx].shape[-1])),
                                      rows[:, np.newaxis],
                                      0) * data[:, np.newaxis]
      buffers[remote_proc_idx].append(relevant_data.T)
  return buffers


def sum_into(fijk_field, buffer, rows, dims):
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
  if not use_wrapper:
    for k_idx in range(fijk_field.shape[-1]):
      res = fijk_field[:, :, :, k_idx].flatten()
      np.add.at(res, rows, buffer[k_idx, :])
      fijk_field[:, :, :, k_idx] = res.reshape(fijk_field.shape[:-1])
  elif wrapper_type == "jax":
    fijk_field = fijk_field.reshape((-1, fijk_field.shape[-1])).at[rows, :].add(buffer.T)
    fijk_field = fijk_field.reshape((*dims["shape"], fijk_field.shape[-1]))
  elif wrapper_type == "torch":
    nfield = fijk_field.shape[-1]
    fijk_field = fijk_field.reshape((-1, fijk_field.shape[-1]))
    fijk_field = fijk_field.scatter_add_(0, rows[:, np.newaxis] * jnp.ones((1, nfield), dtype=jnp.int64), buffer.T)
    fijk_field = fijk_field.reshape((*dims["shape"], nfield))
  return fijk_field


def accumulate_fields_triple(fijk_fields, buffers, vert_redundancy_receive, dims):
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
  for remote_proc_idx in buffers.keys():
    for field_idx in range(len(fijk_fields)):
      (_, rows, _) = vert_redundancy_receive[remote_proc_idx]
      fijk_fields[field_idx] = sum_into(fijk_fields[field_idx],
                                        buffers[remote_proc_idx][field_idx],
                                        rows, dims)
  return fijk_fields
