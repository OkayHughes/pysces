from ..config import np, jnp, has_mpi, use_wrapper, wrapper_type, take_along_axis, mpi_rank
from ..operations_2d.assembly import summation_local_for
from ..operations_2d.operators import inner_prod

if has_mpi:
  from mpi4py import MPI
  from ..config import mpi_comm



def extract_fields_triple(fijk_fields, triples_send):
  """
  Extract values from a list of processor-local 3D fields before
  inter-process communication using assembly triples.

  *This can be used in performance code.*

  Parameters
  ----------
  fijk_fields : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of fields from which to extract redundant dofs to ship off
      to remote processors.
  triples_send : `dict[remote_proc_idx, tuple[Array[tuple[point_idx], Float],\
                                            Array[tuple[point_idx], Int],\
                                            Array[tuple[point_idx], Int]]`
      
      Mapping from `remote_proc_idx` to an assembly triple describing info
      to ship off to `remote_proc_idx`.
      
      Values are extracted as `fijk_fields[field_idx][elem_idx, i_idx, j_idx, :]`

  Returns
  -------
  `dict[remote_proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]`
      A mapping from `remote_proc_idx` to a list
      of buffers of redundant DOFs, preserving the number of levels in each field.

  Notes
  -----
  The number of levels in `fijk_fields[field_idx]` may vary across `field_idx`.

  """
  buffers = {}
  for remote_proc_idx in triples_send.keys():
    buffers[remote_proc_idx] = []
    for field_idx in range(len(fijk_fields)):
      (data, rows, cols) = triples_send[remote_proc_idx]
      relevant_data = take_along_axis(fijk_fields[field_idx].reshape((-1, fijk_fields[field_idx].shape[-1])),
                                      cols[:, np.newaxis],
                                      0) * data[:, np.newaxis]
      buffers[remote_proc_idx].append(relevant_data.T)
  return buffers


def sum_into(fijk_field, buffer, rows, dims):
  """
  Sum redundant dofs into a 3D field using assembly triples.

  Parameters
  ----------
  fijk_field: `Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]`
      Field values into which the values in buffer will be summed
  buffer: `Array[tuple[point_idx, level_idx], Float]`
      Redundant DOFs
  rows: `Array[tuple[point_idx], Int]`
      Indexes of redundant dofs, namely 
      `fijk_field[:, :, :, k_idx].flatten()[row[point_idx]]` corresponds to
      buffer[point_idx, k_idx].
  dims: `frozendict[str, Any]`
      Hashable dictionary containing key "shape", which is the shape of `fijk_field`.

  Returns
  -------
  fijk_field: `Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]`
      Field values into which the values in buffer have been summed
  
  Notes
  -----
  The implementation of this function is allowed to depend on wrapper_type.

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


def accumulate_fields_triple(fijk_fields, buffers, triples_receive, dims):
  """
  Sum non-processor-local redundant DOFs into a list of processor-local 3D fields after
  inter-process communication using assembly triples.

  *Can be used in performance code.*

  Parameters
  ----------
  fijk_fields : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of fields from which to extract redundant dofs to ship off
      to remote processors.
  buffers: `dict[remote_proc_idx, [Array[tuple[point_idx, level_idx], Float]]]`
      a mapping from `remote_proc_idx` to a list
      of buffers of redundant DOFs, preserving the number of levels in each field.
  triples_receive: `dict[remote_proc_idx, tuple[Array[tuple[point_idx], Float],\
                                               Array[tuple[point_idx], Int],\
                                               Array[tuple[point_idx], Int]]`
      Mapping from remote_proc_idx to a list of tuples of processor-local indexes into which
      redundant DOFs will be summed.

  Notes
  -----
  The number of levels in `fijk_fields[field_idx]` may vary across `field_idx`.

  Returns
  -------
  fijk_fields : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of fields into which non-processor-local redundant dofs have been summed.
  """
  for remote_proc_idx in buffers.keys():
    for field_idx in range(len(fijk_fields)):
      (_, rows, _) = triples_receive[remote_proc_idx]
      fijk_fields[field_idx] = sum_into(fijk_fields[field_idx],
                                        buffers[remote_proc_idx][field_idx],
                                        rows, dims)
  return fijk_fields



def extract_fields_for(fijk_fields, vert_redundancy_send):
  """
  Extract values from a list of processor-local 3D fields before
  inter-process communication using a for loop.

  *Designed for testing and debugging only,
  do not use in performance code*

  Parameters
  ----------
  fijk_fields : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of fields from which to extract redundant dofs to ship off
      to remote processors.
  vert_redundancy_send : `dict[remote_proc_idx, list[tuple[elem_idx, gll_idx, gll_idx]]]`
      Mapping from `remote_proc_idx` to a list of tuples of processor_local info
      to ship off to `remote_proc_idx`.
      Values are extracted as `fijk_fields[field_idx][elem_idx, i_idx, j_idx, :]`

  Notes
  -----
  The number of levels in `fijk_fields[field_idx]` may vary across `field_idx`.

  Returns
  -------
  `dict[remote_proc_idx, [Array[tuple[point_idx, level_idx], Float]]]`
      a mapping from `remote_proc_idx` to a list
      of buffers of redundant DOFs, preserving the number of levels in each field.
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
  Sum non-processor-local redundant DOFs into a list of processor-local 3D fields after
  inter-process communication using a for loop.

  *Designed for testing and debugging only,
  do not use in performance code*

  Parameters
  ----------
  fijk_fields : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of fields from which to extract redundant dofs to ship off
      to remote processors.
  buffers: `dict[remote_proc_idx, [Array[tuple[point_idx, level_idx], Float]]]`
      a mapping from `remote_proc_idx` to a list
      of buffers of redundant DOFs, preserving the number of levels in each field.
  vert_redundancy_receive : `dict[remote_proc_idx, list[tuple[elem_idx, gll_idx, gll_idx]]]`
      Mapping from remote_proc_idx to a list of tuples of processor-local indexes into which
      redundant DOFs will be summed.

  Notes
  -----
  The number of levels in `fijk_fields[field_idx]` may vary across `field_idx`.

  Returns
  -------
  fijk_fields : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of fields into which non-processor-local redundant dofs have been summed.
  """
  for remote_proc_idx in buffers.keys():
    for field_idx in range(len(fijk_fields)):
      for col_idx, (target_local_idx, target_i, target_j) in enumerate(vert_redundancy_receive[remote_proc_idx]):
          fijk_fields[field_idx][target_local_idx,
                                 target_i,
                                 target_j, :] += buffers[remote_proc_idx][field_idx][:, col_idx]
  return fijk_fields


def exchange_buffers_stub(buffer_list):
  """
  Exchange buffers between source dofs and target dofs assuming that all grid is processor-local.

  *Only used for testing and debugging, do not use in performance
  code*

  Parameters
  ----------
  buffer_list: `list[dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]]`
      A list of length num_processors, each of which is a buffer struct
      that maps `proc_idx` to a list of arrays containing redundant DOFs to send.

  Returns
  -------
  `list[dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]]`
      A list of length num_processors, each of which is a buffer struct
      that maps proc_idx to a list of arrays containing redundant DOFs that were received.

  Notes
  ------
  This function exchanges the memory reffered to by `buffer_list[proc_idx][remote_proc_idx][field_idx]`
  with `buffer_list[remote_proc_idx][proc_idx][field_idx]`.
  The behavior should be almost identical to how exchange_buffers_mpi 
  behaves when called when has_mpi=True, except for this difference.

  By construction, if any grid point `(elem_idx_source, i_idx_source, j_idx_source) `
  that has a redundancy with `(elem_idx_target, i_idx_target, j_idx_target)`,
  this relation is symmetric. Therefore, the number of grid points
  necessary to send from `proc_idx_1` to `proc_idx_2` is identical
  to the number to send from `proc_idx_2` to `proc_idx_1`.
  The indexes of points in the buffer that is sent
  will be different from those in the buffer that is received,
  but so long as both processes agree on the different orderings,
  this is fine.
  """
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
  Exchange Spectral Element grid non-processor-local redundant DOFS
  between processes using the Message Passing Interface.

  **This function is the only function in the entire codebase
  that will hang indefinitely in the event of, e.g., hardware failures
  on a remote processor, or other distributed-memory shenanigans.**

  Parameters
  ----------
  buffer: `dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]`
      A buffer struct that maps `proc_idx` to a
      list of arrays containing redundant DOFs to send to that processor.

  Returns
  -------
  buffer: `dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]`
      A buffer struct that maps `proc_idx` to a
      list of arrays containing redundant DOFs received from that processor.

  Notes
  -----
  mpi4py is designed to accept objects that buffer properties
  that resemble np.ndarrays. This function can almost certainly
  be designed in a way that can leverage gpu-aware MPI environments,
  but this functionality has not yet been tested.
  Divergence in how this is performed with different wrapper types
  is acceptable.

  Raises
  ------
  Error
    Any error that can be raised by the following two functions:
    * `mpi_comm.Isendrecv_replace`
    * `MPI.Request.Waitall`

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


def assemble_scalar_for_pack(fs_local, grid):
  """
  Extract processor-local list of scalars before
  inter-process communication using a for loop.

  *This is only used for testing correctness of assembly.
  Do not use this in performance-critical code.*

  Parameters
  ----------
  fs_local : `list[Array[tuple[local_elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of 2D scalar fields to extract for communication
  grid : dict[str, Any]
      Processor-local Spectral Element Grid struct.
      Contains "vert_redundancy_send" key, which
      has type `dict[remote_proc_idx, list[tuple[elem_idx, gll_idx, gll_idx]]]`
                                    
  Returns
  -------
  `dict[proc_idx, list[Array[tuple[local_point_idx, 1], Float]]`
      Mapping from the destination processor idx to an array
      of values where `buffer[remote_proc_idx][pt_idx, 1]` is being sent to
      remote_proc_idx and summed on remote_proc_idx into
      `(elem_idx, i, j) = vert_redundancy_receive[local_proc_idx][pt_idx]`

  Notes
  -----
  Grid must be constructed with wrapped=False to contain
  vert_redundancy_send.

  See `se_grid.create_spectral_element_grid`
  for an example of what the grid struct looks like.

  """
  buffers = extract_fields_for([f.reshape((*f.shape, 1)) for f in fs_local], grid["vert_redundancy_send"])
  return buffers


def assemble_scalar_triple_pack(fs_local, grid):
  """
  Extract processor-local list of scalars before
  inter-process communication using assembly triples.

  Assembly triples are designed for use in wrapped code, and
  have the best shot of being compatible with automatic differentiation.

  Parameters
  ----------
  fs_local : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of 2D scalar fields to extract for communication
  grid : `dict[str, Any]`
      Processor-local Spectral Element Grid struct.
      Contains "triples_send" key, which
      has type `dict[remote_proc_idx, tuple[Array[tuple[point_idx], Float],\
                                           Array[tuple[point_idx], Int],\
                                           Array[tuple[point_idx], Int]]]`

  Returns
  -------
  `dict[proc_idx, list[Array[tuple[local_point_idx, 1], Float]]`
      Mapping from the destination processor idx to an array
      of values where `buffer[remote_proc_idx][pt_idx, 1]` is being sent to
      `remote_proc_idx` and summed into
      `point_idx = triples_receive[local_proc_idx][1][pt_idx]`

  Notes
  -----
  See `se_grid.create_spectral_element_grid`
  for an example of what the grid struct looks like.

  """
  buffers = extract_fields_triple([f.reshape((*f.shape, 1)) for f in fs_local], grid["triples_send"])
  return buffers


def assemble_scalar_for_unpack(fs_local, buffers, grid, *args):
  """
  Sum remote processor redundancies into a list of scalars
  after inter-process communication using a for loop.

  *This is only used for testing correctness of assembly.
  Do not use this in performance-critical code.*

  Parameters
  ----------
  fs_local : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of 2D scalar fields to sum redundant DOFs into.
  buffers: `dict[proc_idx, list[Array[tuple[local_point_idx, 1], Float]]`
      Mapping from source processor idx to an array
      of values where `buffer[remote_proc_idx][pt_idx, 1]` is being 
      acumulated on the current process into
      `(elem_idx, i, j) = vert_redundancy_receive[remote_proc_idx][pt_idx]`
  grid : `dict[str, Any]`
      Processor-local Spectral Element Grid struct.
      Contains "vert_redundancy_receive" key, which
      has type `dict[remote_proc_idx, list[tuple[elem_idx, gll_idx, gll_idx]]]`

  Returns
  -------
  `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of scalar fields into which redundant DOF info has been accumulated.

  Notes
  -----
  Grid must be constructed with wrapped=False to contain
  vert_redundancy_receive.

  See `se_grid.create_spectral_element_grid`
  for an example of what the grid struct looks like.

  """
  return [f[:, :, :, 0] for f in accumulate_fields_for([f.reshape((*f.shape, 1)) for f in fs_local],
                                                       buffers, grid["vert_redundancy_receive"])]


def assemble_scalar_triple_unpack(fs_local, buffers, grid, dim, *args):
  """
  Sum non-processor-local redundancies into list of scalars
  after inter-process communication using assembly triples.

  Assembly triples are designed for use in wrapped code, and
  have the best shot of being compatible with automatic differentiation.

  Parameters
  ----------
  fs_local : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of scalar fields to extract for communication
  buffers: `dict[proc_idx, list[Array[tuple[local_point_idx, 1], Float]]`
      Mapping from the (remote) source processor idx to an array
      of values where `buffer[remote_proc_idx][pt_idx, 1]` is being sent to
      `proc_idx` and summed on `proc_idx` into
      `point_idx = triples_receive[proc_idx][1][pt_idx]`
  grid : `dict[str, Any]`
      Processor-local Spectral Element Grid struct.
      Contains "triples_receive" key, which
      has type `dict[remote_proc_idx, tuple[Array[tuple[point_idx], Float],\
                                           Array[tuple[point_idx], Int],\
                                           Array[tuple[point_idx], Int]]]`

  Returns
  -------
  `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of scalar fields into which redundant DOFS have been summed.

  Notes
  -----
  See `se_grid.create_spectral_element_grid`
  for an example of what this struct looks like.

  """
  return [f[:, :, :, 0] for f in accumulate_fields_triple([f.reshape((*f.shape, 1)) for f in fs_local],
                                                       buffers, grid["triples_receive"], dim)]


def project_scalar_for_stub(fs_global, grids):
  """
  Perform continuity projection on a list of scalars assuming all data is processor local
  using stub communicator and a for loop.

  *Only intended for testing and debugging.*

  Parameters
  ----------
  fs_global : `list[list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]]`
      List of length `num_proc`, where fs_global[proc_idx]
      is a list of "processor-local" 2D scalars to perform continuity projection on.
  grids : `list[dict[str, Any]]`
      List of grids containing "vert_redundancy_send",
      "vert_redundancy_receive", and "vert_redundancy_local". 

  Returns
  -------
  `list[list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]]`
      List of length `num_proc`, where `fs_global[proc_idx]`
      is a list of C0 "processor-local" globally continuous 2D scalar has been performed.

  Notes
  -----
  * See `se_grid.create_spectral_element_grid` with `proc_idx=local_proc_idx`
  for how to create the `grids[local_proc_idx]` entries.
  * To calculate `fs_global[proc_idx][field_idx]`,
  use `se_grid.subset_var` to subdivide a given `f` calculated
  using an entire grid, or re-calculate `f` using, e.g.,
  `grids[proc_idx]["physical_coords]`.
  * To create your own dummy grid, see `se_grid.vert_redundancy_triage`
  for how to create "vert_redundancy_send", "vert_redundancy_receive",
  and "vert_redundancy_local" entries.

  """

  buffers = []
  for fs_local, grid in zip(fs_global, grids):
    buffers.append(assemble_scalar_for_pack([f * grid["mass_matrix"] for f in fs_local], grid))

  fs_out = [[summation_local_for(f * grid["mass_matrix"], grid) for f in fs_local]
            for (fs_local, grid) in zip(fs_global, grids)]
  buffers = exchange_buffers_stub(buffers)

  for proc_idx in range(len(fs_out)):
    fs_out[proc_idx] = [f * grids[proc_idx]["mass_matrix_inv"]
                        for f in assemble_scalar_for_unpack(fs_out[proc_idx], buffers[proc_idx], grids[proc_idx])]

  return fs_out


def project_scalar_for_mpi(fs, grid):
  """
  Perform continuity projection on a list of processor-local scalars using a for loop.

  *Only used for testing, do not use in performance-critical code*

  Parameters
  ----------
  fs : `list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]`
      List of processor-local 2D scalars to perform continuity projection on.
  grids : `list[dict[str, Any]]`
      List of grids, each of which contains
      "vert_redundancy_send" and "vert_redundancy_receive", and "vert_redundancy_local"

  Returns
  -------
  `list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]`
      List of continuous processor-local scalars on which continuity projection was performed.

  Notes
  -----
  `create_spectral_element_grid` must be called with `wrapped=False`
  for it to contain "vert_redundancy_send", "vert_redundancy_receive",
  and "vert_redundancy_local"

  * See `se_grid.create_spectral_element_grid` with `proc_idx=local_proc_idx`
  for how to create the `grids[proc_idx]` entries.
  * To create your own dummy grid, see `se_grid.vert_redundancy_triage`
  for how to create "vert_redundancy_send", "vert_redundancy_receive",
  and "vert_redundancy_local".

  Raises
  ------
  Error
    Raises any error that can be raised by exchange_buffers_mpi function.

  """
  # This is primarily for testing!
  # do not use in model code!
  data_scaled = [f * grid["mass_matrix"] for f in fs]
  buffer = assemble_scalar_for_pack(data_scaled, grid)
  buffer = exchange_buffers_mpi(buffer)
  fs_out = [summation_local_for(data_scaled, grid) for f in fs]
  fs = [f * grid["mass_matrix_inv"]
                        for f in assemble_scalar_for_unpack(fs_out, buffer, grid)]
  return fs


def project_scalar_triple_mpi(fs, grid, dim, scaled=True):
  """
  Perform continuity projection on a list of processor-local scalars using projection triples.

  Can be used for performance code.

  Parameters
  ----------
  fs : `list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]`
      List of processor-local 2D scalars to perform continuity projection on.
  grids : `list[dict[str, Any]]`
      List of grids, each of which contains
      "triples_send", "triples_receive", and "assembly_triple"

  Returns
  -------
  `list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]`
      List of continuous processor-local scalars on which continuity projection was performed.

  Notes
  -----
  * See `se_grid.create_spectral_element_grid` with `proc_idx=local_proc_idx`
  for how to create the `grids[proc_idx]` entries.
  * To create your own dummy grid, see `se_grid.vert_redundancy_triage`
  for how to create "vert_redundancy_send", "vert_redundancy_receive",
  and "vert_redundancy_local", then see
  `se_grid.init_assembly_global` and `se_grid.init_assembly_local` for how to generate
  ("triples_send", "triples_receive"), and "assembly_triple", respectively.

  Raises
  ------
  Error
    Raises any error that can be raised by exchange_buffers_mpi function.

  """
  buffer = assemble_scalar_triple_pack([f * grid["mass_matrix"] for f in fs], grid)
  buffer = exchange_buffers_mpi(buffer)
  # TODO: replace with sum_into

  local_buffer = extract_fields_triple([(f * grid["mass_matrix"]).reshape((*dim["shape"], 1)) for f in fs], {mpi_rank: grid["assembly_triple"]})[mpi_rank]
  fs_out = []
  #fs_out = [summation_local_for(f * grid["mass_matrix"], grid) for f in fs]
  #local_buffer = extract_fields_triple([(f * grid["mass_matrix"]).reshape((*dim["shape"], 1)) for f in fs], {mpi_rank: grid["dss_triple"]})[mpi_rank]
  for f, local_buf in zip(fs, local_buffer):
      fs_out.append(sum_into((f * grid["mass_matrix"]).reshape((*dim["shape"], 1)), local_buf, grid["assembly_triple"][1], dim))
  fs = [f.squeeze() * grid["mass_matrix_inv"] for f in assemble_scalar_triple_unpack(fs_out,
                                                                                     buffer,
                                                                                     grid,
                                                                                     dim)]
  return fs


def project_scalar_triple_stub(fs_global, grids, dims):
  """
  Perform continuity projection on a list of scalars assuming all data is processor local
  using stub communicator and projection triples.

  *Only intended for testing and debugging.*

  Parameters
  ----------
  fs_global : `list[list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]]`
      List of length `num_proc`, where fs_global[proc_idx]
      is a list of "processor-local" 2D scalars to perform continuity projection on.
  grids : `list[dict[str, Any]]`
      List of grids containing "triples_send",
      "triples_receive", and "assembly_triple". 

  Returns
  -------
  `list[list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]]`
      List of length `num_proc`, where `fs_global[proc_idx]`
      is a list of C0 "processor-local" 2D scalars on which projection has been performed.

  Notes
  -----
  * See `se_grid.create_spectral_element_grid` with `proc_idx=local_proc_idx`
  for how to create the `grids[proc_idx]` entries.
  * To calculate `fs_global[proc_idx][field_idx]`,
  use `se_grid.subset_var` to subdivide a given `f` calculated
  using an entire grid, or re-calculate `f` using, e.g.,
  `grids[proc_idx]["physical_coords]`.
  * To create your own dummy grid, see `se_grid.vert_redundancy_triage`
  for how to create "vert_redundancy_send", "vert_redundancy_receive",
  and "vert_redundancy_local", then see
  `se_grid.init_assembly_global` and `se_grid.init_assembly_local` for how to generate
  ("triples_send", "triples_receive"), and "assembly_triple", respectively.

  """
  
  buffers = []
  data_scaled = []
  local_buffers = []
  for fs_local, grid, dim in zip(fs_global, grids, dims):
    data_scaled.append([f * grid["mass_matrix"] for f in fs_local])
    buffers.append(assemble_scalar_triple_pack([f * grid["mass_matrix"] for f in fs_local], grid))
    local_buffers.append(extract_fields_triple([(f * grid["mass_matrix"]).reshape((*dim["shape"], 1)) for f in fs_local], {mpi_rank: grid["assembly_triple"]})[mpi_rank])

  # TODO: replace with sum_into
  #fs_out = [[summation_local_for(f * grid["mass_matrix"], grid) for f in fs_local]
  #          for (fs_local, grid) in zip(fs_global, grids)]
  fs_out = []
  for (fs_local, grid, dim, buffer_list) in zip(data_scaled, grids, dims, local_buffers):
    fs_out.append([])
    for f_scaled, buffer in zip(fs_local, buffer_list):
      fs_out[-1].append(sum_into((f_scaled).reshape((*dim["shape"], 1)), buffer, grid["assembly_triple"][1], dim))

  buffers = exchange_buffers_stub(buffers)

  for proc_idx in range(len(fs_out)):
    fs_out[proc_idx] = [f.squeeze() * grids[proc_idx]["mass_matrix_inv"]
                        for f in assemble_scalar_triple_unpack(fs_out[proc_idx], buffers[proc_idx], grids[proc_idx], dims[proc_idx])]

  return fs_out


def global_sum(summand):
  """
  Compute the global sum of a processor-local quantity
  such as a summed integrand.

  Parameters
  ----------
  summand : float
    Processor-local part of the quantity over which reduction is 
    performed.

  Returns
  -------
  integral : float
    Global sum of quantity.
  """
  reqs = []
  if not has_mpi:
    raise NotImplementedError("MPI communication called with has_mpi = False")
  send = np.array(summand)
  recv = np.copy(send)
  req = mpi_comm.Iallreduce(np.array(send),
                            recv,
                            MPI.SUM)
  MPI.Request.Wait(req)
  return recv.item()