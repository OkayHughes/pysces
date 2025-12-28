from .config import np, jnp, has_mpi, put_along_axis_pk, use_wrapper, wrapper_type
from .assembly import summation_local_for


if has_mpi:
  from mpi4py import MPI
  from .config import mpi_comm


def dss_scalar_for_pack(fs_local, grid):
  buffers = extract_fields_for([f.reshape((*f.shape, 1)) for f in fs_local], grid["vert_redundancy_send"])
  return buffers


def dss_scalar_for_unpack(fs_local, buffers, grid, *args):
  return [f[:, :, :, 0] for f in accumulate_fields_for([f.reshape((*f.shape, 1)) for f in fs_local], buffers, grid["vert_redundancy_receive"])]


def dss_scalar_for_stub(fs_global, grids):
  # This is primarily for testing!
  # do not use in model code!

  buffers = []
  for fs_local, grid in zip(fs_global, grids):
    buffers.append(dss_scalar_for_pack([f * grid["mass_matrix"] for f in fs_local], grid))

  fs_out = [[summation_local_for(f * grid["mass_matrix"], grid) for f in fs_local] for (fs_local, grid) in zip(fs_global, grids)]
  buffers = exchange_buffers_stub(buffers)

  for proc_idx in range(len(fs_out)):
    fs_out[proc_idx] = [f * grids[proc_idx]["mass_matrix_inv"] for f in dss_scalar_for_unpack(fs_out[proc_idx], buffers[proc_idx], grids[proc_idx])]

  return fs_out


def dss_scalar_for_mpi(f, grid):
  # This is primarily for testing!
  # do not use in model code!
  buffer = dss_scalar_for_pack(f, grid)
  buffer = exchange_buffers_mpi(buffer)
  f = dss_scalar_for_unscaled(dss_scalar_for_unpack(f, buffer, grid), grid)
  return f


def extract_fields_for(fijk_fields, vert_redundancy_send):
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
  # designed for device code to be tested against, but this is much more transparent
  for remote_proc_idx in buffers.keys():
    for field_idx in range(len(fijk_fields)):
      for col_idx, (target_local_idx, target_i, target_j) in enumerate(vert_redundancy_receive[remote_proc_idx]):
          fijk_fields[field_idx][target_local_idx, target_i, target_j, :] += buffers[remote_proc_idx][field_idx][:, col_idx]
  return fijk_fields


def exchange_buffers_stub(buffer_list):
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
  reqs = []
  if not has_mpi:
    raise NotImplementedError("MPI communication called with has_mpi = False")
  for source_proc_idx in buffer.keys():
    for k_idx in range(len(buffer[source_proc_idx])):
      reqs.append(mpi_comm.Isendrecv_replace(buffer[source_proc_idx][k_idx], source_proc_idx, source=source_proc_idx, sendtag=k_idx, recvtag=k_idx))
  MPI.Request.Waitall(reqs)
  return buffer


def extract_fields_matrix():
  pass


def extract_fields_jax(fijk_fields, vert_redundancy_send):
  buffers = {}
  for remote_proc_idx in vert_redundancy_send.keys():
    buffers[remote_proc_idx] = []
    for field_idx in range(len(fijk_fields)):
      (data, rows, cols) = vert_redundancy_send[remote_proc_idx]
      relevant_data = jnp.take_along_axis(fijk_fields[field_idx].reshape((-1, fijk_fields[field_idx].shape[-1])), rows[:, np.newaxis], axis=0) * data[:, np.newaxis]
      buffers[remote_proc_idx].append(relevant_data.T)
  return buffers

def sum_into(fijk_field, buffer, rows):
  for k_idx in range(fijk_field.shape[-1]):
    res = fijk_field[:, :, :, k_idx].flatten()
    np.add.at(res, rows, buffer[k_idx, :])
    fijk_field[:, :, :, k_idx] = res.reshape(fijk_field.shape[:-1])
  return fijk_field

def accumulate_fields_jax(fijk_fields, buffers, vert_redundancy_receive):
  for remote_proc_idx in buffers.keys():
    for field_idx in range(len(fijk_fields)):
      (_, rows, _) = vert_redundancy_receive[remote_proc_idx]
      tmp = np.copy(fijk_fields[field_idx][:, :, :, 0])
      fijk_fields[field_idx] = sum_into(fijk_fields[field_idx],
                                        buffers[remote_proc_idx][field_idx],
                                        rows)
  return fijk_fields
