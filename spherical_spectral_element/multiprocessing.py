from .config import np, has_mpi
from .assembly import dss_scalar_for
if has_mpi:
  from mpi4py import MPI
  from .config import mpi_comm


def dss_scalar_for_pack(f, grid, *args):
  gll_weights = grid["gll_weights"]
  metdet = grid["met_det"]
  workspace = f * metdet * (gll_weights[np.newaxis, :, np.newaxis] * gll_weights[np.newaxis, np.newaxis, :])
  buffers = extract_fields_for([workspace.reshape((*f.shape, 1))], grid["vert_redundancy_send"])
  return buffers


def dss_scalar_for_unpack(f, buffers, grid, *args):
  return accumulate_fields_for([f.reshape((*f.shape, 1))], buffers, grid["vert_redundancy_receive"])[:, :, :, 0]


def dss_scalar_for_stub(fs_grids):
  # This is primarily for testing!
  # do not use in model code!
  buffers = []
  for f, grid in fs_grids:
    buffers.append(dss_scalar_for_pack(f, grid))
  buffers = exchange_buffers_stub(buffers)
  fs_out = []
  for ((f, grid), buffer) in zip(fs_grids, buffers):
    fs_out.append(dss_scalar_for(dss_scalar_for_unpack(f, buffer, grid), grid))
  return fs_out


def dss_scalar_for_mpi(f, grid):
  # This is primarily for testing!
  # do not use in model code!
  buffer = dss_scalar_for_pack(f, grid)
  buffer = exchange_buffers_mpi(buffer)
  f = dss_scalar_for(dss_scalar_for_unpack(f, buffer, grid), grid)
  return f


def extract_fields_for(fijk_fields, vert_redundancy_send):
  buffers = {}
  for remote_proc_idx in vert_redundancy_send.keys():
    data = []
    for field_idx in range(len(fijk_fields)):
      for (source_local_idx, source_i, source_j) in vert_redundancy_send:
        data.append(fijk_fields[field_idx][source_local_idx, source_i, source_j, :])
    buffers[remote_proc_idx] = np.stack(data, axis=-1)


def accumulate_fields_for(fijk_fields, buffers, vert_redundancy_receive):
  # designed for device code to be tested against, but this is much more transparent
  for remote_proc_idx in buffers.keys():
    col_idx = 0
    for field_idx in range(len(fijk_fields)):
      for (target_local_idx, target_i, target_j) in vert_redundancy_receive:
          fijk_fields[field_idx][target_local_idx, target_i, target_j, :] += buffers[remote_proc_idx][col_idx]
          col_idx += 1
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
    reqs.append(mpi_comm.Isendrecv_replace(buffer[source_proc_idx], source_proc_idx))
  MPI.Request.Waitall(reqs)
  return buffer


def extract_fields_matrix():
  pass


def extract_fields_jax():
  pass
