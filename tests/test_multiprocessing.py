from pysces.equiangular_metric import create_quasi_uniform_grid
from pysces.multiprocessing import dss_scalar_for_stub, exchange_buffers_mpi, dss_scalar_for_pack, dss_scalar_for_unpack, exchange_buffers_stub
from pysces.se_grid import create_spectral_element_grid
from pysces.processor_decomposition import get_decomp, elem_idx_global_to_proc_idx, global_to_local
from pysces.config import jnp, device_unwrapper, np, do_mpi_communication, npt, mpi_rank, mpi_size, mpi_comm
from .handmade_grids import init_test_grid, vert_redundancy_gll


def test_unordered_assembly_stub():
  for nx in range(1, 5):
    for nproc in range(1, 8):
      print(f"dividing nx {nx} grid among {nproc} processors")
      grid_total, dim_total = create_quasi_uniform_grid(nx)
      decomp = get_decomp(dim_total["num_elem"], nproc)
      grids = []
      dims = []
      fs = []
      total_elems = 0
      pairs = {}
      proc_ids = elem_idx_global_to_proc_idx(np.arange(dim_total["num_elem"]), decomp)
      vert_redundancy = grid_total["vert_redundancy"]
      for target_face_idx in vert_redundancy.keys():
        for (target_i, target_j) in vert_redundancy[target_face_idx].keys():
          pair_key = (proc_ids[target_face_idx], global_to_local(target_face_idx, proc_ids[target_face_idx], decomp), target_i, target_j)
          pair_set = set()
          for (source_face_idx, source_i, source_j) in vert_redundancy[target_face_idx][(target_i, target_j)]:
            pair_set.add((proc_ids[source_face_idx], global_to_local(source_face_idx, proc_ids[source_face_idx], decomp), source_i, source_j))
          pairs[pair_key] = pair_set.copy()
      for proc_idx in range(nproc):
        grid, dim = create_spectral_element_grid(grid_total["physical_coords"],
                                                grid_total["jacobian"],
                                                grid_total["jacobian_inv"],
                                                grid_total["recip_met_det"],
                                                grid_total["met_det"],
                                                grid_total["mass_mat"],
                                                grid_total["mass_matrix_inv"],
                                                grid_total["vert_redundancy"],
                                                proc_idx, decomp, jax=False)
        grids.append(grid)
        dims.append(dim)
        total_elems += dim["num_elem"]

      assert (dim_total["num_elem"] == total_elems)
      def zeros_f():
        fs = []
        for grid in grids:
          fs.append([np.zeros_like(grid["physical_coords"][:, :, :, 0])])
        return fs
      # test one scalar at a time
      for pair_key in pairs.keys():
        fs = zeros_f()
        target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
        fs[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
        for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
          fs[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
        fs_ref = [[np.copy(f[0])] for f in fs]
        fs_out = dss_scalar_for_stub(fs, grids)
        for (f, f_new) in zip(fs_ref, fs_out):
          assert (np.allclose(f[0], f_new[0]))

      # test all scalars at once
      fs_ref = [[] for _ in range(nproc)]
      fs = [[] for _ in range(nproc)]
      for pair_key in pairs.keys():
        fs_tmp = zeros_f()
        target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
        fs_tmp[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
        for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
          fs_tmp[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
        for proc_idx in range(nproc):
          fs_ref[proc_idx].append(np.copy(fs_tmp[proc_idx][0]))
          fs[proc_idx].append(np.copy(fs_tmp[proc_idx][0]))
      fs_out = dss_scalar_for_stub(fs, grids)
      for (f, f_new) in zip(fs_ref, fs_out):
        for (f_pair, f_new_pair) in zip(f, f_new):
          assert (np.allclose(f_pair, f_new_pair))
  

def test_stub_exchange():
  NELEM=2
  nproc=2
  se_grid, total_dims = init_test_grid()

  decomp = get_decomp(NELEM, nproc)
  proc_ids = elem_idx_global_to_proc_idx(np.array([0, 1]), decomp)
  # set associated vertices to 1,
  # then test that extract, communicate, accumulate returns multiplicity
  pairs = {}
  for target_face_idx in vert_redundancy_gll.keys():
    for (target_i, target_j) in vert_redundancy_gll[target_face_idx].keys():
      pair_key = (proc_ids[target_face_idx], global_to_local(target_face_idx, proc_ids[target_face_idx], decomp), target_i, target_j)
      pair_set = set()
      for (source_face_idx, source_i, source_j) in vert_redundancy_gll[target_face_idx][(target_i, target_j)]:
        pair_set.add((proc_ids[source_face_idx], global_to_local(source_face_idx, proc_ids[source_face_idx], decomp), source_i, source_j))
      pairs[pair_key] = pair_set.copy()
  grids = []
  dims = []

  for proc_idx in range(nproc):
    grid, dim = create_spectral_element_grid(se_grid["physical_coords"],
                                             se_grid["jacobian"],
                                             se_grid["jacobian_inv"],
                                             se_grid["recip_met_det"],
                                             se_grid["met_det"],
                                             se_grid["mass_mat"],
                                             se_grid["mass_matrix_inv"],
                                             se_grid["vert_redundancy"],
                                             proc_idx,
                                             decomp,
                                             jax=False)
    grids.append(grid)
    dims.append(dim)

  def zeros_f():
    fs = []
    for grid in grids:
      fs.append([np.zeros_like(grid["physical_coords"][:, :, :, 0])])
    return fs

  for pair_key in pairs.keys():
    fs = zeros_f()
    target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
    fs[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
    for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
      fs[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
    fs_ref = [np.copy(f) for f in fs]
    fs_out_dss = dss_scalar_for_stub(fs, grids)
    buffers = []
    for (f, grid) in zip(fs, grids):
      buffers.append(dss_scalar_for_pack(f, grid))
    buffers = exchange_buffers_stub(buffers)
    fs_out = []
    for (f, grid, buffer) in zip(fs, grids, buffers):
      fs_out.append(dss_scalar_for_unpack(f, buffer, grid))
    if (source_i, source_j) in {(0, 0),
                                (3, 3),
                                (0, 3),
                                (3, 0)}:
      mult = 3.0
    elif source_j == 0 or source_j == 3:
      mult = 2.0
    else:
      mult = 1.0
    assert np.allclose(fs_out[target_proc_idx][0][target_local_face_idx, target_i, target_j], mult)
    fs_out[target_proc_idx][0][target_local_face_idx, target_i, target_j] -= mult
    for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
      assert np.allclose(fs_out[source_proc_idx][0][source_local_face_idx, source_i, source_j], mult)
      fs_out[source_proc_idx][0][source_local_face_idx, source_i, source_j] -= mult
    for f in fs_out:
      assert(np.allclose(f[0], 0.0))
    for (f, f_new) in zip(fs_ref, fs_out_dss):
     assert (np.allclose(f, f_new))
        


def test_mpi_exchange():
  if not do_mpi_communication:
    return
  