from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.distributed_memory.multiprocessing import (project_scalar_for_stub,
                                                       assemble_scalar_for_pack, assemble_scalar_for_unpack,
                                                       exchange_buffers_stub, extract_fields_triple, extract_fields_for,
                                                       accumulate_fields_for, accumulate_fields_triple,
                                                       project_scalar_for_mpi, project_scalar_triple_stub,
                                                       project_scalar_triple_mpi)
from pysces.mesh_generation.mesh import vert_red_flat_to_hierarchy
from pysces.operations_2d.se_grid import create_spectral_element_grid
from pysces.mesh_generation.periodic_plane import create_uniform_grid
from pysces.distributed_memory.processor_decomposition import get_decomp, elem_idx_global_to_proc_idx, global_to_local
from pysces.config import device_unwrapper, np, jnp, use_wrapper, device_wrapper, mpi_size, mpi_rank
from ..handmade_grids import init_test_grid, vert_redundancy_gll
from ..context import test_npts
from ..context import seed as global_seed


def test_unordered_assembly_for_stub():
  for npt in test_npts:
    for nx in range(1, 3):
      for nproc in range(1, 6):
        print(f"dividing nx {nx} grid among {nproc} processors")
        grid_total, dim_total = create_quasi_uniform_grid(nx, npt, wrapped=False)
        vert_redundancy = vert_red_flat_to_hierarchy(grid_total["vert_redundancy"])
        decomp = get_decomp(dim_total["num_elem"], nproc)
        grids = []
        dims = []
        fs = []
        total_elems = 0
        pairs = {}
        proc_ids = elem_idx_global_to_proc_idx(np.arange(dim_total["num_elem"]), decomp)
        for target_face_idx in vert_redundancy.keys():
          for (target_i, target_j) in vert_redundancy[target_face_idx].keys():
            pair_key = (proc_ids[target_face_idx],
                        global_to_local(target_face_idx, proc_ids[target_face_idx], decomp),
                        target_i,
                        target_j)
            pair_set = set()
            for (source_face_idx, source_i, source_j) in vert_redundancy[target_face_idx][(target_i, target_j)]:
              pair_set.add((proc_ids[source_face_idx],
                            global_to_local(source_face_idx, proc_ids[source_face_idx], decomp),
                            source_i,
                            source_j))
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
                                                   proc_idx, decomp, wrapped=False)
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
          fs_out = project_scalar_for_stub(fs, grids)
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
        fs_out = project_scalar_for_stub(fs, grids)
        for (f, f_new) in zip(fs_ref, fs_out):
          for (f_pair, f_new_pair) in zip(f, f_new):
            assert (np.allclose(f_pair, f_new_pair))


def test_unordered_assembly_triple_stub():
  for npt in test_npts:
    for nx in range(1, 3):
      for nproc in range(1, 6):
        print(f"dividing nx {nx} grid among {nproc} processors")
        grid_total, dim_total = create_quasi_uniform_grid(nx, npt, wrapped=False)
        decomp = get_decomp(dim_total["num_elem"], nproc)
        grids = []
        dims = []
        fs = []
        total_elems = 0
        pairs = {}
        proc_ids = elem_idx_global_to_proc_idx(np.arange(dim_total["num_elem"]), decomp)
        vert_redundancy = vert_red_flat_to_hierarchy(grid_total["vert_redundancy"])
        for target_face_idx in vert_redundancy.keys():
          for (target_i, target_j) in vert_redundancy[target_face_idx].keys():
            pair_key = (proc_ids[target_face_idx],
                        global_to_local(target_face_idx, proc_ids[target_face_idx], decomp),
                        target_i,
                        target_j)
            pair_set = set()
            for (source_face_idx, source_i, source_j) in vert_redundancy[target_face_idx][(target_i, target_j)]:
              pair_set.add((proc_ids[source_face_idx],
                            global_to_local(source_face_idx, proc_ids[source_face_idx], decomp),
                            source_i,
                            source_j))
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
                                                   proc_idx, decomp, wrapped=use_wrapper)
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
          for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
            fs[source_proc_idx][0] = device_wrapper(fs[source_proc_idx][0])
          fs_ref = [[jnp.copy(f[0])] for f in fs]
          fs_out = project_scalar_triple_stub(fs, grids, dims)
          for (f, f_new) in zip(fs_ref, fs_out):
            assert (np.allclose(device_unwrapper(f[0]), device_unwrapper(f_new[0])))

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
            fs_ref[proc_idx].append(device_wrapper(fs_tmp[proc_idx][0]))
            fs[proc_idx].append(device_wrapper(fs_tmp[proc_idx][0]))
        fs_out = project_scalar_triple_stub(fs, grids, dims)
        for (f, f_new) in zip(fs_ref, fs_out):
          for (f_pair, f_new_pair) in zip(f, f_new):
            assert (np.allclose(device_unwrapper(f_pair), device_unwrapper(f_new_pair)))


def test_stub_exchange():
  NELEM = 2
  nproc = 2
  se_grid, total_dims = init_test_grid()
  npt = total_dims["npt"]

  decomp = get_decomp(NELEM, nproc)
  proc_ids = elem_idx_global_to_proc_idx(np.array([0, 1]), decomp)
  # set associated vertices to 1,
  # then test that extract, communicate, accumulate returns multiplicity
  pairs = {}
  for target_face_idx in vert_redundancy_gll.keys():
    for (target_i, target_j) in vert_redundancy_gll[target_face_idx].keys():
      pair_key = (proc_ids[target_face_idx],
                  global_to_local(target_face_idx, proc_ids[target_face_idx], decomp),
                  target_i,
                  target_j)
      pair_set = set()
      for (source_face_idx, source_i, source_j) in vert_redundancy_gll[target_face_idx][(target_i, target_j)]:
        pair_set.add((proc_ids[source_face_idx],
                      global_to_local(source_face_idx, proc_ids[source_face_idx], decomp),
                      source_i,
                      source_j))
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
                                             wrapped=False)
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
    fs_out_cont = project_scalar_for_stub(fs, grids)
    buffers = []
    for (f, grid) in zip(fs, grids):
      buffers.append(assemble_scalar_for_pack(f, grid))
    buffers = exchange_buffers_stub(buffers)
    fs_out = []
    for (f, grid, buffer) in zip(fs, grids, buffers):
      fs_out.append(assemble_scalar_for_unpack(f, buffer, grid))
    if (source_i, source_j) in {(0, 0),
                                (npt - 1, npt - 1),
                                (0, npt - 1),
                                (npt - 1, 0)}:
      mult = 3.0
    elif source_j == 0 or source_j == npt - 1:
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
    for (f, f_new) in zip(fs_ref, fs_out_cont):
     assert (np.allclose(f, f_new))


def test_extract_fields_triples():
  for npt in test_npts:
    np.random.seed(global_seed)
    for nx in range(1, 3):
      global_grids_nowrapper = [create_uniform_grid(nx, nx + 1, npt, wrapped=False),
                                create_quasi_uniform_grid(nx, npt, wrapped=False)]
      global_grids = [create_uniform_grid(nx, nx + 1, npt, wrapped=use_wrapper),
                      create_quasi_uniform_grid(nx, npt, wrapped=use_wrapper)]
      for ((grid_total, dim_total),
           (grid_total_nowrapper, _)) in zip(global_grids, global_grids_nowrapper):
        for random, num_iters in zip([False, True], [1, 10]):
          for _ in range(num_iters):
            nproc = 2
            decomp = get_decomp(dim_total["num_elem"], nproc)
            grids = []
            grids_nodevice = []
            dims = []
            fs = []
            total_elems = 0
            nlev = 2
            pairs = {}
            proc_ids = elem_idx_global_to_proc_idx(np.arange(dim_total["num_elem"]), decomp)
            vert_redundancy = vert_red_flat_to_hierarchy(grid_total_nowrapper["vert_redundancy"])
            for target_face_idx in vert_redundancy.keys():
              for (target_i, target_j) in vert_redundancy[target_face_idx].keys():
                pair_key = (proc_ids[target_face_idx],
                            global_to_local(target_face_idx, proc_ids[target_face_idx], decomp),
                            target_i,
                            target_j)
                pair_set = set()
                for (source_face_idx, source_i, source_j) in vert_redundancy[target_face_idx][(target_i, target_j)]:
                  pair_set.add((proc_ids[source_face_idx],
                                global_to_local(source_face_idx, proc_ids[source_face_idx], decomp),
                                source_i,
                                source_j))
                pairs[pair_key] = pair_set.copy()
            for proc_idx in range(nproc):
              grid, dim = create_spectral_element_grid(grid_total_nowrapper["physical_coords"],
                                                       grid_total_nowrapper["jacobian"],
                                                       grid_total_nowrapper["jacobian_inv"],
                                                       grid_total_nowrapper["recip_met_det"],
                                                       grid_total_nowrapper["met_det"],
                                                       grid_total_nowrapper["mass_mat"],
                                                       grid_total_nowrapper["mass_matrix_inv"],
                                                       grid_total_nowrapper["vert_redundancy"],
                                                       proc_idx, decomp, wrapped=use_wrapper)
              grid_nodevice, _ = create_spectral_element_grid(grid_total_nowrapper["physical_coords"],
                                                              grid_total_nowrapper["jacobian"],
                                                              grid_total_nowrapper["jacobian_inv"],
                                                              grid_total_nowrapper["recip_met_det"],
                                                              grid_total_nowrapper["met_det"],
                                                              grid_total_nowrapper["mass_mat"],
                                                              grid_total_nowrapper["mass_matrix_inv"],
                                                              grid_total_nowrapper["vert_redundancy"],
                                                              proc_idx, decomp, wrapped=False)
              grids.append(grid)
              grids_nodevice.append(grid_nodevice)
              dims.append(dim)
              if random:
                fs.append([np.random.normal(size=(*grid_nodevice["physical_coords"][:, :, :, 0].shape, nlev)),
                           np.random.normal(size=(*grid_nodevice["physical_coords"][:, :, :, 0].shape, nlev + 1))])
              else:
                f1 = np.arange(1, grid_nodevice["physical_coords"][:, :, :, 0].size + 1)
                f1 = f1.reshape(grid_nodevice["physical_coords"][:, :, :, 0].shape)[:, :, :, np.newaxis]
                fs.append([f1 * np.ones((1, 1, 1, nlev)),
                           f1 * np.ones((1, 1, 1, nlev + 1))])
              total_elems += dim["num_elem"]
            fs_device = [[device_wrapper(f) for f in fs_local] for fs_local in fs]
            buffers_for = []
            buffers_device = []
            for (f, f_device, grid, grid_nodevice, dim) in zip(fs, fs_device, grids, grids_nodevice, dims):
              buffers_for.append(extract_fields_for(f, grid_nodevice["vert_redundancy_send"]))
              buffer_for = buffers_for[-1]
              buffers_device.append(extract_fields_triple(f_device, grid["triples_send"]))
              buffer_device = buffers_device[-1]
              for proc_idx in buffer_for.keys():
                assert proc_idx in buffer_device.keys()
                assert len(buffer_for[proc_idx]) == len(buffer_device[proc_idx])
                for k_idx in range(len(buffer_for[proc_idx])):
                  assert buffer_for[proc_idx][k_idx].shape == buffer_device[proc_idx][k_idx].shape
                  assert np.allclose(buffer_for[proc_idx][k_idx], device_unwrapper(buffer_device[proc_idx][k_idx]))
            buffers_for = exchange_buffers_stub(buffers_for)
            buffers_device = exchange_buffers_stub(buffers_device)
            for (f, f_device, grid, grid_nowrapper, dim, buffer_for, buffer_device) in zip(fs,
                                                                                           fs_device,
                                                                                           grids,
                                                                                           grids_nodevice,
                                                                                           dims,
                                                                                           buffers_for,
                                                                                           buffers_device):
              for remote_idx in buffer_for.keys():
                for field_idx in range(len(buffer_for[remote_idx])):
                  assert np.allclose(buffer_for[remote_idx][field_idx],
                                     device_unwrapper(buffer_device[remote_idx][field_idx]))
                  assert np.allclose(f[field_idx],
                                     device_unwrapper(f_device[field_idx]))
              fijk_fields_for = accumulate_fields_for(f, buffer_for, grid_nowrapper["vert_redundancy_receive"])
              fijk_fields_triple = accumulate_fields_triple(f_device, buffer_device, grid["triples_receive"], dim)
              assert len(fijk_fields_for) == len(fijk_fields_triple)
              for field_idx in range(len(fijk_fields_for)):
                assert np.allclose(f[field_idx], device_unwrapper(f_device[field_idx]))
                assert np.allclose(fijk_fields_for[field_idx], device_unwrapper(fijk_fields_triple[field_idx]))


def test_mpi_exchange_for():
  for npt in test_npts:
    for nx in range(1, 3):
      nproc = mpi_size
      local_proc_idx = mpi_rank
      print(f"dividing nx {nx} grid among {nproc} processors")
      grid_total, dim_total = create_quasi_uniform_grid(nx, npt, wrapped=False)
      decomp = get_decomp(dim_total["num_elem"], nproc)
      grids = []
      dims = []
      fs = []
      total_elems = 0
      pairs = {}
      proc_ids = elem_idx_global_to_proc_idx(np.arange(dim_total["num_elem"]), decomp)
      vert_redundancy = vert_red_flat_to_hierarchy(grid_total["vert_redundancy"])
      for target_face_idx in vert_redundancy.keys():
        for (target_i, target_j) in vert_redundancy[target_face_idx].keys():
          pair_key = (proc_ids[target_face_idx],
                      global_to_local(target_face_idx, proc_ids[target_face_idx], decomp),
                      target_i,
                      target_j)
          pair_set = set()
          for (source_face_idx, source_i, source_j) in vert_redundancy[target_face_idx][(target_i, target_j)]:
            pair_set.add((proc_ids[source_face_idx],
                          global_to_local(source_face_idx, proc_ids[source_face_idx], decomp),
                          source_i,
                          source_j))
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
                                                 proc_idx, decomp, wrapped=False)
        grids.append(grid)
        dims.append(dim)
        total_elems += dim["num_elem"]
      assert (dim_total["num_elem"] == total_elems)
      grid_local = grids[local_proc_idx]

      def zeros_f():
        fs = []
        for grid in grids:
          fs.append([np.zeros_like(grid["physical_coords"][:, :, :, 0])])
        return fs

      def rand_f(seed=global_seed):
        fs = []
        for grid_idx, grid in enumerate(grids):
          np.random.seed(grid_idx + seed)
          fs.append([np.random.uniform(grid["physical_coords"][:, :, :, 0])])
        return fs

      # test one scalar at a time
      for pair_key in pairs.keys():
        fs = zeros_f()
        target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
        fs[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
        for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
          fs[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
        fs_ref = [[np.copy(f[0])] for f in fs]
        fs_out = project_scalar_for_stub(fs, grids)
        f_out = project_scalar_for_mpi(fs[local_proc_idx], grid_local)
        assert (np.allclose(fs_ref[local_proc_idx][0], fs_out[local_proc_idx][0]))
        assert (np.allclose(fs_ref[local_proc_idx][0], f_out))

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
      fs_out = project_scalar_for_stub(fs, grids)
      f_out = project_scalar_for_mpi(fs[local_proc_idx], grid_local)
      for (f_pair, f_new_pair) in zip(fs_out[local_proc_idx], f_out):
        assert (np.allclose(f_pair, f_new_pair))
      num_fields = 20
      fs_rand = [[] for _ in range(nproc)]
      for seed in range(num_fields):
        f_rand = rand_f()
        for proc_idx in range(nproc):
          fs_rand[proc_idx].append(np.copy(f_rand[proc_idx][0]))
      fs_stub_out = project_scalar_for_stub(fs_rand, grids)
      f_out = project_scalar_for_mpi(fs_rand[local_proc_idx], grid_local)
      for (f_stub, f_mpi) in zip(fs_stub_out[local_proc_idx], f_out):
        assert (np.allclose(f_stub, f_mpi))


def test_mpi_exchange_triple():
  for npt in test_npts:
    for nx in range(1, 3):
      nproc = mpi_size
      local_proc_idx = mpi_rank
      print(f"dividing nx {nx} grid among {nproc} processors")
      grid_total_nowrapper, dim_total_nowrapper = create_quasi_uniform_grid(nx, npt, wrapped=False)
      grid_total, dim_total = create_quasi_uniform_grid(nx, npt, wrapped=use_wrapper)
      decomp = get_decomp(dim_total["num_elem"], nproc)
      grids = []
      dims = []
      fs = []
      total_elems = 0
      pairs = {}
      proc_ids = elem_idx_global_to_proc_idx(np.arange(dim_total["num_elem"]), decomp)
      vert_redundancy = vert_red_flat_to_hierarchy(grid_total_nowrapper["vert_redundancy"])
      for target_face_idx in vert_redundancy.keys():
        for (target_i, target_j) in vert_redundancy[target_face_idx].keys():
          pair_key = (proc_ids[target_face_idx],
                      global_to_local(target_face_idx, proc_ids[target_face_idx], decomp),
                      target_i,
                      target_j)
          pair_set = set()
          for (source_face_idx, source_i, source_j) in vert_redundancy[target_face_idx][(target_i, target_j)]:
            pair_set.add((proc_ids[source_face_idx],
                          global_to_local(source_face_idx, proc_ids[source_face_idx], decomp),
                          source_i,
                          source_j))
          pairs[pair_key] = pair_set.copy()
      for proc_idx in range(nproc):
        grid, dim = create_spectral_element_grid(grid_total_nowrapper["physical_coords"],
                                                 grid_total_nowrapper["jacobian"],
                                                 grid_total_nowrapper["jacobian_inv"],
                                                 grid_total_nowrapper["recip_met_det"],
                                                 grid_total_nowrapper["met_det"],
                                                 grid_total_nowrapper["mass_mat"],
                                                 grid_total_nowrapper["mass_matrix_inv"],
                                                 grid_total_nowrapper["vert_redundancy"],
                                                 proc_idx, decomp, wrapped=use_wrapper)
        grids.append(grid)
        dims.append(dim)
        total_elems += dim["num_elem"]
      assert (dim_total["num_elem"] == total_elems)
      grid_local = grids[local_proc_idx]
      dim_local = dims[local_proc_idx]

      def zeros_f():
        fs = []
        for grid in grids:
          fs.append([np.zeros_like(grid["physical_coords"][:, :, :, 0])])
        return fs

      def rand_f(seed=0):
        fs = []
        for grid_idx, grid in enumerate(grids):
          np.random.seed(grid_idx + seed)
          fs.append([np.random.uniform(grid["physical_coords"][:, :, :, 0])])
        return fs

      # test one scalar at a time
      for pair_key in pairs.keys():
        fs = zeros_f()
        target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
        fs[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
        for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
          fs[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
        fs_ref = [[np.copy(f[0])] for f in fs]
        fs_out = project_scalar_triple_stub(fs, grids, dims)
        f_out = project_scalar_triple_mpi(fs[local_proc_idx], grid_local, dim_local)
        assert (np.allclose(fs_ref[local_proc_idx][0], fs_out[local_proc_idx][0]))
        assert (np.allclose(fs_ref[local_proc_idx][0], f_out))

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
      fs_out = project_scalar_triple_stub(fs, grids, dims)
      f_out = project_scalar_triple_mpi(fs[local_proc_idx], grid_local, dim_local)
      for (f_pair, f_new_pair) in zip(fs_out[local_proc_idx], f_out):
        assert (np.allclose(f_pair, f_new_pair))
      num_fields = 20
      fs_rand = [[] for _ in range(nproc)]
      for seed in range(num_fields):
        f_rand = rand_f()
        for proc_idx in range(nproc):
          fs_rand[proc_idx].append(np.copy(f_rand[proc_idx][0]))
      fs_stub_out = project_scalar_triple_stub(fs_rand, grids, dims)
      f_out = project_scalar_triple_mpi(fs_rand[local_proc_idx], grid_local, dim_local)
      for (f_stub, f_mpi) in zip(fs_stub_out[local_proc_idx], f_out):
        assert (np.allclose(f_stub, f_mpi))
