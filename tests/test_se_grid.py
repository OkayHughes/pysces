from pysces.equiangular_metric import gen_metric_from_topo, create_quasi_uniform_grid
from pysces.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from pysces.processor_decomposition import get_decomp, local_to_global
from pysces.se_grid import triage_vert_redundancy, subset_var, init_dss_global, init_dss_matrix_local, create_spectral_element_grid
from .handmade_grids import vert_locals_ref, vert_recvs_ref, vert_sends_ref, vert_redundancy_gll, init_test_grid
from pysces.config import np
from .context import test_npts


def test_vert_triage_artificial():
  NELEM = 2

  # check symmetry first
  nproc = 2
  decomp = get_decomp(NELEM, nproc)
  vert_locals = []
  vert_sends = []
  vert_recvs = []
  for proc_idx in range(nproc):
    vert_red_local, vert_red_send, vert_red_receive = triage_vert_redundancy(vert_redundancy_gll, proc_idx, decomp)
    vert_locals.append(vert_red_local)
    vert_sends.append(vert_red_send)
    vert_recvs.append(vert_red_receive)

  # test pairwise first
  pairs_ref = {}
  for proc_idx in range(nproc):
    for target_face_idx in vert_locals_ref[proc_idx].keys():
      assert(target_face_idx in vert_locals[proc_idx].keys())
      for (target_i, target_j) in vert_locals_ref[proc_idx][target_face_idx].keys():
        assert (target_i, target_j) in vert_locals[proc_idx][target_face_idx].keys()
        for (source_face_idx, source_i, source_j) in vert_locals_ref[proc_idx][target_face_idx][(target_i, target_j)]:
          assert (source_face_idx, source_i, source_j) in vert_locals[proc_idx][target_face_idx][(target_i, target_j)]
          pair = ((target_face_idx, target_i, target_j),
                  (source_face_idx, source_i, source_j))
          if pair not in pairs_ref.keys():
            pairs_ref[pair] = 1
          else:
            pairs_ref[pair] += 1

  for proc_idx in range(nproc):
    for remote_proc_idx in vert_sends[proc_idx].keys():
      for (source_face_idx, source_i, source_j) in vert_sends[proc_idx][remote_proc_idx]:
        assert (source_face_idx, source_i, source_j) in vert_sends_ref[proc_idx][remote_proc_idx]
    for remote_proc_idx in vert_recvs[proc_idx].keys():
      for (target_face_idx, target_i, target_j) in vert_sends[proc_idx][remote_proc_idx]:
        assert (target_face_idx, target_i, target_j) in vert_recvs_ref[proc_idx][remote_proc_idx]

  # test handmade reference soln
  for proc_idx in range(nproc):
    for remote_proc_idx in vert_sends[proc_idx].keys():
      for k in range(len(vert_sends[proc_idx][remote_proc_idx])):
        send_face_idx, send_i, send_j = vert_sends_ref[proc_idx][remote_proc_idx][k]
        recv_face_idx, recv_i, recv_j = vert_recvs_ref[remote_proc_idx][proc_idx][k]
        send_global_idx = local_to_global(send_face_idx, proc_idx, decomp)
        recv_global_idx = local_to_global(recv_face_idx, remote_proc_idx, decomp)
        assert (recv_global_idx, recv_i, recv_j) in vert_redundancy_gll[send_global_idx][(send_i, send_j)]
        assert (send_global_idx, send_i, send_j) in vert_redundancy_gll[recv_global_idx][(recv_i, recv_j)]

  # test generated vert_redundancy structs
  for proc_idx in range(nproc):
    for remote_proc_idx in vert_sends[proc_idx].keys():
      for k in range(len(vert_sends[proc_idx][remote_proc_idx])):
        send_face_idx, send_i, send_j = vert_sends[proc_idx][remote_proc_idx][k]
        recv_face_idx, recv_i, recv_j = vert_recvs[remote_proc_idx][proc_idx][k]
        send_global_idx = local_to_global(send_face_idx, proc_idx, decomp)
        recv_global_idx = local_to_global(recv_face_idx, remote_proc_idx, decomp)
        assert (recv_global_idx, recv_i, recv_j) in vert_redundancy_gll[send_global_idx][(send_i, send_j)]
        assert (send_global_idx, send_i, send_j) in vert_redundancy_gll[recv_global_idx][(recv_i, recv_j)]


def test_vert_red_triage():
  for npt in test_npts:
    for nproc in range(1, 6):
      for nx in range(2, 5):
        face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
        vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
        grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, npt, jax=False)
        decomp = get_decomp(dims["num_elem"], nproc)
        vert_redundancy_gll = grid["vert_redundancy"]
        vert_redundancy_check = {}
        for face_idx in vert_redundancy_gll.keys():
          for (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
            for remote_face_idx, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
              pair = (face_idx, i_idx, j_idx)
              if pair not in vert_redundancy_check.keys():
                vert_redundancy_check[pair] = 1
              else:
                vert_redundancy_check[pair] += 1

        vert_locals = []
        vert_sends = []
        vert_recvs = []
        for proc_idx in range(nproc):
          vert_red_local, vert_red_send, vert_red_receive = triage_vert_redundancy(vert_redundancy_gll, proc_idx, decomp)
          vert_locals.append(vert_red_local)
          vert_sends.append(vert_red_send)
          vert_recvs.append(vert_red_receive)
        # test that all redundancies are accounted for
        vert_red_decomp = {}
        for proc_idx in range(nproc):
          for face_idx in vert_locals[proc_idx].keys():
            for (i_idx, j_idx) in vert_locals[proc_idx][face_idx].keys():
              for remote_face_idx, remote_i, remote_j in vert_locals[proc_idx][face_idx][(i_idx, j_idx)]:
                pair = (local_to_global(face_idx, proc_idx, decomp), i_idx, j_idx)
                if pair not in vert_red_decomp.keys():
                  vert_red_decomp[pair] = 1
                else:
                  vert_red_decomp[pair] += 1
          for remote_proc_idx in vert_recvs[proc_idx].keys():
            for k in range(len(vert_recvs[proc_idx][remote_proc_idx])):
              local_face_idx, local_i, local_j = vert_recvs[proc_idx][remote_proc_idx][k]
              pair = (local_to_global(local_face_idx, proc_idx, decomp), local_i, local_j)
              if pair not in vert_red_decomp.keys():
                vert_red_decomp[pair] = 1
              else:
                vert_red_decomp[pair] += 1
        for key in vert_redundancy_check.keys():
          assert key in vert_red_decomp.keys()
          assert vert_redundancy_check[key] == vert_red_decomp[key]
        # test that ordering of send/receive buffers agree
        # using vert_redundancy_gll as ground truth.
        for proc_idx in range(nproc):
          for remote_proc_idx in vert_sends[proc_idx].keys():
            for k in range(len(vert_sends[proc_idx][remote_proc_idx])):
              send_face_idx, send_i, send_j = vert_sends[proc_idx][remote_proc_idx][k]
              recv_face_idx, recv_i, recv_j = vert_recvs[remote_proc_idx][proc_idx][k]
              send_global_idx = local_to_global(send_face_idx, proc_idx, decomp)
              recv_global_idx = local_to_global(recv_face_idx, remote_proc_idx, decomp)
              assert (recv_global_idx, recv_i, recv_j) in vert_redundancy_gll[send_global_idx][(send_i, send_j)]
              assert (send_global_idx, send_i, send_j) in vert_redundancy_gll[recv_global_idx][(recv_i, recv_j)]


def test_dss_init():
  se_grid, dims = init_test_grid()
  npt = dims["npt"]
  assert np.allclose(np.sum(se_grid["met_det"] *
                             (se_grid["gll_weights"][np.newaxis, :, np.newaxis] *
                              se_grid["gll_weights"][np.newaxis, np.newaxis, :])), 8.0)
  NELEM = 2
  nproc = 2
  decomp = get_decomp(2, 2)
  
  def flip_triple(triple):
    return [x for x in zip(triple[0],
                           triple[1],
                           triple[2])]
  vert_red_total = flip_triple(se_grid["dss_triple"])
  ct = len(vert_red_total)
  for proc_idx in range(nproc):
    vert_red_local, vert_red_send, vert_red_recv = triage_vert_redundancy(se_grid["vert_redundancy"], proc_idx, decomp)
    metdet = subset_var(se_grid["met_det"], proc_idx, decomp)
    _, dss_triple = init_dss_matrix_local(metdet.shape[0], npt, vert_red_local)
    ct -= len(flip_triple(dss_triple))
    triples_send, triples_receive = init_dss_global(metdet.shape[0], npt, vert_red_send, vert_red_recv)
    for remote_proc_idx in triples_send.keys():
      assert (remote_proc_idx in triples_receive.keys())
      # column refers to order of summation, which may not match.
      flip_trip_send = [x for x in flip_triple(triples_send[remote_proc_idx])]
      flip_double_send = [x[:-1] for x in flip_trip_send]
      ct -= len(flip_trip_send)
      flip_trip_recv = [x for x in flip_triple(triples_receive[remote_proc_idx])]
      flip_double_recv = [x[:-1] for x in flip_trip_send]
      for (value, row, _) in flip_trip_send:
        assert (value, row) in flip_double_recv
      for (value, row, _) in flip_trip_recv:
        assert (value, row) in flip_double_send
      
  assert ct == 0


def test_triples_order():
  for npt in test_npts:
    for nx in range(1, 5):
      for nproc in range(1, 3):
        grid_total, dim_total = create_quasi_uniform_grid(nx, npt, jax=False)
        decomp = get_decomp(dim_total["num_elem"], nproc)
        grids = []
        grids_nojax = []
        dims = []
        pairs = {}
        vert_redundancy = grid_total["vert_redundancy"]
        for proc_idx in range(nproc):
          grid, dim = create_spectral_element_grid(grid_total["physical_coords"],
                                                  grid_total["jacobian"],
                                                  grid_total["jacobian_inv"],
                                                  grid_total["recip_met_det"],
                                                  grid_total["met_det"],
                                                  grid_total["mass_mat"],
                                                  grid_total["mass_matrix_inv"],
                                                  grid_total["vert_redundancy"],
                                                  proc_idx, decomp, jax=True)
          grid_nojax, _ = create_spectral_element_grid(grid_total["physical_coords"],
                                                  grid_total["jacobian"],
                                                  grid_total["jacobian_inv"],
                                                  grid_total["recip_met_det"],
                                                  grid_total["met_det"],
                                                  grid_total["mass_mat"],
                                                  grid_total["mass_matrix_inv"],
                                                  grid_total["vert_redundancy"],
                                                  proc_idx, decomp, jax=False)
          grids.append(grid)
          grids_nojax.append(grid_nojax)
          dims.append(dim)
        for local_proc_idx in range(nproc):
          # check that triples and vert redundancy structs are identical
          local_triples_send = grids[local_proc_idx]["triples_send"]
          local_triples_recv = grids[local_proc_idx]["triples_receive"]
          local_vert_red_send = grids_nojax[local_proc_idx]["vert_redundancy_send"]
          local_vert_red_recv = grids_nojax[local_proc_idx]["vert_redundancy_receive"]
          local_coords = grids[local_proc_idx]["physical_coords"]
          for remote_proc_idx in local_triples_send.keys():
            assert remote_proc_idx in local_triples_recv.keys()
            assert remote_proc_idx in local_vert_red_send.keys()
            assert remote_proc_idx in local_vert_red_recv.keys()
            remote_triples_send = grids[remote_proc_idx]["triples_send"]
            remote_triples_recv = grids[remote_proc_idx]["triples_receive"]
            remote_vert_red_send = grids_nojax[remote_proc_idx]["vert_redundancy_send"]
            remote_vert_red_recv = grids_nojax[remote_proc_idx]["vert_redundancy_receive"]
            grid_remote = grids[remote_proc_idx]
            remote_coords = grids[remote_proc_idx]["physical_coords"] 
            for k_idx in range(local_triples_send[remote_proc_idx][0].size):
              f_idx, i_idx, j_idx = local_vert_red_send[remote_proc_idx][k_idx]
              # test that vert_red_send struct and triples_send point to coincident local points
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 0],
                                local_coords[:, :, :, 0].flatten()[local_triples_send[remote_proc_idx][1][k_idx]]))
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 1],
                                local_coords[:, :, :, 1].flatten()[local_triples_send[remote_proc_idx][1][k_idx]]))
              # test that local vert_red_send struct and remote triples_recv point to coincident points
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 0],
                                remote_coords[:, :, :, 0].flatten()[remote_triples_recv[local_proc_idx][1][k_idx]]))
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 1],
                                remote_coords[:, :, :, 1].flatten()[remote_triples_recv[local_proc_idx][1][k_idx]]))
              f_idx, i_idx, j_idx = local_vert_red_recv[remote_proc_idx][k_idx]
              # test that vert_red_recv struct and triples_recv point to coincident local points
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 0],
                                local_coords[:, :, :, 0].flatten()[local_triples_recv[remote_proc_idx][1][k_idx]]))
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 1],
                                local_coords[:, :, :, 1].flatten()[local_triples_recv[remote_proc_idx][1][k_idx]]))
              # test that local vert_red_recv struct and remote triples_send point to coincident points
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 0],
                                remote_coords[:, :, :, 0].flatten()[remote_triples_send[local_proc_idx][1][k_idx]]))
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 1],
                                remote_coords[:, :, :, 1].flatten()[remote_triples_send[local_proc_idx][1][k_idx]]))