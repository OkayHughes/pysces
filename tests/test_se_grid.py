from pysces.config import use_wrapper, npt, np
from pysces.equiangular_metric import gen_metric_from_topo
from pysces.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from pysces.processor_decomposition import get_decomp, local_to_global
from pysces.se_grid import triage_vert_redundancy


def test_vert_triage_artificial():
  NELEM = 2
  # (1, 3, 3)||(1, 0, 3)(1, 1, 3)(1, 2, 3)(1, 3, 3)||(1, 0, 3)
  # =========||====================================||=========
  # (0, 3, 0)||(0, 0, 0)(0, 1, 0)(0, 2, 0)(0, 3, 0)||(0, 0, 0)
  # (0, 3, 1)||(0, 0, 1)(0, 1, 1)(0, 2, 1)(0, 3, 1)||(0, 0, 1)
  # (0, 3, 2)||(0, 0, 2)(0, 1, 2)(0, 2, 2)(0, 3, 2)||(0, 0, 2)
  # (0, 3, 3)||(0, 0, 3)(0, 1, 3)(0, 2, 3)(0, 3, 3)||(0, 0, 3)
  # =========||====================================||=========
  # (1, 3, 0)||(1, 0, 0)(1, 1, 0)(1, 2, 0)(1, 3, 0)||(1, 0, 0)
  # (1, 3, 1)||(1, 0, 1)(1, 1, 1)(1, 2, 1)(1, 3, 1)||(1, 0, 1)
  # (1, 3, 2)||(1, 0, 2)(1, 1, 2)(1, 2, 2)(1, 3, 2)||(1, 0, 2)
  # (1, 3, 3)||(1, 0, 3)(1, 1, 3)(1, 2, 3)(1, 3, 3)||(1, 0, 3)
  # =========||====================================||=========
  # (0, 3, 0)||(0, 0, 0)(0, 1, 0)(0, 2, 0)(0, 3, 0)||(0, 0, 0)
  # do not include self-associations
  vert_redundancy_gll = {0: {(0, 0): {(1, 0, 3),
                                      (1, 3, 3),
                                      (0, 3, 0)},
                             (3, 0): {(1, 3, 3),
                                      (1, 0, 3),
                                      (0, 0, 0)},
                             (3, 3): {(0, 0, 3),
                                      (1, 0, 0),
                                      (1, 3, 0)},
                             (0, 3): {(0, 3, 3),
                                      (1, 3, 0),
                                      (1, 0, 0)},
                             (1, 0): {(1, 1, 3)},
                             (2, 0): {(1, 2, 3)},
                             (3, 1): {(0, 0, 1)},
                             (3, 2): {(0, 0, 2)},
                             (2, 3): {(1, 2, 0)},
                             (1, 3): {(1, 1, 0)},
                             (0, 1): {(0, 3, 1)},
                             (0, 2): {(0, 3, 2)}},
                         1: {(0, 0): {(0, 0, 3),
                                      (0, 3, 3),
                                      (1, 3, 0)},
                             (3, 0): {(1, 0, 0),
                                      (0, 0, 3),
                                      (0, 3, 3)},
                             (3, 3): {(1, 0, 3),
                                      (0, 0, 0),
                                      (0, 3, 0)},
                             (0, 3): {(1, 3, 3),
                                      (0, 3, 0),
                                      (0, 0, 0)},
                             (1, 0): {(0, 1, 3)},
                             (2, 0): {(0, 2, 3)},
                             (3, 1): {(1, 0, 1)},
                             (3, 2): {(1, 0, 2)},
                             (2, 3): {(0, 2, 0)},
                             (1, 3): {(0, 1, 0)},
                             (0, 2): {(1, 3, 2)},
                             (0, 1): {(1, 3, 1)}}}
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
  # note: outermost keys are global face ids
  vert_locals_ref = [{0: {(0, 0): {(0, 3, 0)},
                          (3, 0): {(0, 0, 0)},
                          (3, 3): {(0, 0, 3)},
                          (0, 3): {(0, 3, 3)},
                          (3, 1): {(0, 0, 1)},
                          (3, 2): {(0, 0, 2)},
                          (0, 1): {(0, 3, 1)},
                          (0, 2): {(0, 3, 2)}}},
                     {0: {(0, 0): {(0, 3, 0)},
                          (3, 0): {(0, 0, 0)},
                          (3, 3): {(0, 0, 3)},
                          (0, 3): {(0, 3, 3)},
                          (3, 1): {(0, 0, 1)},
                          (3, 2): {(0, 0, 2)},
                          (0, 1): {(0, 3, 1)},
                          (0, 2): {(0, 3, 2)}}}]
  # note: outermost keys are processor ids
  vert_sends_ref = [{1: [(0, 0, 0),
                         (0, 0, 0),
                         (0, 1, 0),
                         (0, 2, 0),
                         (0, 3, 0),
                         (0, 3, 0),
                         (0, 3, 3),
                         (0, 3, 3),
                         (0, 2, 3),
                         (0, 1, 3),
                         (0, 0, 3),
                         (0, 0, 3)]},
                    {0: [(0, 0, 0),
                         (0, 0, 0),
                         (0, 1, 0),
                         (0, 2, 0),
                         (0, 3, 0),
                         (0, 3, 0),
                         (0, 3, 3),
                         (0, 3, 3),
                         (0, 2, 3),
                         (0, 1, 3),
                         (0, 0, 3),
                         (0, 0, 3)]}]
  vert_recvs_ref = [{1: [(0, 3, 3),
                         (0, 0, 3),
                         (0, 1, 3),
                         (0, 2, 3),
                         (0, 3, 3),
                         (0, 0, 3),
                         (0, 0, 0),
                         (0, 3, 0),
                         (0, 2, 0),
                         (0, 1, 0),
                         (0, 0, 0),
                         (0, 3, 0)]},
                    {0: [(0, 3, 3),
                         (0, 0, 3),
                         (0, 1, 3),
                         (0, 2, 3),
                         (0, 3, 3),
                         (0, 0, 3),
                         (0, 0, 0),
                         (0, 3, 0),
                         (0, 2, 0),
                         (0, 1, 0),
                         (0, 0, 0),
                         (0, 3, 0)]}]

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

  for proc_idx in range(nproc):
    for remote_proc_idx in vert_sends[proc_idx].keys():
      for k in range(len(vert_sends[proc_idx][remote_proc_idx])):
        send_face_idx, send_i, send_j = vert_sends_ref[proc_idx][remote_proc_idx][k]
        recv_face_idx, recv_i, recv_j = vert_recvs_ref[remote_proc_idx][proc_idx][k]
        send_global_idx = local_to_global(send_face_idx, proc_idx, decomp)
        recv_global_idx = local_to_global(recv_face_idx, remote_proc_idx, decomp)
        assert (recv_global_idx, recv_i, recv_j) in vert_redundancy_gll[send_global_idx][(send_i, send_j)]
        assert (send_global_idx, send_i, send_j) in vert_redundancy_gll[recv_global_idx][(recv_i, recv_j)]

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
  for nproc in range(1, 10):
    nx = 5
    face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
    vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
    grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, jax=False)
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
