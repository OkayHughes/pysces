from pysces.config import np, device_wrapper, use_wrapper, wrapper_type
from pysces.mesh_generation.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from pysces.mesh_generation.equiangular_metric import gen_metric_from_topo
from pysces.operations_2d.assembly import project_scalar_for, project_scalar_wrapper, project_scalar_sparse, project_scalar
from ..context import test_npts


def test_projection():
  for npt in test_npts:
    for nx in [3, 4]:
      face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
      vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
      grid, dims = gen_metric_from_topo(face_connectivity,
                                        face_mask,
                                        face_position_2d,
                                        vert_redundancy,
                                        npt,
                                        wrapped=use_wrapper)
      grid_nowrapper, _ = gen_metric_from_topo(face_connectivity,
                                               face_mask,
                                               face_position_2d,
                                               vert_redundancy,
                                               npt,
                                               wrapped=False)
      vert_redundancy_gll = grid_nowrapper["vert_redundancy"]
      fn = np.zeros_like(grid["physical_coords"][:, :, :, 0])
      for face_idx in range(grid["physical_coords"].shape[0]):
        for i_idx in range(npt):
          for j_idx in range(npt):
            fn[:] = 0.0
            fn[face_idx, i_idx, j_idx] = 1.0
            if face_idx in vert_redundancy_gll.keys():
              if (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
                for remote_face_id, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
                  fn[remote_face_id, remote_i, remote_j] = 1.0
            assert (np.allclose((project_scalar(device_wrapper(fn), grid, dims)), fn))


def test_projection_equiv():
  for npt in test_npts:
    for nx in [7, 8]:
      face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
      vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
      grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, npt, wrapped=False)
      grid_wrapped, dims_wrapped = gen_metric_from_topo(face_connectivity,
                                                        face_mask,
                                                        face_position_2d,
                                                        vert_redundancy,
                                                        npt,
                                                        wrapped=use_wrapper)
      fn = device_wrapper(np.cos(grid["physical_coords"][:, :, :, 1]) * np.cos(grid["physical_coords"][:, :, :, 0]))
      assert (np.allclose(project_scalar(fn, grid_wrapped, dims), fn))
      ones = np.ones_like(grid["met_det"])
      ones_out = project_scalar(device_wrapper(ones), grid_wrapped, dims)
      assert (np.allclose(np.asarray(ones_out), ones))
      ones_out_for = project_scalar_for(np.asarray(ones), grid)
      assert (np.allclose(ones_out_for, ones))


def test_projection_equiv_rand():
  for npt in test_npts:
    for nx in [7, 8]:
      face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
      vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
      grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, npt, wrapped=False)
      grid_wrapped, dims_wrapped = gen_metric_from_topo(face_connectivity,
                                                        face_mask,
                                                        face_position_2d,
                                                        vert_redundancy,
                                                        npt,
                                                        wrapped=use_wrapper)
      for _ in range(20):
        fn_rand = np.random.uniform(size=grid["physical_coords"][:, :, :, 1].shape)
        if wrapper_type == "jax":
          assert (np.allclose(np.asarray(project_scalar_wrapper(device_wrapper(fn_rand), grid_wrapped, dims_wrapped)), project_scalar_for(fn_rand, grid)))

        assert (np.allclose(project_scalar_sparse(device_wrapper(fn_rand), grid), project_scalar_for(fn_rand, grid)))
