from spherical_spectral_element.config import np, npt, device_wrapper, use_wrapper, wrapper_type
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.equiangular_metric import gen_metric_from_topo
from spherical_spectral_element.assembly import dss_scalar_for, dss_scalar_jax, dss_scalar_sparse, dss_scalar


def test_dss():
  nx = 3
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, jax=use_wrapper)
  grid_nojax, _ = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, jax=False)
  vert_redundancy_gll = grid_nojax["vert_redundancy"]
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
        assert (np.allclose((dss_scalar(device_wrapper(fn), grid, dims)), fn))


def test_dss_equiv():
  nx = 15
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, jax=False)
  grid_jax, dims_jax = gen_metric_from_topo(face_connectivity,
                                            face_mask,
                                            face_position_2d,
                                            vert_redundancy,
                                            jax=use_wrapper)
  fn = device_wrapper(np.cos(grid["physical_coords"][:, :, :, 1]) * np.cos(grid["physical_coords"][:, :, :, 0]))
  assert (np.allclose(dss_scalar(fn, grid_jax, dims), fn))
  ones = np.ones_like(grid["met_det"])
  ones_out = dss_scalar(device_wrapper(ones), grid_jax, dims)
  assert (np.allclose(np.asarray(ones_out), ones))
  ones_out_for = dss_scalar_for(np.asarray(ones), grid)
  assert (np.allclose(ones_out_for, ones))


def test_dss_equiv_rand():
  nx = 15
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, jax=False)
  grid_jax, dims_jax = gen_metric_from_topo(face_connectivity,
                                            face_mask,
                                            face_position_2d,
                                            vert_redundancy,
                                            jax=use_wrapper)
  for _ in range(20):
    fn_rand = np.random.uniform(size=grid["physical_coords"][:, :, :, 1].shape)
    if wrapper_type == "jax":
      assert (np.allclose(np.asarray(dss_scalar_jax(fn_rand, grid_jax, dims_jax)), dss_scalar_for(fn_rand, grid)))

    assert (np.allclose(dss_scalar_sparse(device_wrapper(fn_rand), grid), dss_scalar_for(fn_rand, grid)))
