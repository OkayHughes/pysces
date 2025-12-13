from spherical_spectral_element.config import np, npt, use_wrapper, device_wrapper, device_unwrapper
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.equiangular_metric import gen_metric_from_topo
from spherical_spectral_element.theta_l.model_state import dss_scalar_3d, dss_scalar_3d_for


def test_dss_3d():
  nx = 3
  nlev = 3
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, jax=use_wrapper)
  grid_nojax, _ = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, jax=False)
  vert_redundancy_gll = grid_nojax["vert_redundancy"]
  fn = np.zeros((*grid["physical_coords"].shape[:-1], nlev))
  for lev_idx in range(nlev):
    for face_idx in range(grid["physical_coords"].shape[0]):
      for i_idx in range(npt):
        for j_idx in range(npt):
          fn[:] = 0.0
          fn[face_idx, i_idx, j_idx, lev_idx] = 1.0
          if face_idx in vert_redundancy_gll.keys():
            if (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
              for remote_face_id, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
                fn[remote_face_id, remote_i, remote_j, lev_idx] = 1.0
            assert (np.allclose(device_unwrapper(dss_scalar_3d(device_wrapper(fn), grid, dims)), fn))


def test_dss_equiv_3d_rand():
  nx = 15
  nlev = 5
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d,
                                    vert_redundancy, jax=False)
  grid_jax, dims_jax = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d,
                                            vert_redundancy, jax=use_wrapper)
  for _ in range(20):
    fn_rand = np.random.uniform(size=(*grid["physical_coords"][:, :, :, 1].shape, nlev))
    assert (np.allclose(device_unwrapper(dss_scalar_3d(device_wrapper(fn_rand), grid_jax, dims_jax)),
                        dss_scalar_3d_for(fn_rand, grid, dims)))
