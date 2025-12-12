from spherical_spectral_element.config import np
from spherical_spectral_element.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from spherical_spectral_element.equiangular_metric import gen_metric_terms_equiangular, gen_metric_from_topo
from spherical_spectral_element.mesh import mesh_to_cart_bilinear, gen_gll_redundancy


def test_gen_metric():
  nx = 7
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  gll_position, gll_jacobian = mesh_to_cart_bilinear(face_position_2d)
  cube_redundancy = gen_gll_redundancy(face_connectivity, vert_redundancy)
  gll_latlon, cube_to_sphere_jacobian = gen_metric_terms_equiangular(face_mask, gll_position, cube_redundancy)

  for elem_idx in cube_redundancy.keys():
    for (i_idx, j_idx) in cube_redundancy[elem_idx].keys():
      for elem_idx_pair, i_idx_pair, j_idx_pair in cube_redundancy[elem_idx][(i_idx, j_idx)]:
        assert (np.max(np.abs(gll_latlon[elem_idx, i_idx, j_idx, :] -
                              gll_latlon[elem_idx_pair, i_idx_pair, j_idx_pair, :])) < 1e-10)


def test_gen_mass_mat():
  nx = 15
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, jax=False)
  assert (np.allclose(np.sum(grid["met_det"] *
                             (grid["gll_weights"][np.newaxis, :, np.newaxis] *
                              grid["gll_weights"][np.newaxis, np.newaxis, :])), 4 * np.pi))
