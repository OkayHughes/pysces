from pysces.config import np
from pysces.mesh_generation.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from pysces.mesh_generation.equiangular_metric import gen_metric_terms_equiangular, gen_metric_from_topo
from pysces.mesh_generation.mesh import mesh_to_cart_bilinear, gen_gll_redundancy
from ..context import test_npts


def test_gen_metric():
  for npt in test_npts:
    for nx in [6, 7]:
      face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
      vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
      gll_position, gll_jacobian = mesh_to_cart_bilinear(face_position_2d, npt)
      cube_redundancy = gen_gll_redundancy(vert_redundancy, npt)
      gll_latlon, cube_to_sphere_jacobian = gen_metric_terms_equiangular(face_mask, gll_position, npt)
      for elem_idx in cube_redundancy.keys():
        for (i_idx, j_idx) in cube_redundancy[elem_idx].keys():
          for elem_idx_pair, i_idx_pair, j_idx_pair in cube_redundancy[elem_idx][(i_idx, j_idx)]:
              assert (np.max(np.abs(gll_latlon[elem_idx, i_idx, j_idx, :] -
                                    gll_latlon[elem_idx_pair, i_idx_pair, j_idx_pair, :])) < 1e-8)


def test_gen_mass_mat():
  for npt in test_npts:
    for nx in [14, 15]:
      face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
      vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
      grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, npt, wrapped=False)
      assert (np.allclose(np.sum(grid["met_det"] *
                                 (grid["gll_weights"][np.newaxis, :, np.newaxis] *
                                  grid["gll_weights"][np.newaxis, np.newaxis, :])), 4 * np.pi))
