from pysces.config import np
from pysces.mesh_generation.unstructured_metric import create_quasi_uniform_grid_unstructured, gen_metric_terms_unstructured
from pysces.mesh_generation.cubed_sphere import gen_cube_topo
from pysces.mesh_generation.mesh import gen_gll_redundancy, mesh_to_cart_bilinear, gen_vert_redundancy
from pysces.mesh_generation.equiangular_metric import gen_metric_terms_equiangular
from ..context import test_npts, get_figdir


def test_gen_metric():
  for npt in test_npts:
    for nx in [6, 7]:
      grid, dims = create_quasi_uniform_grid_unstructured(nx, npt)
      for ((elem_idx, i_idx, j_idx),
           (elem_idx_pair, i_idx_pair, j_idx_pair)) in grid["vert_redundancy"]:
        assert (np.allclose(grid["physical_coords"][elem_idx, i_idx, j_idx, :],
                              grid["physical_coords"][elem_idx_pair, i_idx_pair, j_idx_pair, :]))



def test_gen_mass_mat():
  for npt in test_npts:
    for nx in [14, 15]:
      grid, dims = create_quasi_uniform_grid_unstructured(nx, npt)
      assert (np.allclose(np.sum(grid["met_det"] *
                                 (grid["gll_weights"][np.newaxis, :, np.newaxis] *
                                  grid["gll_weights"][np.newaxis, np.newaxis, :])), 4 * np.pi))
