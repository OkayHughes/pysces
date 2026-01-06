from pysces.config import np
from pysces.mesh_generation.element_local_metric import create_quasi_uniform_grid_elem_local
from ..context import test_npts


def test_gen_metric():
  for npt in test_npts:
    for nx in [6, 7]:
      grid, dims = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
      for ((elem_idx, i_idx, j_idx),
           (elem_idx_pair, i_idx_pair, j_idx_pair)) in grid["vert_redundancy"]:
        assert (np.allclose(grid["physical_coords"][elem_idx, i_idx, j_idx, :],
                            grid["physical_coords"][elem_idx_pair, i_idx_pair, j_idx_pair, :]))


def test_gen_mass_mat():
  for npt in test_npts:
    for nx in [14, 15]:
      grid, dims = create_quasi_uniform_grid_elem_local(nx, npt)
      assert (np.allclose(np.sum(grid["met_det"] *
                                 (grid["gll_weights"][np.newaxis, :, np.newaxis] *
                                  grid["gll_weights"][np.newaxis, np.newaxis, :])), 4 * np.pi))
