from pysces.config import np, device_unwrapper
from pysces.mesh_generation.element_local_metric import (create_quasi_uniform_grid_elem_local,
                                                         create_mobius_like_grid_elem_local)
from ..context import test_npts


def test_gen_metric():
  for npt in test_npts:
    for nx in [6, 7]:
      grid, dims = create_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
      for ((elem_idx, i_idx, j_idx),
           (elem_idx_pair, i_idx_pair, j_idx_pair)) in grid["vertex_redundancy"]:
        assert (np.allclose(grid["physical_coords"][elem_idx, i_idx, j_idx, :],
                            grid["physical_coords"][elem_idx_pair, i_idx_pair, j_idx_pair, :]))


def test_gen_mass_mat():
  for npt in test_npts:
    for nx in [14, 15]:
      grid, dims = create_quasi_uniform_grid_elem_local(nx, npt)
      assert (np.allclose(np.sum(grid["metric_determinant"] *
                                 (grid["gll_weights"][np.newaxis, :, np.newaxis] *
                                  grid["gll_weights"][np.newaxis, np.newaxis, :])), 4 * np.pi))


def test_new_grid_tmp():
  np.random.seed(0)
  for npt in test_npts:
    for nx in [15, 16]:
      for _ in range(10):
        axis_dilation = np.random.uniform(high=1.5, low=1.0, size=(3,))
        offset = np.random.uniform(high=0.25, low=0.0, size=(3,))
        matrix = np.random.normal(size=(3, 3))
        Q, _ = np.linalg.qr(matrix)
        grid, _ = create_mobius_like_grid_elem_local(nx, npt,
                                                     axis_dilation=axis_dilation,
                                                     offset=offset,
                                                     orthogonal_transform=Q)
        assert (np.allclose(np.sum(grid["metric_determinant"] *
                                   (grid["gll_weights"][np.newaxis, :, np.newaxis] *
                                    grid["gll_weights"][np.newaxis, np.newaxis, :])), 4 * np.pi))
        assert not np.any(np.isnan(device_unwrapper(grid["metric_determinant"])))
