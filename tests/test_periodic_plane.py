from pysces.periodic_plane import init_periodic_plane, generate_metric_terms
from pysces.config import np


def test_init_periodic_plane():
  for nx in range(1, 6):
    ny = nx + 1
    NELEM = nx * ny
    physical_coords, ref_to_planar, vert_red = init_periodic_plane(nx, ny)
    physical_coords_test = (physical_coords + 1.0) % (2.0 - 1e-10)
    ct = 0
    for target_face_idx in vert_red.keys():
      for (target_i, target_j) in vert_red[target_face_idx].keys():
        for (source_face_idx, source_i, source_j) in vert_red[target_face_idx][(target_i, target_j)]:
          assert np.allclose(physical_coords_test[target_face_idx, target_i, target_j],
                             physical_coords_test[source_face_idx, source_i, source_j])
          ct += 1
    assert ct == (4 * 4 + 4) * NELEM


def test_metric():
  for nx in range(1, 6):
    ny = nx + 1
    physical_coords, ref_to_planar, vert_red = init_periodic_plane(nx, ny)
    grid, dims = generate_metric_terms(physical_coords, ref_to_planar, vert_red)
    assert (np.allclose(np.sum(grid["met_det"] *
                               (grid["gll_weights"][np.newaxis, :, np.newaxis] *
                                grid["gll_weights"][np.newaxis, np.newaxis, :])), 4.0))
