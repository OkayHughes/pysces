from pysces.models_3d.mass_coordinate import (create_vertical_grid,
                                              mass_from_coordinate_midlev,
                                              mass_from_coordinate_interface)
from pysces.config import jnp, np
from .mass_coordinate_grids import cam30


def test_vcoord():
  v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"])
  n_test = 12
  ps = ((1 + .05 * jnp.sin(jnp.linspace(0, jnp.pi, n_test))[:, np.newaxis, np.newaxis]) *
        v_grid["reference_pressure"])
  p_mid = mass_from_coordinate_midlev(ps, v_grid)
  p_int = mass_from_coordinate_interface(ps, v_grid)
  assert (p_mid.shape == (n_test, 1, 1, 30))
  assert (p_int.shape == (n_test, 1, 1, 31))
  for lev_isobaric in range(0, 12):
    assert (jnp.max(jnp.abs(p_int[0:1, 0, 0, lev_isobaric] -
                            p_int[:, 0, 0, lev_isobaric])) < 1e-8)
    assert (jnp.max(jnp.abs(p_mid[0:1, 0, 0, lev_isobaric] -
                            p_mid[:, 0, 0, lev_isobaric])) < 1e-8)
  for lev_terrain in range(29, 31):
    p_int_scaled = p_int[:, :, :, lev_terrain] / ps
    assert (jnp.max(jnp.abs(p_int_scaled[0:1, 0, 0] -
                            p_int_scaled[:, 0, 0])) < 1e-8)
  assert (jnp.allclose(0.5 * (p_int[:, :, :, :-1] + p_int[:, :, :, 1:]),
                       p_mid))
