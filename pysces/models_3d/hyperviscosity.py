from ..config import vmap_1d_apply, jit
from ..operations_2d.operators import sphere_vec_laplacian_wk, sphere_laplacian_wk


@jit
def scalar_harmonic_3d(scalar, h_grid, config):

  def lap_wk_onearg(scalar):
      return sphere_laplacian_wk(scalar, h_grid, a=config["radius_earth"])

  del2 = vmap_1d_apply(lap_wk_onearg, scalar, -1, -1)
  return del2


@jit
def vector_harmonic_3d(vector, h_grid, config, nu_div_factor):

  def vec_lap_wk_onearg(vector):
      return sphere_vec_laplacian_wk(vector, h_grid, a=config["radius_earth"],
                                     nu_div_fact=nu_div_factor)

  del2 = vmap_1d_apply(vec_lap_wk_onearg, vector, -2, -2)
  return del2
