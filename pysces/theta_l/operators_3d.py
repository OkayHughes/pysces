from ..config import partial, jit, vmap_1d_apply
from ..operators import sphere_divergence, sphere_vorticity
from ..operators import sphere_gradient, sphere_laplacian_wk, sphere_vec_laplacian_wk


@jit
def sphere_divergence_3d(vector, h_grid, config):
  sph_op = partial(sphere_divergence, grid=h_grid, a=config["radius_earth"])
  return vmap_1d_apply(sph_op, vector, -2, -1)


@jit
def sphere_vorticity_3d(vector, h_grid, config):
  sph_op = partial(sphere_vorticity, grid=h_grid, a=config["radius_earth"])
  return vmap_1d_apply(sph_op, vector, -2, -1)


@jit
def sphere_laplacian_wk_3d(scalar, h_grid, config):
  sph_op = partial(sphere_laplacian_wk, grid=h_grid, a=config["radius_earth"])
  return vmap_1d_apply(sph_op, scalar, -1, -1)


@jit
def sphere_vec_laplacian_wk_3d(vector, h_grid, config):
  sph_op = partial(sphere_vec_laplacian_wk, grid=h_grid, a=config["radius_earth"])
  return vmap_1d_apply(sph_op, vector, -2, -2)


@jit
def sphere_gradient_3d(scalar, h_grid, config):
  sph_op = partial(sphere_gradient, grid=h_grid, a=config["radius_earth"])
  return vmap_1d_apply(sph_op, scalar, -1, -2)
