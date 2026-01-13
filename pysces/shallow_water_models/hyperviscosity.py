from ..config import jit, jnp
from ..operations_2d.operators import sphere_vec_laplacian_wk, sphere_laplacian_wk
from .model_state import project_state, create_state_struct, advance_state
from functools import partial

@partial(jit, static_argnames=["dims", "diffusion_config"])
def calc_hypervis_quasi_uniform(state_in, grid, physics_config, diffusion_config, dims):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  a = physics_config["radius_earth"]
  u_tmp = sphere_vec_laplacian_wk(state_in["u"], grid, a=a, damp=True)
  h_tmp = sphere_laplacian_wk(state_in["h"][:, :, :], grid, a=a)
  lap1 = project_state(create_state_struct(u_tmp, h_tmp, state_in["hs"]), grid, dims)
  u_tmp = -diffusion_config["nu"] * sphere_vec_laplacian_wk(lap1["u"], grid, a=a, damp=True)
  h_tmp = -diffusion_config["nu"] * sphere_laplacian_wk(lap1["h"], grid, a=a)
  return project_state(create_state_struct(u_tmp, h_tmp, state_in["hs"]), grid, dims)

@partial(jit, static_argnames=["dims", "diffusion_config"])
def calc_hypervis_quasi_uniform(state_in, grid, physics_config, diffusion_config, dims):
  a = physics_config["radius_earth"]
  u_cart = jnp.einsum("fijs,gs"
  u_tmp = sphere_vec_laplacian_wk(state_in["u"], grid, a=a, damp=True)
  h_tmp = sphere_laplacian_wk(state_in["h"][:, :, :], grid, a=a)
  lap1 = project_state(create_state_struct(u_tmp, h_tmp, state_in["hs"]), grid, dims)
  u_tmp = -diffusion_config["nu"] * sphere_vec_laplacian_wk(lap1["u"], grid, a=a, damp=True)
  h_tmp = -diffusion_config["nu"] * sphere_laplacian_wk(lap1["h"], grid, a=a)
  return project_state(create_state_struct(u_tmp, h_tmp, state_in["hs"]), grid, dims)