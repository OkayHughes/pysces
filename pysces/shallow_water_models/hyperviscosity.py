from ..config import jit, jnp, device_wrapper
from ..operations_2d.operators import manifold_vector_laplacian_weak, manifold_laplacian_weak
from ..operations_2d.tensor_hyperviscosity import quasi_uniform_hypervisc_coeff, variable_resolution_hypervisc_coeff
from ..operations_2d.se_grid import get_global_grid_deformation_metrics
from .model_state import project_state, create_state_struct, advance_state
from ..distributed_memory.global_assembly import project_scalar_global
from functools import partial


def get_hypervis_config_const(ne, config,
                               nu_base=-1.0,
                               nu_dpi=-1.0,
                               nu_div_factor=2.5):
  nu = quasi_uniform_hypervisc_coeff(ne, config["radius_earth"]) if nu_base <= 0 else nu_base
  nu_dpi = nu if nu_dpi < 0 else nu_dpi
  diffusion_config = {"constant_hypervis": 1.0,
                      "nu": device_wrapper(nu),
                      "nu_dpi": device_wrapper(nu_dpi),
                      "nu_div_factor": device_wrapper(nu_div_factor)}
  return diffusion_config


def get_hypervis_config_tensor(h_grid, dims, config,
                                ad_hoc_scale=0.5):
  radius_earth = config["radius_earth"]
  _, max_min_dx, min_max_dx = get_global_grid_deformation_metrics(h_grid, dims)
  nu_tens = variable_resolution_hypervisc_coeff(min_max_dx,
                                                h_grid["hypervis_scaling"],
                                                dims["npt"],
                                                radius_earth=config["radius_earth"])
  nu = device_wrapper(ad_hoc_scale * nu_tens)
  diffusion_config = {"tensor_hypervis": 1.0,
                      "nu": nu,
                      "nu_dpi": nu}
  return diffusion_config


@partial(jit, static_argnames=["dims"])
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
  u_tmp = manifold_vector_laplacian_weak(state_in["u"], grid, a=a, damp=True)
  h_tmp = manifold_laplacian_weak(state_in["h"][:, :, :], grid, a=a)
  lap1 = project_state(create_state_struct(u_tmp, h_tmp, state_in["hs"]), grid, dims, scaled=False)
  u_tmp = diffusion_config["nu"] * manifold_vector_laplacian_weak(lap1["u"], grid,
                                                                   a=a, damp=True,
                                                                   nu_div_fact=diffusion_config["nu_div_factor"])
  h_tmp = diffusion_config["nu_dpi"] * manifold_laplacian_weak(lap1["h"], grid, a=a)
  return project_state(create_state_struct(u_tmp, h_tmp, state_in["hs"]), grid, dims, scaled=False)


@partial(jit, static_argnames=["dims"])
def calc_hypervis_variable_resolution(state_in, grid, physics_config, diffusion_config, dims):
  a = physics_config["radius_earth"]
  u_cart = jnp.einsum("fijs,fijcs->fijc", jnp.flip(state_in["u"], axis=-1), grid["physical_to_cartesian"])
  components_laplace = []
  for cart_idx in range(3):
    components_laplace.append(manifold_laplacian_weak(u_cart[:, :, :, cart_idx], grid, a=a, apply_tensor=False))
  h_laplace = manifold_laplacian_weak(state_in["h"], grid, a=a, apply_tensor=False)
  state_laplace_cont = project_scalar_global([*components_laplace, h_laplace], grid, dims, scaled=False, two_d=True)
  components_biharm = []
  for cart_idx in range(3):
    components_biharm.append(manifold_laplacian_weak(state_laplace_cont[cart_idx], grid, a=a, apply_tensor=True))
  h_biharm = manifold_laplacian_weak(state_laplace_cont[3], grid, a=a, apply_tensor=True)
  state_biharm_cont = project_scalar_global([*components_biharm, h_biharm], grid, dims, scaled=False, two_d=True)

  h_biharm_cont = diffusion_config["nu_dpi"] * state_biharm_cont[3]
  u_cart = jnp.stack(state_biharm_cont[:3], axis=-1)
  u_sph = jnp.einsum("fijc,fijcs->fijs", u_cart,  grid["physical_to_cartesian"])
  u_sph = jnp.flip(diffusion_config["nu"] * u_sph, axis=-1)
  return create_state_struct(u_sph, h_biharm_cont, state_in["hs"])
