from ..config import jit, jnp, device_wrapper, do_mpi_communication
from ..operations_2d.operators import horizontal_weak_vector_laplacian, horizontal_weak_laplacian
from ..operations_2d.tensor_hyperviscosity import (eval_quasi_uniform_hypervisc_coeff,
                                                   eval_variable_resolution_hypervisc_coeff)
from ..horizontal_grid import eval_global_grid_deformation_metrics
from .model_state import project_model_state, wrap_model_state
from ..mpi.global_assembly import project_scalar_global
from ..operations_2d.local_assembly import project_scalar
from functools import partial


def init_hypervis_config_const(ne,
                               config,
                               nu_base=-1.0,
                               nu_d_mass=-1.0,
                               nu_div_factor=2.5):
  nu = eval_quasi_uniform_hypervisc_coeff(ne, config["radius_earth"]) if nu_base <= 0 else nu_base
  nu_d_mass = nu if nu_d_mass < 0 else nu_d_mass
  diffusion_config = {"constant_hypervis": 1.0,
                      "nu": device_wrapper(nu),
                      "nu_d_mass": device_wrapper(nu_d_mass),
                      "nu_div_factor": device_wrapper(nu_div_factor)}
  return diffusion_config


def init_hypervis_config_tensor(h_grid,
                                dims,
                                config,
                                ad_hoc_scale=0.5):
  radius_earth = config["radius_earth"]
  _, max_min_dx, min_max_dx = eval_global_grid_deformation_metrics(h_grid, dims)
  nu_tens = eval_variable_resolution_hypervisc_coeff(min_max_dx,
                                                     h_grid["hypervis_scaling"],
                                                     dims["npt"],
                                                     radius_earth=radius_earth)
  nu = device_wrapper(ad_hoc_scale * nu_tens)
  diffusion_config = {"tensor_hypervis": 1.0,
                      "nu": nu,
                      "nu_d_mass": nu}
  return diffusion_config


@partial(jit, static_argnames=["dims"])
def eval_hypervis_quasi_uniform(state_in,
                                grid,
                                physics_config,
                                diffusion_config,
                                dims):
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
  u_tmp = horizontal_weak_vector_laplacian(state_in["horizontal_wind"], grid, a=a, damp=True)
  h_tmp = horizontal_weak_laplacian(state_in["h"][:, :, :], grid, a=a)
  lap1 = project_model_state(wrap_model_state(u_tmp, h_tmp, state_in["hs"]), grid, dims)
  u_tmp = diffusion_config["nu"] * horizontal_weak_vector_laplacian(lap1["horizontal_wind"],
                                                                    grid,
                                                                    a=a,
                                                                    damp=True,
                                                                    nu_div_fact=diffusion_config["nu_div_factor"])
  h_tmp = diffusion_config["nu_d_mass"] * horizontal_weak_laplacian(lap1["h"], grid, a=a)
  return project_model_state(wrap_model_state(u_tmp, h_tmp, state_in["hs"]), grid, dims)


@partial(jit, static_argnames=["dims"])
def eval_hypervis_variable_resolution(state_in,
                                      grid,
                                      physics_config,
                                      diffusion_config,
                                      dims):
  a = physics_config["radius_earth"]
  u_cart = jnp.einsum("fijs,fijcs->fijc", jnp.flip(state_in["horizontal_wind"], axis=-1), grid["physical_to_cartesian"])
  components_laplace = []
  for cart_idx in range(3):
    components_laplace.append(horizontal_weak_laplacian(u_cart[:, :, :, cart_idx], grid, a=a, apply_tensor=False))
  h_laplace = horizontal_weak_laplacian(state_in["h"], grid, a=a, apply_tensor=False)
  if do_mpi_communication:
    state_laplace_cont = project_scalar_global([*components_laplace, h_laplace], grid, dims, two_d=True)
  else:
    state_laplace_cont = []
    for comp in components_laplace:
      state_laplace_cont.append(project_scalar(comp, grid, dims))
    state_laplace_cont.append(project_scalar(h_laplace, grid, dims))
  components_biharm = []
  for cart_idx in range(3):
    components_biharm.append(horizontal_weak_laplacian(state_laplace_cont[cart_idx], grid, a=a, apply_tensor=True))
  h_biharm = horizontal_weak_laplacian(state_laplace_cont[3], grid, a=a, apply_tensor=True)
  if do_mpi_communication:
    state_biharm_cont = project_scalar_global([*components_biharm, h_biharm], grid, dims, two_d=True)
  else:
    state_biharm_cont = []
    for comp in components_biharm:
      state_biharm_cont.append(project_scalar(comp, grid, dims))
    state_biharm_cont.append(project_scalar(h_biharm, grid, dims))

  h_biharm_cont = diffusion_config["nu_d_mass"] * state_biharm_cont[3]
  u_cart = jnp.stack(state_biharm_cont[:3], axis=-1)
  u_sph = jnp.einsum("fijc,fijcs->fijs", u_cart, grid["physical_to_cartesian"])
  u_sph = jnp.flip(diffusion_config["nu"] * u_sph, axis=-1)
  return wrap_model_state(u_sph, h_biharm_cont, state_in["hs"])
