from ..config import vmap_1d_apply, jit, jnp, device_wrapper, np
from ..operations_2d.operators import sphere_vec_laplacian_wk, sphere_laplacian_wk
from .constants import constant_coeff_hyperviscosity, tensor_hyperviscosity
from ..operations_2d.se_grid import get_global_grid_deformation_metrics
from ..distributed_memory.global_operations import global_max, global_min
from functools import partial

@jit
def scalar_harmonic_3d(scalar, h_grid, config):
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
  def lap_wk_onearg(scalar):
      return sphere_laplacian_wk(scalar, h_grid, a=config["radius_earth"])

  del2 = vmap_1d_apply(lap_wk_onearg, scalar, -1, -1)
  return del2


@jit
def vector_harmonic_3d(vector, h_grid, config, nu_div_factor):
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
  def vec_lap_wk_onearg(vector):
      return sphere_vec_laplacian_wk(vector, h_grid, a=config["radius_earth"],
                                     nu_div_fact=nu_div_factor)

  del2 = vmap_1d_apply(vec_lap_wk_onearg, vector, -2, -2)
  return del2


@partial(jit, static_argnames=["n_sponge"])
def get_nu_ramp(v_grid, n_sponge):
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
  pressure_ratio = ((v_grid["hybrid_a_i"][0] + v_grid["hybrid_b_i"][0]) /
                    (v_grid["hybrid_a_i"][:n_sponge] + v_grid["hybrid_b_i"][:n_sponge]))
  nu_ramp = jnp.minimum(device_wrapper(8.0),
                        (16.0 * pressure_ratio**2 /
                         (pressure_ratio**2 + 1.0))[np.newaxis, np.newaxis, np.newaxis, :])
  return nu_ramp

def init_hypervis_config_const(ne, config,
                               v_grid,
                               nu_top=2.5e5,
                               nu_base=-1.0,
                               nu_phi=-1.0,
                               nu_dpi=-1.0,
                               nu_div_factor=2.5,
                               n_sponge=5):
  nu = constant_coeff_hyperviscosity(ne, config) if nu_base <= 0 else nu_base
  nu_phi = nu_base if nu_phi < 0 else nu_phi
  nu_dpi = nu_base if nu_dpi < 0 else nu_dpi
  nu_ramp = get_nu_ramp(v_grid, n_sponge)
  diffusion_config = {"constant_hypervis": 1.0,
                      "nu": device_wrapper(nu),
                      "nu_phi": device_wrapper(nu_phi),
                      "nu_dpi": device_wrapper(nu_dpi),
                      "nu_div_factor": device_wrapper(nu_div_factor),
                      "nu_top": device_wrapper(nu_top),
                      "nu_ramp": device_wrapper(nu_ramp)}
  return diffusion_config

def init_hypervis_config_tensor(h_grid, v_grid, dims, config,
                                nu_top=2.5e5,
                                ad_hoc_scale=1.0,
                                n_sponge=5):
  nu_ramp = get_nu_ramp(v_grid, n_sponge)
  radius_earth = config["radius_earth"]
  _, max_min_dx, _ = get_global_grid_deformation_metrics(h_grid, dims)
  nu_tens = tensor_hyperviscosity(radius_earth * max_min_dx,
                                  h_grid["hypervis_scaling"],
                                  dims["npt"],
                                  config)
  nu = device_wrapper(ad_hoc_scale * nu_tens)
  diffusion_config = {"tensor_hypervis": 1.0,
                      "nu": nu,
                      "nu_phi": nu,
                      "nu_dpi": nu,
                      "nu_div_factor": device_wrapper(1.0),
                      "nu_top": device_wrapper(nu_top),
                      "nu_ramp": device_wrapper(nu_ramp)}
  return diffusion_config