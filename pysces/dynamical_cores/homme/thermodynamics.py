from ...config import jnp, jit, np, flip
from ..utils_3d import midlevel_to_interface, interface_to_delta, phi_to_r_hat
from functools import partial
from ...model_info import hydrostatic_models, deep_atmosphere_models


@jit
def eval_r_hat_sq_avg(r_hat_i):
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
  r_hat_sq = (r_hat_i[:, :, :, :-1] * r_hat_i[:, :, :, 1:] +
              r_hat_i[:, :, :, :-1] * r_hat_i[:, :, :, :-1] +
              r_hat_i[:, :, :, 1:] * r_hat_i[:, :, :, 1:]) / 3.0
  return r_hat_sq


@jit
def eval_pressure_exner_nonhydrostatic(theta_v_d_mass,
                                       d_phi,
                                       r_hat_sq_avg,
                                       config):
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
  p0 = config["p0"]
  nh_pressure_over_exner = -config["Rgas"] * theta_v_d_mass / d_phi
  nh_pressure_over_exner /= r_hat_sq_avg
  exponent = (1.0 / (1.0 - config["Rgas"] / config["cp"]))
  nh_pressure = p0 * (nh_pressure_over_exner / p0)**exponent
  return nh_pressure, nh_pressure / nh_pressure_over_exner


@partial(jit, static_argnames=["model"])
def eval_mu(state,
            phi_i,
            v_grid,
            config,
            model):
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
  # note: assumes that phi_i is in hydrostatic balance.
  theta_v_d_mass = state["theta_v_d_mass"]
  d_phi = interface_to_delta(phi_i)
  if model in deep_atmosphere_models:
    r_hat_i = phi_to_r_hat(phi_i, config, model)
    r_hat_sq_avg = eval_r_hat_sq_avg(r_hat_i)
  else:
    r_hat_i = 1.0
    r_hat_sq_avg = 1.0
  p_model, exner = eval_pressure_exner_nonhydrostatic(theta_v_d_mass, d_phi, r_hat_sq_avg, config)
  if model in hydrostatic_models:
    d_nh_pressure_d_mass = jnp.ones_like(phi_i)
  else:
    p_top = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"]
    if model in deep_atmosphere_models:
      p_top /= r_hat_i[:, :, :, 0]**2
    d_mass_i = midlevel_to_interface(state["d_mass"])
    d_nh_pressure_d_mass_top = 2 * (p_model[:, :, :, 0] - p_top) / d_mass_i[:, :, :, 0]
    d_nh_pressure_d_mass_bottom = jnp.ones_like(p_model[:, :, :, 0])
    d_nh_pressure_d_mass_int = interface_to_delta(p_model) / d_mass_i[:, :, :, 1:-1]
    d_nh_pressure_d_mass = jnp.concatenate((d_nh_pressure_d_mass_top[:, :, :, np.newaxis],
                                            d_nh_pressure_d_mass_int,
                                            d_nh_pressure_d_mass_bottom[:, :, :, np.newaxis]),
                                           axis=-1)
    if model in deep_atmosphere_models:
      d_nh_pressure_d_mass *= r_hat_i**2
  return p_model, exner, r_hat_i, d_nh_pressure_d_mass


@jit
def eval_midlevel_pressure(state,
                           v_grid):
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
  p = jnp.cumsum(state["d_mass"], axis=-1) + v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"]
  p -= 0.5 * state["d_mass"]
  return p


@jit
def eval_balanced_geopotential(phi_surf,
                               p_mid,
                               theta_v_d_mass,
                               physics_config):
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
  # p = get_p_mid(state, v_grid, config)
  exponent = (physics_config["Rgas"] / physics_config["cp"] - 1.0)
  d_phi = physics_config["Rgas"] * (theta_v_d_mass *
                                    (p_mid / physics_config["p0"])**exponent / physics_config["p0"])
  d_phi_augment = flip(jnp.concatenate((d_phi[:, :, :, :-1],
                                        (d_phi[:, :, :, -1] + phi_surf)[:, :, :, np.newaxis]),
                                       axis=-1), -1)
  phi_i_above_surf = jnp.cumsum(d_phi_augment, axis=-1)
  return jnp.concatenate((flip(phi_i_above_surf, -1), phi_surf[:, :, :, np.newaxis]), axis=-1)
