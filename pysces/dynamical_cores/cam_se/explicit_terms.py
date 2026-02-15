from ..operators_3d import horizontal_gradient_3d, horizontal_vorticity_3d, horizontal_divergence_3d
from ..utils_3d import physical_dot_product
from ...config import jnp, np, jit
from .thermodynamics import (eval_sum_species,
                             eval_midlevel_pressure,
                             eval_interface_pressure,
                             eval_cp_moist,
                             eval_balanced_geopotential,
                             eval_exner_function)
from .thermodynamics import eval_Rgas_dry
from .thermodynamics import eval_cp_dry
from .thermodynamics import eval_virtual_temperature
from ..model_state import wrap_dynamics, wrap_consistency_struct_dynamics
from enum import Enum
from functools import partial

pressure_gradient_options = Enum('pressure_gradient_options',
                                 [("basic", 1),
                                  ("grad_exner", 2),
                                  ("corrected_grad_exner", 3)])


@partial(jit, static_argnames=["model"])
def init_common_variables(dynamics,
                          static_forcing,
                          moisture_species,
                          dry_air_species,
                          h_grid,
                          v_grid,
                          physics_config,
                          model):
  """
  Initialize intermediate quantities that are shared between terms in the adiabatic equations of motion.

  Parameters
  ----------
  dynamics: DynamicsState
      Dict-like containing only the prognostic quantities for adiabatic dynamics, `u`, 
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
  temperature = dynamics["T"]
  wind = dynamics["horizontal_wind"]
  d_mass = dynamics["d_mass"]

  phi_surf = static_forcing["phi_surf"]
  coriolis_param = static_forcing["coriolis_param"]

  R_dry = eval_Rgas_dry(dry_air_species, physics_config)
  cp_dry = eval_cp_dry(dry_air_species, physics_config)
  cp = eval_cp_moist(moisture_species, cp_dry, physics_config)
  total_mixing_ratio = eval_sum_species(moisture_species)
  virtual_temperature = eval_virtual_temperature(temperature,
                                                 moisture_species,
                                                 total_mixing_ratio,
                                                 R_dry,
                                                 physics_config)

  ptop = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"]
  d_pressure = total_mixing_ratio * d_mass
  p_int = eval_interface_pressure(d_pressure, ptop)
  pressure_model_lev = eval_midlevel_pressure(p_int)
  grad_pressure = horizontal_gradient_3d(pressure_model_lev, h_grid, physics_config)
  density_inv = R_dry * virtual_temperature / pressure_model_lev
  phi = eval_balanced_geopotential(virtual_temperature,
                                   d_pressure,
                                   pressure_model_lev,
                                   R_dry,
                                   phi_surf)
  return {"horizontal_wind": wind,
          "temperature": temperature,
          "phi": phi,
          "d_mass": d_mass,
          "d_pressure": d_pressure,
          "virtual_temperature": virtual_temperature,
          "cp": cp,
          "cp_dry": cp_dry,
          "R_dry": R_dry,
          "grad_pressure": grad_pressure,
          "density_inv": density_inv,
          "pressure_midlevel": pressure_model_lev,
          "phi_surf": phi_surf,
          "coriolis_param": coriolis_param}


@jit
def eval_temperature_horiz_advection_term(common_variables,
                                          h_grid,
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
  grad_temperature = horizontal_gradient_3d(common_variables["temperature"], h_grid, physics_config)
  u_dot_grad_temperature = physical_dot_product(common_variables["horizontal_wind"], grad_temperature)
  return -u_dot_grad_temperature


@jit
def eval_temperature_vertical_advection_term(common_variables,
                                             h_grid,
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
  u = common_variables["horizontal_wind"]
  cp = common_variables["cp"]
  d_pressure_u = common_variables["d_pressure"][:, :, :, :, np.newaxis] * u
  div_d_pressure_u = horizontal_divergence_3d(d_pressure_u, h_grid, physics_config)
  u_dot_grad_pressure = physical_dot_product(u, common_variables["grad_pressure"])
  d_pressure_tend = -jnp.cumsum(div_d_pressure_u, axis=3) + 0.5 * div_d_pressure_u
  vertical_pressure_velocity = d_pressure_tend + u_dot_grad_pressure
  return common_variables["density_inv"] * vertical_pressure_velocity / cp


@jit
def eval_d_mass_divergence_term(common_variables,
                                h_grid,
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
  d_mass_u = common_variables["d_mass"][:, :, :, :, np.newaxis] * common_variables["horizontal_wind"]
  div_d_mass_u = horizontal_divergence_3d(d_mass_u, h_grid, physics_config)
  return -div_d_mass_u


@jit
def eval_energy_gradient_term(common_variables,
                              h_grid,
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
  u = common_variables["horizontal_wind"]
  phi = common_variables["phi"]
  kinetic_energy = physical_dot_product(u, u) / 2.0
  return -horizontal_gradient_3d(kinetic_energy[:, :, :] + phi, h_grid, physics_config)


@jit
def eval_vorticity_term(common_variables,
                        h_grid,
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
  u = common_variables["horizontal_wind"]
  coriolis_parameter = common_variables["coriolis_param"]
  vorticity = horizontal_vorticity_3d(u, h_grid, physics_config)
  return jnp.stack((u[:, :, :, :, 1] * (coriolis_parameter[:, :, :, np.newaxis] + vorticity),
                    -u[:, :, :, :, 0] * (coriolis_parameter[:, :, :, np.newaxis] + vorticity)), axis=-1)


@partial(jit, static_argnames=["pgf_formulation"])
def eval_pressure_gradient_force_term(common_variables,
                                      h_grid,
                                      v_grid,
                                      physics_config,
                                      pgf_formulation):
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
  cp_dry = common_variables["cp_dry"]
  R_dry = common_variables["R_dry"]
  exner = eval_exner_function(common_variables["pressure_midlevel"],
                              R_dry,
                              cp_dry,
                              physics_config)
  theta_v = common_variables["virtual_temperature"] / exner
  grad_exner = horizontal_gradient_3d(exner, h_grid, physics_config)
  lapse_rate = .0065
  # balanced ref profile correction:
  # reference temperature profile (Simmons and Jiabin, 1991, QJRMS, Section 2a)
  #
  #  Tref = T0+T1*Exner
  #  T1 = .0065*Tref*Cp/g ! = ~191
  #  T0 = Tref-T1         ! = ~97
  T_ref = 288.0

  T1 = (lapse_rate * T_ref * physics_config["cp"] / physics_config["gravity"])
  T0 = T_ref - T1
  pgf_term_grad_exner = cp_dry[:, :, :, :, np.newaxis] * theta_v[:, :, :, :, np.newaxis] * grad_exner
  grad_logexner = horizontal_gradient_3d(jnp.log(exner), h_grid, physics_config)
  pgf_correction = cp_dry[:, :, :, :, np.newaxis] * T0 * (grad_logexner - grad_exner / exner[:, :, :, :, np.newaxis])
  basic_pgf = common_variables["density_inv"][:, :, :, :, np.newaxis] * common_variables["grad_pressure"]
  if pgf_formulation == pressure_gradient_options.grad_exner:
      pgf_term = - pgf_term_grad_exner
  elif pgf_formulation == pressure_gradient_options.basic:
      pgf_term = - basic_pgf
  elif pgf_formulation == pressure_gradient_options.corrected_grad_exner:
      lower_levels_pgf = pgf_term_grad_exner + pgf_correction
      # only apply away from constant pressure levels
      pgf_term = -jnp.where(v_grid["hybrid_b_m"][np.newaxis, np.newaxis, np.newaxis, :, np.newaxis] > 1e-9,
                            lower_levels_pgf,
                            basic_pgf)
  else:
    raise ValueError("Pressure gradient not implemented")
  return pgf_term


@jit
def eval_tracer_consistency_term(common_variables):
   return common_variables["d_mass"][:, :, :, :, jnp.newaxis] * common_variables["u"]


@partial(jit, static_argnames=["model", "pgf_formulation"])
def eval_explicit_tendency(dynamics,
                           static_forcing,
                           moist_species,
                           dry_air_species,
                           h_grid,
                           v_grid,
                           physics_config,
                           model,
                           pgf_formulation=pressure_gradient_options.corrected_grad_exner):
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
  common_variables = init_common_variables(dynamics,
                                           static_forcing,
                                           moist_species,
                                           dry_air_species,
                                           h_grid,
                                           v_grid,
                                           physics_config,
                                           model)

  velocity_tend = (eval_vorticity_term(common_variables, h_grid, physics_config) +
                   eval_energy_gradient_term(common_variables, h_grid, physics_config) +
                   eval_pressure_gradient_force_term(common_variables,
                                                     h_grid,
                                                     v_grid,
                                                     physics_config,
                                                     pgf_formulation))
  temperature_tend = (eval_temperature_horiz_advection_term(common_variables, h_grid, physics_config) +
                      eval_temperature_vertical_advection_term(common_variables, h_grid, physics_config))
  d_mass_tend = eval_d_mass_divergence_term(common_variables, h_grid, physics_config)

  dynamics = wrap_dynamics(velocity_tend,
                           temperature_tend,
                           d_mass_tend,
                           model)
  tracer_consistency = wrap_consistency_struct_dynamics(eval_tracer_consistency_term(common_variables))
  return dynamics, tracer_consistency
