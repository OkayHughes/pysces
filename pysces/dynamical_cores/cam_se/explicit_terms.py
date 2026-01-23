from ...operations_2d.operators import horizontal_gradient, horizontal_vorticity, horizontal_divergence
from ..operators_3d import horizontal_gradient_3d, horizontal_vorticity_3d, horizontal_divergence_3d
from ..utils_3d import sphere_dot
from ...config import jnp, np
from .thermodynamics import sum_species, midpoint_pressure, interface_pressure, cp_moist, hydrostatic_geopotential, exner_function
from .thermodynamics import Rgas_dry as Rgas_dry_fn
from .thermodynamics import cp_dry as cp_dry_fn
from .thermodynamics import virtual_temperature as virtual_temperature_fn
from ..model_state import wrap_dynamics_struct
from enum import Enum

pressure_gradient_options = Enum('pressure_gradient_options',
                                 [("basic", 1),
                                  ("grad_exner", 2),
                                  ("corrected_grad_exner", 3)])


def init_common_variables(dynamics, static_forcing, moisture_species, dry_air_species, h_grid, v_grid, physics_config, model):
  temperature = dynamics["T"]
  u = dynamics["u"]
  d_mass = dynamics["d_mass"]

  phi_surf = static_forcing["phi_surf"]
  coriolis_param = static_forcing["coriolis_param"]

  R_dry = Rgas_dry_fn(dry_air_species, physics_config, model)
  cp_dry = cp_dry_fn(dry_air_species, physics_config, model)
  cp = cp_moist(moisture_species, cp_dry, physics_config)
  total_mixing_ratio = sum_species(moisture_species)
  virtual_temperature = virtual_temperature_fn(temperature, moisture_species, total_mixing_ratio, R_dry, physics_config)

  ptop = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"]
  d_pressure = total_mixing_ratio * d_mass
  p_int = interface_pressure(d_pressure, ptop)
  pressure_model_lev = midpoint_pressure(p_int)
  grad_pressure = horizontal_gradient_3d(pressure_model_lev, h_grid, physics_config)

  density_inv = R_dry * virtual_temperature / pressure_model_lev
  return {"u": u,
          "temperature": temperature,
          "d_mass": d_mass,
          "d_pressure": d_pressure,
          "virtual_temperature": virtual_temperature,
          "cp": cp,
          "cp_dry": cp_dry,
          "R_dry": R_dry,
          "grad_pressure": grad_pressure,
          "density_inv": density_inv,
          "pressure_midpoint": pressure_model_lev,
          "phi_surf": phi_surf,
          "coriolis_param": coriolis_param}


def temperature_horiz_advection_term(common_variables, h_grid, physics_config):
  grad_temperature = horizontal_gradient_3d(common_variables["temperature"], h_grid, physics_config)
  u_dot_grad_temperature = sphere_dot(common_variables["u"], grad_temperature)
  return -u_dot_grad_temperature


def temperature_vertical_advection_term(common_variables, h_grid, physics_config):
  u = common_variables["u"]
  cp = common_variables["cp"]
  d_pressure_u = common_variables["d_pressure"][:, :, :, :, np.newaxis] * u
  div_d_pressure_u = horizontal_divergence_3d(d_pressure_u, h_grid, physics_config)
  u_dot_grad_pressure = sphere_dot(u, common_variables["grad_pressure"])
  d_pressure_tend = -jnp.cumsum(div_d_pressure_u, axis=3) + 0.5 * div_d_pressure_u
  vertical_pressure_velocity = d_pressure_tend + u_dot_grad_pressure 
  return common_variables["density_inv"] * vertical_pressure_velocity / cp


def d_mass_divergence_term(common_variables, h_grid, physics_config):
  d_mass_u = common_variables["d_mass"][:, :, :, :, np.newaxis] * common_variables["u"]
  div_d_mass_u = horizontal_divergence_3d(d_mass_u, h_grid, physics_config)
  return -div_d_mass_u


def energy_gradient_term(common_variables, h_grid, physics_config):
  u = common_variables["u"]
  phi_surf = common_variables["phi_surf"]
  kinetic_energy = sphere_dot(u, u) / 2.0 
  phi = hydrostatic_geopotential(common_variables["virtual_temperature"],
                                 common_variables["d_pressure"],
                                 common_variables["pressure_midpoint"],
                                 common_variables["R_dry"],
                                 phi_surf)
  return -horizontal_gradient_3d(kinetic_energy[:, :, :] + phi, h_grid, physics_config)


def vorticity_term(common_variables, h_grid, physics_config):
  u = common_variables["u"]
  coriolis_parameter = common_variables["coriolis_param"]
  vorticity = horizontal_vorticity_3d(u, h_grid, physics_config)
  return jnp.stack((u[:, :, :, :, 1] * (coriolis_parameter[:, :, :, np.newaxis] + vorticity),
                    -u[:, :, :, :, 0] * (coriolis_parameter[:, :, :, np.newaxis] + vorticity)), axis=-1)


def pressure_gradient_force_term(common_variables, h_grid, v_grid, physics_config, pgf_formulation):
  cp_dry = common_variables["cp_dry"]
  R_dry = common_variables["R_dry"]
  exner = exner_function(common_variables["pressure_midpoint"],
                         R_dry, cp_dry, physics_config)
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


def explicit_tendency(dynamics, static_forcing, moist_species, dry_air_species, h_grid, v_grid, physics_config,  model, pgf_formulation=pressure_gradient_options.corrected_grad_exner):

  common_variables = init_common_variables(dynamics, static_forcing, moist_species, dry_air_species, h_grid, v_grid, physics_config, model)

  velocity_tend = (vorticity_term(common_variables, h_grid, physics_config) +
                   energy_gradient_term(common_variables, h_grid, physics_config) +
                   pressure_gradient_force_term(common_variables, h_grid, v_grid, physics_config, pgf_formulation))
  temperature_tend = (temperature_horiz_advection_term(common_variables, h_grid, physics_config) +
                      temperature_vertical_advection_term(common_variables, h_grid, physics_config))
  d_mass_tend = d_mass_divergence_term(common_variables, h_grid, physics_config)

  return wrap_dynamics_struct(velocity_tend,
                              temperature_tend,
                              d_mass_tend,
                              model)
