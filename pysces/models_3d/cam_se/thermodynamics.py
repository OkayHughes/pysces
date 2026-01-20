from ...config import jnp, np
from ..model_info import variable_kappa_models


def sum_species(moisture_species_per_dry_mass):
  sum_species = jnp.ones_like(moisture_species_per_dry_mass[moisture_species_per_dry_mass.keys()[0]])
  for species_name in moisture_species_per_dry_mass.keys():
      sum_species += sum_species + (moisture_species_per_dry_mass[species_name])
  return sum_species


def dry_to_moist_conversion_factor(sum_species):
   pass


def cp_moist(moisture_species_per_dry_mass, cp_dry, physics_config):
  sum_cp = cp_dry
  for species_name in moisture_species_per_dry_mass.keys():
    sum_cp += physics_config["water_species_cp"][species_name] * moisture_species_per_dry_mass[species_name]
  return sum_cp


def d_pressure(d_mass, moisture_species_per_dry_mass):
  dp_moist = 1.0 * d_mass
  for species_name in moisture_species_per_dry_mass.keys():
    dp_moist += d_mass * moisture_species_per_dry_mass[species_name]


def surface_pressure(d_mass, moisture_species_per_dry_mass, p_top):
  ps = jnp.sum(d_pressure(d_mass, moisture_species_per_dry_mass), axis=3) + p_top
  return ps


def interface_pressure(d_pressure, p_top):
  p_int_lower = p_top + jnp.cumsum(d_pressure, axis=3)
  p_int = jnp.stack((p_top * jnp.ones(d_pressure.shape[:-1]),
                     p_int_lower), axis=-1)
  return p_int


def midpoint_pressure(p_int):
  p_mid = (p_int[:, :, :, :-1] + p_int[:, :, :, 1:]) / 2.0
  return p_mid


def virtual_temperature(temperature, moisture_species_per_dry_mass, sum_species, R_dry, physics_config):
  Rgas_total = jnp.copy(R_dry)
  for species_name in moisture_species_per_dry_mass.keys():
      Rgas_total += moisture_species_per_dry_mass[species_name] * physics_config["water_species_Rgas"][species_name]
  virtual_temperature = temperature * Rgas_total / (R_dry * sum_species)
  return virtual_temperature


def Rgas_dry(dry_species_per_dry_mass, physics_config, model):
  if model in variable_kappa_models:
    Rgas_total = jnp.zeros_like(dry_species_per_dry_mass[dry_species_per_dry_mass.keys()[0]])
    for species_name in dry_species_per_dry_mass.keys():
      Rgas_total += dry_species_per_dry_mass[species_name] * physics_config["dry_species_Rgas"][species_name]
  else:
    Rgas_total = physics_config["Rgas"] * jnp.ones_like(dry_species_per_dry_mass[dry_species_per_dry_mass.keys()[0]])
  return Rgas_total


def cp_dry(dry_species_per_dry_mass, physics_config, model):
  if model in variable_kappa_models:
    cp_total = jnp.zeros_like(dry_species_per_dry_mass[dry_species_per_dry_mass.keys()[0]])
    for species_name in dry_species_per_dry_mass.keys():
      cp_total += dry_species_per_dry_mass[species_name] * physics_config["dry_species_cp"][species_name]
  else:
    cp_total = physics_config["cp"] * jnp.ones_like(dry_species_per_dry_mass[dry_species_per_dry_mass.keys()[0]])
  return cp_total


def hydrostatic_geopotential(T_v, dp, p_mid, R_dry, phi_surf):
  d_phi = -R_dry * T_v * dp / p_mid
  phi_i = jnp.cumsum(jnp.flip(d_phi, axis=-1), axis=-1) + phi_surf[:, :, :, np.newaxis]
  phi_m = jnp.flip(phi_i, axis=-1) - 0.5 * d_phi
  return phi_m
