from ...config import jit, jnp
from functools import partial
from ..model_state import wrap_dynamics_struct, init_static_forcing, wrap_tracer_struct, wrap_model_state
from ...model_info import variable_kappa_models

@partial(jit, static_argnames=["dims", "model"])
def init_model_struct(u, T, d_mass, phi_surf, moisture_species, tracers, h_grid, dims, physics_config, model, dry_air_species=None):
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
  if model not in variable_kappa_models:
    dry_air_species = {"dry_air": jnp.ones_like(T)}
  dynamics = wrap_dynamics_struct(u, T, d_mass, model)
  static_forcing = init_static_forcing(phi_surf, h_grid, physics_config, dims, model)
  tracers = wrap_tracer_struct(moisture_species, tracers, model, dry_air_species=dry_air_species)
  return wrap_model_state(dynamics, static_forcing, tracers)
