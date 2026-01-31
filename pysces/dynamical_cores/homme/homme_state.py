from ...config import jnp, jit
from ..model_state import wrap_model_state, wrap_tracers, init_static_forcing, wrap_dynamics
from functools import partial


@partial(jit, static_argnames=["dims", "model"])
def init_model_struct(u,
                      theta_v_d_mass,
                      d_mass,
                      phi_surf,
                      moisture_species,
                      tracers,
                      h_grid,
                      dims,
                      physics_config,
                      model,
                      phi_i=None,
                      w_i=None,
                      f_plane_center=jnp.pi / 4.0):
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
  dynamics = wrap_dynamics(u,
                           theta_v_d_mass,
                           d_mass,
                           model,
                           phi_i=phi_i,
                           w_i=w_i)
  static_forcing = init_static_forcing(phi_surf,
                                       h_grid,
                                       physics_config,
                                       dims,
                                       model,
                                       f_plane_center=f_plane_center)
  tracers = wrap_tracers(moisture_species,
                         tracers,
                         model)
  return wrap_model_state(dynamics,
                          static_forcing,
                          tracers)


# TODO 12/23/25: add wrapper functions that apply
# summation, and project_scalar so model interface remains identical.
# also: refactor into separate file, since these are non-jittable
