from ..config import jit, jnp, np
from ..operations_2d.operators import sphere_vorticity, sphere_gradient, sphere_divergence
from .model_state import create_state_struct


@jit
def calc_rhs(state_in, grid, config):
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
  coriolis = (-jnp.cos(grid["physical_coords"][:, :, :, 1]) *
              jnp.cos(grid["physical_coords"][:, :, :, 0]) *
              jnp.sin(config["alpha"]) +
              jnp.sin(grid["physical_coords"][:, :, :, 0]) *
              jnp.cos(config["alpha"]))
  abs_vort = sphere_vorticity(state_in["u"], grid, a=config["radius_earth"]) + 2 * config["earth_period"] * coriolis
  energy = 0.5 * (state_in["u"][:, :, :, 0]**2 +
                  state_in["u"][:, :, :, 1]**2) + config["gravity"] * (state_in["h"] + state_in["hs"])
  energy_grad = sphere_gradient(energy, grid, a=config["radius_earth"])
  u_tend = abs_vort * state_in["u"][:, :, :, 1] - energy_grad[:, :, :, 0]
  v_tend = -abs_vort * state_in["u"][:, :, :, 0] - energy_grad[:, :, :, 1]
  h_tend = -sphere_divergence(state_in["h"][:, :, :, np.newaxis] * state_in["u"], grid, a=config["radius_earth"])
  return create_state_struct(jnp.stack((u_tend, v_tend), axis=-1), h_tend, state_in["hs"])
