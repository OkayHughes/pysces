from ..config import jnp, jit, np, flip
from functools import partial
from ..model_info import deep_atmosphere_models


@jit
def midlevel_to_interface_vel(field_model,
                              d_mass,
                              d_mass_int):
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
  scaled_sum = (d_mass[:, :, :, :-1, np.newaxis] * field_model[:, :, :, :-1, :] +
                d_mass[:, :, :, 1:, np.newaxis] * field_model[:, :, :, 1:, :])
  mid_levels = scaled_sum / (2.0 * d_mass_int[:, :, :, 1:-1, np.newaxis])
  return jnp.concatenate((field_model[:, :, :, 0:1, :],
                          mid_levels,
                          field_model[:, :, :, -1:, :]), axis=-2)


@jit
def midlevel_to_interface(field_model):
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
  mid_levels = (field_model[:, :, :, :-1] + field_model[:, :, :, 1:]) / 2.0
  return jnp.concatenate((field_model[:, :, :, 0:1],
                          mid_levels,
                          field_model[:, :, :, -1:]), axis=-1)


@jit
def interface_to_midlevel(field_interface):
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
  return (field_interface[:, :, :, 1:] +
          field_interface[:, :, :, :-1]) / 2.0


@jit
def interface_to_midlevel_vec(vec_interface):
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
  return (vec_interface[:, :, :, 1:, :] +
          vec_interface[:, :, :, :-1, :]) / 2.0


@jit
def interface_to_delta(field_interface):
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
  return field_interface[:, :, :, 1:] - field_interface[:, :, :, :-1]


@jit
def cumulative_sum(dfield_model,
                   val_surf_top):
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
  return jnp.concatenate((flip(jnp.cumsum(flip(dfield_model, -1), axis=-1), -1) +
                          val_surf_top[:, :, :, np.newaxis],
                          val_surf_top[:, :, :, np.newaxis]), axis=-1)


@partial(jit, static_argnames=["model"])
def phi_to_z(phi,
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
  gravity = config["gravity"]
  radius_earth = config["radius_earth"]
  if model in deep_atmosphere_models:
    b = (2 * phi * radius_earth - gravity * radius_earth**2)
    z = -2 * phi * radius_earth**2 / (b - jnp.sqrt(b**2 - 4 * phi**2 * radius_earth**2))
  else:
    z = phi / gravity
  return z


@partial(jit, static_argnames=["model"])
def z_to_g(z,
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
  radius_earth = config["radius_earth"]
  if model in deep_atmosphere_models:
    g = config["gravity"] * (radius_earth /
                             (z + radius_earth))**2
  else:
    g = config["gravity"]
  return g


@partial(jit, static_argnames=["model"])
def phi_to_g(phi,
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
  z = phi_to_z(phi, config, model)
  return z_to_g(z, config, model)


@partial(jit, static_argnames=["model"])
def phi_to_r_hat(phi,
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
  radius_earth = config["radius_earth"]
  if model in deep_atmosphere_models:
    r_hat = (phi_to_z(phi, config, model) + radius_earth) / radius_earth
  else:
    r_hat = jnp.ones_like(phi)
  return r_hat


@jit
def physical_dot_product(u, v):
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
  return (u[:, :, :, :, 0] * v[:, :, :, :, 0] +
          u[:, :, :, :, 1] * v[:, :, :, :, 1])
