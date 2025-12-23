from ..config import jnp, jit, np, flip
from functools import partial


exit_codes = {"success": (0, "Operation completed successfully"),
              "nan": (1, "NaN found in state")}


def err_code(message):
  return (1, message)


def succeeded(code):
  return True if code[0] == exit_codes["success"][0] else False


@jit
def vel_model_to_interface(field_model, dpi, dpi_i):
  mid_levels = (dpi[:, :, :, :-1, np.newaxis] * field_model[:, :, :, :-1, :] +
                dpi[:, :, :, 1:, np.newaxis] * field_model[:, :, :, 1:, :]) / (2.0 * dpi_i[:, :, :, 1:-1, np.newaxis])
  return jnp.concatenate((field_model[:, :, :, 0:1, :],
                          mid_levels,
                          field_model[:, :, :, -1:, :]), axis=-2)


@jit
def model_to_interface(field_model):
  mid_levels = (field_model[:, :, :, :-1] + field_model[:, :, :, 1:]) / 2.0
  return jnp.concatenate((field_model[:, :, :, 0:1],
                          mid_levels,
                          field_model[:, :, :, -1:]), axis=-1)


@jit
def interface_to_model(field_interface):
  return (field_interface[:, :, :, 1:] +
          field_interface[:, :, :, :-1]) / 2.0


@jit
def interface_to_model_vec(vec_interface):
  return (vec_interface[:, :, :, 1:, :] +
          vec_interface[:, :, :, :-1, :]) / 2.0


@jit
def get_delta(field_interface):
  return field_interface[:, :, :, 1:] - field_interface[:, :, :, :-1]


@jit
def get_surface_sum(dfield_model, val_surf):
  return jnp.concatenate((flip(jnp.cumsum(flip(dfield_model, -1), axis=-1), -1) +
                          val_surf[:, :, :, np.newaxis],
                          val_surf[:, :, :, np.newaxis]), axis=-1)


@partial(jit, static_argnames=["deep"])
def z_from_phi(phi, config, deep=False):
  gravity = config["gravity"]
  radius_earth = config["radius_earth"]
  if deep:
    b = (2 * phi * radius_earth - gravity * radius_earth**2)
    z = -2 * phi * radius_earth**2 / (b - jnp.sqrt(b**2 - 4 * phi**2 * radius_earth**2))
  else:
    z = phi / gravity
  return z


@partial(jit, static_argnames=["deep"])
def g_from_z(z, config, deep=False):
  radius_earth = config["radius_earth"]
  if deep:
    g = config["gravity"] * (radius_earth /
                             (z + radius_earth))**2
  else:
    g = config["gravity"]
  return g


@partial(jit, static_argnames=["deep"])
def g_from_phi(phi, config, deep=False):
  z = z_from_phi(phi, config, deep=deep)
  return g_from_z(z, config, deep=deep)


@partial(jit, static_argnames=["deep"])
def r_hat_from_phi(phi, config, deep=False):
  radius_earth = config["radius_earth"]
  if deep:
    r_hat = (z_from_phi(phi, config, deep=deep) + radius_earth) / radius_earth
  else:
    r_hat = jnp.ones_like(phi)
  return r_hat


@jit
def sphere_dot(u, v):
  return (u[:, :, :, :, 0] * v[:, :, :, :, 0] +
          u[:, :, :, :, 1] * v[:, :, :, :, 1])
