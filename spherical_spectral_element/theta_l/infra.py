from ..config import jnp


exit_codes = {"success": (0, "Operation completed successfully"),
              "nan": (1, "NaN found in state")}

def err_code(message):
  return (1, message)
def succeeded(code):
  return True if code[0] == exit_codes["success"][0] else False

def vel_model_to_interface(field_model, dpi, dpi_i):
  mid_levels = (dpi[:, :, :, :-1, jnp.newaxis] * field_model[:, :, :, :-1, :] +
                dpi[:, :, :, 1:, jnp.newaxis] * field_model[:, :, :, 1:, :]) / (2.0 * dpi_i[:, :, :, 1:-1, jnp.newaxis])
  return jnp.concatenate((field_model[:, :, :, 0:1, :],
                          mid_levels,
                          field_model[:, :, :, -1:, :]), axis=-2)


def model_to_interface(field_model):
  mid_levels = (field_model[:, :, :, :-1] + field_model[:, :, :, 1:]) / 2.0
  return jnp.concatenate((field_model[:, :, :, 0:1],
                          mid_levels,
                          field_model[:, :, :, -1:]), axis=-1)


def interface_to_model(field_interface):
  return (field_interface[:, :, :, 1:] +
          field_interface[:, :, :, :-1]) / 2.0


def interface_to_model_vec(vec_interface):
  return (vec_interface[:, :, :, 1:, :] +
          vec_interface[:, :, :, :-1, :]) / 2.0


def get_delta(field_interface):
  return field_interface[:, :, :, 1:] - field_interface[:, :, :, :-1]

def get_surface_sum(dfield_model, val_surf):
  return jnp.concatenate((jnp.cumsum(dfield_model[:, :, :, ::-1], axis=-1)[:, :, :, ::-1] + val_surf[:, :, :, jnp.newaxis],
                          val_surf[:, :, :, jnp.newaxis]), axis=-1)

def z_from_phi(phi, config, deep=False):
  gravity = config["gravity"]
  radius_earth = config["radius_earth"]
  if deep:
    b = (2 * phi * radius_earth - gravity * radius_earth**2)
    z = -2 * phi * radius_earth**2 / (b - jnp.sqrt(b**2 - 4 * phi**2 * radius_earth**2))
  else:
    z = phi / gravity
  return z


def g_from_z(z, config, deep=False):
  radius_earth = config["radius_earth"]
  if deep:
    g = config["gravity"] * (radius_earth /
                             (z + radius_earth))**2
  else:
    g = config["gravity"]
  return g


def g_from_phi(phi, config, deep=False):
  z = z_from_phi(phi, config, deep=deep)
  return g_from_z(z, config, deep=deep)


def r_hat_from_phi(phi, config, deep=False):
  radius_earth = config["radius_earth"]
  if deep:
    r_hat = (z_from_phi(phi, config, deep=deep) + radius_earth) / radius_earth
  else:
    r_hat = jnp.ones_like(phi)
  return r_hat


def sphere_dot(u, v):
  return (u[:, :, :, :, 0] * v[:, :, :, :, 0] +
          u[:, :, :, :, 1] * v[:, :, :, :, 1])
