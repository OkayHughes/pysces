from ..config import jnp, device_wrapper
from .model_state import init_model_struct, init_tracer_struct
from .infra import get_delta
from .vertical_coordinate import mass_from_coordinate_midlev, mass_from_coordinate_interface


def z_from_p_monotonic(pressures, p_given_z, eps=1e-5, z_top=80e3):
  z_guesses = p_given_z(z_top * jnp.ones_like(pressures))
  not_converged = jnp.logical_not(jnp.abs((p_given_z(z_guesses) - pressures)) / pressures < eps)
  frac = 0.5
  while jnp.any(not_converged):
    p_guess = p_given_z(z_guesses)
    too_high = p_guess < pressures
    z_guesses = jnp.where(not_converged,
                          jnp.where(jnp.logical_and(not_converged, too_high),
                                    z_guesses - frac * z_top,
                                    z_guesses + frac * z_top), z_guesses)
    not_converged = jnp.logical_not(jnp.abs((p_given_z(z_guesses) - pressures)) / pressures < eps)
    frac *= 0.5
  return z_guesses


def init_model_pressure(z_pi_surf_func, p_func, Tv_func, u_func, v_func, Q_func,
                        h_grid, v_grid, config, dims,
                        hydrostatic=True, w_func=lambda lat, lon, z: 0.0, eps=1e-8):
  lat = h_grid["physical_coords"][:, :, :, 0]
  lon = h_grid["physical_coords"][:, :, :, 1]
  z_surf, pi_surf = z_pi_surf_func(lat, lon)
  p_mid = mass_from_coordinate_midlev(pi_surf, v_grid)
  p_int = mass_from_coordinate_interface(pi_surf, v_grid)
  z_mid = z_from_p_monotonic(p_mid, p_func, eps=eps)
  phi_surf = z_surf * config["gravity"]
  if not hydrostatic:
    z_int = z_from_p_monotonic(p_int, p_func, eps=eps)
    w_i = w_func(lat, lon, z_int)
    phi_i = z_int * config["gravity"]
  else:
    w_i = 0.0
    phi_i = 0.0
  dpi = get_delta(p_int)
  vtheta = Tv_func(lat, lon, z_mid) * (config["p0"] / p_mid)**(config["Rgas"] / config["cp"])
  vtheta_dpi = vtheta * dpi
  u = u_func(lat, lon, z_mid)
  v = v_func(lat, lon, z_mid)
  initial_state = init_model_struct(device_wrapper(jnp.stack((u, v), axis=-1)),
                                    device_wrapper(vtheta_dpi),
                                    device_wrapper(dpi),
                                    device_wrapper(phi_surf),
                                    device_wrapper(phi_i),
                                    device_wrapper(w_i),
                                    h_grid,
                                    dims,
                                    config)
  tracer_struct = init_tracer_struct(Q_func(lat, lon, z_mid))
  return initial_state, tracer_struct
