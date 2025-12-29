from pysces.config import np, jnp, device_unwrapper, device_wrapper, use_wrapper, wrapper_type

from pysces.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from pysces.equiangular_metric import gen_metric_from_topo
from pysces.theta_l.initialization.umjs14 import (get_umjs_config,
                                                  evaluate_pressure_temperature,
                                                  evaluate_state)
from pysces.theta_l.infra import g_from_z


def test_shallow():
  npt = 4
  if use_wrapper and wrapper_type == "torch":
    # getting double precision init is not a priority
    return
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, npt)
  config_shallow = get_umjs_config(pertu0=0.0,
                                   pertup=0.0)
  lat = grid["physical_coords"][:, :, :, 0]
  eps = device_wrapper(1e-3)

  for z in jnp.linspace(0, 40e3, 10):
    z_above = device_wrapper((z + eps) * jnp.ones((*lat.shape, 1)))
    pressure_above, _ = evaluate_pressure_temperature(z_above, lat, config_shallow, deep=False)
    z_below = device_wrapper((z - eps) * jnp.ones((*lat.shape, 1)))
    pressure_below, _ = evaluate_pressure_temperature(z_below, lat, config_shallow, deep=False)
    z_center = device_wrapper(z * jnp.ones((*lat.shape, 1)))
    pressure, temperature = evaluate_pressure_temperature(z_center, lat, config_shallow, deep=False)
    rho = pressure / (config_shallow["Rgas"] * temperature)
    dp_dz = (pressure_above - pressure_below) / (2 * eps)
    assert (np.max(np.abs(device_unwrapper(g_from_z(z, config_shallow, deep=False) * rho + dp_dz))) < 0.001)


def test_deep():
  npt = 4
  if use_wrapper and wrapper_type == "torch":
    # getting double precision init is not a priority
    return
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, npt)
  lat = grid["physical_coords"][:, :, :, 0]
  lon = grid["physical_coords"][:, :, :, 1]
  eps = device_wrapper(1e-3)
  for alpha in [0.4, 0.5, 0.8]:
    config_deep = get_umjs_config(pertu0=0.0,
                                  pertup=0.0,
                                  radius_earth=6371e3 / 20.0,
                                  period_earth=7.292e-5 * 20.0,
                                  alpha=alpha)
    for z in jnp.linspace(0, 40e3, 10):
      z_above = device_wrapper((z + eps) * jnp.ones((*lat.shape, 1)))
      pressure_above, _ = evaluate_pressure_temperature(z_above, lat, config_deep, deep=True)
      z_below = device_wrapper((z - eps) * jnp.ones((*lat.shape, 1)))
      pressure_below, _ = evaluate_pressure_temperature(z_below, lat, config_deep, deep=True)
      z_center = device_wrapper(z * jnp.ones((*lat.shape, 1)))
      u, v, pressure, temperature, _ = evaluate_state(lat, lon, z_center, config_deep, deep=True)
      rho = pressure / (config_deep["Rgas"] * temperature)
      dp_dz = (pressure_above - pressure_below) / (2 * eps)
      metric_terms = -(u**2 + v**2) / (z_center + config_deep["radius_earth"])
      ncts = -u * 2.0 * config_deep["period_earth"] * jnp.cos(lat)[:, :, :, np.newaxis]
      assert (np.max(np.abs(device_unwrapper(dp_dz / rho + g_from_z(z_center,
                                                                    config_deep,
                                                                    deep=True) +
                                             metric_terms + ncts))) < 1e-3)
