from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.operations_2d.operators import sphere_gradient
from pysces.operations_2d.assembly import project_scalar
from .mass_coordinate_grids import cam30
from pysces.models_3d.mass_coordinate import (create_vertical_grid,
                                              mass_from_coordinate_interface)
from pysces.models_3d.theta_l.init_model import z_from_p_monotonic, init_model_pressure
from pysces.models_3d.constants import init_config
from pysces.models_3d.utils_3d import get_delta
from pysces.config import jnp, device_wrapper
from pysces.models_3d.initialization.umjs14 import (get_umjs_config,
                                                    evaluate_surface_state,
                                                    evaluate_pressure_temperature,
                                                    evaluate_state)


def get_umjs_state(h_grid, v_grid,
                   model_config, test_config, dims,
                   deep=False, mountain=False, hydrostatic=True, eps=1e-10, pert_type="none"):
  lat = h_grid["physical_coords"][:, :, :, 0]

  def z_pi_surf_func(lat, lon):
    return evaluate_surface_state(lat, lon, test_config, deep=deep, mountain=mountain)

  def Q_func(lat, lon, z):
    return jnp.zeros_like(lat)

  def p_func(z):
    return evaluate_pressure_temperature(z, lat, test_config, deep=deep)[0]

  def u_func(lat, lon, z):
    return evaluate_state(lat, lon, z, test_config, deep=deep, pert_type=pert_type)[0]

  def v_func(lat, lon, z):
    return evaluate_state(lat, lon, z, test_config, deep=deep, pert_type=pert_type)[1]

  def Tv_func(lat, lon, z):
    return evaluate_state(lat, lon, z, test_config, deep=deep)[3]

  def w_func(lat, lon, z):
    if hydrostatic:
      return 0.0
    else:
      return jnp.zeros_like(z)

  model_state, tracer_state = init_model_pressure(z_pi_surf_func,
                                                  p_func,
                                                  Tv_func,
                                                  u_func,
                                                  v_func,
                                                  Q_func,
                                                  h_grid, v_grid,
                                                  model_config,
                                                  dims,
                                                  w_func=w_func,
                                                  hydrostatic=hydrostatic,
                                                  eps=eps)
  return model_state, tracer_state


def test_z_p_func():
  config = init_config()
  pressures = device_wrapper(jnp.linspace(config["p0"], 100, 10))
  T0 = 300

  def p_given_z(z):
    return config["p0"] * jnp.exp(-config["gravity"] * z /
                                  (config["Rgas"] * T0))

  def z_given_p(p):
    return -T0 * config["Rgas"] / config["gravity"] * jnp.log(p / config["p0"])

  heights = z_from_p_monotonic(pressures, p_given_z, eps=1e-10, z_top=80e3)
  assert (jnp.max(jnp.abs(heights - z_given_p(pressures)) / pressures) < 1e-5)


def test_init():
  npt = 4
  nx = 15
  h_grid, dims = create_quasi_uniform_grid(nx, npt)
  v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"])
  lat = h_grid["physical_coords"][:, :, :, 0]
  lon = h_grid["physical_coords"][:, :, :, 1]
  for mountain in [False, True]:
    model_config = init_config()
    test_config = get_umjs_config(model_config=model_config)
    model_state, _ = get_umjs_state(h_grid, v_grid, model_config, test_config, dims, mountain=mountain)
    z_surf, ps = evaluate_surface_state(lat, lon, test_config, mountain=mountain)
    phi_surf = model_config["gravity"] * z_surf
    grad_phi_surf = sphere_gradient(phi_surf,
                                    h_grid, a=model_config["radius_earth"])
    grad_phi_surf_cont = jnp.stack((project_scalar(grad_phi_surf[:, :, :, 0], h_grid, dims),
                                    project_scalar(grad_phi_surf[:, :, :, 1], h_grid, dims)), axis=-1)
    p_int = mass_from_coordinate_interface(ps, v_grid)
    dpi = get_delta(p_int)
    assert (jnp.allclose(dpi, model_state["dpi"]))
    assert (jnp.allclose(phi_surf, model_state["phi_surf"]))
    assert (jnp.allclose(grad_phi_surf_cont, model_state["grad_phi_surf"]))
