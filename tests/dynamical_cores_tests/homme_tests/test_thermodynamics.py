from pysces.analytic_initialization.moist_baroclinic_wave import (get_umjs_config,
                                                    evaluate_pressure_temperature)
from pysces.dynamical_cores.physics_config import init_physics_config
from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.config import jnp
from ..mass_coordinate_grids import vertical_grid_finite_diff
from pysces.dynamical_cores.utils_3d import interface_to_model
from pysces.dynamical_cores.mass_coordinate import create_vertical_grid
from pysces.dynamical_cores.homme.thermodynamics import get_mu, get_p_mid, get_balanced_phi
from ..test_init import get_umjs_state
from pysces.model_info import models


def test_eos_hydro():
  npt = 4
  nx = 5
  h_grid, dims = create_quasi_uniform_grid(nx, npt)
  lat = h_grid["physical_coords"][:, :, :, 0]
  vstruct = vertical_grid_finite_diff(200)
  model = models.homme_hydrostatic
  v_grid = create_vertical_grid(vstruct["hybrid_a_i"],
                                vstruct["hybrid_b_i"],
                                vstruct["p0"],
                                model)
  for mountain in [False, True]:
    model_config = init_physics_config(model)
    test_config = get_umjs_config(model_config=model_config)
    lat = h_grid["physical_coords"][:, :, :, 0]
    model_state = get_umjs_state(h_grid, v_grid, model_config, test_config, dims, model, mountain=mountain)
    p_mid = get_p_mid(model_state["dynamics"],
                      v_grid)
    phi_i = get_balanced_phi(model_state["static_forcing"]["phi_surf"],
                             p_mid,
                             model_state["dynamics"]["theta_v_d_mass"], model_config)

    z_i = phi_i / model_config['gravity']
    z_mid = interface_to_model(z_i)

    p_model, exner, r_hat_i, mu = get_mu(model_state["dynamics"], phi_i, v_grid, model_config, model)
    Tv = (model_state["dynamics"]["theta_v_d_mass"] / model_state["dynamics"]
          ["d_mass"]) * exner
    assert (jnp.max(jnp.abs(p_model - p_mid)) < 1e-8)
    assert (jnp.max(jnp.abs(p_model - p_mid)) < 1e-8)
    p_int_state, t_int_state = evaluate_pressure_temperature(z_mid, lat, test_config)
    p_mid_state, t_mid_state = evaluate_pressure_temperature(z_mid, lat, test_config)
    assert (jnp.max(jnp.abs(p_model - p_mid_state) / p_model) < 0.01)
    assert (jnp.max(jnp.abs(Tv - t_mid_state) / Tv) < 0.01)
    assert (jnp.max(jnp.abs(mu - 1)) < 1e-5)


def test_eos_nonhydro():
  npt = 4
  nx = 5
  model = models.homme_nonhydrostatic
  h_grid, dims = create_quasi_uniform_grid(nx, npt)
  lat = h_grid["physical_coords"][:, :, :, 0]
  vstruct = vertical_grid_finite_diff(100)
  v_grid = create_vertical_grid(vstruct["hybrid_a_i"],
                                vstruct["hybrid_b_i"],
                                vstruct["p0"],
                                model)
  for mountain in [False, True]:
    model_config = init_physics_config(model)
    test_config = get_umjs_config(model_config=model_config)
    lat = h_grid["physical_coords"][:, :, :, 0]
    model_state = get_umjs_state(h_grid, v_grid, model_config, test_config, dims, model,
                                 mountain=mountain)
    phi_i = model_state["dynamics"]["phi_i"]
    p_mid = get_p_mid(model_state["dynamics"],
                      v_grid)

    z_i = phi_i / model_config['gravity']
    z_mid = interface_to_model(z_i)

    p_model, exner, r_hat_i, mu = get_mu(model_state["dynamics"], phi_i, v_grid, model_config, model)
    p_int_state, t_int_state = evaluate_pressure_temperature(z_mid, lat, test_config)
    p_mid_state, t_mid_state = evaluate_pressure_temperature(z_mid, lat, test_config)
    Tv = (model_state["dynamics"]["theta_v_d_mass"] / model_state["dynamics"]["d_mass"]) * exner
    assert (jnp.max(jnp.abs(p_model - p_mid) / p_model) < .01)
    assert (jnp.max(jnp.abs(p_model - p_mid) / p_model) < .01)
    assert (jnp.max(jnp.abs(p_model - p_mid_state) / p_model) < 0.01)
    assert (jnp.max(jnp.abs(Tv - t_mid_state) / Tv) < 0.01)
    assert (jnp.max(jnp.abs(mu - 1)) < 1e-2)
