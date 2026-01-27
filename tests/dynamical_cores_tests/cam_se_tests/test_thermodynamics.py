from pysces.model_info import models, cam_se_models, variable_kappa_models
from pysces.config import jnp, np
from pysces.mesh_generation.element_local_metric import create_quasi_uniform_grid_elem_local
from pysces.dynamical_cores.mass_coordinate import create_vertical_grid
from pysces.dynamical_cores.physics_config import init_physics_config
from pysces.dynamical_cores.cam_se.thermodynamics import Rgas_dry, cp_dry, cp_moist, virtual_temperature, sum_species, d_pressure, surface_pressure, interface_pressure, midpoint_pressure, exner_function, hydrostatic_geopotential
from pysces.dynamical_cores.utils_3d import get_delta
from pysces.dynamical_cores.mass_coordinate import mass_from_coordinate_interface, surface_mass_from_coordinate, mass_from_coordinate_midlev, top_interface_mass
from ..test_init import get_umjs_state, get_umjs_config, evaluate_pressure_temperature
from pysces.analytic_initialization.moist_baroclinic_wave import evaluate_state, evaluate_pressure_temperature
from pysces.initialization import z_from_p_monotonic_moist
from ..mass_coordinate_grids import cam30, vertical_grid_finite_diff
from ...context import get_figdir


def test_sum_species():
  npt = 4
  nx = 2
  for model in cam_se_models:
    h_grid, dims = create_quasi_uniform_grid_elem_local(nx, npt)
    v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                  cam30["hybrid_b_i"],
                                  cam30["p0"],
                                  model)
    physics_config = init_physics_config(model)
    test_config = get_umjs_config(model_config=physics_config)
    model_state = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False, moist=True, eps=1e-5)
    cp_dry_vals = cp_dry(model_state["tracers"]["dry_air_species"],
                        physics_config)
    total_mixing_ratio = sum_species(model_state["tracers"]["moisture_species"])
    cp = cp_moist(model_state["tracers"]["moisture_species"], cp_dry_vals, physics_config)
    assert jnp.allclose(cp, cp_dry_vals + physics_config["moisture_species_cp"]["water_vapor"] * model_state["tracers"]["moisture_species"]["water_vapor"])
    assert jnp.allclose(total_mixing_ratio, 1.0 + model_state["tracers"]["moisture_species"]["water_vapor"])


def test_dry_Rgas_cp():
  # init upper atmosphere, lower atmosphere models
  npt = 4
  nx = 2
  model_low = models.cam_se
  model_high = models.cam_se_whole_atmosphere
  h_grid, dims = create_quasi_uniform_grid_elem_local(nx, npt)
  v_grid_low = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model_low)
  v_grid_high = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model_high)

  physics_config_low = init_physics_config(model_low)
  test_config = get_umjs_config(model_config=physics_config_low)
  model_state_low = get_umjs_state(h_grid, v_grid_low, physics_config_low, test_config, dims, model_low, mountain=False, moist=False, eps=1e-5)
  physics_config_high = init_physics_config(model_high)
  model_state_high = get_umjs_state(h_grid, v_grid_high, physics_config_high, test_config, dims, model_high, mountain=False, moist=False, eps=1e-5)

  Rgas_low = Rgas_dry(model_state_low["tracers"]["dry_air_species"],
                         physics_config_low)
  cp_low = cp_dry(model_state_low["tracers"]["dry_air_species"],
                         physics_config_low)
  assert jnp.allclose(physics_config_low["Rgas"], Rgas_low)
  assert jnp.allclose(physics_config_low["cp"], cp_low)
  Rgas_high = Rgas_dry(model_state_high["tracers"]["dry_air_species"],
                         physics_config_high)
  cp_high = cp_dry(model_state_high["tracers"]["dry_air_species"],
                   physics_config_high)
  assert jnp.max(jnp.abs(Rgas_high - physics_config_low["Rgas"])/physics_config_low["Rgas"]) < 0.01
  assert jnp.max(jnp.abs(cp_high - physics_config_low["cp"])/physics_config_low["cp"]) < 0.01


def test_moist_cp_virtual_temperature():
  # Many of these tests can be tightened up once we come up with
  # a way to compare atmospheric states 
  npt = 4
  nx = 2
  model_low = models.cam_se
  model_high = models.cam_se_whole_atmosphere
  h_grid, dims = create_quasi_uniform_grid_elem_local(nx, npt)
  v_grid_low = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model_low)
  v_grid_high = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model_high)
  physics_config_low = init_physics_config(model_low)
  test_config = get_umjs_config(model_config=physics_config_low)
  model_state_dry_low =  get_umjs_state(h_grid, v_grid_low, physics_config_low, test_config, dims, model_low, mountain=False, moist=False, eps=1e-8)
  model_state_low = get_umjs_state(h_grid, v_grid_low, physics_config_low, test_config, dims, model_low, mountain=False, moist=True, eps=1e-8)
  physics_config_high = init_physics_config(model_high)
  model_state_high = get_umjs_state(h_grid, v_grid_low, physics_config_high, test_config, dims, model_high, mountain=False, moist=True, eps=1e-8)
  Rgas_dry_low = Rgas_dry(model_state_low["tracers"]["dry_air_species"],
                         physics_config_low)
  cp_dry_low = cp_dry(model_state_low["tracers"]["dry_air_species"],
                         physics_config_low)
  cp_low = cp_moist(model_state_low["tracers"]["moisture_species"], cp_dry_low, physics_config_low)
  Rgas_dry_high = Rgas_dry(model_state_high["tracers"]["dry_air_species"],
                         physics_config_high)
  cp_dry_high = cp_dry(model_state_high["tracers"]["dry_air_species"],
                   physics_config_high)
  cp_high = cp_moist(model_state_high["tracers"]["moisture_species"], cp_dry_high, physics_config_high)
  total_mixing_ratio = sum_species(model_state_dry_low["tracers"]["moisture_species"])
  virtual_temperature_dry_low = virtual_temperature(model_state_dry_low["dynamics"]["T"],
                                                    model_state_dry_low["tracers"]["moisture_species"],
                                                    total_mixing_ratio,
                                                    Rgas_dry_low,
                                                    physics_config_low)
  assert jnp.allclose(model_state_dry_low["dynamics"]["T"], virtual_temperature_dry_low)
  total_mixing_ratio = sum_species(model_state_low["tracers"]["moisture_species"])
  virtual_temperature_low = virtual_temperature(model_state_low["dynamics"]["T"],
                                                model_state_low["tracers"]["moisture_species"],
                                                total_mixing_ratio,
                                                Rgas_dry_low,
                                                physics_config_low)
  # should be equivalent
  max_temp_diff_virtual_temps = jnp.max(jnp.abs(virtual_temperature_low - virtual_temperature_dry_low))
  # should not be equivalent
  max_temp_diff_vtemp_vs_temp = jnp.max(jnp.abs(virtual_temperature_dry_low - model_state_low["dynamics"]["T"]))
  assert  max_temp_diff_virtual_temps < 0.2 * max_temp_diff_vtemp_vs_temp
  assert max_temp_diff_virtual_temps < 0.5
  total_mixing_ratio = sum_species(model_state_high["tracers"]["moisture_species"])
  virtual_temperature_high = virtual_temperature(model_state_high["dynamics"]["T"],
                                                 model_state_high["tracers"]["moisture_species"],
                                                 total_mixing_ratio,
                                                 Rgas_dry_high,
                                                 physics_config_high)
  assert jnp.max(jnp.abs(virtual_temperature_low - virtual_temperature_high)) < 1e-2

  assert jnp.max(jnp.abs(cp_high - cp_low)/physics_config_low["cp"]) < 0.01


def test_pressure_quantities():
  npt = 4
  nx = 3
  for model in cam_se_models:
    for mountain in [True, False]:
      errs = []
      nlev = 50
      v_tmp = vertical_grid_finite_diff(nlev)
      h_grid, dims = create_quasi_uniform_grid_elem_local(nx, npt)
      v_grid = create_vertical_grid(v_tmp["hybrid_a_i"],
                                    v_tmp["hybrid_b_i"],
                                    v_tmp["p0"],
                                    model)
      lat = h_grid["physical_coords"][:, :, :, 0]
      physics_config = init_physics_config(model)
      test_config = get_umjs_config(model_config=physics_config)
      model_state = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=mountain, moist=True, eps=1e-7)
      p_top = top_interface_mass(v_grid)
      moisture_species = model_state["tracers"]["moisture_species"]
      d_pres = d_pressure(model_state["dynamics"]["d_mass"], model_state["tracers"]["moisture_species"])
      p_int = p_top * jnp.ones_like(d_pres[:, :, :, 0])
      R_dry_vals = Rgas_dry(model_state["tracers"]["dry_air_species"], physics_config)
      cp_dry_vals = cp_dry(model_state["tracers"]["dry_air_species"], physics_config)
      interface_pressures = interface_pressure(d_pres, p_top)
      assert jnp.allclose(interface_pressures[:, :, :, 0], p_int)
      for lev_idx in range(nlev):
        p_int += d_pres[:, :, :, lev_idx]
        assert jnp.allclose(interface_pressures[:, :, :, lev_idx+1], p_int)
      p_mid = midpoint_pressure(interface_pressures)
      exner_test = (p_mid / physics_config["p0"])**(R_dry_vals/cp_dry_vals)
      assert jnp.allclose(exner_function(p_mid, R_dry_vals, cp_dry_vals, physics_config), exner_test)
      total_mass = sum_species(moisture_species)
      Tv = model_state["dynamics"]["T"] * (moisture_species["water_vapor"] + physics_config["epsilon"]) / (physics_config["epsilon"] * (1 + moisture_species["water_vapor"]))
      Tv_se =  virtual_temperature(model_state["dynamics"]["T"], moisture_species, total_mass, R_dry_vals, physics_config)
      if model in variable_kappa_models:
        eps = 1e-2
      else:
        eps = 1e-7
      for lev_idx in range(nlev):
        assert jnp.max(jnp.abs(Tv[:, :, :, lev_idx]-Tv_se[:, :, :, lev_idx])) < eps


def test_hydrostatic_geopotential():
  npt = 4
  nx = 2
  h_grid, dims = create_quasi_uniform_grid_elem_local(nx, npt)
  for moist in [True, False]:
    for mountain in [True, False]:
      for model in cam_se_models:
          print(f"testing hydrostatic geopotential, moist: {moist}, mountain: {mountain}, model: {model}")
          nlev = 60
          v_tmp = vertical_grid_finite_diff(nlev)
          v_grid = create_vertical_grid(v_tmp["hybrid_a_i"],
                                        v_tmp["hybrid_b_i"],
                                        v_tmp["p0"],
                                        model)
          physics_config = init_physics_config(model)
          test_config = get_umjs_config(model_config=physics_config)
          model_state = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=mountain, moist=moist, eps=1e-5)
          moisture_species = model_state["tracers"]["moisture_species"]
          dry_air_species = model_state["tracers"]["dry_air_species"]
          R_dry = Rgas_dry(dry_air_species, physics_config)

          d_pres = d_pressure(model_state["dynamics"]["d_mass"], moisture_species)
          p_int = interface_pressure(d_pres, top_interface_mass(v_grid))
          p_mid = midpoint_pressure(p_int)
          total_mass = sum_species(moisture_species)
          Tv = virtual_temperature(model_state["dynamics"]["T"], moisture_species, total_mass, R_dry, physics_config)

          z_midlev = hydrostatic_geopotential(Tv,
                                              d_pres,
                                              p_mid,
                                              R_dry,
                                              model_state["static_forcing"]["phi_surf"]) / physics_config["gravity"]
          def p_given_z(z):
            return evaluate_pressure_temperature(z, h_grid["physical_coords"][:, :, :, 0], test_config, deep=False)[0]
          z_midlev_analytic = z_from_p_monotonic_moist(p_mid, p_given_z)
          norm_const = jnp.mean(z_midlev[:, :, :, :-1] - z_midlev[:, :, :, 1:], axis=(0, 1, 2))
          norm_const = jnp.concatenate((norm_const, jnp.array([norm_const[-1]])), axis=0)
          if model in variable_kappa_models:
            eps = 0.05
          else:
            eps = 0.01
          for lev in range(nlev):
            assert jnp.max(jnp.abs(z_midlev[:, :, :, lev] - z_midlev_analytic[:, :, :, lev])/norm_const[lev]) < eps
      
