from pysces.initialization import init_baroclinic_wave_state
from .mass_coordinate_grids import cam30
from pysces.config import jnp
from pysces.dynamical_cores.physics_config import init_physics_config
from pysces.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config
from pysces.time_step import time_step_options
from pysces.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pysces.dynamical_cores.mass_coordinate import init_vertical_grid
from pysces.dynamical_cores.model_state import sum_dynamics_series, sum_dynamics
from pysces.model_info import models, cam_se_models, homme_models
from pysces.dynamical_cores.time_stepping import advance_dynamics_euler, advance_dynamics_ullrich_5stage
from frozendict import frozendict


def get_dummy_time_stepping_config(dynamics_tstep_type=time_step_options.RK3_5STAGE,
                                   dt_dynamics=1.0):
  return frozendict(dynamics=frozendict(step_type=dynamics_tstep_type,
                                        dt=dt_dynamics))


epsilons = {"u": 3e-3,
            "T": 2e-4,
            "theta_v_d_mass": 2e-3,
            "d_mass": 2e-2,
            "phi_i": .3,
            "w_i": 1e-3}


def test_steady_state_euler(nx15_np4_dry_homme_hydro, nx15_np4_dry_se):
  for model, struct in zip([models.cam_se, models.homme_hydrostatic],
                           [nx15_np4_dry_se, nx15_np4_dry_homme_hydro]):
    model_state = struct["model_state"]
    physics_config = struct["physics_config"]
    dims = struct["dims"]
    h_grid = struct["h_grid"]
    v_grid = struct["v_grid"]
    dt = 10
    timestep_config = get_dummy_time_stepping_config(dynamics_tstep_type=time_step_options.Euler, dt_dynamics=dt)
    if model in cam_se_models:
      dry_species = model_state["tracers"]["dry_air_species"]
      moisture_species = model_state["tracers"]["moisture_species"]
    else:
      dry_species = None
      moisture_species = None
    dynamics_new = advance_dynamics_euler(model_state["dynamics"],
                                          model_state["static_forcing"],
                                          h_grid,
                                          v_grid,
                                          physics_config,
                                          timestep_config,
                                          dims,
                                          model,
                                          moisture_species=moisture_species,
                                          dry_air_species=dry_species)
    for s_idx in range(10):
      dynamics_new = advance_dynamics_euler(dynamics_new,
                                            model_state["static_forcing"],
                                            h_grid,
                                            v_grid,
                                            physics_config,
                                            timestep_config,
                                            dims,
                                            model,
                                            moisture_species=moisture_species,
                                            dry_air_species=dry_species)
    dynamics_diff = sum_dynamics(dynamics_new, model_state["dynamics"], 1.0, -1.0, model)
    if model in homme_models:
      dynamics_diff["theta_v_d_mass"] = dynamics_diff["theta_v_d_mass"] / model_state["dynamics"]["d_mass"]
    for field in dynamics_diff.keys():
      assert jnp.max(jnp.abs(dynamics_diff[field])) < epsilons[field]


def test_steady_state_ullrich():
  for model in [models.homme_hydrostatic, models.cam_se]:
    npt = 4
    nx = 16
    h_grid, dims = init_quasi_uniform_grid(nx, npt)
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)

    dt = 10
    physics_config = init_physics_config(model)
    test_config = init_baroclinic_wave_config(model_config=physics_config)
    model_state = init_baroclinic_wave_state(h_grid,
                                             v_grid,
                                             physics_config,
                                             test_config,
                                             dims,
                                             model,
                                             mountain=False)
    timestep_config = get_dummy_time_stepping_config(dynamics_tstep_type=time_step_options.SSPRK3, dt_dynamics=dt)
    if model in cam_se_models:
      dry_species = model_state["tracers"]["dry_air_species"]
      moisture_species = model_state["tracers"]["moisture_species"]
    else:
      dry_species = None
      moisture_species = None
    for s_idx in range(10):
      model_state["dynamics"] = advance_dynamics_ullrich_5stage(model_state["dynamics"],
                                                                model_state["static_forcing"],
                                                                h_grid,
                                                                v_grid,
                                                                physics_config,
                                                                timestep_config,
                                                                dims,
                                                                model,
                                                                moisture_species=moisture_species,
                                                                dry_air_species=dry_species)
    model_state_compare = init_baroclinic_wave_state(h_grid,
                                                     v_grid,
                                                     physics_config,
                                                     test_config,
                                                     dims,
                                                     model,
                                                     mountain=False)
    dynamics_diff = sum_dynamics_series(model_state["dynamics"], model_state_compare["dynamics"], 1.0, -1.0, model)
    if model in homme_models:
      dynamics_diff["theta_v_d_mass"] = dynamics_diff["theta_v_d_mass"] / model_state["dynamics"]["d_mass"]
    for field in dynamics_diff.keys():
      assert jnp.max(jnp.abs(dynamics_diff[field])) < epsilons[field]


def test_steady_state_nonhydro():
  model = models.homme_nonhydrostatic
  npt = 4
  nx = 16
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)

  dt = .5
  physics_config = init_physics_config(model)
  test_config = init_baroclinic_wave_config(model_config=physics_config)
  model_state = init_baroclinic_wave_state(h_grid,
                                           v_grid,
                                           physics_config,
                                           test_config,
                                           dims,
                                           model,
                                           mountain=False,
                                           enforce_hydrostatic=True)
  timestep_config = get_dummy_time_stepping_config(dynamics_tstep_type=time_step_options.SSPRK3, dt_dynamics=dt)
  dry_species = None
  moisture_species = None
  for s_idx in range(100):
    model_state["dynamics"] = advance_dynamics_ullrich_5stage(model_state["dynamics"],
                                                              model_state["static_forcing"],
                                                              h_grid,
                                                              v_grid,
                                                              physics_config,
                                                              timestep_config,
                                                              dims,
                                                              model,
                                                              moisture_species=moisture_species,
                                                              dry_air_species=dry_species)
  model_state_compare = init_baroclinic_wave_state(h_grid,
                                                   v_grid,
                                                   physics_config,
                                                   test_config,
                                                   dims,
                                                   model,
                                                   mountain=False,
                                                   enforce_hydrostatic=True)
  dynamics_diff = sum_dynamics_series(model_state["dynamics"], model_state_compare["dynamics"], 1.0, -1.0, model)
  dynamics_diff["theta_v_d_mass"] = dynamics_diff["theta_v_d_mass"] / model_state["dynamics"]["d_mass"]
  for field in dynamics_diff.keys():
    assert jnp.max(jnp.abs(dynamics_diff[field])) < epsilons[field] / 2.0
