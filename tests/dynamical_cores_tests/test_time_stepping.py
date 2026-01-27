from .test_init import get_umjs_state
from .mass_coordinate_grids import cam30
from ..context import get_figdir
from pysces.config import device_unwrapper, jnp, np
from pysces.dynamical_cores.physics_config import init_physics_config
from pysces.operations_2d.operators import inner_product
from pysces.analytic_initialization.moist_baroclinic_wave import get_umjs_config
from pysces.time_step import time_step_options
from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.dynamical_cores.mass_coordinate import create_vertical_grid
from pysces.dynamical_cores.model_state import sum_dynamics_states
from pysces.dynamical_cores.utils_3d import sphere_dot
from pysces.model_info import models, cam_se_models, thermodynamic_variable_names, homme_models
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


def test_steady_state_euler():
  for model in [models.cam_se, models.homme_hydrostatic]:
    npt = 4
    nx = 16
    h_grid, dims = create_quasi_uniform_grid(nx, npt)
    v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                  cam30["hybrid_b_i"],
                                  cam30["p0"],
                                  model)

    dt = 10
    physics_config = init_physics_config(model)
    test_config = get_umjs_config(model_config=physics_config)
    model_state = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False)
    timestep_config = get_dummy_time_stepping_config(dynamics_tstep_type=time_step_options.Euler, dt_dynamics=dt)
    if model in cam_se_models:
      dry_species = model_state["tracers"]["dry_air_species"]
      moisture_species = model_state["tracers"]["moisture_species"]
    else:
      dry_species = None
      moisture_species = None
    for s_idx in range(10):
      model_state["dynamics"] = advance_dynamics_euler(model_state["dynamics"], model_state["static_forcing"], h_grid, v_grid, physics_config, timestep_config, dims, model,
                                                       moisture_species=moisture_species,
                                                       dry_air_species=dry_species)
    model_state_compare = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False)
    dynamics_diff = sum_dynamics_states(model_state["dynamics"], model_state_compare["dynamics"], 1.0, -1.0, model)
    if model in homme_models:
      dynamics_diff["theta_v_d_mass"] = dynamics_diff["theta_v_d_mass"] / model_state["dynamics"]["d_mass"]
    for field in dynamics_diff.keys():
      assert jnp.max(jnp.abs(dynamics_diff[field])) < epsilons[field]


def test_steady_state_ullrich():
  for model in [models.homme_hydrostatic, models.cam_se]:
    npt = 4
    nx = 16
    h_grid, dims = create_quasi_uniform_grid(nx, npt)
    v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                  cam30["hybrid_b_i"],
                                  cam30["p0"],
                                  model)

    dt = 10
    physics_config = init_physics_config(model)
    test_config = get_umjs_config(model_config=physics_config)
    model_state = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False)
    timestep_config = get_dummy_time_stepping_config(dynamics_tstep_type=time_step_options.SSPRK3, dt_dynamics=dt)
    if model in cam_se_models:
      dry_species = model_state["tracers"]["dry_air_species"]
      moisture_species = model_state["tracers"]["moisture_species"]
    else:
      dry_species = None
      moisture_species = None
    for s_idx in range(10):
      model_state["dynamics"] = advance_dynamics_ullrich_5stage(model_state["dynamics"], model_state["static_forcing"], h_grid, v_grid, physics_config, timestep_config, dims, model,
                                                                moisture_species=moisture_species,
                                                                dry_air_species=dry_species)
    model_state_compare = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False)
    dynamics_diff = sum_dynamics_states(model_state["dynamics"], model_state_compare["dynamics"], 1.0, -1.0, model)
    if model in homme_models:
      dynamics_diff["theta_v_d_mass"] = dynamics_diff["theta_v_d_mass"] / model_state["dynamics"]["d_mass"]
    for field in dynamics_diff.keys():
      assert jnp.max(jnp.abs(dynamics_diff[field])) < epsilons[field]



def test_steady_state_nonhydro():
  model = models.homme_nonhydrostatic
  npt = 4
  nx = 16
  h_grid, dims = create_quasi_uniform_grid(nx, npt)
  v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)

  dt = .5
  physics_config = init_physics_config(model)
  test_config = get_umjs_config(model_config=physics_config)
  model_state = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False, enforce_hydrostatic=True)
  timestep_config = get_dummy_time_stepping_config(dynamics_tstep_type=time_step_options.SSPRK3, dt_dynamics=dt)
  dry_species = None
  moisture_species = None
  for s_idx in range(100):
    model_state["dynamics"] = advance_dynamics_ullrich_5stage(model_state["dynamics"], model_state["static_forcing"], h_grid, v_grid, physics_config, timestep_config, dims, model,
                                                              moisture_species=moisture_species,
                                                              dry_air_species=dry_species)
  model_state_compare = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False, enforce_hydrostatic=True)
  dynamics_diff = sum_dynamics_states(model_state["dynamics"], model_state_compare["dynamics"], 1.0, -1.0, model)
  dynamics_diff["theta_v_d_mass"] = dynamics_diff["theta_v_d_mass"] / model_state["dynamics"]["d_mass"]
  for field in dynamics_diff.keys():
    assert jnp.max(jnp.abs(dynamics_diff[field])) < epsilons[field] / 2.0
  

