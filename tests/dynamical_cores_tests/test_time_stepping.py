from .test_init import get_umjs_state
from .mass_coordinate_grids import cam30
from ..context import get_figdir
from pysces.config import device_unwrapper, jnp, np
from pysces.dynamical_cores.physics_config import init_physics_config
from pysces.analytic_initialization.moist_baroclinic_wave import get_umjs_config
from pysces.time_step import time_step_options
from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.dynamical_cores.mass_coordinate import create_vertical_grid
from pysces.dynamical_cores.model_state import sum_dynamics_states
from pysces.model_info import models, cam_se_models, thermodynamic_variable_names
from pysces.dynamical_cores.time_stepping import advance_dynamics_euler
from frozendict import frozendict

def get_dummy_time_stepping_config(dynamics_tstep_type=time_step_options.RK3_5STAGE,
                                   dt_dynamics=1.0):
  return frozendict(dynamics=frozendict(step_type=dynamics_tstep_type,
                                        dt=dt_dynamics))

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
      dry_species = model_state["tracers"]["dry_species"]
      moisture_species = model_state["tracers"]["moisture_species"]
    else:
      dry_species = None
      moisture_species = None
    for _ in range(10):
      model_state["dynamics"] = advance_dynamics_euler(model_state["dynamics"], model_state["static_forcing"], h_grid, v_grid, physics_config, timestep_config, dims, model,
                                                       moisture_species=moisture_species,
                                                       dry_air_species=dry_species)
    model_state_compare = get_umjs_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False)
    dynamics_diff = sum_dynamics_states(model_state["dynamics"], model_state_compare["dynamics"], 1.0, -1.0, model)
    for field in dynamics_diff.keys():
      print(f"Maximum difference for field {field}: {jnp.max(jnp.abs(dynamics_diff[field]))}")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                    h_grid["physical_coords"][:, :, :, 0].flatten(),
                    dynamics_diff["u"][:, :, :, 10, 0].flatten())
    plt.colorbar()
    plt.savefig(f"{get_figdir()}/test_timestep_u_diff_{model}.pdf")
    plt.figure()
    plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                    h_grid["physical_coords"][:, :, :, 0].flatten(),
                    dynamics_diff["u"][:, :, :, 10, 1].flatten())
    plt.colorbar()
    plt.savefig(f"{get_figdir()}/test_timestep_v_diff_{model}.pdf")
    plt.figure()
    plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                    h_grid["physical_coords"][:, :, :, 0].flatten(),
                    dynamics_diff["d_mass"][:, :, :, 10].flatten())
    plt.colorbar()
    plt.savefig(f"{get_figdir()}/test_timestep_d_mass_diff_{model}.pdf")
    plt.figure()
    plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                    h_grid["physical_coords"][:, :, :, 0].flatten(),
                    dynamics_diff[thermodynamic_variable_names[model]][:, :, :, 10].flatten())
    plt.colorbar()
    plt.savefig(f"{get_figdir()}/test_timestep_thermo_diff_{model}.pdf")
    
