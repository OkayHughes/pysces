from ...test_data.mass_coordinate_grids import cam30
from pysces.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config, perturbation_opts
from pysces.run_dycore import init_simulator
from pysces.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pysces.dynamical_cores.mass_coordinate import init_vertical_grid
from pysces.model_info import models
from pysces.dynamical_cores.model_config import init_default_config, hypervis_opts
from pysces.initialization import init_baroclinic_wave_state


def test_theta_steady_state():
  for model in [models.cam_se, models.homme_hydrostatic]:
    npt = 4
    nx = 7
    h_grid, dims = init_quasi_uniform_grid(nx, npt, calc_smooth_tensor=True)
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)

    total_time = (3600.0 * 6.0)
    for diffusion in [hypervis_opts.none, hypervis_opts.quasi_uniform, hypervis_opts.variable_resolution]:
      print("=" * 10)
      print(f"starting {diffusion}")
      print("=" * 10)
      physics_config, diffusion_config, timestep_config = init_default_config(nx, h_grid, v_grid,
                                                                              dims, model,
                                                                              hypervis_type=diffusion)
      diffusion_config["nu_top"] = 0.0
      test_config = init_baroclinic_wave_config(model_config=physics_config)
      model_state = init_baroclinic_wave_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False)
      simulator = init_simulator(h_grid, v_grid,
                                 physics_config,
                                 diffusion_config,
                                 timestep_config,
                                 dims,
                                 model)

      t = 0.0

      for t, state in simulator(model_state):
        print(t)
        if t > total_time:
          break


def test_theta_baro_wave_topo():
  npt = 4
  nx = 7
  h_grid, dims = init_quasi_uniform_grid(nx, npt, calc_smooth_tensor=True)
  model = models.homme_hydrostatic
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)

  total_time = (3600.0 * 6.0)
  for diffusion in [hypervis_opts.variable_resolution, hypervis_opts.quasi_uniform, hypervis_opts.none]:
    physics_config, diffusion_config, timestep_config = init_default_config(nx, h_grid, v_grid, dims, model,
                                                                            hypervis_type=diffusion)
    test_config = init_baroclinic_wave_config(model_config=physics_config)
    model_state = init_baroclinic_wave_state(h_grid, v_grid, physics_config, test_config,
                                             dims, model, mountain=True,
                                             pert_type=perturbation_opts.none)
    simulator = init_simulator(h_grid, v_grid,
                               physics_config,
                               diffusion_config,
                               timestep_config,
                               dims,
                               model)

    t = 0.0
    for t, state in simulator(model_state):
      print(t)
      if t > total_time:
        break
