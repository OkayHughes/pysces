from .mass_coordinate_grids import cam30
from ..context import get_figdir
from pysces.config import device_unwrapper, jnp, np
from pysces.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config, perturbation_opts
from pysces.run_dycore import init_simulator
from pysces.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pysces.dynamical_cores.mass_coordinate import init_vertical_grid
from pysces.horizontal_grid import postprocess_grid
from pysces.model_info import models, cam_se_models, homme_models
from pysces.dynamical_cores.model_config import init_default_config, hypervis_opts
from pysces.initialization import init_baroclinic_wave_state


def test_theta_steady_state():
  for model in [models.cam_se, models.homme_hydrostatic]:
    npt = 4
    nx = 15
    h_grid, dims = init_quasi_uniform_grid(nx, npt)
    h_grid = postprocess_grid(h_grid, dims)
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)

    total_time = (3600.0 * 24.0)
    for diffusion in [hypervis_opts.variable_resolution, hypervis_opts.quasi_uniform, hypervis_opts.none]:
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
      import matplotlib.pyplot as plt

      for t, state in simulator(model_state):
        print(t)
        if t > total_time:
          break

      end_state = state["dynamics"]
      ps = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] + jnp.sum(end_state["d_mass"], axis=-1)
      ps_begin = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] + jnp.sum(end_state["d_mass"], axis=-1)
      figdir = get_figdir()
      if model in homme_models:
        thermo = end_state["theta_v_d_mass"][:, :, :, 12] / end_state["d_mass"][:, :, :, 12]
      elif model in cam_se_models:
        thermo = end_state["T"][:, :, :, 12]
      plt.figure()
      plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                      device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                      device_unwrapper(ps).flatten())
      plt.colorbar()
      plt.savefig(f"{figdir}/final_state_hv_{diffusion}_{model}.pdf")
      plt.figure()
      plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                      device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                      device_unwrapper(ps - ps_begin).flatten())
      plt.colorbar()
      plt.savefig(f"{figdir}/ps_diff_hv_{diffusion}_{model}.pdf")
      plt.figure()
      plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                      device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                      device_unwrapper(end_state["u"][:, :, :, 12, 1]).flatten())
      plt.colorbar()
      plt.savefig(f"{figdir}/v_end_hv_{diffusion}_{model}.pdf")
      plt.figure()
      plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                      device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                      device_unwrapper(end_state["u"][:, :, :, 12, 0]).flatten())
      plt.colorbar()
      plt.savefig(f"{figdir}/u_end_hv_{diffusion}_{model}.pdf")
      plt.figure()
      plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                      device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                      device_unwrapper(thermo).flatten())
      plt.colorbar()
      plt.savefig(f"{figdir}/thermo_end_hv_{diffusion}_{model}.pdf")
      if not diffusion:
        assert(jnp.max(jnp.abs(end_state["u"][:, :, :, :, 1])) < 0.2)
      else:
        assert(not np.any(np.isnan(device_unwrapper(end_state["u"][:, :, :, :, 1]))))


def test_theta_baro_wave():
  npt = 4
  nx = 15
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  h_grid = postprocess_grid(h_grid, dims)
  model = models.homme_hydrostatic
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)

  total_time = (3600.0 * 24.0)
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

    end_state = state["dynamics"]
    ps = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] + jnp.sum(end_state["d_mass"], axis=-1)
    theta_v = end_state["theta_v_d_mass"][:, :, :, 12] / end_state["d_mass"][:, :, :, 12]
    import matplotlib.pyplot as plt
    figdir = get_figdir()
    plt.figure()
    plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                    device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                    device_unwrapper(ps).flatten())
    plt.colorbar()
    plt.savefig(f"{figdir}/final_state_bw_{diffusion}.pdf")
    plt.figure()
    plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                    device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                    device_unwrapper(end_state["u"][:, :, :, 12, 1]).flatten())
    plt.colorbar()
    plt.savefig(f"{figdir}/v_end_bw_{diffusion}.pdf")
    plt.figure()
    plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                    device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                    device_unwrapper(end_state["u"][:, :, :, 12, 0]).flatten())
    plt.colorbar()
    plt.savefig(f"{figdir}/u_end_bw_{diffusion}.pdf")
    plt.figure()
    plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                    device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                    device_unwrapper(theta_v).flatten())
    plt.colorbar()
    plt.savefig(f"{figdir}/theta_v_end_bw_{diffusion}.pdf")
