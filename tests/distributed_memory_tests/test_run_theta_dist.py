from pysces.initialization import init_baroclinic_wave_state
from pysces.dynamical_cores.physics_config import init_physics_config
from ..dynamical_cores_tests.mass_coordinate_grids import cam30
from ..context import get_figdir, test_division_factor
from pysces.config import device_unwrapper, jnp, np, mpi_rank
from pysces.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config
from pysces.run_dycore import init_simulator
from pysces.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pysces.dynamical_cores.mass_coordinate import init_vertical_grid
from pysces.model_info import models


def test_theta_steady_state():
  npt = 4
  nx = 16
  model = models.homme_hydrostatic
  h_grid, dims = init_quasi_uniform_grid(nx, npt, proc_idx=mpi_rank)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)
  model_config = init_physics_config(model)
  test_config = init_baroclinic_wave_config(model_config=model_config)
  model_state = init_baroclinic_wave_state(h_grid,
                                           v_grid,
                                           model_config,
                                           test_config,
                                           dims,
                                           mountain=False,
                                           hydrostatic=False)
  total_time = (3600.0 * 24.0 * 10.0) / (test_division_factor)
  for diffusion in [False, True]:
    end_state = init_simulator(total_time, nx, model_state,
                               h_grid, v_grid,
                               model_config, dims,
                               hydrostatic=True,
                               deep=False,
                               diffusion=diffusion,
                               step_type="ull5")
    ps = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] + jnp.sum(end_state["d_mass"], axis=-1)
    ps_begin = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] + jnp.sum(model_state["d_mass"], axis=-1)
    theta_v = device_unwrapper(end_state["theta_v_d_mass"][:, :, :, 12] / end_state["d_mass"][:, :, :, 12]).flatten()
    import matplotlib.pyplot as plt
    figdir = get_figdir()
    plt.figure()
    plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                    device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                    device_unwrapper(ps).flatten())
    plt.colorbar()
    plt.savefig(f"{figdir}/dist_final_state_hv_{diffusion}.pdf")
    plt.figure()
    plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                    device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                    device_unwrapper(ps - ps_begin).flatten())
    plt.colorbar()
    plt.savefig(f"{figdir}/dist_ps_diff_hv_{diffusion}.pdf")
    plt.figure()
    plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                    device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                    device_unwrapper(end_state["u"][:, :, :, 12, 1]).flatten())
    plt.colorbar()
    plt.savefig(f"{figdir}/dist_v_end_hv_{diffusion}.pdf")
    plt.figure()
    plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                    device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                    device_unwrapper(end_state["u"][:, :, :, 12, 0]).flatten())
    plt.colorbar()
    plt.savefig(f"{figdir}/dist_u_end_hv_{diffusion}.pdf")
    plt.figure()
    plt.tricontourf(device_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                    device_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                    theta_v)
    plt.colorbar()
    plt.savefig(f"{figdir}/dist_theta_v_end_hv_{diffusion}.pdf")
    if not diffusion:
      assert(jnp.max(jnp.abs(end_state["u"][:, :, :, :, 1])) < 0.2)
    else:
      assert(not np.any(np.isnan(device_unwrapper(end_state["u"][:, :, :, :, 1]))))


def test_theta_baro_wave():
  npt = 4
  nx = 30
  model = models.homme_hydrostatic
  h_grid, dims = init_quasi_uniform_grid(nx, npt, proc_idx=mpi_rank)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"])
  model_config = init_physics_config()
  test_config = init_baroclinic_wave_config(model_config=model_config)
  model_state = init_baroclinic_wave_state(h_grid, v_grid, model_config,
                                           test_config, dims, mountain=False, hydrostatic=False,
                                           pert_type="exponential")
  total_time = (3600.0 * 24.0 * 30.0) / (test_division_factor)
  end_state = init_simulator(total_time, nx, model_state,
                             h_grid, v_grid,
                             model_config, dims,
                             hydrostatic=True,
                             deep=False,
                             diffusion=False,
                             step_type="ull5",
                             sponge_split=3)
