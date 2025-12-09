from .test_init import get_umjs_state
from .vertical_grids import cam30
from ..context import get_figdir
from spherical_spectral_element.config import jax_unwrapper, jnp
from spherical_spectral_element.theta_l.constants import init_config
from spherical_spectral_element.theta_l.initialization.umjs14 import get_umjs_config
from spherical_spectral_element.theta_l.run_model import simulate_theta
from spherical_spectral_element.equiangular_metric import create_quasi_uniform_grid
from spherical_spectral_element.theta_l.vertical_coordinate import create_vertical_grid
from spherical_spectral_element.theta_l.infra import succeeded


def test_theta_steady_state():
  nx = 8
  h_grid, dims = create_quasi_uniform_grid(nx)
  v_grid = create_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"])
  model_config = init_config()
  test_config = get_umjs_config(model_config=model_config)
  model_state, _ = get_umjs_state(h_grid, v_grid, model_config, test_config, dims, mountain=False, hydrostatic=False)
  total_time = (3600.0 * 24.0 * 30)/360
  end_state, err_code = simulate_theta(total_time, nx, model_state,
                                       h_grid, v_grid,
                                       model_config, dims,
                                       hydrostatic=True,
                                       deep=False,
                                       diffusion=False,
                                       step_type="ull5")
  ps = v_grid["hybrid_a_i"][0] * v_grid["reference_pressure"] + jnp.sum(end_state["dpi"], axis=-1)
  ps_begin = v_grid["hybrid_a_i"][0] * v_grid["reference_pressure"] + jnp.sum(model_state["dpi"], axis=-1)
  import matplotlib.pyplot as plt
  figdir = get_figdir()
  plt.figure()
  plt.tricontourf(jax_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                  jax_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                  jax_unwrapper(ps).flatten())
  plt.colorbar()
  plt.savefig(f"{figdir}/final_state.pdf")
  plt.figure()
  plt.tricontourf(jax_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                  jax_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                  jax_unwrapper(ps-ps_begin).flatten())
  plt.colorbar()
  plt.savefig(f"{figdir}/ps_diff.pdf")
  plt.figure()
  plt.tricontourf(jax_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                  jax_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                  jax_unwrapper(end_state["u"][:, :, :, 12, 1]).flatten())
  plt.colorbar()
  plt.savefig(f"{figdir}/v_end.pdf")
  plt.figure()
  plt.tricontourf(jax_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                  jax_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                  jax_unwrapper(end_state["u"][:, :, :, 12, 0]).flatten())
  plt.colorbar()
  plt.savefig(f"{figdir}/u_end.pdf")
  plt.figure()
  plt.tricontourf(jax_unwrapper(h_grid["physical_coords"][:, :, :, 1]).flatten(),
                  jax_unwrapper(h_grid["physical_coords"][:, :, :, 0]).flatten(),
                  jax_unwrapper(end_state["vtheta_dpi"][:, :, :, 12] / end_state["dpi"][:, :, :, 12]).flatten())
  plt.colorbar()
  plt.savefig(f"{figdir}/vtheta_end.pdf")
  print(err_code)
  assert succeeded(err_code)
