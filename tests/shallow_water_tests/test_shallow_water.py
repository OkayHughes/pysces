from pysces.config import jnp, np, DEBUG, device_unwrapper, device_wrapper
from pysces.shallow_water_models.run_shallow_water import simulate_sw
from pysces.shallow_water_models.model_state import create_state_struct
from pysces.shallow_water_models.constants import get_physics_config_sw
from pysces.shallow_water_models.williamson_init import (get_williamson_steady_config,
                                                         williamson_tc2_h,
                                                         williamson_tc2_hs,
                                                         williamson_tc2_u)
from pysces.shallow_water_models.galewsky_init import get_galewsky_config, galewsky_wind, galewsky_hs, galewsky_h
from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.operations_2d.operators import inner_product, manifold_vorticity
from pysces.operations_2d.local_assembly import project_scalar
from ..context import get_figdir, test_division_factor
from os import makedirs
from os.path import join

if DEBUG:
  import matplotlib.pyplot as plt


def test_sw_model():
  npt = 4
  nx = 15
  grid, dims = create_quasi_uniform_grid(nx, npt)
  config = get_physics_config_sw(alpha=jnp.pi / 4, ne=15)
  test_config = get_williamson_steady_config(config)
  u_init = device_wrapper(williamson_tc2_u(grid["physical_coords"][:, :, :, 0],
                                           grid["physical_coords"][:, :, :, 1],
                                           test_config))
  h_init = device_wrapper(williamson_tc2_h(grid["physical_coords"][:, :, :, 0],
                                           grid["physical_coords"][:, :, :, 1],
                                           test_config))
  hs_init = device_wrapper(williamson_tc2_hs(grid["physical_coords"][:, :, :, 0],
                                             grid["physical_coords"][:, :, :, 1],
                                             test_config))
  print(u_init.dtype)
  init_state = create_state_struct(u_init, h_init, hs_init)

  T = 4000.0
  final_state = simulate_sw(T, nx, init_state, grid, config, dims)
  print(final_state["u"].dtype)

  diff_u = u_init - final_state["u"]
  diff_h = h_init - final_state["h"]
  assert (inner_product(diff_u[:, :, :, 0], diff_u[:, :, :, 0], grid) < 1e-5)
  assert (inner_product(diff_u[:, :, :, 1], diff_u[:, :, :, 1], grid) < 1e-5)
  assert (inner_product(diff_h, diff_h, grid) / jnp.max(h_init) < 1e-5)
  if DEBUG:
    fig_dir = get_figdir()
    makedirs(fig_dir, exist_ok=True)
    plt.figure()
    plt.title("U at time {t}")
    lon = device_unwrapper(grid["physical_coords"][:, :, :, 1])
    lat = device_unwrapper(grid["physical_coords"][:, :, :, 0])
    plt.tricontourf(lon.flatten(),
                    lat.flatten(),
                    device_unwrapper(final_state["u"][:, :, :, 0].flatten()))
    plt.colorbar()
    plt.savefig(join(fig_dir, "U_final.pdf"))
    plt.figure()
    plt.title("V at time {t}")
    plt.tricontourf(lon.flatten(),
                    lat.flatten(),
                    device_unwrapper(final_state["u"][:, :, :, 1].flatten()))
    plt.colorbar()
    plt.savefig(join(fig_dir, "V_final.pdf"))
    plt.figure()
    plt.title("h at time {t}")
    plt.tricontourf(lon.flatten(),
                    lat.flatten(),
                    device_unwrapper(final_state["h"].flatten()))
    plt.colorbar()
    plt.savefig(join(fig_dir, "h_final.pdf"))


def test_galewsky():
  npt = 4
  nx = 61
  grid, dims = create_quasi_uniform_grid(nx, npt)

  config = get_physics_config_sw(ne=15)
  test_config = get_galewsky_config(config)

  T = (144 * 3600) / test_division_factor
  u_init = device_wrapper(galewsky_wind(grid["physical_coords"][:, :, :, 0],
                                        grid["physical_coords"][:, :, :, 1],
                                        test_config))
  h_init = device_wrapper(galewsky_h(grid["physical_coords"][:, :, :, 0],
                                     grid["physical_coords"][:, :, :, 1],
                                     test_config))
  hs_init = device_wrapper(galewsky_hs(grid["physical_coords"][:, :, :, 0],
                                       grid["physical_coords"][:, :, :, 1],
                                       test_config))
  init_state = create_state_struct(u_init, h_init, hs_init)
  final_state = simulate_sw(T, nx, init_state, grid, config, dims, diffusion=True)
  mass_init = inner_product(h_init, h_init, grid)
  mass_final = inner_product(final_state["h"], final_state["h"], grid)

  assert (jnp.abs(mass_init - mass_final) / mass_final < 1e-6)
  # assert (not jnp.any(jnp.isnan(final_state["u"])))

  if DEBUG:
    fig_dir = get_figdir()
    makedirs(fig_dir, exist_ok=True)
    lon = device_unwrapper(grid["physical_coords"][:, :, :, 1])
    lat = device_unwrapper(grid["physical_coords"][:, :, :, 0])
    levels = np.arange(-10 + 1e-4, 101, 10)
    vort = project_scalar(manifold_vorticity(final_state["u"], grid, a=config["radius_earth"]), grid, dims)
    plt.figure()
    plt.title(f"U at time {T}s")
    plt.tricontourf(lon.flatten(), lat.flatten(),
                    device_unwrapper(final_state["u"][:, :, :, 0].flatten()), levels=levels)
    plt.colorbar()
    plt.savefig(join(fig_dir, "galewsky_U_final.pdf"))
    plt.figure()
    plt.title(f"V at time {T}s")
    plt.tricontourf(lon.flatten(), lat.flatten(),
                    device_unwrapper(final_state["u"][:, :, :, 1].flatten()))
    plt.colorbar()
    plt.savefig(join(fig_dir, "galewsky_V_final.pdf"))
    plt.figure()
    plt.title(f"h at time {T}s")
    plt.tricontourf(lon.flatten(), lat.flatten(),
                    device_unwrapper(final_state["h"].flatten()))
    plt.colorbar()
    plt.savefig(join(fig_dir, "galewsky_h_final.pdf"))
    plt.figure()
    plt.title(f"vorticity at time {T}s")
    plt.tricontourf(lon.flatten(), lat.flatten(),
                    device_unwrapper(vort.flatten()),
                    vmin=-0.0002, vmax=0.0002)
    plt.colorbar()
    plt.savefig(join(fig_dir, "galewsky_vort_final.pdf"))
