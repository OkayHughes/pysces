from pysces.config import jnp, np, DEBUG, device_unwrapper, device_wrapper
from pysces.shallow_water_models.shallow_water_sphere_model import get_config_sw, create_state_struct, simulate_sw
from pysces.shallow_water_models.williamson_init import (get_williamson_steady_config,
                                                         williamson_tc2_h,
                                                         williamson_tc2_hs,
                                                         williamson_tc2_u)
from pysces.shallow_water_models.galewsky_init import get_galewsky_config, galewsky_wind, galewsky_hs, galewsky_h
from pysces.mesh_generation.element_local_metric import create_quasi_uniform_grid_elem_local
from pysces.operations_2d.operators import inner_prod, sphere_vorticity
from pysces.operations_2d.local_assembly import project_scalar
from ..context import get_figdir, test_division_factor
from os import makedirs
from os.path import join

if DEBUG:
  import matplotlib.pyplot as plt


def test_galewsky():
  npt = 4
  nx = 31
  grid, dims = create_quasi_uniform_grid_elem_local(nx, npt)

  config = get_config_sw(ne=15)
  test_config = get_galewsky_config(config)

  T = (96 * 3600) # four days
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
  mass_init = inner_prod(h_init, h_init, grid)
  mass_final = inner_prod(final_state["h"], final_state["h"], grid)

  if DEBUG:
    fig_dir = get_figdir()
    makedirs(fig_dir, exist_ok=True)
    lon = device_unwrapper(grid["physical_coords"][:, :, :, 1])
    lat = device_unwrapper(grid["physical_coords"][:, :, :, 0])
    levels = np.arange(-10 + 1e-4, 101, 10)
    vort = project_scalar(sphere_vorticity(final_state["u"], grid, a=config["radius_earth"]), grid, dims)
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
