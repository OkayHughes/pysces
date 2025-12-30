from pysces.config import jnp, np, DEBUG, device_unwrapper, device_wrapper
from pysces.shallow_water_models.shallow_water_sphere_model import get_config_sw, create_state_struct, simulate_sw
from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.operations_2d.operators import inner_prod, sphere_vorticity
from pysces.operations_2d.assembly import dss_scalar
from ..context import get_figdir, test_division_factor
from os import makedirs
from os.path import join

if DEBUG:
  import matplotlib.pyplot as plt


def test_sw_model():
  npt = 4
  nx = 15
  grid, dims = create_quasi_uniform_grid(nx, npt)
  config = get_config_sw(alpha=jnp.pi / 4, ne=15)
  u0 = 2.0 * jnp.pi * config["radius_earth"] / (12.0 * 24.0 * 60.0 * 60.0)
  h0 = 2.94e4 / config["gravity"]

  def williamson_tc2_u(lat, lon):
    wind = jnp.stack((u0 * (jnp.cos(lat) * jnp.cos(config["alpha"]) +
                            jnp.cos(lon) * jnp.sin(lat) * jnp.sin(config["alpha"])),
                     -u0 * (jnp.sin(lon) * jnp.sin(config["alpha"]))), axis=-1)
    return wind

  def williamson_tc2_h(lat, lon):
    h = jnp.zeros_like(lat)
    h += h0
    second_factor = (-jnp.cos(lon) * jnp.cos(lat) * jnp.sin(config["alpha"]) +
                     jnp.sin(lat) * jnp.cos(config["alpha"]))**2
    h -= (config["radius_earth"] * config["earth_period"] * u0 + u0**2 / 2.0) / config["gravity"] * second_factor
    return h

  def williamson_tc2_hs(lat, lon):
    return jnp.zeros_like(lat)

  u_init = device_wrapper(williamson_tc2_u(grid["physical_coords"][:, :, :, 0], grid["physical_coords"][:, :, :, 1]))
  h_init = device_wrapper(williamson_tc2_h(grid["physical_coords"][:, :, :, 0], grid["physical_coords"][:, :, :, 1]))
  hs_init = device_wrapper(williamson_tc2_hs(grid["physical_coords"][:, :, :, 0], grid["physical_coords"][:, :, :, 1]))
  print(u_init.dtype)
  init_state = create_state_struct(u_init, h_init, hs_init)

  T = 4000.0
  final_state = simulate_sw(T, nx, init_state, grid, config, dims)
  print(final_state["u"].dtype)

  diff_u = u_init - final_state["u"]
  diff_h = h_init - final_state["h"]
  assert (inner_prod(diff_u[:, :, :, 0], diff_u[:, :, :, 0], grid) < 1e-5)
  assert (inner_prod(diff_u[:, :, :, 1], diff_u[:, :, :, 1], grid) < 1e-5)
  assert (inner_prod(diff_h, diff_h, grid) / jnp.max(h_init) < 1e-5)
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

  config = get_config_sw(ne=15)

  deg = 100
  pts, weights = device_wrapper(np.polynomial.legendre.leggauss(deg))
  pts = (pts + 1.0) / 2.0
  weights /= 2.0
  u_max = 80
  phi0 = np.pi / 7
  phi1 = np.pi / 2 - phi0
  e_norm = np.exp(-4 / (phi1 - phi0)**2)
  a = config["radius_earth"]
  Omega = config["earth_period"]
  h0 = 1e4
  hat_h = 120.0
  alpha = 1.0 / 3.0
  beta = 1.0 / 15.0
  pert_center = np.pi / 4

  def galewsky_u(lat):
    u = jnp.zeros_like(lat)
    mask = jnp.logical_and(lat > phi0, lat < phi1)
    u = jnp.where(mask, u_max / e_norm * jnp.exp(1 / ((lat - phi0) * (lat - phi1))), u)
    return u

  def galewsky_wind(lat, lon):
    u = jnp.stack((galewsky_u(lat),
                   jnp.zeros_like(lat)), axis=-1)
    return u

  def galewsky_h(lat, lon):
    quad_amount = lat + jnp.pi / 2.0
    weights_quad = quad_amount.reshape([*lat.shape, 1]) * weights.reshape((*[1 for _ in lat.shape], deg))
    phi_quad = quad_amount.reshape([*lat.shape, 1]) * pts.reshape((*[1 for _ in lat.shape], deg)) - np.pi / 2
    u_quad = galewsky_u(phi_quad)
    f = 2.0 * Omega * jnp.sin(phi_quad)
    integrand = a * u_quad * (f + jnp.tan(phi_quad) / a * u_quad)
    h = h0 - 1.0 / config["gravity"] * jnp.sum(integrand * weights_quad, axis=-1)
    h_prime = hat_h * jnp.cos(lat) * jnp.exp(-(lon / alpha)**2) * jnp.exp(-((pert_center - lat) / beta)**2)
    return h + h_prime

  def galewsky_hs(lat, lon):
    return jnp.zeros_like(lat)

  T = (144 * 3600) / test_division_factor
  u_init = device_wrapper(galewsky_wind(grid["physical_coords"][:, :, :, 0], grid["physical_coords"][:, :, :, 1]))
  h_init = device_wrapper(galewsky_h(grid["physical_coords"][:, :, :, 0], grid["physical_coords"][:, :, :, 1]))
  hs_init = device_wrapper(galewsky_hs(grid["physical_coords"][:, :, :, 0], grid["physical_coords"][:, :, :, 1]))
  init_state = create_state_struct(u_init, h_init, hs_init)
  final_state = simulate_sw(T, nx, init_state, grid, config, dims, diffusion=True)
  mass_init = inner_prod(h_init, h_init, grid)
  mass_final = inner_prod(final_state["h"], final_state["h"], grid)

  assert (jnp.abs(mass_init - mass_final) / mass_final < 1e-6)
  assert (not jnp.any(jnp.isnan(final_state["u"])))

  if DEBUG:
    fig_dir = get_figdir()
    makedirs(fig_dir, exist_ok=True)
    lon = device_unwrapper(grid["physical_coords"][:, :, :, 1])
    lat = device_unwrapper(grid["physical_coords"][:, :, :, 0])
    levels = np.arange(-10 + 1e-4, 101, 10)
    vort = dss_scalar(sphere_vorticity(final_state["u"], grid, a=config["radius_earth"]), grid, dims)
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
