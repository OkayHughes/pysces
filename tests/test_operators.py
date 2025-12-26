from pysces.config import np, jnp, eps, device_wrapper, device_unwrapper
from pysces.equiangular_metric import create_quasi_uniform_grid
from pysces.assembly import dss_scalar
from pysces.operators import sphere_gradient, sphere_divergence, sphere_vorticity, inner_prod
from pysces.operators import sphere_divergence_wk, sphere_gradient_wk_cov, sphere_vec_laplacian_wk
from pysces.periodic_plane import create_uniform_grid

def test_vector_identites_sphere():
  nx = 31
  grid, dims = create_quasi_uniform_grid(nx)
  fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
  grad = sphere_gradient(fn, grid)
  vort = sphere_vorticity(grad, grid)

  iprod_vort = inner_prod(vort, vort, grid)
  assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
  v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                 jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)

  grad = sphere_gradient(fn, grid)
  discrete_divergence_thm = (inner_prod(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                             inner_prod(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                             inner_prod(fn, sphere_divergence(v, grid), grid))
  assert (jnp.allclose(discrete_divergence_thm, jnp.zeros_like(discrete_divergence_thm), atol=eps))

def test_vector_identities_plane():
  nx, ny = (31, 33)
  grid, dims = create_uniform_grid(nx, ny, length_x=jnp.pi, length_y=jnp.pi)
  fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
  grad = sphere_gradient(fn, grid)
  vort = sphere_vorticity(grad, grid)

  iprod_vort = inner_prod(vort, vort, grid)
  assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
  v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                 jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)

  grad = sphere_gradient(fn, grid)
  discrete_divergence_thm = (inner_prod(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                             inner_prod(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                             inner_prod(fn, sphere_divergence(v, grid), grid))
  assert (jnp.allclose(discrete_divergence_thm, jnp.zeros_like(discrete_divergence_thm), atol=eps))


def test_vector_identites_rand():
  np.random.seed(0)
  nx = 31
  grid, dims = create_quasi_uniform_grid(nx)
  for _ in range(10):
    fn = device_wrapper(np.random.normal(scale=10, size=grid["physical_coords"][:, :, :, 0].shape))
    fn = dss_scalar(fn, grid, dims)
    grad = sphere_gradient(fn, grid)
    vort = sphere_vorticity(grad, grid)
    grad = jnp.stack((dss_scalar(grad[:, :, :, 0], grid, dims),
                      dss_scalar(grad[:, :, :, 1], grid, dims)), axis=-1)
    vort = dss_scalar(vort, grid, dims)
    iprod_vort = inner_prod(vort, vort, grid)
    assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
    v = device_wrapper(np.random.normal(scale=1, size=grid["physical_coords"].shape))
    v = jnp.stack((dss_scalar(v[:, :, :, 0], grid, dims),
                   dss_scalar(v[:, :, :, 1], grid, dims)), axis=-1)
    div = sphere_divergence(v, grid)
    div = dss_scalar(sphere_divergence(v, grid), grid, dims)
    discrete_divergence_thm = (inner_prod(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                               inner_prod(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                               inner_prod(fn, div, grid))
    assert (jnp.allclose(discrete_divergence_thm, jnp.zeros_like(discrete_divergence_thm), atol=eps))


def test_divergence():
  nx = 61
  grid, dims = create_quasi_uniform_grid(nx)
  vec = np.zeros_like(grid["physical_coords"])
  lat = grid["physical_coords"][:, :, :, 0]
  lon = grid["physical_coords"][:, :, :, 1]
  vec[:, :, :, 0] = np.cos(lat)**2 * np.cos(lon)**3
  vec[:, :, :, 1] = np.cos(lat)**2 * np.cos(lon)**3
  vec = device_wrapper(vec)

  vort_analytic = device_wrapper((-3.0 * np.cos(lon)**2 * np.sin(lon) * np.cos(lat) +
                                  3.0 * np.cos(lat) * np.sin(lat) * np.cos(lon)**3))

  div_analytic = device_wrapper((-3.0 * np.cos(lon)**2 * np.sin(lon) * np.cos(lat) -
                                 3.0 * np.cos(lat) * np.sin(lat) * np.cos(lon)**3))
  div = dss_scalar(sphere_divergence(vec, grid), grid, dims)
  div_wk = dss_scalar(sphere_divergence_wk(vec, grid), grid, dims, scaled=False)
  vort = dss_scalar(sphere_vorticity(vec, grid), grid, dims)
  assert (inner_prod(div_wk - div, div_wk - div, grid) < 1e-6)
  assert (inner_prod(div_analytic - div, div_analytic - div, grid) < 1e-6)
  assert (inner_prod(vort_analytic - vort, vort_analytic - vort, grid) < 1e-6)


def test_analytic_soln():
  nx = 61
  grid, dims = create_quasi_uniform_grid(nx)
  fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
  grad_f_numerical = sphere_gradient(fn, grid)
  sph_grad_wk = sphere_gradient_wk_cov(fn, grid)
  sph_grad_wk = jnp.stack((dss_scalar(sph_grad_wk[:, :, :, 0], grid, dims, scaled=False),
                           dss_scalar(sph_grad_wk[:, :, :, 1], grid, dims, scaled=False)), axis=-1)
  grad_diff = sph_grad_wk - grad_f_numerical

  sph_grad_lat = -jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.sin(grid["physical_coords"][:, :, :, 0])
  sph_grad_lon = -jnp.sin(grid["physical_coords"][:, :, :, 1])
  assert ((inner_prod(grad_diff[:, :, :, 0], grad_diff[:, :, :, 0], grid) +
           inner_prod(grad_diff[:, :, :, 1], grad_diff[:, :, :, 1], grid)) < 1e-5)
  assert (jnp.max(jnp.abs(sph_grad_lat - grad_f_numerical[:, :, :, 1])) < 1e-5)
  assert (jnp.max(jnp.abs(sph_grad_lon - grad_f_numerical[:, :, :, 0])) < 1e-5)


def test_vector_laplacian():
  nx = 61
  grid, dims = create_quasi_uniform_grid(nx)
  v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                 jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)
  laplace_v_wk = sphere_vec_laplacian_wk(v, grid)
  laplace_v_wk = jnp.stack((dss_scalar(laplace_v_wk[:, :, :, 0], grid, dims, scaled=False),
                            dss_scalar(laplace_v_wk[:, :, :, 1], grid, dims, scaled=False)), axis=-1)

  lap_diff = laplace_v_wk + 2 * v
  assert ((inner_prod(lap_diff[:, :, :, 0], lap_diff[:, :, :, 0], grid) +
           inner_prod(lap_diff[:, :, :, 1], lap_diff[:, :, :, 1], grid)) < 1e-5)
  v = jnp.stack((np.cos(grid["physical_coords"][:, :, :, 0])**2,
                 np.cos(grid["physical_coords"][:, :, :, 0])**2), axis=-1)
  laplace_v_wk = sphere_vec_laplacian_wk(v, grid)
  laplace_v_wk = jnp.stack((dss_scalar(laplace_v_wk[:, :, :, 0], grid, dims, scaled=False),
                            dss_scalar(laplace_v_wk[:, :, :, 1], grid, dims, scaled=False)), axis=-1)
  lap_diff = laplace_v_wk + 3.0 * (np.cos(2 * grid["physical_coords"][:, :, :, 0]))[:, :, :, np.newaxis]
  lap_diff *= np.cos(grid["physical_coords"][:, :, :, 0])[:, :, :, np.newaxis]**2  # hack to negate pole point

  assert ((inner_prod(lap_diff[:, :, :, 0], lap_diff[:, :, :, 0], grid) +
           inner_prod(lap_diff[:, :, :, 1], lap_diff[:, :, :, 1], grid)) < 1e-5)
