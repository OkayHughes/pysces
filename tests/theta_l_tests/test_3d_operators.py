
from pysces.config import np, jnp, eps, device_wrapper, device_unwrapper
from pysces.equiangular_metric import create_quasi_uniform_grid
from pysces.operators import inner_prod
from pysces.theta_l.operators_3d import (sphere_divergence_3d,
                                         sphere_gradient_3d,
                                         sphere_vorticity_3d,
                                         sphere_vec_laplacian_wk_3d)
from pysces.theta_l.model_state import dss_scalar_3d


def threedify(field, nlev, axis=-1):
  if axis < 0:
    shape_out = list(field.shape)
    shape_out.reverse()
    shape_out.insert(-axis - 1, 1)
    shape_out.reverse()
  else:
    shape_out = field.shape
    shape_out.insert(axis, 1)
  ones_shape = [1 for _ in range(len(field.shape) + 1)]
  ones_shape[axis] = nlev
  return field.reshape(shape_out) * jnp.ones(ones_shape)


def test_vector_identites():
  npt = 4
  nx = 31
  nlev = 3
  grid, dims = create_quasi_uniform_grid(nx, npt)
  config = {"radius_earth": 1.0}
  fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
  fn_3d = device_wrapper(threedify(fn, nlev) * jnp.arange(nlev).reshape((1, 1, 1, -1)))
  v = np.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)
  v_3d = threedify(device_wrapper(v), nlev, axis=-2)

  grad = sphere_gradient_3d(fn_3d, grid, config)
  vort = sphere_vorticity_3d(grad, grid, config)
  div = sphere_divergence_3d(v_3d, grid, config)
  for k_idx in range(nlev):
    iprod_vort = inner_prod(vort[:, :, :, k_idx], vort[:, :, :, k_idx], grid)
    assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))

    discrete_divergence_thm = (inner_prod(v_3d[:, :, :, k_idx, 0], grad[:, :, :, k_idx, 0], grid) +
                               inner_prod(v_3d[:, :, :, k_idx, 1], grad[:, :, :, k_idx, 1], grid) -
                               inner_prod(fn_3d[:, :, :, k_idx], div[:, :, :, k_idx], grid))
    assert (jnp.allclose(discrete_divergence_thm, jnp.zeros_like(discrete_divergence_thm), atol=eps))


def test_divergence():
  npt = 4
  nx = 31
  nlev = 3
  grid, dims = create_quasi_uniform_grid(nx, npt)
  config = {"radius_earth": 1.0}
  vec = np.zeros_like(grid["physical_coords"])
  lat = grid["physical_coords"][:, :, :, 0]
  lon = grid["physical_coords"][:, :, :, 1]
  vec[:, :, :, 0] = np.cos(lat)**2 * np.cos(lon)**3
  vec[:, :, :, 1] = np.cos(lat)**2 * np.cos(lon)**3

  vec_3d = threedify(device_wrapper(vec), nlev, axis=-2)

  vort_analytic = (-3.0 * np.cos(lon)**2 * np.sin(lon) * np.cos(lat) +
                   3.0 * np.cos(lat) * np.sin(lat) * np.cos(lon)**3)

  div_analytic = (-3.0 * np.cos(lon)**2 * np.sin(lon) * np.cos(lat) -
                  3.0 * np.cos(lat) * np.sin(lat) * np.cos(lon)**3)
  div = dss_scalar_3d(sphere_divergence_3d(vec_3d, grid, config), grid, dims)
  vort = dss_scalar_3d(sphere_vorticity_3d(vec_3d, grid, config), grid, dims)
  for lev_idx in range(nlev):
    assert (inner_prod(div_analytic - div[:, :, :, lev_idx],
                       div_analytic - div[:, :, :, lev_idx], grid) < 1e-5)
    assert (inner_prod(vort_analytic - vort[:, :, :, lev_idx],
                       vort_analytic - vort[:, :, :, lev_idx], grid) < 1e-5)


def test_analytic_soln():
  npt = 4
  nx = 31
  nlev = 3
  grid, dims = create_quasi_uniform_grid(nx, npt)
  config = {"radius_earth": 1.0}

  fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
  fn_3d = threedify(device_wrapper(fn), nlev)
  grad_f_numerical = sphere_gradient_3d(fn_3d, grid, config)

  sph_grad_lat = -jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.sin(grid["physical_coords"][:, :, :, 0])
  sph_grad_lon = -jnp.sin(grid["physical_coords"][:, :, :, 1])
  for lev_idx in range(nlev):
    assert (np.max(np.abs(device_unwrapper(sph_grad_lat - grad_f_numerical[:, :, :, lev_idx, 1]))) < 1e-4)
    assert (np.max(np.abs(device_unwrapper(sph_grad_lon - grad_f_numerical[:, :, :, lev_idx, 0]))) < 1e-4)


def test_vector_laplacian():
  npt = 4
  nx = 31
  nlev = 3
  grid, dims = create_quasi_uniform_grid(nx, npt)
  vec = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                   jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)
  config = {"radius_earth": 1.0}
  vec_3d = threedify(vec, nlev, axis=-2)
  laplace_v_wk = sphere_vec_laplacian_wk_3d(vec_3d, grid, config)
  laplace_v_wk = jnp.stack((dss_scalar_3d(laplace_v_wk[:, :, :, :, 0], grid, dims, scaled=False),
                            dss_scalar_3d(laplace_v_wk[:, :, :, :, 1], grid, dims, scaled=False)), axis=-1)

  lap_diff = laplace_v_wk + 2 * vec_3d
  for lev_idx in range(nlev):
    assert ((inner_prod(lap_diff[:, :, :, lev_idx, 0], lap_diff[:, :, :, lev_idx, 0], grid) +
             inner_prod(lap_diff[:, :, :, lev_idx, 1], lap_diff[:, :, :, lev_idx, 1], grid)) < 1e-5)
  vec = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0])**2,
                   jnp.cos(grid["physical_coords"][:, :, :, 0])**2), axis=-1)
  vec_3d = threedify(vec, nlev, axis=-2)
  laplace_v_wk = sphere_vec_laplacian_wk_3d(vec_3d, grid, config)
  laplace_v_wk = jnp.stack((dss_scalar_3d(laplace_v_wk[:, :, :, :, 0], grid, dims, scaled=False),
                            dss_scalar_3d(laplace_v_wk[:, :, :, :, 1], grid, dims, scaled=False)), axis=-1)
  lap_diff = laplace_v_wk + 3 * (np.cos(2 * grid["physical_coords"][:, :, :, 0]))[:, :, :, np.newaxis, np.newaxis]
  # hack to negate pole point
  lap_diff *= np.cos(grid["physical_coords"][:, :, :, 0])[:, :, :, np.newaxis, np.newaxis]**2
  for lev_idx in range(nlev):
    assert ((inner_prod(lap_diff[:, :, :, lev_idx, 0], lap_diff[:, :, :, lev_idx, 0], grid) +
             inner_prod(lap_diff[:, :, :, lev_idx, 1], lap_diff[:, :, :, lev_idx, 1], grid)) < 1e-5)
