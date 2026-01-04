from pysces.config import np, jnp, eps, device_wrapper, device_unwrapper, mpi_rank, mpi_size
from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.operations_2d.assembly import project_scalar
from pysces.operations_2d.se_grid import subset_var
from pysces.operations_2d.operators import sphere_gradient, sphere_divergence, sphere_vorticity, inner_prod
from pysces.mesh_generation.periodic_plane import create_uniform_grid
from pysces.distributed_memory.multiprocessing import project_scalar_triple_mpi, global_sum
from pysces.distributed_memory.processor_decomposition import get_decomp
from ..context import test_npts, seed


def test_vector_identites_sphere():
  for npt in test_npts:
    for nx in [30, 31]:
      grid, dims = create_quasi_uniform_grid(nx, npt, proc_idx=mpi_rank)
      fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
      grad = sphere_gradient(fn, grid)
      grad = jnp.stack(project_scalar_triple_mpi([grad[:, :, :, 0],
                                                  grad[:, :, :, 1]], grid, dims), axis=-1)
      vort = project_scalar_triple_mpi([sphere_vorticity(grad, grid)], grid, dims)[0]

      iprod_vort = inner_prod(vort, vort, grid)
      assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
      v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                     jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)

      grad = sphere_gradient(fn, grid)
      discrete_divergence_thm = (inner_prod(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                                 inner_prod(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                                 inner_prod(fn, sphere_divergence(v, grid), grid))
      assert (jnp.allclose(global_sum(discrete_divergence_thm), jnp.zeros((1,)), atol=eps))


def test_vector_identities_plane():
  for npt in test_npts:
    nx, ny = (31, 33)
    grid, dims = create_uniform_grid(nx, ny, npt, length_x=jnp.pi, length_y=jnp.pi, proc_idx=mpi_rank)
    fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
    grad = sphere_gradient(fn, grid)
    grad = jnp.stack(project_scalar_triple_mpi([grad[:, :, :, 0],
                                                grad[:, :, :, 1]], grid, dims), axis=-1)
    vort = project_scalar_triple_mpi([sphere_vorticity(grad, grid)], grid, dims)[0]

    iprod_vort = inner_prod(vort, vort, grid)
    assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
    v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                  jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)

    grad = sphere_gradient(fn, grid)
    discrete_divergence_thm = (inner_prod(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                               inner_prod(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                               inner_prod(fn, sphere_divergence(v, grid), grid))
    assert (jnp.allclose(global_sum(discrete_divergence_thm), jnp.zeros((1,)), atol=eps))


def test_vector_identites_rand_sphere():
  for npt in test_npts:
    np.random.seed(seed)
    for nx in [30, 31]:
      grid, dims = create_quasi_uniform_grid(nx, npt, proc_idx=mpi_rank)
      for _ in range(10):
        fn = device_wrapper(np.random.normal(scale=10, size=grid["physical_coords"][:, :, :, 0].shape))
        fn = project_scalar_triple_mpi([fn], grid, dims, (1))[0]
        grad = sphere_gradient(fn, grid)
        grad = jnp.stack(project_scalar_triple_mpi([grad[:, :, :, 0],
                                                    grad[:, :, :, 1]], grid, dims), axis=-1)
        vort = project_scalar_triple_mpi([sphere_vorticity(grad, grid)], grid, dims)[0]
        iprod_vort = inner_prod(vort, vort, grid)
        assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
        v = device_wrapper(np.random.normal(scale=1, size=grid["physical_coords"].shape))
        v = jnp.stack(project_scalar_triple_mpi([v[:, :, :, 0], v[:, :, :, 1]], grid, dims), axis=-1)
        div = sphere_divergence(v, grid)
        div = project_scalar_triple_mpi([sphere_divergence(v, grid)], grid, dims, (1,))[0]
        discrete_divergence_thm = (inner_prod(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                                   inner_prod(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                                   inner_prod(fn, div, grid))
        assert (jnp.allclose(global_sum(discrete_divergence_thm), jnp.zeros((1,)), atol=eps))


def test_vector_identites_rand_plane():
  for npt in test_npts:
    np.random.seed(seed)
    nx, ny = (31, 33)
    grid, dims = create_uniform_grid(nx, ny, npt, proc_idx=mpi_rank)
    for _ in range(10):
      fn = device_wrapper(np.random.normal(scale=10, size=grid["physical_coords"][:, :, :, 0].shape))
      fn = project_scalar_triple_mpi([fn], grid, dims, (1,))[0]
      grad = sphere_gradient(fn, grid)
      vort = sphere_vorticity(grad, grid)
      grad = jnp.stack(project_scalar_triple_mpi([grad[:, :, :, 0],
                                                  grad[:, :, :, 1]], grid, dims), axis=-1)
      vort = project_scalar_triple_mpi([sphere_vorticity(grad, grid)], grid, dims)[0]
      iprod_vort = inner_prod(vort, vort, grid)
      assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
      v = device_wrapper(np.random.normal(scale=1, size=grid["physical_coords"].shape))
      v = jnp.stack(project_scalar_triple_mpi([v[:, :, :, 0], v[:, :, :, 1]], grid, dims), axis=-1)
      div = sphere_divergence(v, grid)
      div = project_scalar_triple_mpi([sphere_divergence(v, grid)], grid, dims, (1,))[0]
      discrete_divergence_thm = (inner_prod(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                                 inner_prod(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                                 inner_prod(fn, div, grid))
      assert (jnp.allclose(global_sum(discrete_divergence_thm), jnp.zeros((1,)), atol=eps))


def test_equivalence_rand_sphere():
  for npt in test_npts:
    np.random.seed(seed)
    for nx in [6, 9]:
      grid_total, dims_total = create_quasi_uniform_grid(nx, npt)
      grid, dims = create_quasi_uniform_grid(nx, npt, proc_idx=mpi_rank)
      decomp = get_decomp(dims_total["num_elem"], mpi_size)
      for _ in range(10):
        fn_total = device_wrapper(np.random.normal(scale=10, size=grid_total["physical_coords"][:, :, :, 0].shape))
        u_total = device_wrapper(np.random.normal(scale=10, size=grid_total["physical_coords"][:, :, :, 0].shape))
        v_total = device_wrapper(np.random.normal(scale=10, size=grid_total["physical_coords"][:, :, :, 0].shape))
        vec_total = jnp.stack((project_scalar(u_total, grid_total, dims_total),
                               project_scalar(v_total, grid_total, dims_total)), axis=-1)
        fn_total = project_scalar(fn_total, grid_total, dims_total)
        fn = subset_var(fn_total, mpi_rank, decomp)
        u = subset_var(u_total, mpi_rank, decomp)
        v = subset_var(v_total, mpi_rank, decomp)
        fn = project_scalar_triple_mpi([fn], grid, dims)[0]

        vec = jnp.stack(project_scalar_triple_mpi([u, v], grid, dims), axis=-1)
        grad = sphere_gradient(fn, grid)
        vort = sphere_vorticity(vec, grid)
        div = sphere_divergence(vec, grid)
        grad_total = sphere_gradient(fn_total, grid_total)
        vort_total = sphere_vorticity(vec_total, grid_total)
        div_total = sphere_divergence(vec_total, grid_total)
        assert np.allclose(subset_var(vec_total, mpi_rank, decomp), vec)

        assert np.allclose(subset_var(grad_total, mpi_rank, decomp), grad)

        assert np.allclose(subset_var(vort_total, mpi_rank, decomp), vort)

        assert np.allclose(subset_var(div_total, mpi_rank, decomp), div)
