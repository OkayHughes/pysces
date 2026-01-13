from pysces.config import np, jnp, eps, device_wrapper, device_unwrapper, mpi_rank, mpi_size
from pysces.mesh_generation.equiangular_metric import create_quasi_uniform_grid
from pysces.operations_2d.local_assembly import project_scalar
from pysces.operations_2d.se_grid import subset_var
from pysces.operations_2d.operators import manifold_gradient, manifold_divergence, manifold_vorticity, inner_product
from pysces.mesh_generation.periodic_plane import create_uniform_grid
from pysces.distributed_memory.global_assembly import project_scalar_global
from pysces.distributed_memory.global_operations import global_sum
from pysces.distributed_memory.processor_decomposition import get_decomp
from ..context import test_npts, seed


def test_vector_identites_sphere():
  for npt in test_npts:
    for nx in [30, 31]:
      grid, dims = create_quasi_uniform_grid(nx, npt, proc_idx=mpi_rank)
      fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
      grad = manifold_gradient(fn, grid)
      grad = jnp.stack(project_scalar_global([grad[:, :, :, 0],
                                                  grad[:, :, :, 1]], grid, dims), axis=-1)
      vort = project_scalar_global([manifold_vorticity(grad, grid)], grid, dims)[0]

      iprod_vort = inner_product(vort, vort, grid)
      assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
      v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                     jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)

      grad = manifold_gradient(fn, grid)
      discrete_divergence_thm = (inner_product(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                                 inner_product(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                                 inner_product(fn, manifold_divergence(v, grid), grid))
      assert (np.allclose(global_sum(discrete_divergence_thm), jnp.zeros((1,)), atol=eps))


def test_vector_identities_plane():
  for npt in test_npts:
    nx, ny = (31, 33)
    grid, dims = create_uniform_grid(nx, ny, npt, length_x=jnp.pi, length_y=jnp.pi, proc_idx=mpi_rank)
    fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
    grad = manifold_gradient(fn, grid)
    grad = jnp.stack(project_scalar_global([grad[:, :, :, 0],
                                                grad[:, :, :, 1]], grid, dims), axis=-1)
    vort = project_scalar_global([manifold_vorticity(grad, grid)], grid, dims)[0]

    iprod_vort = inner_product(vort, vort, grid)
    assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
    v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                  jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)

    grad = manifold_gradient(fn, grid)
    discrete_divergence_thm = (inner_product(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                               inner_product(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                               inner_product(fn, manifold_divergence(v, grid), grid))
    assert (np.allclose(global_sum(discrete_divergence_thm), jnp.zeros((1,)), atol=eps))


def test_vector_identites_rand_sphere():
  for npt in test_npts:
    np.random.seed(seed)
    for nx in [30, 31]:
      grid, dims = create_quasi_uniform_grid(nx, npt, proc_idx=mpi_rank)
      for _ in range(10):
        fn = device_wrapper(np.random.normal(scale=10, size=grid["physical_coords"][:, :, :, 0].shape))
        fn = project_scalar_global([fn], grid, dims, (1))[0]
        grad = manifold_gradient(fn, grid)
        grad = jnp.stack(project_scalar_global([grad[:, :, :, 0],
                                                    grad[:, :, :, 1]], grid, dims), axis=-1)
        vort = project_scalar_global([manifold_vorticity(grad, grid)], grid, dims)[0]
        iprod_vort = inner_product(vort, vort, grid)
        assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
        v = device_wrapper(np.random.normal(scale=1, size=grid["physical_coords"].shape))
        v = jnp.stack(project_scalar_global([v[:, :, :, 0], v[:, :, :, 1]], grid, dims), axis=-1)
        div = manifold_divergence(v, grid)
        div = project_scalar_global([manifold_divergence(v, grid)], grid, dims, (1,))[0]
        discrete_divergence_thm = (inner_product(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                                   inner_product(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                                   inner_product(fn, div, grid))
        assert (np.allclose(global_sum(discrete_divergence_thm), jnp.zeros((1,)), atol=eps))


def test_vector_identites_rand_plane():
  for npt in test_npts:
    np.random.seed(seed)
    nx, ny = (31, 33)
    grid, dims = create_uniform_grid(nx, ny, npt, proc_idx=mpi_rank)
    for _ in range(10):
      fn = device_wrapper(np.random.normal(scale=10, size=grid["physical_coords"][:, :, :, 0].shape))
      fn = project_scalar_global([fn], grid, dims, (1,))[0]
      grad = manifold_gradient(fn, grid)
      vort = manifold_vorticity(grad, grid)
      grad = jnp.stack(project_scalar_global([grad[:, :, :, 0],
                                                  grad[:, :, :, 1]], grid, dims), axis=-1)
      vort = project_scalar_global([manifold_vorticity(grad, grid)], grid, dims)[0]
      iprod_vort = inner_product(vort, vort, grid)
      assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
      v = device_wrapper(np.random.normal(scale=1, size=grid["physical_coords"].shape))
      v = jnp.stack(project_scalar_global([v[:, :, :, 0], v[:, :, :, 1]], grid, dims), axis=-1)
      div = manifold_divergence(v, grid)
      div = project_scalar_global([manifold_divergence(v, grid)], grid, dims, (1,))[0]
      discrete_divergence_thm = (inner_product(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                                 inner_product(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                                 inner_product(fn, div, grid))
      assert (np.allclose(global_sum(discrete_divergence_thm), jnp.zeros((1,)), atol=eps))


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
        fn = project_scalar_global([fn], grid, dims)[0]

        vec = jnp.stack(project_scalar_global([u, v], grid, dims), axis=-1)
        grad = manifold_gradient(fn, grid)
        vort = manifold_vorticity(vec, grid)
        div = manifold_divergence(vec, grid)
        grad_total = manifold_gradient(fn_total, grid_total)
        vort_total = manifold_vorticity(vec_total, grid_total)
        div_total = manifold_divergence(vec_total, grid_total)
        assert np.allclose(subset_var(vec_total, mpi_rank, decomp), vec)

        assert np.allclose(subset_var(grad_total, mpi_rank, decomp), grad)

        assert np.allclose(subset_var(vort_total, mpi_rank, decomp), vort)

        assert np.allclose(subset_var(div_total, mpi_rank, decomp), div)
