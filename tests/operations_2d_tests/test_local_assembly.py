from pysces.config import np, device_wrapper, use_wrapper, jnp, get_global_array
from pysces.mesh_generation.cubed_sphere import init_cube_topo
from pysces.mesh_generation.mesh import init_element_corner_vert_redundancy
from pysces.horizontal_grid import shard_grid
from pysces.mesh_generation.equiangular_metric import init_grid_from_topo
from pysces.mesh_generation.mesh import vert_red_flat_to_hierarchy
from pysces.operations_2d.local_assembly import (project_scalar_wrapper,
                                                 project_scalar)
from ..context import test_npts


def project_scalar_for(f,
                       grid,
                       *args):
  """
  Project a potentially discontinuous scalar onto the continuous
  subspace using a for loop, assuming all data is processor-local.

  *This is used for testing. Do not use in performance code*

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Scalar field to project
  grid : `SpectralElementGrid`
    Spectral element grid struct that contains coordinate and metric data.

  Notes
  -----
  This is the most human-readable way to perform projection, and can be used to test
  if your grid is topologically malformed.

  Returns
  -------
  f_cont
      The globally continous scalar closest in norm to f.
  """
  # assumes that values from remote processors have already been accumulated
  metdet = grid["metric_determinant"]
  inv_mass_mat = grid["mass_matrix_denominator"]
  vert_redundancy_gll = grid["vertex_redundancy"]
  gll_weights = grid["gll_weights"]
  workspace = f.copy()
  workspace *= metdet * (gll_weights[np.newaxis, :, np.newaxis] * gll_weights[np.newaxis, np.newaxis, :])
  for ((local_face_idx, local_i, local_j),
       (remote_face_id, remote_i, remote_j)) in vert_redundancy_gll:
    workspace[remote_face_id, remote_i, remote_j] += (metdet[local_face_idx, local_i, local_j] *
                                                      f[local_face_idx, local_i, local_j] *
                                                      (gll_weights[local_i] * gll_weights[local_j]))
  # this line works even for multi-processor decompositions.
  workspace *= inv_mass_mat
  return workspace


def test_projection():
  for npt in test_npts:
    for nx in [3, 4]:
      face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
      vert_redundancy = init_element_corner_vert_redundancy(nx, face_connectivity, face_position)
      raw_grid, dims = init_grid_from_topo(face_connectivity,
                                           face_mask,
                                           face_position_2d,
                                           vert_redundancy,
                                           npt,
                                           wrapped=use_wrapper)
      grid = shard_grid(raw_grid, dims)
      grid_nowrapper, _ = init_grid_from_topo(face_connectivity,
                                              face_mask,
                                              face_position_2d,
                                              vert_redundancy,
                                              npt,
                                              wrapped=False)

      def add_one_point(field, f, i, j):
        axis_1 = jnp.eye(field.shape[0])[f, :].squeeze()
        axis_2 = jnp.eye(field.shape[1])[i, :].squeeze()
        axis_3 = jnp.eye(field.shape[2])[j, :].squeeze()
        return (axis_1[:, jnp.newaxis, jnp.newaxis] *
                axis_2[jnp.newaxis, :, jnp.newaxis] *
                axis_3[jnp.newaxis, jnp.newaxis, :])

      vert_redundancy_gll = vert_red_flat_to_hierarchy(grid_nowrapper["vertex_redundancy"])
      for face_idx in range(grid["physical_coords"].shape[0]):
        for i_idx in range(npt):
          for j_idx in range(npt):
            fn = jnp.zeros_like(grid["physical_coords"][:, :, :, 0])
            fn += add_one_point(fn, face_idx, i_idx, j_idx)
            if face_idx in vert_redundancy_gll.keys():
              if (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
                for remote_face_id, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
                  fn += add_one_point(fn, remote_face_id, remote_i, remote_j)
            cont_fn = project_scalar(fn, grid, dims)
            assert np.allclose(get_global_array(cont_fn, dims), get_global_array(fn, dims))


def test_projection_equiv():
  for npt in test_npts:
    for nx in [7, 8]:
      face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
      vert_redundancy = init_element_corner_vert_redundancy(nx, face_connectivity, face_position)
      raw_grid, dims = init_grid_from_topo(face_connectivity,
                                           face_mask,
                                           face_position_2d,
                                           vert_redundancy,
                                           npt, wrapped=False)
      grid_wrapped, dims_wrapped = init_grid_from_topo(face_connectivity,
                                                       face_mask,
                                                       face_position_2d,
                                                       vert_redundancy,
                                                       npt, wrapped=use_wrapper)
      grid_wrapped = shard_grid(grid_wrapped, dims_wrapped)
      fn = device_wrapper(jnp.cos(grid_wrapped["physical_coords"][:, :, :, 1]) *
                          jnp.cos(grid_wrapped["physical_coords"][:, :, :, 0]))
      assert (np.allclose(get_global_array(project_scalar(fn, grid_wrapped, dims_wrapped), dims_wrapped),
                          get_global_array(fn, dims_wrapped)))
      ones = jnp.ones_like(grid_wrapped["metric_determinant"])
      ones_out = project_scalar(device_wrapper(ones), grid_wrapped, dims_wrapped)
      assert (np.allclose(get_global_array(ones_out, dims_wrapped), get_global_array(ones, dims_wrapped)))
      ones_out_for = project_scalar_for(get_global_array(ones, dims_wrapped), raw_grid)
      assert (np.allclose(ones_out_for, get_global_array(ones, dims_wrapped)))


def test_projection_equiv_rand():
  for npt in test_npts:
    for nx in [7, 8]:
      face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
      vert_redundancy = init_element_corner_vert_redundancy(nx, face_connectivity, face_position)
      raw_grid, dims = init_grid_from_topo(face_connectivity,
                                           face_mask,
                                           face_position_2d,
                                           vert_redundancy,
                                           npt, wrapped=False)
      grid_wrapped, dims_wrapped = init_grid_from_topo(face_connectivity,
                                                       face_mask,
                                                       face_position_2d,
                                                       vert_redundancy,
                                                       npt,
                                                       wrapped=use_wrapper)
      grid_wrapped = shard_grid(grid_wrapped, dims_wrapped)
      for _ in range(20):
        fn_rand = np.random.uniform(size=grid_wrapped["physical_coords"][:, :, :, 1].shape)
        assert np.allclose(get_global_array(project_scalar_wrapper(device_wrapper(fn_rand, elem_sharding_axis=0),
                                                                   grid_wrapped,
                                                                   dims_wrapped),
                                            dims_wrapped),
                           project_scalar_for(fn_rand[:dims["num_elem"], :, :], raw_grid))
