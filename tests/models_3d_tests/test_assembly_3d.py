from pysces.config import np, use_wrapper, device_wrapper, device_unwrapper
from pysces.mesh_generation.cubed_sphere import gen_cube_topo, gen_vert_redundancy
from pysces.mesh_generation.equiangular_metric import gen_metric_from_topo
from pysces.models_3d.theta_l.model_state import project_scalar_3d, project_scalar_3d_for
from pysces.mesh_generation.mesh import vert_red_flat_to_hierarchy

def test_project_3d():
  npt = 4
  nx = 3
  nlev = 3
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity,
                                    face_mask,
                                    face_position_2d,
                                    vert_redundancy,
                                    npt,
                                    wrapped=use_wrapper)
  grid_nowrapper, _ = gen_metric_from_topo(face_connectivity,
                                           face_mask,
                                           face_position_2d,
                                           vert_redundancy,
                                           npt,
                                           wrapped=False)
  vert_redundancy_gll = vert_red_flat_to_hierarchy(grid_nowrapper["vert_redundancy"])
  fn = np.zeros((*grid["physical_coords"].shape[:-1], nlev))
  for lev_idx in range(nlev):
    for face_idx in range(grid["physical_coords"].shape[0]):
      for i_idx in range(npt):
        for j_idx in range(npt):
          fn[:] = 0.0
          fn[face_idx, i_idx, j_idx, lev_idx] = 1.0
          if face_idx in vert_redundancy_gll.keys():
            if (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
              for remote_face_id, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
                fn[remote_face_id, remote_i, remote_j, lev_idx] = 1.0
            assert (np.allclose(device_unwrapper(project_scalar_3d(device_wrapper(fn), grid, dims)), fn))


def test_project_equiv_3d_rand():
  npt = 4
  nx = 15
  nlev = 5
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  grid, dims = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d,
                                    vert_redundancy, npt, wrapped=False)
  grid_wrapped, dims_wrapped = gen_metric_from_topo(face_connectivity, face_mask, face_position_2d,
                                                    vert_redundancy, npt, wrapped=use_wrapper)
  for _ in range(20):
    fn_rand = np.random.uniform(size=(*grid["physical_coords"][:, :, :, 1].shape, nlev))
    assert (np.allclose(device_unwrapper(project_scalar_3d(device_wrapper(fn_rand), grid_wrapped, dims_wrapped)),
                        project_scalar_3d_for(fn_rand, grid, dims)))
