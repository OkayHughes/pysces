from ..config import use_wrapper, np
from .mesh_definitions import (TOP_EDGE, BOTTOM_EDGE, LEFT_EDGE, RIGHT_EDGE, MAX_VERT_DEGREE)
from .mesh import edge_to_vert, gen_vert_redundancy, gen_gll_redundancy, mesh_to_cart_bilinear, generate_metric_terms
from .equiangular_metric import gen_metric_terms_equiangular
from .cubed_sphere import gen_cube_topo


def sphere_to_cart(latlon):
  lat = latlon[:, :, :, 0]
  lon = latlon[:, :, :, 1]
  cos_lat = np.cos(lat)
  cart = np.stack((cos_lat * np.cos(lon),
                          cos_lat * np.sin(lon),
                          np.sin(lat)), axis=-1)
  return cart
def cart_to_sphere_unit_sphere(xyz):
  latlon = np.stack((np.asin(xyz[:, :, :, 2]),
                     np.atan2(xyz[:, :, :, 1],
                              xyz[:, :, :, 0])), axis=-1)
  return latlon

def gen_metric_terms_unstructured(latlon_corners, npt):
  cart_corners = sphere_to_cart(latlon_corners)
  cart_points_3d, gll_to_cart_jacobian = mesh_to_cart_bilinear(cart_corners, npt)
  norm_cart =  np.linalg.norm(cart_points_3d, axis=-1)[:, :, :, np.newaxis]
  gll_xyz = cart_points_3d / norm_cart
  # 1/‖p‖³ (‖p‖² I − pp⊤)
  cart_to_unit_sphere_jacobian = 1.0/norm_cart**3 * (np.eye(3)[np.newaxis, np.newaxis, np.newaxis, :, :] -
                                                     np.einsum("fijc,fijd->fijcd", cart_points_3d, cart_points_3d))
  x = gll_xyz[:, :, :, 0]
  y = gll_xyz[:, :, :, 1]
  z = gll_xyz[:, :, :, 2]
  gll_latlon = cart_to_sphere_unit_sphere(gll_xyz)
  normsq_2d = x**2 + y**2
  unit_sphere_to_sph_coords_jacobian = np.zeros((latlon_corners.shape[0], npt, npt, 2, 3))
  unit_sphere_to_sph_coords_jacobian[:, :, :, 0, 2] = 1.0 / np.sqrt(1 - z**2)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 0, 0] = -y / normsq_2d
  unit_sphere_to_sph_coords_jacobian[:, :, :, 0, 1] = x / normsq_2d
  cart_to_sphere_jacobian = np.einsum("fijcd,fijsc->fijsd", cart_to_unit_sphere_jacobian, unit_sphere_to_sph_coords_jacobian)

  return gll_latlon, gll_to_cart_jacobian, cart_to_sphere_jacobian




def create_quasi_uniform_grid_unstructured(nx, npt, wrapped=use_wrapper, proc_idx=None):
  face_connectivity, face_mask, face_position, _ = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  gll_position_equi, gll_jacobian = mesh_to_cart_bilinear(face_position, npt)
  cube_redundancy = gen_gll_redundancy(vert_redundancy, npt)
  gll_latlon_equi, _ = gen_metric_terms_equiangular(face_mask, gll_position_equi, npt)
  latlon_corners = np.zeros((gll_latlon_equi.shape[0], 2, 2, 2))
  for i_in in [0, 1]:
    for j_in in [0, 1]:
      latlon_corners[:, i_in, j_in, :] = gll_latlon_equi[:, i_in*(npt-1), j_in*(npt-1), :]

  gll_latlon, gll_to_cart_jacobian, cart_to_sphere_jacobian = gen_metric_terms_unstructured(latlon_corners, npt)

  return generate_metric_terms(gll_latlon, gll_to_cart_jacobian, cart_to_sphere_jacobian, cube_redundancy, npt,
                               wrapped=wrapped, proc_idx=proc_idx)








