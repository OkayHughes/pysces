from ..config import use_wrapper, np
from .mesh_definitions import (TOP_EDGE, BOTTOM_EDGE, LEFT_EDGE, RIGHT_EDGE, MAX_VERT_DEGREE)
from .mesh import edge_to_vert, gen_vert_redundancy, gen_gll_redundancy, mesh_to_cart_bilinear, generate_metric_terms
from .equiangular_metric import gen_metric_terms_equiangular
from .cubed_sphere import gen_cube_topo


def sphere_to_cart(latlon):
  lat = np.take(latlon, 0, axis=-1)
  lon = np.take(latlon, 1, axis=-1)
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


def gen_metric_terms_unstructured(latlon_corners, npt, rotate=False):
  cart_corners = sphere_to_cart(latlon_corners)
  print(cart_corners.shape)
  if rotate:
    theta = 1e-3

    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, np.cos(theta), -np.sin(theta)],
                                (0.0, np.sin(theta), np.cos(theta))])
    cart_corners = np.einsum("kl, fvk->fvl", rotation_matrix, cart_corners)
  cart_points_3d, gll_to_cart_jacobian = mesh_to_cart_bilinear(cart_corners, npt)
  norm_cart =  np.linalg.norm(cart_points_3d, axis=-1)
  gll_xyz = cart_points_3d / norm_cart[:, :, :, np.newaxis]
  # 1/‖p‖³ (‖p‖² I − pp⊤)
  cart_to_unit_sphere_jacobian = 1.0/norm_cart[:, :, :, np.newaxis, np.newaxis]**3 * (norm_cart[:, :, :, np.newaxis, np.newaxis]**2 * np.eye(3)[np.newaxis, np.newaxis, np.newaxis, :, :] -
                                                                                      np.einsum("fijc,fijd->fijcd", cart_points_3d, cart_points_3d))
  x = gll_xyz[:, :, :, 0]
  y = gll_xyz[:, :, :, 1]
  z = gll_xyz[:, :, :, 2]
  gll_latlon = cart_to_sphere_unit_sphere(gll_xyz)
  normsq_2d = x**2 + y**2
  unit_sphere_to_sph_coords_jacobian = np.zeros((latlon_corners.shape[0], npt, npt, 2, 3))
  unit_sphere_to_sph_coords_jacobian[:, :, :, 0, 2] = 1.0 / np.sqrt(1 - z**2)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 1, 0] = -y / normsq_2d
  unit_sphere_to_sph_coords_jacobian[:, :, :, 1, 1] = x / normsq_2d
  # unit_sphere_to_sph_coords_jacobian[:, :, :, 0, 2][np.abs(z) > 1-1e-8] = 0.0
  # unit_sphere_to_sph_coords_jacobian[:, :, :, 1, 0][normsq_2d < 1e-8] = -1.0
  # unit_sphere_to_sph_coords_jacobian[:, :, :, 1, 1][normsq_2d < 1e-8] = 1.0

  cart_to_sphere_jacobian = np.einsum("fijcd,fijsc->fijds", cart_to_unit_sphere_jacobian, unit_sphere_to_sph_coords_jacobian)
  #cart_to_sphere_jacobian = np.einsum("fijdc, fijsd->fijcs", cart_to_unit_sphere_jacobian, sphere_to_sphere_jacobian)
  # gll_latlon[:, :, :, 1] = np.mod(gll_latlon[:, :, :, 1], 2 * np.pi - 1e-9)
  # too_close_to_top = np.abs(gll_latlon[:, :, :, 0] - np.pi / 2) < 1e-8
  # too_close_to_bottom = np.abs(gll_latlon[:, :, :, 0] + np.pi / 2) < 1e-8
  # mask = np.logical_or(too_close_to_top,
  #                      too_close_to_bottom)
  # gll_latlon[:, :, :, 1] = np.where(mask, 0.0, gll_latlon[:, :, :, 1])
  return gll_latlon, gll_to_cart_jacobian, cart_to_sphere_jacobian


def create_quasi_uniform_grid_unstructured(nx, npt, wrapped=use_wrapper, proc_idx=None, rotate=True):
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  gll_position_equi, gll_jacobian = mesh_to_cart_bilinear(face_position_2d, npt)
  cube_redundancy = gen_gll_redundancy(vert_redundancy, npt)
  gll_latlon_equi, _ = gen_metric_terms_equiangular(face_mask, gll_position_equi, npt)
  latlon_corners = np.zeros((gll_latlon_equi.shape[0], 4, 2))
  for vert_idx, (i_in, j_in) in enumerate([(0, 0), (npt-1, 0), (0, npt-1), (npt-1, npt-1)]):
      latlon_corners[:, vert_idx, :] = gll_latlon_equi[:, i_in, j_in, :]

  # too_close_to_top = np.abs(latlon_corners[:, :, 0] - np.pi / 2) < 1e-8
  # too_close_to_bottom = np.abs(latlon_corners[:, :, 0] + np.pi / 2) < 1e-8
  # mask = np.logical_or(too_close_to_top,
  #                      too_close_to_bottom)
  # latlon_corners[:, :, 1] = np.where(mask, 0.0, latlon_corners[:, :, 1])

  gll_latlon, gll_to_cart_jacobian, cart_to_sphere_jacobian = gen_metric_terms_unstructured(latlon_corners, npt, rotate=rotate)
  print(gll_to_cart_jacobian.shape)
  print(cart_to_sphere_jacobian.shape)


  return generate_metric_terms(gll_latlon, gll_to_cart_jacobian, cart_to_sphere_jacobian, cube_redundancy, npt,
                               wrapped=wrapped, proc_idx=proc_idx)
