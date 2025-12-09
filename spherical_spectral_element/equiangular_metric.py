from .config import np, npt, DEBUG, use_jax
from .spectral import deriv
from .mesh import mesh_to_cart_bilinear, gen_gll_redundancy
from .grid_definitions import TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE
from .se_grid import create_spectral_element_grid
from .cubed_sphere import gen_cube_topo, gen_vert_redundancy
from textwrap import dedent


def gen_metric_terms_equiangular(face_mask, cube_points_2d, cube_redundancy):
  NFACES = cube_points_2d.shape[0]

  top_face_mask = (face_mask == TOP_FACE)[:, np.newaxis, np.newaxis]
  bottom_face_mask = (face_mask == BOTTOM_FACE)[:, np.newaxis, np.newaxis]
  left_face_mask = (face_mask == LEFT_FACE)[:, np.newaxis, np.newaxis]
  right_face_mask = (face_mask == RIGHT_FACE)[:, np.newaxis, np.newaxis]
  front_face_mask = (face_mask == FRONT_FACE)[:, np.newaxis, np.newaxis]
  back_face_mask = (face_mask == BACK_FACE)[:, np.newaxis, np.newaxis]

  gll_latlon = np.zeros(shape=(NFACES, npt, npt, 2))
  cube_to_sphere_jacobian = np.zeros(shape=(NFACES, npt, npt, 2, 2))
  if DEBUG:
    gll_latlon_pert = np.zeros(shape=(NFACES, npt, npt, 2))
    cube_points_pert = np.zeros_like(cube_points_2d)

  if DEBUG:
    n_mask = 0
    for m1, mask1 in enumerate([top_face_mask,
                                bottom_face_mask,
                                front_face_mask,
                                back_face_mask,
                                left_face_mask,
                                right_face_mask]):
      for m2, mask2 in enumerate([top_face_mask,
                                  bottom_face_mask,
                                  front_face_mask,
                                  back_face_mask,
                                  left_face_mask,
                                  right_face_mask]):
        if m1 != m2:
          ct = np.sum(np.logical_and(mask1, mask2))
          assert (ct == 0)
      n_mask += np.sum(mask1)
    assert (n_mask == NFACES)

  def set_jac_eq(jac, lat, lon, mask, flip_x=1.0, flip_y=1.0):
    jac[:, :, :, 0, 1] += flip_x * np.cos(lon[:, :, :])**2 * mask
    jac[:, :, :, 0, 0] += flip_x * -1 / 4 * np.sin(2 * lon[:, :, :]) * np.sin(2 * lat[:, :, :]) * mask
    jac[:, :, :, 1, 0] += flip_y * np.cos(lon[:, :, :]) * np.cos(lat[:, :, :])**2 * mask

  def set_jac_pole(jac, lat, lon, mask, k, flip_x=1.0, flip_y=1.0):
    jac[:, :, :, 0, 1] += flip_x * k * np.cos(lon) * np.tan(lat) * mask
    jac[:, :, :, 0, 0] += flip_x * -k * np.sin(lon) * np.sin(lat)**2 * mask
    jac[:, :, :, 1, 1] += flip_y * np.sin(lon) * np.tan(lat) * mask
    jac[:, :, :, 1, 0] += flip_y * np.cos(lon) * np.sin(lat)**2 * mask

  def dlatlon_dcube(latlon_fn, latlon_idx, cube_idx, mask):
    gll_latlon_pert[:] = 0
    cube_points_pert[:] = cube_points_2d[:]
    cube_points_pert[:, :, :, cube_idx] *= 0.99999
    gll_latlon_pert[:, :, :, latlon_idx] += latlon_fn(cube_points_pert[:, :, :, 0], cube_points_pert[:, :, :, 1]) * mask
    result = ((gll_latlon_pert[:, :, :, latlon_idx] - gll_latlon[:, :, :, latlon_idx]) /
              (cube_points_pert[:, :, :, cube_idx] - cube_points_2d[:, :, :, cube_idx]))
    return result

  def test_face(lat_fn, lon_fn, mask):
    dlat_dx = dlatlon_dcube(lat_fn, 0, 0, mask)
    dlat_dy = dlatlon_dcube(lat_fn, 0, 1, mask)
    dlon_dx = dlatlon_dcube(lon_fn, 1, 0, mask)
    dlon_dy = dlatlon_dcube(lon_fn, 1, 1, mask)
    jac_tmp = np.zeros((2, 2))
    check1 = mask * (dlat_dx - cube_to_sphere_jacobian[:, :, :, 0, 0])
    check2 = mask * (dlat_dy - cube_to_sphere_jacobian[:, :, :, 1, 0])
    check3 = mask * (dlon_dx - cube_to_sphere_jacobian[:, :, :, 0, 1])
    check4 = mask * (dlon_dy - cube_to_sphere_jacobian[:, :, :, 1, 1])
    try:
      assert (np.max(check1) < 1e-3)
      assert (np.max(check2) < 1e-3)
      assert (np.max(check3) < 1e-3)
      assert (np.max(check4) < 1e-3)
    except AssertionError:
      for face_idx in range(NFACES):
        if mask[face_idx]:
          i_idx, j_idx = (1, 1)
          jac_tmp[0, 0] = dlat_dx[face_idx, i_idx, j_idx]
          jac_tmp[0, 1] = dlon_dx[face_idx, i_idx, j_idx]
          jac_tmp[1, 0] = dlat_dy[face_idx, i_idx, j_idx]
          jac_tmp[1, 1] = dlon_dy[face_idx, i_idx, j_idx]
          err_str = f"""Face: {face_idx},\nvvvvvvvvvvvvvvvvvvv\nNumerical jac: \n{jac_tmp}, \nAnalytic jac: \n{cube_to_sphere_jacobian[face_idx, i_idx, j_idx, :, :]}\n^^^^^^^^^^^^^^^^^\n """
          print(dedent(err_str))

  # front face
  def front_lat(x, y):
    return (np.arctan2(y, np.sqrt(1 + x**2)))

  def front_lon(x, y):
    return np.mod((np.arctan(x) + 2 * np.pi), 2 * np.pi)

  gll_latlon[:, :, :, 0] += front_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * front_face_mask
  gll_latlon[:, :, :, 1] += front_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * front_face_mask
  set_jac_eq(cube_to_sphere_jacobian, gll_latlon[:, :, :, 0], gll_latlon[:, :, :, 1], front_face_mask)
  if DEBUG:
    test_face(front_lat, front_lon, front_face_mask)

  # right face
  def right_lat(x, y):
    return (np.arctan2(y, np.sqrt(1 + x**2)))

  def right_lon(x, y):
    return np.arctan(x) + np.pi / 2

  gll_latlon[:, :, :, 0] += right_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * right_face_mask
  gll_latlon[:, :, :, 1] += right_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * right_face_mask
  set_jac_eq(cube_to_sphere_jacobian, gll_latlon[:, :, :, 0], -np.pi / 2 + gll_latlon[:, :, :, 1], right_face_mask)
  if DEBUG:
    test_face(right_lat, right_lon, right_face_mask)

  # back face
  def back_lat(x, y):
    return (np.arctan2(y, np.sqrt(1 + x**2)))

  def back_lon(x, y):
    return (np.arctan(x) + np.pi)
  gll_latlon[:, :, :, 0] += back_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * back_face_mask
  gll_latlon[:, :, :, 1] += back_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * back_face_mask
  set_jac_eq(cube_to_sphere_jacobian, gll_latlon[:, :, :, 0], -np.pi + gll_latlon[:, :, :, 1], back_face_mask)
  if DEBUG:
    test_face(back_lat, back_lon, back_face_mask)

  # left face
  def left_lat(x, y):
    return (np.arctan2(y, np.sqrt(1 + x**2)))

  def left_lon(x, y):
    return (np.arctan(x) + 3 * np.pi / 2)

  gll_latlon[:, :, :, 0] += left_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * left_face_mask
  gll_latlon[:, :, :, 1] += left_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * left_face_mask
  set_jac_eq(cube_to_sphere_jacobian, gll_latlon[:, :, :, 0], -3 * np.pi / 2 + gll_latlon[:, :, :, 1], left_face_mask)
  if DEBUG:
    test_face(left_lat, left_lon, left_face_mask)

  # top face
  def top_lat(x, y):
    return (np.arctan2(1, np.sqrt(x**2 + y**2)))

  def top_lon(x, y):
    return np.mod((np.arctan2(x, -y)), 2 * np.pi)

  gll_latlon[:, :, :, 0] += top_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * top_face_mask
  gll_latlon[:, :, :, 1] += top_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * top_face_mask
  set_jac_pole(cube_to_sphere_jacobian, gll_latlon[:, :, :, 0], gll_latlon[:, :, :, 1], top_face_mask, 1.0)
  if DEBUG:
    test_face(top_lat, top_lon, top_face_mask)

  # bottom face
  def bottom_lat(x, y):
    return -np.arctan2(1, np.sqrt(x**2 + y**2))

  def bottom_lon(x, y):
    return np.mod((np.arctan2(x, y)), 2 * np.pi)

  gll_latlon[:, :, :, 0] += bottom_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * bottom_face_mask
  gll_latlon[:, :, :, 1] += bottom_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * bottom_face_mask
  set_jac_pole(cube_to_sphere_jacobian, gll_latlon[:, :, :, 0], gll_latlon[:, :, :, 1], bottom_face_mask, -1.0)

  if DEBUG:
    test_face(bottom_lat, bottom_lon, bottom_face_mask)
  

  return gll_latlon, cube_to_sphere_jacobian


def generate_metric_terms(gll_latlon, gll_to_cube_jacobian, cube_to_sphere_jacobian, vert_redundancy_gll, jax=use_jax):
  gll_to_sphere_jacobian = np.einsum("fijpg,fijps->fijgs", cube_to_sphere_jacobian, gll_to_cube_jacobian)
  gll_to_sphere_jacobian[:, :, :, 1, :] *= np.cos(gll_latlon[:, :, :, 0])[:, :, :, np.newaxis]
  too_close_to_top = np.abs(gll_latlon[:, :, :, 0] - np.pi/2) < 1e-8
  too_close_to_bottom = np.abs(gll_latlon[:, :, :, 0] + np.pi/2) < 1e-8
  for i_idx, j_idx, entry in zip([0, 1, 0, 1],
                                 [0, 1, 1, 0],
                                 [1.0, 1.0, 0.0, 0.0]):
    gll_to_sphere_jacobian[:, :, :,
                           i_idx, j_idx] = np.where(np.logical_or(too_close_to_top,
                                                                  too_close_to_bottom),
                                                    entry,
                                                    gll_to_sphere_jacobian[:, :, :, i_idx, j_idx])
  gll_to_sphere_jacobian_inv = np.linalg.inv(gll_to_sphere_jacobian)

  rmetdet = np.linalg.det(gll_to_sphere_jacobian_inv)

  metdet = 1.0 / rmetdet

  mass_mat = metdet.copy() * (deriv["gll_weights"][np.newaxis, :, np.newaxis] *
                              deriv["gll_weights"][np.newaxis, np.newaxis, :])

  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        mass_mat[remote_face_id, remote_i, remote_j] += (metdet[local_face_idx, local_i, local_j] *
                                                         (deriv["gll_weights"][local_i] *
                                                          deriv["gll_weights"][local_j]))

  inv_mass_mat = 1.0 / mass_mat

  return create_spectral_element_grid(gll_latlon,
                                      gll_to_sphere_jacobian,
                                      gll_to_sphere_jacobian_inv,
                                      rmetdet, metdet, mass_mat,
                                      inv_mass_mat, vert_redundancy_gll, jax=jax)


def gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, jax=use_jax):
  gll_position, gll_jacobian = mesh_to_cart_bilinear(face_position_2d)
  cube_redundancy = gen_gll_redundancy(face_connectivity, vert_redundancy)
  gll_latlon, cube_to_sphere_jacobian = gen_metric_terms_equiangular(face_mask, gll_position, cube_redundancy)
  return generate_metric_terms(gll_latlon, gll_jacobian, cube_to_sphere_jacobian, cube_redundancy, jax=jax)


def create_quasi_uniform_grid(nx):
  face_connectivity, face_mask, face_position, face_position_2d = gen_cube_topo(nx)
  vert_redundancy = gen_vert_redundancy(nx, face_connectivity, face_position)
  return gen_metric_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy)
