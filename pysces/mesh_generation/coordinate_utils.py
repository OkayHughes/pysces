from ..config import np


def bilinear(v0, v1, v2, v3, alpha, beta):
  """
  Compute bilinear mapping for unstructured arrays of
  topological quadrilaterals in arbitrary cartesian dimension.

  Parameters
  ----------
  v0: Array[*Shape, Float]
    Upper left vertex, final index is cartesian dimension
  v1: Array[*Shape, Float]
    Upper right vertex, final index is cartesian dimension
  v2: Array[*Shape, Float]
    Bottom left vertex, final index is cartesian dimension
  v3: Array[*Shape, Float]
    Bottom right vertex, final index is cartesian dimension
  alpha: Float
    First coordinate within reference element
  beta: Float
    Second coordinate position within reference element

  Returns
  -------
  Array[*Shape, Float]
      Interpolated positions.

  Notes
  -----
  The reference element is assumed to be [-1, 1]^2.
  """
  #   v0---α---v1
  #   |    :    |
  #   |    β    |
  #   |    :    |
  #   v2---α---v3
  aprime = (alpha + 1) / 2
  bprime = (beta + 1) / 2
  top_point = aprime * v0 + (1 - aprime) * v1
  bottom_point = aprime * v2 + (1 - aprime) * v3
  return (bprime * top_point + (1 - bprime) * bottom_point)


def bilinear_jacobian(v0, v1, v2, v3, alpha, beta):
  """
  Compute jacobian of the bilinear mapping for unstructured arrays of
  topological quadrilaterals in arbitrary cartesian dimension.

  Parameters
  ----------
  v0: Array[*Shape, Float]
    Upper left vertex, final index is cartesian dimension
  v1: Array[*Shape, Float]
    Upper right vertex, final index is cartesian dimension
  v2: Array[*Shape, Float]
    Bottom left vertex, final index is cartesian dimension
  v3: Array[*Shape, Float]
    Bottom right vertex, final index is cartesian dimension
  alpha: Float
    First coordinate within reference element
  beta: Float
    Second coordinate within reference element

  Returns
  -------
  dphys_dalpha: Array[*Shape, Float]
    Derivative of each cartesian dimension
    w.r.t. the first coordinate on the reference element
  dphys_dalpha: Array[*Shape, Float]
    Derivative of each cartesian dimension
    w.r.t. the second coordinate on the reference element

  Notes
  -----
  The reference element is assumed to be [-1, 1]^2.
  """
  aprime = (alpha + 1) / 2
  bprime = (beta + 1) / 2
  dphys_dalpha = 1 / 2.0 * (bprime * (v0 - v1) + (1 - bprime) * (v2 - v3))
  dphys_dbeta = 1 / 2.0 * (aprime * v0 + (1 - aprime) * v1 - (aprime * v2 + (1 - aprime) * v3))
  return dphys_dalpha, dphys_dbeta


def unit_sphere_to_cart(latlon):
  lat = np.take(latlon, 0, axis=-1)
  lon = np.take(latlon, 1, axis=-1)
  cos_lat = np.cos(lat)
  cart = np.stack((cos_lat * np.cos(lon),
                   cos_lat * np.sin(lon),
                   np.sin(lat)), axis=-1)
  return cart


def cart_to_unit_sphere(xyz):
  latlon = np.stack((np.asin(xyz[:, :, :, 2]),
                     np.mod(np.atan2(xyz[:, :, :, 1],
                                     xyz[:, :, :, 0]) + 2 * np.pi,
                            2 * np.pi)), axis=-1)
  return latlon


def unit_sphere_to_cart_coords_jacobian(latlon):
  lat = latlon[:, :, :, 0]
  lon = latlon[:, :, :, 1]
  unit_sphere_to_sph_coords_jacobian = np.zeros((*lat.shape[:3], 3, 2))
  unit_sphere_to_sph_coords_jacobian[:, :, :, 0, 0] = -np.sin(lat) * np.cos(lon)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 0, 1] = np.cos(lat) * -np.sin(lon)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 1, 0] = -np.sin(lat) * np.sin(lon)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 1, 1] = np.cos(lat) * np.cos(lon)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 2, 0] = np.cos(lat)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 2, 1] = 0.0
  return unit_sphere_to_sph_coords_jacobian


def cart_to_unit_sphere_coords_jacobian(xyz):
  x = xyz[:, :, :, 0]
  y = xyz[:, :, :, 1]
  z = xyz[:, :, :, 2]
  normsq_2d = x**2 + y**2
  unit_sphere_to_sph_coords_jacobian = np.zeros((*x.shape[:3], 2, 3))
  unit_sphere_to_sph_coords_jacobian[:, :, :, 0, 2] = 1.0 / np.sqrt(1 - z**2)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 1, 0] = -y / normsq_2d
  unit_sphere_to_sph_coords_jacobian[:, :, :, 1, 1] = x / normsq_2d
  return unit_sphere_to_sph_coords_jacobian
