from ..config import np


def unit_sphere_to_cart_coords(latlon):
  lat = np.take(latlon, 0, axis=-1)
  lon = np.take(latlon, 1, axis=-1)
  cos_lat = np.cos(lat)
  cart = np.stack((cos_lat * np.cos(lon),
                   cos_lat * np.sin(lon),
                   np.sin(lat)), axis=-1)
  return cart


def cart_to_unit_sphere_coords(xyz):
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
