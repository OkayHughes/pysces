from ..config import jnp, device_wrapper, np, remainder
from enum import Enum

perturbation_opts = Enum('perturbation_type',
                         [("none", 1),
                          ("exponential", 2),
                          ("streamfunction", 3)])

def get_umjs_config(T0E=310,
                    T0P=240,
                    B=2.0,
                    K=3.0,
                    lapse=0.005,
                    pertu0=0.5,
                    pertr=1.0 / 6.0,
                    pertup=1.0,
                    pertexpr=0.1,
                    pertlon=jnp.pi / 9.0,
                    pertlat=2.0 * jnp.pi / 9.0,
                    pertz=15000,
                    moistqlat=2.0 * jnp.pi / 9.0,
                    moistqp=34000.0,
                    moisttr=0.1,
                    moistq0=0.018,
                    moistqr=0.9,
                    moisteps=0.622,
                    moistT0=273.16,
                    moistE0Ast=610.78,
                    p0=1e5,
                    radius_earth=6371e3,
                    period_earth=7.292e-5,
                    Rgas=287.1,
                    R_water_vapor=461.50,
                    gravity=9.81,
                    model_config=None,
                    alpha=0.5,
                    mountain_heights=[2000.0, 2000.0],
                    mountain_lats=[jnp.pi / 4.0, jnp.pi / 4.0],
                    mountain_lons=[(7.0 / 9.0) * jnp.pi,
                                   0.4 * jnp.pi],
                    mountain_lat_widths=[40.0 * jnp.pi / 180.0,
                                         40.0 * jnp.pi / 180.0],
                    mountain_lon_widths=[7.0 * jnp.pi / 180.0,
                                         7.0 * jnp.pi / 180.0]):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  moistqs = 1e-12
  dx_epsilon = 1e-5
  if model_config:
    radius_earth = model_config["radius_earth"]
    period_earth = model_config["period_earth"]
    Rgas = model_config["Rgas"]
    R_water_vapor = model_config["moisture_species_Rgas"]["water_vapor"]
    gravity = model_config["gravity"]
  mountain_lat_scales = [lat_width / (2.0 * (-np.log(0.1))**(1.0 / 6.0))
                         for lat_width in mountain_lat_widths]
  mountain_lon_scales = [lon_width / (2.0 * (-np.log(0.1))**(1.0 / 2.0))
                         for lon_width in mountain_lon_widths]
  return {"T0E": device_wrapper(T0E),
          "T0P": device_wrapper(T0P),
          "B": device_wrapper(B),
          "K": device_wrapper(K),
          "lapse": device_wrapper(lapse),
          "pertu0": device_wrapper(pertu0),
          "pertr": device_wrapper(pertr),
          "pertup": device_wrapper(pertup),
          "pertexpr": device_wrapper(pertexpr),
          "pertlon": device_wrapper(pertlon),
          "pertlat": device_wrapper(pertlat),
          "pertz": device_wrapper(pertz),
          "dx_epsilon": device_wrapper(dx_epsilon),
          "moistqlat": device_wrapper(moistqlat),
          "moistqp": device_wrapper(moistqp),
          "moisttr": device_wrapper(moisttr),
          "moistqs": device_wrapper(moistqs),
          "moistq0": device_wrapper(moistq0),
          "moistqr": device_wrapper(moistqr),
          "moisteps": device_wrapper(moisteps),
          "moistT0": device_wrapper(moistT0),
          "moistE0Ast": device_wrapper(moistE0Ast),
          "p0": device_wrapper(p0),
          "radius_earth": device_wrapper(radius_earth),
          "period_earth": device_wrapper(period_earth),
          "Rgas": device_wrapper(Rgas),
          "R_water_vapor": device_wrapper(R_water_vapor),
          "gravity": device_wrapper(gravity),
          "alpha": device_wrapper(alpha),
          "mountain_heights": device_wrapper(mountain_heights),
          "mountain_lats": device_wrapper(mountain_lats),
          "mountain_lons": device_wrapper(mountain_lons),
          "mountain_lat_scales": device_wrapper(mountain_lat_scales),
          "mountain_lon_scales": device_wrapper(mountain_lon_scales),
          }


def get_T0(config):
  return (config["alpha"] * config["T0E"] +
          (1.0 - config["alpha"]) * config["T0P"])


def get_constH(config):
  return config["Rgas"] * get_T0(config) / config["gravity"]


def get_constA(config):
  return 1.0 / config["lapse"]


def get_constB(config):
  T0 = get_T0(config)
  return (T0 - config["T0P"]) / (T0 * config["T0P"])


def get_scaledZ(z, config):
  return z / (config["B"] * get_constH(config))


def get_inttau2(z, config):
  return (get_constC(config) * z *
          jnp.exp(-get_scaledZ(z, config)**2))


def get_constC(config):
  T0E = config["T0E"]
  T0P = config["T0P"]
  return (0.5 * (config["K"] + 2.0) *
          (T0E - T0P) / (T0E * T0P))


def get_r_hat(z, config, deep=False):
  # note: should be separate from model code.
  # so constant-g equation set can be used
  if deep:
    r_hat = (z + config["radius_earth"]) / config["radius_earth"]
  else:
    r_hat = jnp.ones_like(z)
  return r_hat


def get_z_surface(lat, lon, config, mountain=False):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  if mountain:
    zs = jnp.zeros_like(lat)
    for (mountain_height,
         mountain_lat,
         mountain_lon,
         mountain_lat_scale,
         mountain_lon_scale) in zip(config["mountain_heights"],
                                    config["mountain_lats"],
                                    config["mountain_lons"],
                                    config["mountain_lat_scales"],
                                    config["mountain_lon_scales"]):
      d0 = remainder(lon - mountain_lon, 2.0 * jnp.pi)
      d0 = jnp.minimum(d0, 2.0 * jnp.pi - d0)
      zs += mountain_height * jnp.exp(-(((lat - mountain_lat) / mountain_lat_scale)**6 +
                                        (d0 / mountain_lon_scale)**2))
  else:
    zs = jnp.zeros_like(lat)
  return zs


def evaluate_pressure_temperature(z, lat, config, deep=False):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  lapse = config["lapse"]
  K = config["K"]
  T0 = get_T0(config)
  constA = get_constA(config)
  constB = get_constB(config)
  constC = get_constC(config)
  scaledZ = get_scaledZ(z, config)

  # note: this can be optimized for numpy so
  # scaledZ**2 quantities are not recomputed

  tau1 = (constA * lapse / T0 * jnp.exp(lapse * z / T0) +
          constB * (1.0 - 2.0 * scaledZ**2) * jnp.exp(-scaledZ**2))
  tau2 = constC * (1.0 - 2.0 * scaledZ**2) * jnp.exp(-scaledZ**2)

  inttau1 = (constA * (jnp.exp(lapse * z / T0) - device_wrapper(1.0)) +
             constB * z * jnp.exp(-scaledZ**2))
  inttau2 = get_inttau2(z, config)

  r_hat = get_r_hat(z, config, deep=deep)

  inttermT = ((r_hat * jnp.cos(lat)[:, :, :, np.newaxis])**K -
              K / (K + 2.0) * (r_hat * jnp.cos(lat)[:, :, :, np.newaxis])**(K + 2))

  temperature = 1.0 / (r_hat**2 * (tau1 - tau2 * inttermT))
  pressure = config["p0"] * jnp.exp(-config["gravity"] / config["Rgas"] *
                                    (inttau1 - inttau2 * inttermT))
  return pressure, temperature


def evaluate_surface_state(lat, lon, config, deep=False, mountain=False):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  z_surface = get_z_surface(lat, lon, config, mountain=mountain)
  p_surface = evaluate_pressure_temperature(z_surface[:, :, :, np.newaxis],
                                            lat, config, deep=deep)[0][:, :, :, 0]
  return z_surface, p_surface


def evaluate_state(lat, lon, z, config, deep=False, moist=False, pert_type=perturbation_opts.none):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  K = config["K"]
  inttau2 = get_inttau2(z, config)
  r_hat = get_r_hat(z, config, deep=deep)
  cos_lat = jnp.cos(lat)[:, :, :, np.newaxis]
  inttermU = ((r_hat * cos_lat)**(K - 1.0) -
              (r_hat * cos_lat)**(K + 1.0))
  pressure, temp = evaluate_pressure_temperature(z, lat, config, deep=deep)
  bigU = (config["gravity"] / config["radius_earth"] * K *
          inttau2 * inttermU * temp)

  if deep:
    rcoslat = config["radius_earth"] * cos_lat
  else:
    rcoslat = (z + config["radius_earth"]) * cos_lat
  solid_body_rotation = config["period_earth"] * rcoslat
  u = -solid_body_rotation + jnp.sqrt(solid_body_rotation**2 +
                                      rcoslat * bigU)
  v = jnp.zeros_like(u)

  if moist:
    p0 = config["p0"]
    eta = pressure / p0
    q_vapor = config["moistq0"] * (jnp.exp(-(lat / config["moistqlat"])**4) *
                                   jnp.exp((-(eta - 1.0) * p0 / config["moistqp"])**2))
    Mvap = config["R_water_vapor"] / config["Rgas"] - 1.0
    temp_v = temp / (1.0 + Mvap * q_vapor)
  else:
    q_vapor = jnp.zeros_like(z)
    temp_v = temp

  # todo: handle pert
  if pert_type == perturbation_opts.exponential:
    print("using exponential perturbation type")
    u += evaluate_exponential(lat, lon, z, config)
  elif pert_type == perturbation_opts.streamfunction:
    print("using streamfunction perturbation type")
    eps = 1e-5
    sf_lat_above = evaluate_streamfunction(lat + eps, lon, z, config)
    sf_lat_below = evaluate_streamfunction(lat - eps, lon, z, config)
    sf_lon_above = evaluate_streamfunction(lat, lon + eps, z, config)
    sf_lon_below = evaluate_streamfunction(lat, lon - eps, z, config)
    u += - (sf_lat_above - sf_lat_below) / (2 * eps)
    v += (sf_lon_above - sf_lon_below) / (2 * eps)
  return u, v, pressure, temp_v, q_vapor


def great_circle_dist(lat, lon, config):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  return (1.0 / config["pertexpr"] *
          jnp.arccos(jnp.sin(config["pertlat"]) *
                     jnp.sin(lat) +
                     jnp.cos(config["pertlat"]) *
                     jnp.cos(lat) *
                     jnp.cos(lon - config["pertlon"])))


def taper_fn(z, config):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  pertz = config["pertz"]
  taper_below_pertz = (1.0 - 3.0 * z**2 / pertz**2 + 2.0 * z**3 / pertz**3)
  return jnp.where(z < pertz,
                   taper_below_pertz,
                   jnp.zeros_like(z))


def evaluate_exponential(lat, lon, z, config):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  greatcircle_dist = great_circle_dist(lat, lon, config)[:, :, :, np.newaxis]
  taper = taper_fn(z, config)

  pert_inside_circle = (config["pertup"] *
                        taper *
                        jnp.exp(-greatcircle_dist**2))
  return jnp.where(greatcircle_dist < 1.0,
                   pert_inside_circle,
                   jnp.zeros_like(z))


def evaluate_streamfunction(lat, lon, z, config):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  greatcircle_dist = great_circle_dist(lat, lon, config)
  taper = taper_fn(z, config)
  pert_inside_circle = jnp.cos(0.5 * jnp.pi * greatcircle_dist)
  return jnp.where(greatcircle_dist < 1.0,
                   -config["pertu0"] * config["pertr"] * taper * pert_inside_circle**4,
                   jnp.zeros_like(z))
