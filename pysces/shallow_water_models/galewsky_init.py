from ..config import device_wrapper, np, jnp

def get_galewsky_config(model_config):
  config = {}
  config["deg"] = 100
  pts, weights = device_wrapper(np.polynomial.legendre.leggauss(config["deg"]))
  pts = (pts + 1.0) / 2.0
  weights /= 2.0
  config["pts"] = pts
  config["weights"] = weights
  config["u_max"] = 80
  config["phi0"] = np.pi / 7
  config["phi1"] = np.pi / 2 - config["phi0"]
  config["e_norm"] = np.exp(-4 / (config["phi1"] - config["phi0"])**2)
  config["radius_earth"] = model_config["radius_earth"]
  config["earth_period"] = model_config["earth_period"]
  config["h0"] = 1e4
  config["hat_h"] = 120.0
  config["pert_alpha"] = 1.0 / 3.0
  config["pert_beta"] = 1.0 / 15.0
  config["pert_center"] = np.pi / 4
  config["gravity"] = model_config["gravity"]
  return config

def galewsky_u(lat, config):
  u = jnp.zeros_like(lat)
  mask = jnp.logical_and(lat > config["phi0"], lat < config["phi1"])
  u = jnp.where(mask, config["u_max"] / config["e_norm"] * jnp.exp(1 / ((lat - config["phi0"]) * (lat - config["phi1"]))), u)
  return u

def galewsky_wind(lat, lon, config):
  u = jnp.stack((galewsky_u(lat, config),
                  jnp.zeros_like(lat)), axis=-1)
  return u

def galewsky_h(lat, lon, config):
  quad_amount = lat + jnp.pi / 2.0
  weights_quad = quad_amount.reshape([*lat.shape, 1]) * config["weights"].reshape((*[1 for _ in lat.shape], config["deg"]))
  phi_quad = quad_amount.reshape([*lat.shape, 1]) * config["pts"].reshape((*[1 for _ in lat.shape], config["deg"])) - np.pi / 2
  u_quad = galewsky_u(phi_quad, config)
  f = 2.0 * config["earth_period"] * jnp.sin(phi_quad)
  integrand = config["radius_earth"] * u_quad * (f + jnp.tan(phi_quad) / config["radius_earth"] * u_quad)
  h = config["h0"] - 1.0 / config["gravity"] * jnp.sum(integrand * weights_quad, axis=-1)
  h_prime = (config["hat_h"] * jnp.cos(lat) * jnp.exp(-(lon / config["pert_alpha"])**2) *
             jnp.exp(-((config["pert_center"] - lat) / config["pert_beta"])**2))
  return h + h_prime

def galewsky_hs(lat, lon, config):
  return jnp.zeros_like(lat)