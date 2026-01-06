from ..config import jnp


def get_williamson_steady_config(model_config):
  config = {}
  config["u0"] = 2.0 * jnp.pi * model_config["radius_earth"] / (12.0 * 24.0 * 60.0 * 60.0)
  config["h0"] = 2.94e4 / model_config["gravity"]
  config["alpha"] = model_config["alpha"]
  config["gravity"] = model_config["gravity"]
  config["radius_earth"] = model_config["radius_earth"]
  config["earth_period"] = model_config["earth_period"]
  return config


def williamson_tc2_u(lat, lon, config):
  wind = jnp.stack((config["u0"] * (jnp.cos(lat) * jnp.cos(config["alpha"]) +
                                    jnp.cos(lon) * jnp.sin(lat) * jnp.sin(config["alpha"])),
                    -config["u0"] * (jnp.sin(lon) * jnp.sin(config["alpha"]))), axis=-1)
  return wind


def williamson_tc2_h(lat, lon, config):
  h = jnp.zeros_like(lat)
  h += config["h0"]
  second_factor = (-jnp.cos(lon) * jnp.cos(lat) * jnp.sin(config["alpha"]) +
                   jnp.sin(lat) * jnp.cos(config["alpha"]))**2
  h -= (config["radius_earth"] * config["earth_period"] *
        config["u0"] + config["u0"]**2 / 2.0) / config["gravity"] * second_factor
  return h


def williamson_tc2_hs(lat, lon, config):
  return jnp.zeros_like(lat)
