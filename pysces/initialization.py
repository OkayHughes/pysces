from .config import jnp, device_wrapper, np
from .dynamical_cores.homme.homme_state import init_model_struct as init_model_struct_homme
from .dynamical_cores.cam_se.se_state import init_model_struct as init_model_struct_se
from .dynamical_cores.homme.thermodynamics import eval_balanced_geopotential
from .dynamical_cores.utils_3d import interface_to_delta
from .dynamical_cores.mass_coordinate import (surface_mass_to_midlevel_mass,
                                              surface_mass_to_interface_mass,
                                              eval_top_interface_mass)
from .dynamical_cores.physics_config import typical_mass_ratios
from .model_info import (hydrostatic_models,
                         moist_mixing_ratio_models,
                         cam_se_models,
                         homme_models,
                         dry_mixing_ratio_models,
                         variable_kappa_models,
                         deep_atmosphere_models)
from .analytic_initialization.moist_baroclinic_wave import (eval_surface_state,
                                                            eval_state,
                                                            eval_pressure_temperature,
                                                            perturbation_opts)


gauss_points = (jnp.array([-0.97390652851717,
                           -0.865063366689,
                           -0.67940956829902,
                           -0.4333953941292,
                           -0.14887433898163,
                           0.14887433898163,
                           0.4333953941292,
                           0.679409568299,
                           0.86506336668898,
                           0.97390652851717]) + 1.0) / 2.0

gauss_weights = jnp.array([0.06667134430869,
                           0.1494513491506,
                           0.219086362516,
                           0.26926671931,
                           0.29552422471475,
                           0.2955242247148,
                           0.26926671931,
                           0.21908636251598,
                           0.1494513491506,
                           0.0666713443087]) / 2.0


def integrate_weight_of_vapor(p_moist_given_z,
                              Tv_given_z,
                              q_given_z,
                              z,
                              z_top,
                              config):
  weight = jnp.zeros_like(z)
  for gauss_point, gauss_weight in zip(gauss_points, gauss_weights):
    z_quad = z_top - gauss_point * (z_top - z)
    rho_water = p_moist_given_z(z_quad) / (config["Rgas"] * Tv_given_z(z_quad)) * q_given_z(z_quad)
    weight += gauss_weight * rho_water * config["gravity"] * (z_top - z)
  return weight


def z_from_p_monotonic_dry(pressures,
                           p_moist_given_z,
                           q_given_z,
                           Tv_given_z,
                           v_grid,
                           config,
                           eps=1e-5,
                           z_top=80e3):
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
  p_top = eval_top_interface_mass(v_grid)
  z_top = z_from_p_monotonic_moist(p_top * jnp.ones_like(pressures), p_moist_given_z, eps)

  def p_dry_given_z(z):
    return p_moist_given_z(z) - integrate_weight_of_vapor(p_moist_given_z, Tv_given_z, q_given_z, z, z_top, config)

  z_guesses = jnp.zeros_like(pressures)
  not_converged = jnp.logical_not(jnp.abs((p_dry_given_z(z_guesses) - pressures)) / pressures < eps)
  frac = 1.0
  ct = 0
  while jnp.any(not_converged):
    p_guess = p_dry_given_z(z_guesses)
    too_high = p_guess < pressures
    z_guesses = jnp.where(not_converged,
                          jnp.where(jnp.logical_and(not_converged, too_high),
                                    z_guesses - frac * z_top,
                                    z_guesses + frac * z_top), z_guesses)
    not_converged = jnp.logical_not(jnp.abs((p_dry_given_z(z_guesses) - pressures)) / pressures < eps)
    print(ct)
    frac *= 0.5
    if ct > 30:
      break
    ct += 1
  return z_guesses


def z_from_p_monotonic_moist(pressures,
                             p_given_z,
                             eps=1e-5,
                             z_top=80e3):
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

  z_guesses = 0.0 * p_given_z(z_top * jnp.ones_like(pressures))
  not_converged = jnp.logical_not(jnp.abs((p_given_z(z_guesses) - pressures)) / pressures < eps)
  frac = 1.0
  ct = 0
  while jnp.any(not_converged):
    p_guess = p_given_z(z_guesses)
    too_high = p_guess < pressures
    z_guesses = jnp.where(not_converged,
                          jnp.where(jnp.logical_and(not_converged, too_high),
                                    z_guesses - frac * z_top,
                                    z_guesses + frac * z_top), z_guesses)
    not_converged = jnp.logical_not(jnp.abs((p_given_z(z_guesses) - pressures)) / pressures < eps)
    frac *= 0.5
    if ct > 30:
      break
    ct += 1
  return z_guesses


def init_model_pressure(z_pi_surf_func,
                        p_moist_func,
                        Tv_func,
                        u_func,
                        v_func,
                        Q_func,
                        h_grid,
                        v_grid,
                        config,
                        dims,
                        model,
                        w_func=lambda lat, lon, z: 0.0,
                        eps=1e-8,
                        enforce_hydrostatic=False):
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
  lat = h_grid["physical_coords"][:, :, :, 0]
  lon = h_grid["physical_coords"][:, :, :, 1]
  z_surf, surface_mass = z_pi_surf_func(lat, lon)

  if model in dry_mixing_ratio_models:
    p_top = eval_top_interface_mass(v_grid)
    z_top = z_from_p_monotonic_moist(p_top * jnp.ones_like(surface_mass)[:, :, :, np.newaxis], p_moist_func, eps)
    weight_of_vapor = integrate_weight_of_vapor(p_moist_func,
                                                lambda z: Tv_func(lat, lon, z),
                                                lambda z: Q_func(lat, lon, z),
                                                z_surf[:, :, :, np.newaxis],
                                                z_top,
                                                config)
    surface_mass -= weight_of_vapor.squeeze()
  p_mid = surface_mass_to_midlevel_mass(surface_mass, v_grid)
  p_int = surface_mass_to_interface_mass(surface_mass, v_grid)
  if model in moist_mixing_ratio_models:
    z_mid = z_from_p_monotonic_moist(p_mid, p_moist_func, eps=eps)
    z_int = z_from_p_monotonic_moist(p_int, p_moist_func, eps=eps)
  elif model in dry_mixing_ratio_models:
    z_mid = z_from_p_monotonic_dry(p_mid,
                                   p_moist_func,
                                   lambda z: Q_func(lat, lon, z),
                                   lambda z: Tv_func(lat, lon, z),
                                   v_grid,
                                   config,
                                   eps=eps)
    z_int = z_from_p_monotonic_dry(p_int,
                                   p_moist_func,
                                   lambda z: Q_func(lat, lon, z),
                                   lambda z: Tv_func(lat, lon, z),
                                   v_grid,
                                   config,
                                   eps=eps)
  if model not in hydrostatic_models:
    w_i = device_wrapper(w_func(lat, lon, z_int))
    phi_i = device_wrapper(z_int * config["gravity"])
  else:
    w_i = None
    phi_i = None

  phi_surf = z_surf * config["gravity"]
  d_mass = interface_to_delta(p_int)

  u = u_func(lat, lon, z_mid)
  v = v_func(lat, lon, z_mid)
  wind = jnp.stack((u, v), axis=-1)
  if model in cam_se_models:
    p_int_moist = p_moist_func(z_int)
    virtual_temperature = Tv_func(lat, lon, z_mid)
    d_mass_moist = (p_int_moist[:, :, :, 1:] - p_int_moist[:, :, :, :-1])
    d_mass_dry = (p_int[:, :, :, 1:] - p_int[:, :, :, :-1])
    moisture_dry_ratio = (d_mass_moist / d_mass_dry) - 1.0
    temperature = virtual_temperature * ((1.0 + moisture_dry_ratio) /
                                         (1.0 + (1.0 / config["epsilon"]) * moisture_dry_ratio))
    moisture_species = {"water_vapor": device_wrapper(moisture_dry_ratio)}
    if model in variable_kappa_models:
      species_names = config["dry_air_species_Rgas"].keys()
      ratios = typical_mass_ratios[frozenset(species_names)]
      dry_air_species = {}
      for species in species_names:
        dry_air_species[species] = ratios[species] * jnp.ones_like(moisture_species["water_vapor"])
    else:
      dry_air_species = None
    initial_state = init_model_struct_se(device_wrapper(wind),
                                         device_wrapper(temperature),
                                         device_wrapper(d_mass),
                                         device_wrapper(phi_surf),
                                         moisture_species,
                                         {},
                                         h_grid,
                                         dims,
                                         config, model,
                                         dry_air_species=dry_air_species)

  elif model in homme_models:
    theta_v = Tv_func(lat, lon, z_mid) * (config["p0"] / p_mid)**(config["Rgas"] / config["cp"])
    moisture_moist_ratio = Q_func(lat, lon, z_mid)
    theta_v_d_mass = theta_v * d_mass
    if enforce_hydrostatic and model not in hydrostatic_models:
      phi_i = eval_balanced_geopotential(phi_surf, p_mid, theta_v_d_mass, config)

    moisture_species = {"water_vapor": device_wrapper(moisture_moist_ratio)}
    initial_state = init_model_struct_homme(device_wrapper(wind),
                                            device_wrapper(theta_v_d_mass),
                                            device_wrapper(d_mass),
                                            device_wrapper(phi_surf),
                                            moisture_species,
                                            {},
                                            h_grid,
                                            dims,
                                            config,
                                            model,
                                            phi_i=phi_i,
                                            w_i=w_i)
  return initial_state


def init_baroclinic_wave_state(h_grid,
                               v_grid,
                               model_config,
                               test_config,
                               dims,
                               model,
                               mountain=False,
                               moist=False,
                               eps=1e-6,
                               pert_type=perturbation_opts.none,
                               enforce_hydrostatic=False):
  lat = h_grid["physical_coords"][:, :, :, 0]
  deep = model in deep_atmosphere_models

  def z_pi_surf_func(lat,
                     lon):
    return eval_surface_state(lat, lon, test_config, mountain=mountain)

  def Q_func(lat, lon, z):
    return eval_state(lat,
                      lon,
                      z,
                      test_config,
                      moist=moist,
                      deep=deep,
                      pert_type=pert_type)[4]

  def p_func(z):
    return eval_pressure_temperature(z,
                                     lat,
                                     test_config,
                                     deep=deep)[0]

  def u_func(lat, lon, z):
    return eval_state(lat,
                      lon,
                      z,
                      test_config,
                      moist=moist,
                      deep=deep,
                      pert_type=pert_type)[0]

  def v_func(lat, lon, z):
    return eval_state(lat,
                      lon,
                      z,
                      test_config,
                      moist=moist,
                      deep=deep,
                      pert_type=pert_type)[1]

  def Tv_func(lat, lon, z):
    return eval_state(lat,
                      lon,
                      z,
                      test_config,
                      moist=moist,
                      deep=deep)[3]

  def w_func(lat, lon, z):
    return jnp.zeros_like(z)

  model_state = init_model_pressure(z_pi_surf_func,
                                    p_func,
                                    Tv_func,
                                    u_func,
                                    v_func,
                                    Q_func,
                                    h_grid, v_grid,
                                    model_config,
                                    dims,
                                    model,
                                    w_func=w_func,
                                    eps=eps,
                                    enforce_hydrostatic=enforce_hydrostatic)
  return model_state
