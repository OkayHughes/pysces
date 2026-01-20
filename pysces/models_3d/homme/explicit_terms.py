from ...config import jnp, jit, np, device_wrapper
from ..utils_3d import vel_model_to_interface, model_to_interface, interface_to_model, interface_to_model_vec
from ..utils_3d import z_from_phi, g_from_z, g_from_phi, sphere_dot
from .thermodynamics import get_mu, get_balanced_phi, get_p_mid
from ..operators_3d import horizontal_gradient_3d, horizontal_vorticity_3d, horizontal_divergence_3d
from .homme_state import wrap_dynamics_struct
from .homme_state import project_scalar_3d
from functools import partial
from ..model_info import hydrostatic_models, deep_atmosphere_models


@partial(jit, static_argnames=["model"])
def init_common_variables(dynamics, static_forcing, h_grid, v_grid, physics_config, model):
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
  if model in hydrostatic_models:
    p_mid = get_p_mid(dynamics, v_grid, physics_config)
    phi_i = get_balanced_phi(static_forcing["phi_surf"],
                             p_mid,
                             dynamics["theta_v_d_mass"],
                             physics_config)
  else:
    phi_i = dynamics["phi_i"]
    w_i = dynamics["w_i"]

  d_mass = dynamics["d_mass"]
  u = dynamics["u"]
  radius_earth = physics_config["radius_earth"]
  theta_v_d_mass = dynamics["theta_v_d_mass"]

  d_mass_i = model_to_interface(d_mass)
  phi = interface_to_model(phi_i)
  pnh, exner, r_hat_i, mu = get_mu(dynamics, phi_i, v_grid, physics_config, model)
  if model in deep_atmosphere_models:
    r_hat_m = interface_to_model(r_hat_i)
    z = z_from_phi(phi_i, physics_config, model)
    r_m = interface_to_model(z + radius_earth)
    g = g_from_z(z, physics_config, model)
  else:
    r_hat_m = device_wrapper(jnp.ones((1, 1, 1, 1)))
    r_m = radius_earth * device_wrapper(jnp.ones((1, 1, 1, 1)))
    g = physics_config["gravity"] * device_wrapper(jnp.ones((1, 1, 1, 1)))
  if model not in  hydrostatic_models:
    w_m = interface_to_model(w_i)
    grad_w_i = horizontal_gradient_3d(w_i, h_grid, physics_config)
  else:
    w_m = None
    grad_w_i = None

  grad_exner = horizontal_gradient_3d(exner, h_grid, physics_config) / r_hat_m
  theta_v = theta_v_d_mass / d_mass
  grad_phi_i = horizontal_gradient_3d(phi_i, h_grid, physics_config)
  v_over_r_hat_i = vel_model_to_interface(u / r_hat_m[:, :, :, np.newaxis],
                                          d_mass, d_mass_i)
  div_dp = horizontal_divergence_3d(d_mass[:, :, :, :, np.newaxis] * u /
                                r_hat_m[:, :, :, :, np.newaxis], h_grid, physics_config)
  u_i = vel_model_to_interface(u, d_mass, d_mass_i)
  common_variables = {"phi_i": phi_i,
                      "phi": phi,
                      "d_mass_i": d_mass_i,
                      "pnh": pnh,
                      "exner": exner,
                      "r_hat_i": r_hat_i,
                      "mu": mu,
                      "r_hat_m": r_hat_m,
                      "r_m": r_m,
                      "g": g,
                      "fcor": static_forcing["coriolis_param"],
                      "grad_exner": grad_exner,
                      "theta_v": theta_v,
                      "grad_phi_i": grad_phi_i,
                      "v_over_r_hat_i": v_over_r_hat_i,
                      "div_d_mass": div_dp,
                      "u_i": u_i,
                      "u": u,
                      "theta_v_d_mass": theta_v_d_mass}
  if model not in hydrostatic_models:
    common_variables["w_i"] = w_i
    common_variables["w_m"] = w_m
    common_variables["grad_w_i"] = grad_w_i
  if model in deep_atmosphere_models:
    common_variables["nontrad_coriolis_param"] = static_forcing["nontrad_coriolis_param"]
  return common_variables


@jit
def vorticity_term(common_variables, h_grid, config):
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
  u = common_variables["u"]
  fcor = common_variables["coriolis_param"]
  vort = horizontal_vorticity_3d(u, h_grid, config)
  vort /= common_variables["r_hat_m"]
  vort_term = jnp.stack((u[:, :, :, :, 1] * (fcor[:, :, :, np.newaxis] + vort),
                         -u[:, :, :, :, 0] * (fcor[:, :, :, np.newaxis] + vort)), axis=-1)
  return vort_term


@jit
def grad_kinetic_energy_h_term(common_variables, h_grid, config):
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
  u = common_variables["u"]
  grad_kinetic_energy = horizontal_gradient_3d((u[:, :, :, :, 0]**2 +
                                                u[:, :, :, :, 1]**2) / 2.0, h_grid, config)
  return -grad_kinetic_energy / common_variables["r_hat_m"]


@jit
def grad_kinetic_energy_v_term(common_variables, h_grid, config):
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
  w_i = common_variables["w_i"]
  w_sq_m = interface_to_model(w_i * w_i) / 2.0
  w2_grad_sph = horizontal_gradient_3d(w_sq_m, h_grid, config) / common_variables["r_hat_m"]
  return -w2_grad_sph


@jit
def w_vorticity_correction_term(common_variables):
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
  w_grad_w_m = interface_to_model_vec(common_variables["w_i"][:, :, :, :, np.newaxis] *
                                      common_variables["grad_w_i"])
  w_grad_w_m /= common_variables["r_hat_m"][:, :, :, :, np.newaxis]
  return w_grad_w_m


@jit
def u_metric_term(common_variables):
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
  return -(common_variables["w_m"][:, :, :, :, np.newaxis] * common_variables["u"] /
           common_variables["r_m"][:, :, :, np.newaxis])


@jit
def u_nct_term(common_variables):
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
  w_m = common_variables["w_m"]
  fcorcos = common_variables["nontrad_coriolis_param"]
  return -jnp.stack((w_m, jnp.zeros_like(w_m)), axis=-1) * fcorcos[:, :, :, np.newaxis, np.newaxis]


@jit
def pgrad_pressure_term(common_variables, h_grid, config):
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
  theta_v = common_variables["theta_v"]
  exner = common_variables["exner"]
  r_hat_m = common_variables["r_hat_m"]
  grad_p_term_1 = config["cp"] * theta_v[:, :, :, :, np.newaxis] * common_variables["grad_exner"]
  grad_theta_v_exner = horizontal_gradient_3d(theta_v * exner, h_grid, config) / r_hat_m
  grad_theta_v = horizontal_gradient_3d(theta_v, h_grid, config) / r_hat_m
  grad_p_term_2 = config["cp"] * (grad_theta_v_exner - exner[:, :, :, :, np.newaxis] * grad_theta_v)
  return -(grad_p_term_1 + grad_p_term_2) / 2.0


@jit
def pgrad_phi_term(common_variables):
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
  pgf_grad_phi_m = interface_to_model_vec(common_variables["mu"][:, :, :, :, np.newaxis] * common_variables["grad_phi_i"])
  pgf_grad_phi_m /= common_variables["r_hat_m"][:, :, :, :, np.newaxis]
  return -pgf_grad_phi_m


@jit
def w_advection_term(common_variables):
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
  v_over_r_hat_i = common_variables["v_over_r_hat_i"]
  grad_w_i = common_variables["grad_w_i"]
  v_grad_w_i = (v_over_r_hat_i[:, :, :, :, 0] * grad_w_i[:, :, :, :, 0] +
                v_over_r_hat_i[:, :, :, :, 1] * grad_w_i[:, :, :, :, 1])
  return -v_grad_w_i


@jit
def w_metric_term(common_variables):
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
  v_sq_over_r_i = vel_model_to_interface(common_variables["u"]**2 / common_variables["r_m"],
                                         common_variables["d_mass"],
                                         common_variables["d_mass_i"])
  return (v_sq_over_r_i[:, :, :, :, 0] + v_sq_over_r_i[:, :, :, :, 1])


@jit
def w_nct_term(common_variables):
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
  fcorcos = common_variables["nontrad_coriolis_param"]
  return common_variables["u_i"][:, :, :, :, 0] * fcorcos[:, :, :, np.newaxis]


@jit
def w_buoyancy_term(common_variables):
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
  return -common_variables["g"] * (1 - common_variables["mu"])


@jit
def phi_advection_term(common_variables):
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
  v_over_r_hat_i = common_variables["v_over_r_hat_i"]
  grad_phi_i = common_variables["grad_phi_i"]
  v_grad_phi_i = (v_over_r_hat_i[:, :, :, :, 0] * grad_phi_i[:, :, :, :, 0] +
                  v_over_r_hat_i[:, :, :, :, 1] * grad_phi_i[:, :, :, :, 1])
  return -v_grad_phi_i


@jit
def phi_acceleration_v_term(common_variables):
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
  return common_variables["g"] * common_variables["w_i"]


@jit
def theta_v_divergence_term(common_variables, h_grid, config):
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
  r_hat_m = common_variables["r_hat_m"]
  theta_v = common_variables["theta_v"]
  u = common_variables["u"]
  div_d_mass = common_variables["div_d_mass"]
  d_mass = common_variables["d_mass"]
  v_theta_v = common_variables["u"] * common_variables["theta_v_d_mass"][:, :, :, :, np.newaxis]
  v_theta_v /= r_hat_m
  div_v_theta_v = horizontal_divergence_3d(v_theta_v, h_grid, config) / 2.0
  grad_theta_v = horizontal_gradient_3d(theta_v, h_grid, config)
  grad_theta_v /= r_hat_m

  div_v_theta_v += (theta_v * div_d_mass + (d_mass * (u[:, :, :, :, 0] * grad_theta_v[:, :, :, :, 0] +
                                                      u[:, :, :, :, 1] * grad_theta_v[:, :, :, :, 1]))) / 2.0
  return -div_v_theta_v


@jit
def d_mass_divergence_term(common_variables):
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
  return -common_variables["div_d_mass"]


@partial(jit, static_argnames=["model"])
def explicit_tendency(dynamics, static_forcing, h_grid, v_grid, config, model):
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

  common_variables = init_common_variables(dynamics, static_forcing, h_grid,
                                           v_grid, config,
                                           model)

  u_tend = (vorticity_term(common_variables, h_grid, config) +
            grad_kinetic_energy_h_term(common_variables, h_grid, config) +
            pgrad_pressure_term(common_variables, h_grid, config) +
            pgrad_phi_term(common_variables))

  if model not in hydrostatic_models:
    u_tend += (grad_kinetic_energy_v_term(common_variables, h_grid, config) +
               w_vorticity_correction_term(common_variables))
    w_tend = (w_advection_term(common_variables) +
              w_buoyancy_term(common_variables))
    phi_tend = (phi_advection_term(common_variables) +
                phi_acceleration_v_term(common_variables))
  else:
    w_tend = None
    phi_tend = None

  if model in deep_atmosphere_models:
      u_tend += (u_metric_term(common_variables) +
                 u_nct_term(common_variables))
      w_tend += (w_metric_term(common_variables) +
                 w_nct_term(common_variables))

  theta_v_d_mass_tend = theta_v_divergence_term(common_variables, h_grid, config)
  d_mass_tend = d_mass_divergence_term(common_variables)
  return wrap_dynamics_struct(u_tend,
                              theta_v_d_mass_tend,
                              d_mass_tend,
                              phi_i=phi_tend,
                              w_i=w_tend)


@partial(jit, static_argnames=["dims", "model"])
def calc_energy_quantities(dynamics, static_forcing, h_grid, v_grid, config, dims, model):
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
  common_variables = init_common_variables(dynamics, static_forcing, h_grid,
                                           v_grid, config,
                                           model)

  # !!!!!!!!!!!!!!!!!!!!!!!!!!
  # todo: incorporate mu correction.
  # !!!!!!!!!!!!!!!!!!!!!!!!!!
  d_mass_i = common_variables["d_mass_i"]

  d_mass_i_integral = jnp.concatenate((d_mass_i[:, :, :, 0:1] / 2.0,
                                       d_mass_i[:, :, :, 1:-1],
                                       d_mass_i[:, :, :, -1:] / 2.0), axis=-1)

  u = dynamics["u"]
  d_mass = dynamics["d_mass"]
  w_i = dynamics["w_i"]
  u1 = u[:, :, :, :, 0]
  u2 = u[:, :, :, :, 1]
  u_sq = sphere_dot(u, u)
  g = common_variables["g"]
  mu = common_variables["mu"]
  exner = common_variables["exner"]
  phi = common_variables["phi"]

  grad_kinetic_energy_h = grad_kinetic_energy_h_term(common_variables, h_grid, config)
  d_mass_divergence = d_mass_divergence_term(common_variables)
  phi_acceleration_v = phi_acceleration_v_term(common_variables)
  w_buoyancy = w_buoyancy_term(common_variables)
  pgrad_pressure = pgrad_pressure_term(common_variables, h_grid, config)
  pgrad_phi = pgrad_phi_term(common_variables)
  theta_v_divergence = theta_v_divergence_term(common_variables, h_grid, config)
  w_vorticity = w_vorticity_correction_term(common_variables)
  w_advection = w_advection_term(common_variables)
  u_metric = u_metric_term(common_variables)
  w_metric = w_metric_term(common_variables)
  u_nct = u_nct_term(common_variables)
  w_nct = w_nct_term(common_variables)
  grad_kinetic_energy_v = grad_kinetic_energy_v_term(common_variables, h_grid, config)
  vorticity = vorticity_term(common_variables, h_grid, config)
  phi_advection = phi_advection_term(common_variables)

  ke_ke_1_a = jnp.sum(d_mass * sphere_dot(u, grad_kinetic_energy_h), axis=-1)
  ke_ke_1_b = jnp.sum(1.0 / 2.0 * u_sq * project_scalar_3d(d_mass_divergence, h_grid, dims), axis=-1)

  ke_ke_2_a = jnp.sum(d_mass * (u1 * grad_kinetic_energy_v[:, :, :, :, 0] +
                             u2 * grad_kinetic_energy_v[:, :, :, :, 1]), axis=-1)
  ke_ke_2_b = jnp.sum(1.0 / 2.0 * interface_to_model(w_i**2) * d_mass_divergence, axis=-1)

  ke_pe_1_a = jnp.sum(d_mass_i_integral * w_i * (w_buoyancy - mu * g), axis=-1)
  ke_pe_1_b = jnp.sum(d_mass_i_integral * phi_acceleration_v, axis=-1)

  ke_ie_1_a = jnp.sum(d_mass_i_integral * -mu * phi_acceleration_v, axis=-1)
  ke_ie_1_b = jnp.sum(d_mass_i_integral * w_i * (w_buoyancy + g), axis=-1)

  ke_ie_2_a = jnp.sum(d_mass * (u1 * pgrad_pressure[:, :, :, :, 0] +
                             u2 * pgrad_pressure[:, :, :, :, 1]), axis=-1)
  ke_ie_2_b = jnp.sum(config["cp"] * exner * theta_v_divergence, axis=-1)

  ke_ie_3_a = jnp.sum(d_mass * (u1 * pgrad_phi[:, :, :, :, 0] +
                             u2 * pgrad_phi[:, :, :, :, 1]), axis=-1)
  ke_ie_3_b = jnp.sum(d_mass_i_integral * -mu * phi_advection, axis=-1)

  ke_ke_3_a = jnp.sum(d_mass * (u1 * w_vorticity[:, :, :, :, 0] +
                             u2 * w_vorticity[:, :, :, :, 1]), axis=-1)
  ke_ke_3_b = jnp.sum(d_mass_i_integral * w_i * w_advection, axis=-1)

  ke_ke_4_a = jnp.sum(d_mass * u1 * vorticity[:, :, :, :, 0], axis=-1)
  ke_ke_4_b = jnp.sum(d_mass * u2 * vorticity[:, :, :, :, 1], axis=-1)

  pe_pe_1_a = jnp.sum(phi * d_mass_divergence, axis=-1)
  pe_pe_1_b = jnp.sum(d_mass_i_integral * phi_advection, axis=-1)

  ke_ke_5_a = jnp.sum(d_mass * (u1 * u_metric[:, :, :, :, 0] +
                             u2 * u_metric[:, :, :, :, 1]), axis=-1)
  ke_ke_5_b = jnp.sum(d_mass_i_integral * w_i * w_metric, axis=-1)

  ke_ke_6_a = jnp.sum(d_mass * (u1 * u_nct[:, :, :, :, 0] +
                             u2 * u_nct[:, :, :, :, 1]), axis=-1)
  ke_ke_6_b = jnp.sum(d_mass_i_integral * w_i * w_nct, axis=-1)

  tends = explicit_tendency(dynamics, static_forcing, h_grid, v_grid, config, model)
  u_tend = tends["u"]

  ke_tend_emp = jnp.sum(d_mass * (u1 * u_tend[:, :, :, :, 0] +
                               u2 * u_tend[:, :, :, :, 1]), axis=-1)
  ke_tend_emp += jnp.sum(d_mass_i_integral * w_i * tends["w_i"], axis=-1)

  ke_tend_emp += jnp.sum(u_sq / 2.0 * tends["d_mass"], axis=-1)
  ke_tend_emp += jnp.sum(interface_to_model(w_i**2) / 2.0 * tends["d_mass"], axis=-1)

  pe_tend_emp = jnp.sum(phi * tends["d_mass"], axis=-1)
  pe_tend_emp += jnp.sum(d_mass_i_integral * tends["phi_i"], axis=-1)

  ie_tend_emp = jnp.sum(config["cp"] * exner * tends["theta_v_d_mass"], axis=-1)
  ie_tend_emp -= jnp.sum(mu * d_mass_i_integral * tends["phi_i"], axis=-1)

  pairs = {"ke_ke_1": (ke_ke_1_a, ke_ke_1_b),
           "ke_ke_2": (ke_ke_2_a, ke_ke_2_b),
           "ke_ke_3": (ke_ke_3_a, ke_ke_3_b),
           "ke_ke_4": (ke_ke_4_a, ke_ke_4_b),
           "ke_ke_5": (ke_ke_5_a, ke_ke_5_b),
           "ke_ke_6": (ke_ke_6_a, ke_ke_6_b),
           "ke_pe_1": (ke_pe_1_a, ke_pe_1_b),
           "pe_pe_1": (pe_pe_1_a, pe_pe_1_b),
           "ke_ie_1": (ke_ie_1_a, ke_ie_1_b),
           "ke_ie_2": (ke_ie_2_a, ke_ie_2_b),
           "ke_ie_3": (ke_ie_3_a, ke_ie_3_b)}
  empirical_tendencies = {"ke": ke_tend_emp,
                          "ie": ie_tend_emp,
                          "pe": pe_tend_emp}
  return pairs, empirical_tendencies


@partial(jit, static_argnames=["model"])
def correct_state(dynamics, static_forcing, dt, config, model):
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
  if model in hydrostatic_models:
    return dynamics
  u_lowest_new, w_lowest_new, mu_update = lower_boundary_correction(dynamics,
                                                                    static_forcing,
                                                                    dt,
                                                                    config,
                                                                    model)
  u_new = jnp.append((dynamics["u"][:, :, :, :-1, :],
                      u_lowest_new), axis=-2)
  if model not in hydrostatic_models:
    w_new = jnp.append((dynamics["w_i"][:, :, :, :-1],
                        w_lowest_new), axis=-1)
  else:
    w_new = dynamics["w_i"]
  return wrap_dynamics_struct(u_new,
                              dynamics["theta_v_d_mass"],
                              dynamics["d_mass"],
                              dynamics["phi_surf"],
                              dynamics["grad_phi_surf"],
                              dynamics["phi_i"],
                              w_new)


@partial(jit, static_argnames=["model"])
def lower_boundary_correction(dynamics, static_forcing, dt, config, model):
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
  # we need to pass in original state. Something is wrong here.
  if model in hydrostatic_models:
    u_corrected = dynamics["u"][:, :, :, -1, :]
    w_corrected = 0.0
    mu_surf = 1.0
  else:
    u_lowest = dynamics["u"][:, :, :, -1, :]
    w_lowest = dynamics["w_i"][:, :, :, -1]
    grad_phi_surf = static_forcing["grad_phi_surf"]
    g_surf = g_from_phi(static_forcing["phi_surf"], config, model)
    mu_surf = ((u_lowest[:, :, :, 0] * grad_phi_surf[:, :, :, 0] +
                u_lowest[:, :, :, 1] * grad_phi_surf[:, :, :, 1]) / g_surf - w_lowest)
    mu_surf /= (g_surf + 1.0 / (2.0 * g_surf) * (grad_phi_surf[:, :, :, 0]**2 +
                                                 grad_phi_surf[:, :, :, 1]**2))
    mu_surf /= dt
    mu_surf += 1.0

    w_corrected = w_lowest + dt * g_surf * (mu_surf - 1)
    u_corrected = u_lowest - dt * (mu_surf[:, :, :, np.newaxis] - 1) * grad_phi_surf / 2.0

  return u_corrected, w_corrected, mu_surf
