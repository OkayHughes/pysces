from ...config import jnp, jit, np, device_wrapper
from ..utils_3d import vel_model_to_interface, model_to_interface, interface_to_model, interface_to_model_vec
from ..utils_3d import z_from_phi, g_from_z, g_from_phi, sphere_dot
from .eqn_of_state import get_mu, get_balanced_phi, get_p_mid
from ..operators_3d import sphere_gradient_3d, sphere_vorticity_3d, sphere_divergence_3d
from .model_state import wrap_model_struct
from .model_state import dss_scalar_3d
from functools import partial


@partial(jit, static_argnames=["hydrostatic", "deep"])
def calc_shared_quantities(state, h_grid, v_grid, config, hydrostatic=True, deep=False):
  if hydrostatic:
    p_mid = get_p_mid(state, v_grid, config)
    phi_i = get_balanced_phi(state["phi_surf"],
                             p_mid,
                             state["vtheta_dpi"],
                             config)
  else:
    phi_i = state["phi_i"]
  w_i = state["w_i"]
  dpi = state["dpi"]
  u = state["u"]
  radius_earth = config["radius_earth"]
  period_earth = config["period_earth"]
  lat = h_grid["physical_coords"][:, :, :, 0]

  dpi_i = model_to_interface(dpi)
  phi = interface_to_model(phi_i)
  pnh, exner, r_hat_i, mu = get_mu(state, phi_i, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  if deep:
    r_hat_m = interface_to_model(r_hat_i)
    z = z_from_phi(phi_i, config, deep=deep)
    r_m = interface_to_model(z + radius_earth)
    g = g_from_z(z, config, deep=deep)
  else:
    r_hat_m = device_wrapper(jnp.ones((1, 1, 1, 1)))
    r_m = radius_earth * device_wrapper(jnp.ones((1, 1, 1, 1)))
    g = config["gravity"]
  fcor = 2.0 * period_earth * jnp.sin(lat)
  fcorcos = 2.0 * period_earth * jnp.cos(lat)
  if not hydrostatic:
    w_m = interface_to_model(w_i)
    grad_w_i = sphere_gradient_3d(w_i, h_grid, config)
  else:
    w_m = 0.0
    grad_w_i = 0.0

  grad_exner = sphere_gradient_3d(exner, h_grid, config) / r_hat_m
  vtheta = state["vtheta_dpi"] / dpi
  grad_phi_i = sphere_gradient_3d(phi_i, h_grid, config)
  v_over_r_hat_i = vel_model_to_interface(u / r_hat_m[:, :, :, np.newaxis],
                                          dpi, dpi_i)
  div_dp = sphere_divergence_3d(dpi[:, :, :, :, np.newaxis] * u /
                                r_hat_m[:, :, :, :, np.newaxis], h_grid, config)
  u_i = vel_model_to_interface(u, dpi, dpi_i)
  return (phi_i, phi, dpi_i, pnh, exner,
          r_hat_i, mu, r_hat_m, r_m, g,
          fcor, fcorcos, w_m,
          grad_w_i, grad_exner, vtheta,
          grad_phi_i, v_over_r_hat_i,
          div_dp, u_i)


@jit
def vorticity_term(u, fcor, r_hat_m, h_grid, config):
  vort = sphere_vorticity_3d(u, h_grid, config)
  vort /= r_hat_m
  vort_term = jnp.stack((u[:, :, :, :, 1] * (fcor[:, :, :, np.newaxis] + vort),
                         -u[:, :, :, :, 0] * (fcor[:, :, :, np.newaxis] + vort)), axis=-1)
  return vort_term


@jit
def grad_kinetic_energy_h_term(u, r_hat_m, h_grid, config):
  grad_kinetic_energy = sphere_gradient_3d((u[:, :, :, :, 0]**2 +
                                            u[:, :, :, :, 1]**2) / 2.0, h_grid, config)
  return -grad_kinetic_energy / r_hat_m


@jit
def grad_kinetic_energy_v_term(w_i, r_hat_m, h_grid, config):
  w_sq_m = interface_to_model(w_i * w_i) / 2.0
  w2_grad_sph = sphere_gradient_3d(w_sq_m, h_grid, config) / r_hat_m
  return -w2_grad_sph


@jit
def w_vorticity_correction_term(w_i, grad_w_i, r_hat_m):
  w_grad_w_m = interface_to_model_vec(w_i[:, :, :, :, np.newaxis] * grad_w_i)
  w_grad_w_m /= r_hat_m[:, :, :, :, np.newaxis]
  return w_grad_w_m


@jit
def u_metric_term(u, w_m, r_m):
  return -w_m[:, :, :, :, np.newaxis] * u / r_m[:, :, :, np.newaxis]


@jit
def u_nct_term(w_m, fcorcos):
  return -jnp.stack((w_m, jnp.zeros_like(w_m)), axis=-1) * fcorcos[:, :, :, np.newaxis, np.newaxis]


@jit
def pgrad_pressure_term(vtheta, grad_exner, exner, r_hat_m, h_grid, config):
  grad_p_term_1 = config["cp"] * vtheta[:, :, :, :, np.newaxis] * grad_exner
  grad_vtheta_exner = sphere_gradient_3d(vtheta * exner, h_grid, config) / r_hat_m
  grad_vtheta = sphere_gradient_3d(vtheta, h_grid, config) / r_hat_m
  grad_p_term_2 = config["cp"] * (grad_vtheta_exner - exner[:, :, :, :, np.newaxis] * grad_vtheta)
  return -(grad_p_term_1 + grad_p_term_2) / 2.0


@jit
def pgrad_phi_term(mu, grad_phi_i, r_hat_m):
  pgf_gradphi_m = interface_to_model_vec(mu[:, :, :, :, np.newaxis] * grad_phi_i)
  pgf_gradphi_m /= r_hat_m[:, :, :, :, np.newaxis]
  return -pgf_gradphi_m


@jit
def w_advection_term(v_over_r_hat_i, grad_w_i):
  v_grad_w_i = (v_over_r_hat_i[:, :, :, :, 0] * grad_w_i[:, :, :, :, 0] +
                v_over_r_hat_i[:, :, :, :, 1] * grad_w_i[:, :, :, :, 1])
  return -v_grad_w_i


@jit
def w_metric_term(u, r_m, dpi, dpi_i):
  v_sq_over_r_i = vel_model_to_interface(u**2 / r_m, dpi, dpi_i)
  return (v_sq_over_r_i[:, :, :, :, 0] + v_sq_over_r_i[:, :, :, :, 1])


@jit
def w_nct_term(u_i, fcorcos):
  return u_i[:, :, :, :, 0] * fcorcos[:, :, :, np.newaxis]


@jit
def w_buoyancy_term(g, mu):
  return -g * (1 - mu)


@jit
def phi_advection_term(v_over_r_hat_i, grad_phi_i):
  v_grad_phi_i = (v_over_r_hat_i[:, :, :, :, 0] * grad_phi_i[:, :, :, :, 0] +
                  v_over_r_hat_i[:, :, :, :, 1] * grad_phi_i[:, :, :, :, 1])
  return -v_grad_phi_i


@jit
def phi_acceleration_v_term(g, w_i):
  return g * w_i


@jit
def vtheta_divergence_term(u, vtheta_dpi, vtheta, div_dp, dpi, r_hat_m, h_grid, config):
  v_vtheta = u * vtheta_dpi[:, :, :, :, np.newaxis]
  v_vtheta /= r_hat_m
  div_v_vtheta = sphere_divergence_3d(v_vtheta, h_grid, config) / 2.0
  grad_vtheta = sphere_gradient_3d(vtheta, h_grid, config)
  grad_vtheta /= r_hat_m

  div_v_vtheta += (vtheta * div_dp + (dpi * (u[:, :, :, :, 0] * grad_vtheta[:, :, :, :, 0] +
                                             u[:, :, :, :, 1] * grad_vtheta[:, :, :, :, 1]))) / 2.0
  return -div_v_vtheta


@jit
def dpi_divergence_term(div_dp):
  return -div_dp


@partial(jit, static_argnames=["hydrostatic", "deep"])
def explicit_tendency(state, h_grid, v_grid, config, hydrostatic=True, deep=False):
  dpi = state["dpi"]
  u = state["u"]
  w_i = state["w_i"]
  vtheta_dpi = state["vtheta_dpi"]

  (phi_i, phi, dpi_i, pnh, exner,
   r_hat_i, mu, r_hat_m, r_m, g,
   fcor, fcorcos, w_m,
   grad_w_i, grad_exner, vtheta,
   grad_phi_i, v_over_r_hat_i,
   div_dp, u_i) = calc_shared_quantities(state, h_grid,
                                         v_grid, config,
                                         hydrostatic=hydrostatic,
                                         deep=deep)
  if hydrostatic:
    mu = device_wrapper(mu)[np.newaxis, np.newaxis, np.newaxis, np.newaxis]

  u_tend = (vorticity_term(u, fcor, r_hat_m, h_grid, config) +
            grad_kinetic_energy_h_term(u, r_hat_m, h_grid, config) +
            pgrad_pressure_term(vtheta, grad_exner, exner, r_hat_m, h_grid, config) +
            pgrad_phi_term(mu, grad_phi_i, r_hat_m))
  if not hydrostatic:
    u_tend += (grad_kinetic_energy_v_term(w_i, r_hat_m, h_grid, config) +
               w_vorticity_correction_term(w_i, grad_w_i, r_hat_m))
    if deep:
      u_tend += (u_metric_term(u, w_m, r_m) +
                 u_nct_term(w_m, fcorcos))
  if not hydrostatic:
    w_tend = (w_advection_term(v_over_r_hat_i, grad_w_i) +
              w_buoyancy_term(g, mu))
    if deep:
      w_tend += (w_metric_term(u, r_m, dpi, dpi_i) +
                 w_nct_term(u_i, fcorcos))
  else:
    w_tend = 0.0
  if not hydrostatic:
    phi_tend = (phi_advection_term(v_over_r_hat_i, grad_phi_i) +
                phi_acceleration_v_term(g, w_i))
  else:
    phi_tend = 0.0

  vtheta_dpi_tend = vtheta_divergence_term(u, vtheta_dpi, vtheta, div_dp, dpi, r_hat_m, h_grid, config)
  dpi_tend = dpi_divergence_term(div_dp)
  return wrap_model_struct(u_tend,
                           vtheta_dpi_tend,
                           dpi_tend,
                           state["phi_surf"],
                           state["grad_phi_surf"],
                           phi_tend,
                           w_tend)


@partial(jit, static_argnames=["dims", "deep"])
def calc_energy_quantities(state, h_grid, v_grid, config, dims, deep=False):
  (phi_i, phi, dpi_i, pnh, exner,
   r_hat_i, mu, r_hat_m, r_m, g,
   fcor, fcorcos, w_m,
   grad_w_i, grad_exner, vtheta,
   grad_phi_i, v_over_r_hat_i,
   div_dp, u_i) = calc_shared_quantities(state, h_grid,
                                         v_grid, config,
                                         hydrostatic=False,
                                         deep=deep)

  # !!!!!!!!!!!!!!!!!!!!!!!!!!
  # todo: incorporate mu correction.
  # !!!!!!!!!!!!!!!!!!!!!!!!!!

  dpi_i_integral = jnp.concatenate((dpi_i[:, :, :, 0:1] / 2.0,
                                    dpi_i[:, :, :, 1:-1],
                                    dpi_i[:, :, :, -1:] / 2.0), axis=-1)

  u = state["u"]
  dpi = state["dpi"]
  w_i = state["w_i"]
  vtheta_dpi = state["vtheta_dpi"]
  u1 = u[:, :, :, :, 0]
  u2 = u[:, :, :, :, 1]
  u_sq = sphere_dot(u, u)

  grad_kinetic_energy_h = grad_kinetic_energy_h_term(u, r_hat_m, h_grid, config)
  dpi_divergence = dpi_divergence_term(div_dp)
  phi_acceleration_v = phi_acceleration_v_term(g, w_i)
  w_buoyancy = w_buoyancy_term(g, mu)
  pgrad_pressure = pgrad_pressure_term(vtheta, grad_exner, exner, r_hat_m, h_grid, config)
  pgrad_phi = pgrad_phi_term(mu, grad_phi_i, r_hat_m)
  vtheta_divergence = vtheta_divergence_term(u, vtheta_dpi, vtheta, div_dp, dpi, r_hat_m, h_grid, config)
  w_vorticity = w_vorticity_correction_term(w_i, grad_w_i, r_hat_m)
  w_advection = w_advection_term(v_over_r_hat_i, grad_w_i)
  u_metric = u_metric_term(u, w_m, r_m)
  w_metric = w_metric_term(u, r_m, dpi, dpi_i)
  u_nct = u_nct_term(w_m, fcorcos)
  w_nct = w_nct_term(u_i, fcorcos)
  grad_kinetic_energy_v = grad_kinetic_energy_v_term(w_i, r_hat_m, h_grid, config)
  vorticity = vorticity_term(u, fcor, r_hat_m, h_grid, config)
  phi_advection = phi_advection_term(v_over_r_hat_i, grad_phi_i)
  ke_ke_1_a = jnp.sum(dpi * sphere_dot(u, grad_kinetic_energy_h), axis=-1)
  ke_ke_1_b = jnp.sum(1.0 / 2.0 * u_sq * dss_scalar_3d(dpi_divergence, h_grid, dims), axis=-1)

  ke_ke_2_a = jnp.sum(dpi * (u1 * grad_kinetic_energy_v[:, :, :, :, 0] +
                             u2 * grad_kinetic_energy_v[:, :, :, :, 1]), axis=-1)
  ke_ke_2_b = jnp.sum(1.0 / 2.0 * interface_to_model(w_i**2) * dpi_divergence, axis=-1)

  ke_pe_1_a = jnp.sum(dpi_i_integral * w_i * (w_buoyancy - mu * g), axis=-1)
  ke_pe_1_b = jnp.sum(dpi_i_integral * phi_acceleration_v, axis=-1)

  ke_ie_1_a = jnp.sum(dpi_i_integral * -mu * phi_acceleration_v, axis=-1)
  ke_ie_1_b = jnp.sum(dpi_i_integral * w_i * (w_buoyancy + g), axis=-1)

  ke_ie_2_a = jnp.sum(dpi * (u1 * pgrad_pressure[:, :, :, :, 0] +
                             u2 * pgrad_pressure[:, :, :, :, 1]), axis=-1)
  ke_ie_2_b = jnp.sum(config["cp"] * exner * vtheta_divergence, axis=-1)

  ke_ie_3_a = jnp.sum(dpi * (u1 * pgrad_phi[:, :, :, :, 0] +
                             u2 * pgrad_phi[:, :, :, :, 1]), axis=-1)
  ke_ie_3_b = jnp.sum(dpi_i_integral * -mu * phi_advection, axis=-1)

  ke_ke_3_a = jnp.sum(dpi * (u1 * w_vorticity[:, :, :, :, 0] +
                             u2 * w_vorticity[:, :, :, :, 1]), axis=-1)
  ke_ke_3_b = jnp.sum(dpi_i_integral * w_i * w_advection, axis=-1)

  ke_ke_4_a = jnp.sum(dpi * u1 * vorticity[:, :, :, :, 0], axis=-1)
  ke_ke_4_b = jnp.sum(dpi * u2 * vorticity[:, :, :, :, 1], axis=-1)

  pe_pe_1_a = jnp.sum(phi * dpi_divergence, axis=-1)
  pe_pe_1_b = jnp.sum(dpi_i_integral * phi_advection, axis=-1)

  ke_ke_5_a = jnp.sum(dpi * (u1 * u_metric[:, :, :, :, 0] +
                             u2 * u_metric[:, :, :, :, 1]), axis=-1)
  ke_ke_5_b = jnp.sum(dpi_i_integral * w_i * w_metric, axis=-1)

  ke_ke_6_a = jnp.sum(dpi * (u1 * u_nct[:, :, :, :, 0] +
                             u2 * u_nct[:, :, :, :, 1]), axis=-1)
  ke_ke_6_b = jnp.sum(dpi_i_integral * w_i * w_nct, axis=-1)

  tends = explicit_tendency(state, h_grid, v_grid, config, hydrostatic=False, deep=deep)
  u_tend = tends["u"]

  ke_tend_emp = jnp.sum(dpi * (u1 * u_tend[:, :, :, :, 0] +
                               u2 * u_tend[:, :, :, :, 1]), axis=-1)
  ke_tend_emp += jnp.sum(dpi_i_integral * w_i * tends["w_i"], axis=-1)

  ke_tend_emp += jnp.sum(u_sq / 2.0 * tends["dpi"], axis=-1)
  ke_tend_emp += jnp.sum(interface_to_model(w_i**2) / 2.0 * tends["dpi"], axis=-1)

  pe_tend_emp = jnp.sum(phi * tends["dpi"], axis=-1)
  pe_tend_emp += jnp.sum(dpi_i_integral * tends["phi_i"], axis=-1)

  ie_tend_emp = jnp.sum(config["cp"] * exner * tends["vtheta_dpi"], axis=-1)
  ie_tend_emp -= jnp.sum(mu * dpi_i_integral * tends["phi_i"], axis=-1)

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


@partial(jit, static_argnames=["hydrostatic", "deep"])
def correct_state(state_in, dt, config, hydrostatic=True, deep=False):
  if hydrostatic:
    return state_in
  u_lowest_new, w_lowest_new, mu_update = lower_boundary_correction(state_in,
                                                                    dt,
                                                                    config,
                                                                    hydrostatic=hydrostatic,
                                                                    deep=deep)
  u_new = jnp.append((state_in["u"][:, :, :, :-1, :],
                      u_lowest_new), axis=-2)
  if not hydrostatic:
    w_new = jnp.append((state_in["w_i"][:, :, :, :-1],
                        w_lowest_new), axis=-1)
  else:
    w_new = state_in["w_i"]
  return wrap_model_struct(u_new,
                           state_in["vtheta_dpi"],
                           state_in["dpi"],
                           state_in["phi_surf"],
                           state_in["grad_phi_surf"],
                           state_in["phi_i"],
                           w_new)


@partial(jit, static_argnames=["hydrostatic", "deep"])
def lower_boundary_correction(state_in, dt, config, hydrostatic=True, deep=False):
  # we need to pass in original state. Something is wrong here.
  if hydrostatic:
    u_corrected = state_in["u"][:, :, :, -1, :]
    w_corrected = 0.0
    mu_surf = 1.0
  else:
    u_lowest = state_in["u"][:, :, :, -1, :]
    w_lowest = state_in["w_i"][:, :, :, -1]
    grad_phi_surf = state_in["grad_phi_surf"]
    g_surf = g_from_phi(state_in["phi_surf"], config, deep=deep)
    mu_surf = ((u_lowest[:, :, :, 0] * grad_phi_surf[:, :, :, 0] +
                u_lowest[:, :, :, 1] * grad_phi_surf[:, :, :, 1]) / g_surf - w_lowest)
    mu_surf /= (g_surf + 1.0 / (2.0 * g_surf) * (grad_phi_surf[:, :, :, 0]**2 +
                                                 grad_phi_surf[:, :, :, 1]**2))
    mu_surf /= dt
    mu_surf += 1.0

    w_corrected = w_lowest + dt * g_surf * (mu_surf - 1)
    u_corrected = u_lowest - dt * (mu_surf[:, :, :, np.newaxis] - 1) * grad_phi_surf / 2.0

  return u_corrected, w_corrected, mu_surf
