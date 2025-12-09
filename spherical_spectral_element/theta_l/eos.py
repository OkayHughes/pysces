from ..config import jnp, jit
from .infra import model_to_interface, get_delta, r_hat_from_phi
from functools import partial


@jit
def get_r_hat_sq_avg(r_hat_i):
  r_hat_sq = (r_hat_i[:, :, :, :-1] * r_hat_i[:, :, :, 1:] +
              r_hat_i[:, :, :, :-1] * r_hat_i[:, :, :, :-1] +
              r_hat_i[:, :, :, 1:] * r_hat_i[:, :, :, 1:]) / 3.0
  return r_hat_sq


@jit
def p_exner_nonhydrostatic(vtheta_dpi, dphi, r_hat_sq_avg, config):
  p0 = config["p0"]
  pnh_over_exner = -config["Rgas"] * vtheta_dpi / dphi
  pnh_over_exner /= r_hat_sq_avg
  pnh = p0 * (pnh_over_exner / p0)**(1.0 /
                                     (1.0 - config["Rgas"] /
                                      config["cp"]))
  return pnh, pnh / pnh_over_exner


@partial(jit, static_argnames=["hydrostatic", "deep"])
def get_mu(state, phi_i, v_grid, config, deep=False, hydrostatic=True):
  # note: assumes that phi_i is in hydrostatic balance.
  vtheta_dpi = state["vtheta_dpi"]
  dphi = get_delta(phi_i)
  if deep:
    r_hat_i = r_hat_from_phi(phi_i, config, deep=deep)
    r_hat_sq_avg = get_r_hat_sq_avg(r_hat_i)
  else:
    r_hat_i = 1.0
    r_hat_sq_avg = 1.0
  p_model, exner = p_exner_nonhydrostatic(vtheta_dpi, dphi, r_hat_sq_avg, config)
  if hydrostatic:
    dpnh_dpi = 1.0
  else:
    p_top = v_grid["hybrid_a_i"][0] * v_grid["reference_pressure"]
    if deep:
      p_top /= r_hat_i[:, :, :, 0]**2
    dpi_i = model_to_interface(state["dpi"])
    dpnh_dpi_top = 2 * (p_model[:, :, :, 0] - p_top) / dpi_i[:, :, :, 0]
    dpnh_dpi_bottom = jnp.ones_like(p_model[:, :, :, 0])
    dpnh_dpi_int = get_delta(p_model) / dpi_i[:, :, :, 1:-1]
    dpnh_dpi = jnp.concatenate((dpnh_dpi_top[:, :, :, jnp.newaxis],
                                dpnh_dpi_int,
                                dpnh_dpi_bottom[:, :, :, jnp.newaxis]), axis=-1)
    if deep:
      dpnh_dpi *= r_hat_i**2
  return p_model, exner, r_hat_i, dpnh_dpi


@jit
def get_p_mid(state, v_grid, config):
  p = jnp.cumsum(state["dpi"], axis=-1) + v_grid["hybrid_a_i"][0] * v_grid["reference_pressure"]
  p -= 0.5 * state["dpi"]
  return p


@jit
def get_balanced_phi(phi_surf, p_mid, vtheta_dpi, config):
  #p = get_p_mid(state, v_grid, config)
  dphi = config["Rgas"] * (vtheta_dpi *
                           (p_mid / config["p0"])**(config["Rgas"] / config["cp"] - 1.0) / config["p0"])
  dphi_augment = jnp.concatenate((dphi[:, :, :, :-1],
                                  (dphi[:, :, :, -1] + phi_surf)[:, :, :, jnp.newaxis]),
                                 axis=-1)[:, :, :, ::-1]
  phi_i_above_surf = jnp.cumsum(dphi_augment, axis=-1)
  return jnp.concatenate((phi_i_above_surf[:, :, :, ::-1], phi_surf[:, :, :, jnp.newaxis]), axis=-1)
