from ..config import jnp, vmap_1d_apply, jit, np
from .infra import get_delta, interface_to_model
from ..operators import sphere_vec_laplacian_wk, sphere_laplacian_wk
from .model_state import wrap_model_struct, dss_model_state
from .eos import get_balanced_phi
from .vertical_coordinate import mass_from_coordinate_interface
from functools import partial


@jit
def get_ref_states(phi_surf, v_grid, config):
  # could eventually only be called once.
  # due to low cost, if we end up going the "vmap over nelem" route,
  # then this should probably be recomputed from element-local (and ideally SM-local) phi_surf
  reference_params = config["reference_profiles"]
  ps_ref = v_grid["reference_pressure"] * jnp.exp(-phi_surf / (config["Rgas"] * reference_params["T_ref"]))
  pressure_int = mass_from_coordinate_interface(ps_ref, v_grid)
  dpi_ref = get_delta(pressure_int)
  p_mid = interface_to_model(pressure_int)
  exner = (p_mid / config["p0"])**(config["Rgas"] / config["cp"])
  T1 = reference_params["T_ref_lapse"] * reference_params["T_ref"] * config["cp"] / config["gravity"]
  T0 = reference_params["T_ref"] - T1
  theta_ref = T0 + T0 * (1 - exner) + T1
  return {"dpi": dpi_ref,
          "vtheta": theta_ref,
          "phi_i": get_balanced_phi(phi_surf, p_mid, theta_ref * dpi_ref, config)}


@jit
def scalar_harmonic_3d(scalar, h_grid, config):

  def lap_wk_onearg(scalar):
      return sphere_laplacian_wk(scalar, h_grid, a=config["radius_earth"])

  del2 = vmap_1d_apply(lap_wk_onearg, scalar, -1, -1)
  return del2


@jit
def vector_harmonic_3d(vector, h_grid, config, nu_div_factor):

  def vec_lap_wk_onearg(vector):
      return sphere_vec_laplacian_wk(vector, h_grid, a=config["radius_earth"],
                                     nu_div_fact=nu_div_factor)

  del2 = vmap_1d_apply(vec_lap_wk_onearg, vector, -2, -2)
  return del2


@partial(jit, static_argnames=["apply_nu", "hydrostatic"])
def calc_state_harmonic(state, h_grid, config, apply_nu=True, hydrostatic=True):
  if not hydrostatic:
    hyperdiff_phi_i = scalar_harmonic_3d(state["phi_i"],
                                         h_grid, config)
    hyperdiff_w_i = scalar_harmonic_3d(state["w_i"],
                                       h_grid, config)
  else:
    hyperdiff_phi_i = 0.0
    hyperdiff_w_i = 0.0

  hyperdiff_vtheta = scalar_harmonic_3d(state["vtheta_dpi"] / state["dpi"],
                                        h_grid, config)
  hyperdiff_dpi = scalar_harmonic_3d(state["dpi"], h_grid, config)
  if apply_nu:
    diff_params = config["diffusion"]
    nu_default = diff_params["nu"]
    nu_phi = diff_params["nu_phi"]
    nu_dpi = diff_params["nu_dpi"]
    nu_div_factor = diff_params["nu_div_factor"]
  else:
    nu_default = 1.0
    nu_phi = 1.0
    nu_dpi = 1.0
    nu_div_factor = 1.0

  hyperdiff_u = vector_harmonic_3d(state["u"],
                                   h_grid, config, nu_div_factor)
  return wrap_model_struct(nu_default * hyperdiff_u,
                           nu_default * hyperdiff_vtheta,
                           nu_dpi * hyperdiff_dpi,
                           state["phi_surf"],
                           state["grad_phi_surf"],
                           nu_phi * hyperdiff_phi_i,
                           nu_default * hyperdiff_w_i)


@partial(jit, static_argnames=["n_sponge"])
def get_nu_ramp(v_grid, n_sponge):
  pressure_ratio = ((v_grid["hybrid_a_i"][0] + v_grid["hybrid_b_i"][0])/
                    (v_grid["hybrid_a_i"][:n_sponge] + v_grid["hybrid_b_i"][:n_sponge]))
  nu_ramp = jnp.minimum(8.0, (16.0 * pressure_ratio**2 / (pressure_ratio**2 + 1.0))[np.newaxis, np.newaxis, np.newaxis, :])
  return nu_ramp


@partial(jit, static_argnames=["dims", "n_sponge", "hydrostatic"])
def sponge_layer(state, dt, h_grid, v_grid, config, dims, n_sponge, hydrostatic=True):
  nu_top = config["diffusion"]["nu_top"]
  nu_ramp = nu_top * get_nu_ramp(v_grid, n_sponge)
  if not hydrostatic:
    hyperdiff_phi_i = nu_ramp * scalar_harmonic_3d(state["phi_i"][:, :, :, :n_sponge],
                                                   h_grid, config)
    hyperdiff_w_i = nu_ramp * scalar_harmonic_3d(state["w_i"][:, :, :, :n_sponge],
                                                 h_grid, config)
  else:
    hyperdiff_phi_i = 0.0
    hyperdiff_w_i = 0.0

  hyperdiff_vtheta = scalar_harmonic_3d(state["vtheta_dpi"][:, :, :, :n_sponge],
                                        h_grid, config)
  hyperdiff_vtheta *= nu_ramp
  hyperdiff_dpi = scalar_harmonic_3d(state["dpi"][:, :, :, :n_sponge], h_grid, config)
  hyperdiff_dpi *= nu_ramp
  hyperdiff_u = vector_harmonic_3d(state["u"][:, :, :, :n_sponge, :],
                                   h_grid, config, 1.0)
  hyperdiff_u *= nu_ramp[:, :, :, :, np.newaxis]
  hyperdiff_state =  wrap_model_struct(hyperdiff_u,
                           hyperdiff_vtheta,
                           hyperdiff_dpi,
                           state["phi_surf"],
                           state["grad_phi_surf"],
                           hyperdiff_phi_i,
                           hyperdiff_w_i)
  hyperdiff_state = dss_model_state(hyperdiff_state,
                              h_grid,
                              dims,
                              scaled=False,
                              hydrostatic=hydrostatic)

  u_out = jnp.concatenate((dt * hyperdiff_state["u"] + state["u"][:, :, :, :n_sponge, :],
                           state["u"][:, :, :, n_sponge:, :]), axis=-2)
  vtheta_out = jnp.concatenate((dt * hyperdiff_state["vtheta_dpi"] + state["vtheta_dpi"][:, :, :, :n_sponge],
                                state["vtheta_dpi"][:, :, :, n_sponge:]), axis=-1)
  dpi_out = jnp.concatenate((dt * hyperdiff_state["dpi"] + state["dpi"][:, :, :, :n_sponge],
                             state["dpi"][:, :, :, n_sponge:]), axis=-1)
  if not hydrostatic:
    phi_i_out = jnp.concatenate((dt * hyperdiff_state["phi_i"] + state["phi_i"][:, :, :, :n_sponge],
                                 state["phi_i"][:, :, :, n_sponge:]), axis=-1)
    w_i_out = jnp.concatenate((dt * hyperdiff_state["phi_i"] + state["w_i"][:, :, :, :n_sponge],
                               state["w_i"][:, :, :, n_sponge:]), axis=-1)
  else:
    phi_i_out = 0.0
    w_i_out = 0.0

  if not hydrostatic:
    phi_i_out = jnp.concatenate((state["phi_i"][:, :, :, :n_sponge],
                                 state["phi_i"][:, :, :, n_sponge:]), axis=-1)
    w_i_out = jnp.concatenate((state["w_i"][:, :, :, :n_sponge],
                               state["w_i"][:, :, :, n_sponge:]), axis=-1)
  else:
    phi_i_out = 0.0
    w_i_out = 0.0
  struct = wrap_model_struct(u_out,
                           vtheta_out,
                           dpi_out,
                           state["phi_surf"],
                           state["grad_phi_surf"],
                           phi_i_out,
                           w_i_out)
  return struct


@partial(jit, static_argnames=["hydrostatic", "dims"])
def hypervis_terms(state, ref_state, h_grid, dims, config, hydrostatic=True):
  if hydrostatic:
    phi_i_pert = state["phi_i"] - ref_state["phi_i"]
  else:
    phi_i_pert = 0.0
  state_rhs = wrap_model_struct(state["u"],
                                state["vtheta_dpi"] / state["dpi"] - ref_state["vtheta"],
                                state["dpi"] - ref_state["dpi"],
                                state["phi_surf"],
                                state["grad_phi_surf"],
                                phi_i_pert,
                                state["w_i"])
  for apply_nu in [True, False]:
    struct_rhs = calc_state_harmonic(state_rhs,
                                     h_grid,
                                     config,
                                     apply_nu=apply_nu,
                                     hydrostatic=hydrostatic)
    struct_rhs = dss_model_state(struct_rhs,
                                 h_grid,
                                 dims,
                                 scaled=False,
                                 hydrostatic=hydrostatic)
  return struct_rhs
