from ..config import jit
from .infra import exit_codes
from .model_state import wrap_model_struct, dss_model_state
from .explicit_terms import explicit_tendency, correct_state
from .hyperviscosity import hypervis_terms
from functools import partial


@jit
def rfold_state(state1, state2, fold_coeff1, fold_coeff2):
  return wrap_model_struct(state1["u"] * fold_coeff1 + state2["u"] * fold_coeff2,
                           state1["vtheta_dpi"] * fold_coeff1 + state2["vtheta_dpi"] * fold_coeff2,
                           state1["dpi"] * fold_coeff1 + state2["dpi"] * fold_coeff2,
                           state1["phi_surf"],
                           state1["grad_phi_surf"],
                           state1["phi_i"] * fold_coeff1 + state2["phi_i"] * fold_coeff2,
                           state1["w_i"] * fold_coeff1 + state2["w_i"] * fold_coeff2)


#
# def accumulate_avg_explicit_terms(averaging_weight, state_c0, tracer_struct):
#   return wrap_tracer_avg_struct(tracer_struct["avg_u"] + averaging_weight *
#                                 state_c0["u"] *
#                                 state_c0["dpi"][:, :, :, :, jnp.newaxis],
#                                 tracer_struct["avg_dpi"],
#                                 tracer_struct["avg_dpi_dissip"])


@jit
def advance_state(states, coeffs):
  state_out = rfold_state(states[0],
                          states[1],
                          coeffs[0],
                          coeffs[1])
  for coeff_idx in range(2, len(states)):
    state_out = rfold_state(state_out,
                            states[coeff_idx],
                            1.0,
                            coeffs[coeff_idx])
  return state_out


def check_nan(state):
  #
  # is_nan = False
  # for field in ["u", "vtheta_dpi", "dpi", "w_i", "phi_i"]:
  #   is_nan = is_nan or jnp.any(jnp.isnan(state[field]))
  return exit_codes["success"]


@partial(jit, static_argnames=["dims", "hydrostatic", "deep"])
def advance_euler(state_in, dt, h_grid, v_grid, config, dims, hydrostatic=True, deep=False):
  u_tend = explicit_tendency(state_in, h_grid, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)
  u1 = advance_state([state_in, u_tend_c0], [1.0, dt])
  u1_cons = correct_state(u1, dt, config, hydrostatic=hydrostatic, deep=deep)
  return u1_cons


@partial(jit, static_argnames=["dims", "n_subcycle", "hydrostatic"])
def advance_euler_hypervis(state_in, dt, h_grid, v_grid, config, dims, ref_state, n_subcycle=1, hydrostatic=True):
  state_out = state_in
  for _ in range(n_subcycle):
    hypervis_rhs = hypervis_terms(state_in, ref_state,
                                  h_grid, dims,
                                  config,
                                  hydrostatic=hydrostatic)
    state_out = advance_state([state_in, hypervis_rhs], [1.0, dt / n_subcycle])
  return state_out


@partial(jit, static_argnames=["dims", "hydrostatic", "deep"])
def ullrich_5stage(state_in, dt, h_grid, v_grid, config, dims, hydrostatic=True, deep=False):
  u_tend = explicit_tendency(state_in, h_grid, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)

  u1 = advance_state([state_in, u_tend_c0], [1.0, dt / 5.0])
  u1 = correct_state(u1, dt / 5.0, config, hydrostatic=hydrostatic, deep=deep)

  u_tend = explicit_tendency(u1, h_grid, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)

  u2 = advance_state([state_in, u_tend_c0], [1.0, dt / 5.0])
  u2 = correct_state(u2, dt / 5.0, config, hydrostatic=hydrostatic, deep=deep)

  u_tend = explicit_tendency(u2, h_grid, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)

  u3 = advance_state([state_in, u_tend_c0], [1.0, dt / 3.0])
  u3 = correct_state(u3, dt / 3.0, config, hydrostatic=hydrostatic, deep=deep)

  u_tend = explicit_tendency(u3, h_grid, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)

  u4 = advance_state([state_in, u_tend_c0], [1.0, 2.0 * dt / 3.0])
  u4 = correct_state(u4, 2.0 * dt / 3.0, config, hydrostatic=hydrostatic, deep=deep)

  u_tend = explicit_tendency(u4, h_grid, v_grid, config, hydrostatic=hydrostatic)
  u_tend_c0 = dss_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)

  final_state = advance_state([state_in, u1, u_tend_c0], [-1.0 / 4.0,
                                                          5.0 / 4.0,
                                                          3.0 * dt / 4.0])
  final_state = correct_state(final_state, 2.0 * dt / 3.0, config, hydrostatic=hydrostatic, deep=deep)

  return final_state
