from ..config import jnp
from .model_state import wrap_model_struct, dss_model_state, wrap_tracer_avg_struct
from .explicit_terms import explicit_tendency, lower_boundary_correction
from .hyperviscosity import hypervis_terms


def rfold_state(state1, state2, fold_coeff1, fold_coeff2):
  return wrap_model_struct(state1["u"] * fold_coeff1 + state2("u") * fold_coeff2,
                           state1["vtheta_dpi"] * fold_coeff1 + state2("vtheta_dpi") * fold_coeff2,
                           state1["dpi"] * fold_coeff1 + state2["vtheta_dpi"] * fold_coeff2,
                           state1["phi_surf"],
                           state1["grad_phi_surf"],
                           state1["phi_i"] * fold_coeff1 + state2["phi_i"] * fold_coeff2,
                           state1["w_i"] * fold_coeff1 + state2["w_i"] * fold_coeff2)


def accumulate_avg_explicit_terms(averaging_weight, state_c0, tracer_struct):
  return wrap_tracer_avg_struct(tracer_struct["avg_u"] + averaging_weight *
                                state_c0["u"] *
                                state_c0["dpi"][:, :, :, :, jnp.newaxis],
                                tracer_struct["avg_dpi"],
                                tracer_struct["avg_dpi_dissip"])


def advance_state(states, coeffs):
  for coeff_idx in len(states) - 1:
    state_out = rfold_state(states[coeff_idx],
                            states[coeff_idx + 1],
                            coeffs[coeff_idx],
                            coeffs[coeff_idx + 1])
  return state_out


def advance_euler(state_in, dt, h_grid, v_grid, config, dims, hydrostatic=True, deep=False):
  u_tend = explicit_tendency(state_in, h_grid, v_grid, config, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)
  u1 = advance_state([state_in, u_tend_c0], [1.0, dt])
  return lower_boundary_correction(state_in, u1, dt, h_grid, v_grid, dims, config, hydrostatic=hydrostatic, deep=deep)


def advance_euler_hypervis(state_in, dt, h_grid, v_grid, config, dims, ref_state, n_subcycle=1, hydrostatic=True):
  hypervis_rhs = hypervis_terms(state_in, ref_state,
                                h_grid, dims,
                                config,
                                hydrostatic=hydrostatic)
  


def ullrich_5stage(state_in, dt, h_grid, v_grid, config, dims, hydrostatic=True, deep=False):
  u_tend = explicit_tendency(state_in, h_grid, v_grid, config, dims, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)
  u1 = advance_state([state_in, u_tend_c0], [1.0, dt / 5.0])
  u1 = lower_boundary_correction(state_in, u1, dt / 5.0, h_grid, v_grid, hydrostatic=hydrostatic, deep=deep)
  u_tend = explicit_tendency(u1, h_grid, v_grid, config, dims, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)
  u2 = advance_state([state_in, u_tend_c0], [1.0, dt / 5.0])
  u2 = lower_boundary_correction(state_in, u2, dt / 5.0, h_grid, v_grid, config, dims,
                                 hydrostatic=hydrostatic, deep=deep)
  u_tend = explicit_tendency(u2, h_grid, v_grid, config, dims, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)
  u3 = advance_state([state_in, u_tend_c0], [1.0, dt / 3.0])
  u3 = lower_boundary_correction(state_in, u3, dt / 3.0, h_grid, v_grid, config, dims,
                                 hydrostatic=hydrostatic, deep=deep)
  u_tend = explicit_tendency(u3, h_grid, v_grid, config, dims, hydrostatic=hydrostatic, deep=deep)
  u_tend_c0 = dss_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)
  u4 = advance_state([state_in, u_tend_c0], [1.0, 2.0 * dt / 3.0])
  u4 = lower_boundary_correction(state_in, u4, 2.0 * dt / 3.0, h_grid, v_grid, config, dims,
                                 hydrostatic=hydrostatic, deep=deep)
  u_tend = explicit_tendency(u4, h_grid, v_grid, config, dims, hydrostatic=hydrostatic)
  u_tend_c0 = dss_model_state(u_tend, h_grid, dims, hydrostatic=hydrostatic)
  return advance_state([state_in, u1, u_tend_c0], [-1.0 / 4.0,
                                                   5.0 / 4.0,
                                                   3.0 * dt / 4.0])
