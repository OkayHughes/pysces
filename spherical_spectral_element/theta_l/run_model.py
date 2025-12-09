from ..config import jnp
from .infra import succeeded, exit_codes
from .hyperviscosity import get_ref_states
from .time_stepping import advance_euler, advance_euler_hypervis, ullrich_5stage


def simulate_theta(end_time, ne_min, state_in,
                   h_grid, v_grid, config,
                   dims, hydrostatic=True, deep=False,
                   diffusion=False, step_type="euler"):
  dt = 100.0 * (30.0 / ne_min)  # todo: automatically calculate CFL from sw dispersion relation
  state_n = state_in
  ref_states = get_ref_states(state_in["phi_surf"], v_grid, config)
  t = 0.0
  times = jnp.arange(0.0, end_time, dt)
  k = 0
  for t in times:
    print(f"{k/len(times-1)*100}%")
    if step_type == "euler":
      state_tmp, err_code = advance_euler(state_n, dt, h_grid, v_grid, config, dims, hydrostatic=hydrostatic, deep=False)
      state_np1 = state_tmp
    elif step_type == "ull5":
      state_tmp, err_code = ullrich_5stage(state_n, dt, h_grid, v_grid, config, dims, hydrostatic=hydrostatic, deep=False)
      state_np1 = state_tmp
    else:
      return state_n, 
    if diffusion:
      state_np1, err_code = advance_euler_hypervis(state_tmp, dt, h_grid, v_grid,
                                                   config, dims, ref_states,
                                                   n_subcycle=1, hydrostatic=hydrostatic)
    if not succeeded(err_code):
      return state_n, err_code
    state_n, state_np1 = state_np1, state_n

    # versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["u"]))))
    # versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["h"]))))
    k += 1
  return state_n, exit_codes["success"]