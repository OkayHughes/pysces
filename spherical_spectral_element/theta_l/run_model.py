from ..config import jnp
from .hyperviscosity import get_ref_states
from .time_stepping import advance_euler, advance_euler_hypervis, ullrich_5stage, advance_euler_sponge
from .model_state import remap_state

def check_nan(state):
  is_nan = False
  for field in ["u", "vtheta_dpi", "dpi", "w_i", "phi_i"]:
    is_nan = is_nan or jnp.any(jnp.isnan(state[field]))
  return is_nan

def simulate_theta(end_time, ne_min, state_in,
                   h_grid, v_grid, config,
                   dims, hydrostatic=True, deep=False,
                   diffusion=False, step_type="euler", rsplit=3, hvsplit=3, sponge_split=0, n_sponge=5):
  dt = 250.0 * (30.0 / ne_min)  # todo: automatically calculate CFL from sw dispersion relation
  state_n = state_in
  ref_states = get_ref_states(state_in["phi_surf"], v_grid, config)
  t = 0.0
  times = jnp.arange(0.0, end_time, dt)
  k = 0
  for t in times:
    print(f"{k/len(times-1)*100}%")
    if step_type == "euler":
      state_tmp = advance_euler(state_n, dt, h_grid, v_grid, config, dims, hydrostatic=hydrostatic, deep=False)
      state_np1 = state_tmp
    elif step_type == "ull5":
      state_tmp = ullrich_5stage(state_n, dt, h_grid, v_grid, config, dims, hydrostatic=hydrostatic, deep=False)
      state_np1 = state_tmp
    else:
      return state_n
    if diffusion:
      state_np1 = advance_euler_hypervis(state_tmp, dt, h_grid, v_grid,
                                         config, dims, ref_states,
                                         n_subcycle=hvsplit, hydrostatic=hydrostatic)
      if sponge_split != 0:
        state_np1 = advance_euler_sponge(state_np1, dt, h_grid, v_grid, config, dims,
                                         n_subcycle_sponge=sponge_split, n_sponge=n_sponge,
                                         hydrostatic=hydrostatic)
    #assert not check_nan(state_np1)
    if k % rsplit == 0:
      state_np1 = remap_state(state_np1, v_grid, config, len(v_grid["hybrid_b_m"]), hydrostatic=hydrostatic, deep=deep)
    state_n, state_np1 = state_np1, state_n

    # versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["u"]))))
    # versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["h"]))))
    k += 1
  return state_n
