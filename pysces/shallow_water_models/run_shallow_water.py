from ..config import jnp, versatile_assert, is_main_proc
from .time_stepping import advance_step_euler, advance_step_ssprk3, advance_hypervis_euler, advance_tracers_rk2
from ..time_step import time_step_options
from .model_state import unwrap_split_transport, wrap_split_transport
from sys import stdout


def simulate_shallow_water(end_time,
                           state_in,
                           grid,
                           physics_config,
                           diffusion_config,
                           timestep_config,
                           dims,
                           diffusion=True,
                           split_transport=False,
                           tracers_in=None):
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
  state_n = state_in
  if tracers_in is not None:
    tracers_n = tracers_in
  t = 0.0
  times = jnp.arange(0.0, end_time, timestep_config["dt_coupling"])
  k = 0
  for t in times:
    if is_main_proc:
      print(f"{k/len(times-1)*100}%")
      stdout.flush()
    for dyn_subcycle_idx in range(timestep_config["dynamics_subcycle"]):
      step_type = timestep_config["dynamics"]["step_type"]
      if split_transport:
        state_n = wrap_split_transport(state_n)
      if step_type == time_step_options.SSPRK3:
        state_tmp, avg_struct = advance_step_ssprk3(state_n, grid, physics_config, timestep_config, dims)
      elif step_type == time_step_options.Euler:
        state_tmp, avg_struct = advance_step_euler(state_n, grid, physics_config, timestep_config, dims)
      if tracers_in is not None:
        tracers_n = advance_tracers_rk2(tracers_n, state_n, avg_struct, grid, physics_config, timestep_config, dims)
      if split_transport:
        state_tmp = unwrap_split_transport(state_tmp)

      if diffusion:
        state_np1 = advance_hypervis_euler(state_tmp, grid, physics_config, diffusion_config, timestep_config, dims)
      else:
        state_np1 = state_tmp
      state_n, state_np1 = state_np1, state_n

      versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["u"]))))
      versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["h"]))))
    k += 1
  ret = {"dynamics": state_n}
  if tracers_in is not None:
    ret["tracers"] = tracers_n
  return ret


