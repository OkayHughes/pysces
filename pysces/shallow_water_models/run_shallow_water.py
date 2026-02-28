from ..config import jnp, versatile_assert, is_main_proc
from .time_stepping import advance_step_euler, advance_step_ssprk3, advance_hypervis_euler
from .tracers import advance_tracers_shallow_water
from ..time_step import time_step_options
from sys import stdout


def simulate_shallow_water(end_time,
                           state_in,
                           grid,
                           physics_config,
                           diffusion_config,
                           timestep_config,
                           dims,
                           diffusion=True,
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
      tracer_init_struct = {"d_mass_init": state_n["h"]}
      if step_type == time_step_options.SSPRK3:
        state_tmp, tracer_consist_dyn = advance_step_ssprk3(state_n, grid, physics_config, timestep_config, dims)
      elif step_type == time_step_options.Euler:
        state_tmp, tracer_consist_dyn = advance_step_euler(state_n, grid, physics_config, timestep_config, dims)
      if diffusion:
        state_np1, tracer_consist_hypervis = advance_hypervis_euler(state_tmp, grid, physics_config, diffusion_config, timestep_config, dims)
      else:
        state_np1 = state_tmp
        tracer_consist_hypervis = None
      tracer_init_struct["d_mass_end"] = state_n["h"]
      if tracers_in is not None:
        tracers_n = advance_tracers_shallow_water(tracers_n,
                                                  tracer_consist_dyn,
                                                  tracer_init_struct,
                                                  grid,
                                                  dims,
                                                  physics_config,
                                                  diffusion_config,
                                                  timestep_config,
                                                  tracer_consist_hypervis=tracer_consist_hypervis)


      state_n, state_np1 = state_np1, state_n

      versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["horizontal_wind"]))))
      versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["h"]))))
    k += 1
  ret = {"dynamics": state_n}
  if tracers_in is not None:
    ret["tracers"] = tracers_n
  return ret


