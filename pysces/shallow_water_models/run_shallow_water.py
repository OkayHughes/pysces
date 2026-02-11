from ..config import jnp, versatile_assert, is_main_proc, jit
from .time_stepping import advance_step_euler, advance_step_ssprk3, advance_hypervis_euler
from ..time_step import time_step_options
from sys import stdout
from functools import partial

@partial(jit, static_argnames=["diffusion", "timestep_config", "dims"])
def advance_coupling_step(states_in, grid, physics_config, diffusion_config, timestep_config, dims, diffusion=True):
  states_n = states_in
  for dyn_subcycle_idx in range(timestep_config["dynamics_subcycle"]):
    step_type = timestep_config["dynamics"]["step_type"]
    if step_type == time_step_options.SSPRK3:
      states_tmp = advance_step_ssprk3(states_n, grid, physics_config, timestep_config, dims)
    elif step_type == time_step_options.Euler:
      states_tmp = advance_step_euler(states_n, grid, physics_config, timestep_config, dims)

    if diffusion:
      states_np1 = advance_hypervis_euler(states_tmp, grid, physics_config, diffusion_config, timestep_config, dims)
    else:
      states_np1 = states_tmp
    states_n, states_np1 = states_np1, states_n
  return states_n

def simulate_shallow_water(end_time,
                           states_in,
                           grid,
                           physics_config,
                           diffusion_config,
                           timestep_config,
                           dims,
                           diffusion=True):
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
  states_n = states_in
  t = 0.0
  times = jnp.arange(0.0, end_time, timestep_config["dt_coupling"])
  k = 0
  for t in times:
    if is_main_proc:
      print(f"{k/len(times-1)*100}%")
      stdout.flush()
      states_n = advance_coupling_step(states_n, grid, physics_config, diffusion_config, timestep_config, dims, diffusion=diffusion)
    k += 1
  return states_n

def init_simulator(grid,
                   physics_config,
                   diffusion_config,
                   timestep_config,
                   dims,
                   diffusion=True):
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
  def simulator(state_in):
    state_n = state_in
    t = 0.0
    while True:
      state_n = advance_coupling_step(state_n, grid, physics_config, diffusion_config, timestep_config, dims, diffusion=diffusion)
      t += timestep_config["dt_coupling"]
      yield t, state_n
  return simulator