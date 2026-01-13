from ..config import jnp, versatile_assert, np, is_main_proc
from .time_stepping import advance_step_euler, advance_step_ssprk3, advance_hypervis_euler
from sys import stdout


def simulate_sw(end_time, ne, state_in, grid, config, dims, diffusion=False, step_type="ssprk3"):
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
  dt = 120.0 * (30.0 / ne)  # todo: automatically calculate CFL from sw dispersion relation
  state_n = state_in
  t = 0.0
  times = jnp.arange(0.0, end_time, dt)
  k = 0
  for t in times:
    if is_main_proc:
      print(f"{k/len(times-1)*100}%")
      stdout.flush()

    if step_type == "ssprk3":
      state_tmp = advance_step_ssprk3(state_n, dt, grid, config, dims)
    elif step_type == "euler":
      state_tmp = advance_step_euler(state_n, dt, grid, config)

    if diffusion:
      state_np1 = advance_hypervis_euler(state_tmp, dt, grid, config, dims, substeps=1)
    else:
      state_np1 = state_tmp
    state_n, state_np1 = state_np1, state_n

    versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["u"]))))
    versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["h"]))))
    k += 1
  return state_n
