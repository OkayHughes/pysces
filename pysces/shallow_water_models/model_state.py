from ..config import jit, jnp, do_mpi_communication
from ..mpi.global_assembly import project_scalar_global
from ..operations_2d.local_assembly import project_scalar
from functools import partial


def wrap_model_state(horizontal_wind,
                     h,
                     hs):
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
  return {"horizontal_wind": horizontal_wind,
          "h": h,
          "hs": hs}


@partial(jit, static_argnames=["dims"])
def project_model_state(state,
                        grid,
                        dims):
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
  if do_mpi_communication:
    u, v, h = project_scalar_global([state["horizontal_wind"][:, :, :, 0], state["horizontal_wind"][:, :, :, 1], state["h"][:, :, :]],
                                    grid, dims, two_d=True)
  else:
    u = project_scalar(state["horizontal_wind"][:, :, :, 0], grid, dims)
    v = project_scalar(state["horizontal_wind"][:, :, :, 1], grid, dims)
    h = project_scalar(state["h"][:, :, :], grid, dims)
  return wrap_model_state(jnp.stack((u, v), axis=-1), h, state["hs"])


@jit
def sum_state_series(states_in,
                     coeffs):
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
  state_res = wrap_model_state(states_in[0]["horizontal_wind"] * coeffs[0],
                               states_in[0]["h"] * coeffs[0],
                               states_in[0]["hs"])
  for state_idx in range(1, len(coeffs)):
    state = states_in[state_idx]
    coeff = coeffs[state_idx]
    state_res = wrap_model_state(state_res["horizontal_wind"] + state["horizontal_wind"] * coeff,
                                 state_res["h"] + state["h"] * coeff,
                                 state_res["hs"])
  return state_res
