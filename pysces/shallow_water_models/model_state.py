from ..config import jit, jnp, do_mpi_communication
from ..mpi.global_assembly import project_scalar_global
from ..operations_2d.local_assembly import project_scalar
from functools import partial


def wrap_model_state(u,
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
  return {"u": u,
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
    u, v, h = project_scalar_global([state["u"][:, :, :, 0], state["u"][:, :, :, 1], state["h"][:, :, :]],
                                    grid, dims, two_d=True)
  else:
    u = project_scalar(state["u"][:, :, :, 0], grid, dims)
    v = project_scalar(state["u"][:, :, :, 1], grid, dims)
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
  state_res = wrap_model_state(states_in[0]["u"] * coeffs[0],
                               states_in[0]["h"] * coeffs[0],
                               states_in[0]["hs"])
  for state_idx in range(1, len(coeffs)):
    state = states_in[state_idx]
    coeff = coeffs[state_idx]
    state_res = wrap_model_state(state_res["u"] + state["u"] * coeff,
                                 state_res["h"] + state["h"] * coeff,
                                 state_res["hs"])
  return state_res
