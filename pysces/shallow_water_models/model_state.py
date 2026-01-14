from ..config import jit, jnp
from ..distributed_memory.global_assembly import project_scalar_global
from functools import partial


def create_state_struct(u, h, hs):
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
def project_state(state, grid, dims, scaled=True):
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
  u, v, h = project_scalar_global([state["u"][:, :, :, 0], state["u"][:, :, :, 1], state["h"][:, :, :]],
                                  grid, dims, two_d=True, scaled=scaled)
  return create_state_struct(jnp.stack((u, v), axis=-1), h, state["hs"])


@jit
def advance_state(states_in, coeffs):
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
  state_res = create_state_struct(states_in[0]["u"] * coeffs[0],
                                  states_in[0]["h"] * coeffs[0],
                                  states_in[0]["hs"])
  for state_idx in range(1, len(coeffs)):
    state = states_in[state_idx]
    coeff = coeffs[state_idx]
    state_res = create_state_struct(state_res["u"] + state["u"] * coeff,
                                    state_res["h"] + state["h"] * coeff,
                                    state_res["hs"])
  return state_res