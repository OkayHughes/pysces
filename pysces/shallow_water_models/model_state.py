from ..config import jit, jnp, do_mpi_communication
from ..mpi.global_assembly import project_scalar_global
from ..operations_2d.local_assembly import project_scalar
from functools import partial


def wrap_model_state(u,
                     h_like,
                     hs,
                     h_name="h"):
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
          h_name: h_like,
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
  if "half_h" in state.keys():
    h_name = "half_h"
  else:
    h_name = "h"
  if do_mpi_communication:
    u, v, h = project_scalar_global([state["u"][:, :, :, 0], state["u"][:, :, :, 1], state[h_name][:, :, :]],
                                    grid, dims, two_d=True)
  else:
    u = project_scalar(state["u"][:, :, :, 0], grid, dims)
    v = project_scalar(state["u"][:, :, :, 1], grid, dims)
    h = project_scalar(state[h_name][:, :, :], grid, dims)
  return wrap_model_state(jnp.stack((u, v), axis=-1), h, state["hs"], h_name=h_name)


@jit
def sum_avg_struct(struct_1, struct_2, coeff_1, coeff_2):
  struct_out = {}
  for field in struct_1.keys():
    struct_out[field] = struct_1[field] * coeff_1 + struct_2[field] * coeff_2
  return struct_out

@jit
def extract_average(state_tendency):
  out = {}
  if "half_h" in state_tendency.keys():
    out["half_h"] = state_tendency["half_h"]
    out["half_h_wind"] = state_tendency["half_h"][:, :, :, jnp.newaxis] * state_tendency["u"]
  else:
    out["h"] = state_tendency["h"]
    out["h_wind"] = state_tendency["h"][:, :, :, jnp.newaxis] * state_tendency["u"]
  out["u"] = state_tendency["u"]

  return out


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
  if "half_h" in states_in[0].keys():
    h_name = "half_h"
  else:
    h_name = "h"
  state_res = wrap_model_state(states_in[0]["u"] * coeffs[0],
                               states_in[0][h_name] * coeffs[0],
                               states_in[0]["hs"],
                               h_name=h_name)
  for state_idx in range(1, len(coeffs)):
    state = states_in[state_idx]
    coeff = coeffs[state_idx]
    state_res = wrap_model_state(state_res["u"] + state["u"] * coeff,
                                 state_res[h_name] + state[h_name] * coeff,
                                 state_res["hs"],
                                 h_name=h_name)
  return state_res


def calc_h(state):
  if "half_h" in state.keys():
    h = state["half_h"]**2
  else:
    h = state["h"]
  return h


def wrap_split_transport(state):
  return wrap_model_state(state["u"],
                          jnp.sqrt(state["h"]),
                          state["hs"],
                          h_name="half_h")


def unwrap_split_transport(state):
  return wrap_model_state(state["u"],
                          state["half_h"]**2,
                          state["hs"])
