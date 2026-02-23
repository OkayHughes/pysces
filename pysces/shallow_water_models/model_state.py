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
  if "half_h" in state.keys():
    h_name = "half_h"
  else:
    h_name = "h"
  if do_mpi_communication:
    u, v, h = project_scalar_global([state["horizontal_wind"][:, :, :, 0], state["horizontal_wind"][:, :, :, 1], state["h"][:, :, :]],
                                    grid, dims, two_d=True)
  else:
    u = project_scalar(state["horizontal_wind"][:, :, :, 0], grid, dims)
    v = project_scalar(state["horizontal_wind"][:, :, :, 1], grid, dims)
    h = project_scalar(state["h"][:, :, :], grid, dims)
  return wrap_model_state(jnp.stack((u, v), axis=-1), h, state["hs"])


@jit
def sum_avg_struct(struct_1, struct_2, coeff_1, coeff_2):
  struct_out = {}
  for field in struct_1.keys():
    struct_out[field] = struct_1[field] * coeff_1 + struct_2[field] * coeff_2
  return struct_out


@jit
def extract_average_dyn(state_in, state_tendency):
  out = {}
  out["u_d_mass_avg"] = state_in["h"] * state_in["horizontal_wind"]
  out["d_mass_tend_dyn"] = state_tendency["h"]
  return out

@jit
def extract_average_hypervis(state_in, state_tendency, diffusion_config):
  out = {}
  out["d_mass_hypervis_avg"] = state_in["h"]
  if diffusion_config["nu_d_mass"] > 0.0:
    nu = diffusion_config["nu_d_mass"]
  else:
    nu = 1.0
  out["d_mass_hypervis_tend"] = state_tendency["h"] / nu
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
